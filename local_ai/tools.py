"""The agent's tools. Deliberately small and read-only:

- list_documents     what is in the index
- search_documents   semantic search over the index (local embeddings)
- read_passage       the full text of one passage by id
- calculator         arithmetic, parsed with Python's ast (never eval)
- today              the local date

There is no shell, no file writing, no network and no path argument anywhere: a tool can only
reach passages that are already in the index, by id. A model that asks for anything else
gets a polite error back as the tool result and has to work with what it has.
"""
import ast
import datetime as dt
import math
import operator
import re
from typing import Callable, Dict, List

MAX_RESULT_CHARS = 2500


class ToolError(ValueError):
    pass


# --- calculator -----------------------------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max, "sqrt": math.sqrt}
_CONSTS = {"pi": math.pi}


def _percent(expr: str) -> str:
    """Turn percentages into plain arithmetic. "17.5% of 2340" and "17.5% * 2340" both mean
    0.175 * 2340. A bare "a % b" is refused: a real local model wrote "17.5 % 2340" for
    "17.5% of 2340", which Python would read as a remainder (17.5) and quietly get wrong."""
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s+", r"(\1/100)*", expr, flags=re.I)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*[\d(.])", r"(\1/100)", expr)
    if "%" in expr:
        raise ToolError("'%' is ambiguous here. For a percentage write it as 17.5/100*2340 or '17.5% of 2340'.")
    return expr


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 100 or abs(left) > 1e6):
            raise ToolError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ToolError(f"not allowed in a calculation: {ast.dump(node)[:60]}")


def calculate(expression: str) -> str:
    expression = str(expression or "").strip()
    if not expression:
        raise ToolError("empty expression")
    if len(expression) > 200:
        raise ToolError("expression too long")
    cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
    cleaned = _percent(re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned))  # 1,200 -> 1200; round(x, 2) untouched
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        raise ToolError(f"could not read {expression!r} as arithmetic")
    try:
        value = _eval(tree)
    except ZeroDivisionError:
        raise ToolError("division by zero")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError("result is not a finite number")
        value = round(value, 10)
        if value.is_integer():
            value = int(value)
    return f"{expression} = {value}"


# --- the tool table -------------------------------------------------------------------------
def tool_specs() -> List[dict]:
    """JSON-schema tool definitions in the shape Ollama and OpenAI share; the Claude backend
    converts them (input_schema instead of parameters)."""
    def fn(name, description, props, required):
        return {"type": "function", "function": {"name": name, "description": description, "parameters": {
            "type": "object", "properties": props, "required": required}}}

    return [
        fn("list_documents", "List the documents that are indexed, with how many passages each has.", {}, []),
        fn("search_documents", "Search the indexed documents. Returns passages with ids and sources. "
           "Use this before answering any question about the documents.",
           {"query": {"type": "string", "description": "what to look for, in plain words"},
            "k": {"type": "integer", "description": "how many passages, 1 to 8 (default 4)"}}, ["query"]),
        fn("read_passage", "Read the full text of one passage by the id that search_documents returned.",
           {"passage_id": {"type": "integer"}}, ["passage_id"]),
        fn("calculator", "Do arithmetic exactly. Supports + - * / ** ( ) round abs min max sqrt, and "
           "percentages written as '17.5% of 2340'. Use it for every calculation instead of working it out yourself.",
           {"expression": {"type": "string"}}, ["expression"]),
        fn("today", "Today's date on this computer.", {}, []),
    ]


class Toolbox:
    def __init__(self, store, today: Callable[[], dt.date] = dt.date.today):
        self.store = store
        self._today = today
        self.handlers: Dict[str, Callable[[dict], str]] = {
            "list_documents": self._list,
            "search_documents": self._search,
            "read_passage": self._read,
            "calculator": lambda a: calculate(a.get("expression", "")),
            "today": lambda a: self._today().isoformat(),
        }
        self.sources_seen: List[str] = []

    @property
    def names(self) -> List[str]:
        return list(self.handlers)

    def run(self, name: str, arguments: dict) -> str:
        """Run a tool and return its text. Never raises: errors go back to the model as text."""
        if name not in self.handlers:
            return f"ERROR: there is no tool called {name!r}. Available tools: {', '.join(self.names)}."
        if not isinstance(arguments, dict) or "_raw" in arguments:
            return "ERROR: the arguments must be a JSON object."
        try:
            out = self.handlers[name](arguments)
        except ToolError as e:
            return f"ERROR: {e}"
        except Exception as e:  # a bug in a tool must not crash the agent loop
            return f"ERROR: {type(e).__name__}: {e}"
        return out if len(out) <= MAX_RESULT_CHARS else out[:MAX_RESULT_CHARS] + "\n[truncated]"

    def _list(self, args) -> str:
        files = self.store.files()
        return "\n".join(f"- {f['path']} ({f['chunks']} passages)" for f in files) or "No documents are indexed."

    def _search(self, args) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("query is required")
        try:
            k = int(args.get("k") or 4)
        except (TypeError, ValueError):
            k = 4
        hits = self.store.search(query, k=max(1, min(8, k)))
        if not hits:
            return "No passages matched."
        lines = []
        for h in hits:
            self.sources_seen.append(h.chunk.source)
            lines.append(f"[passage {h.chunk_id}] {h.chunk.label} (score {h.score:.2f})\n{h.chunk.text[:600]}")
        return "\n\n".join(lines)

    def _read(self, args) -> str:
        try:
            pid = int(args.get("passage_id"))
        except (TypeError, ValueError):
            raise ToolError("passage_id must be a number from search_documents")
        chunk = self.store.get(pid)
        if chunk is None:
            raise ToolError(f"no passage {pid}")
        self.sources_seen.append(chunk.source)
        return f"[passage {pid}] {chunk.label}\n{chunk.text}"
