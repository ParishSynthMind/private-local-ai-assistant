"""A small web page on 127.0.0.1: ask the documents, or give the agent a task.

Bound to this machine only. Requests whose Host header is not localhost/127.0.0.1 are refused,
which stops a web page elsewhere from reaching it through DNS rebinding.
"""
import html

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ALLOWED_HOSTS = {"127.0.0.1", "localhost", "testserver"}


class AskIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list = Field(default_factory=list, max_length=12)


class AgentIn(BaseModel):
    task: str = Field(min_length=1, max_length=2000)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Private AI assistant</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:860px;margin:32px auto;padding:0 18px;
color:#1f2328;background:#fff}
h1{font-size:22px;margin:0 0 4px}.sub{color:#59636e;margin:0 0 18px;font-size:14px}
.badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600}
.local{background:#dafbe1;color:#116329}.cloud{background:#fff1e5;color:#953800}
form{display:flex;gap:8px;margin:14px 0}input{flex:1;font-size:16px;padding:10px 12px;border:1px solid #d1d9e0;
border-radius:8px}button{font-size:15px;padding:10px 16px;border-radius:8px;border:0;background:#1f6feb;color:#fff;
cursor:pointer}button.alt{background:#6639ba}.card{border:1px solid #d1d9e0;border-radius:10px;padding:14px 16px;
margin:12px 0}.q{font-weight:600;margin-bottom:6px}.src{font-size:13px;color:#59636e;margin-top:8px}
.src div{margin:2px 0}.meta{font-size:12px;color:#818b98;margin-top:8px}.step{font-family:SF Mono,Menlo,monospace;
font-size:12px;color:#59636e;margin:2px 0}.warn{color:#9a6700}
</style></head><body>
<h1>Private AI assistant</h1>
<p class="sub">__COMPANY__ documents, answered by <b>__MODEL__</b>
<span class="badge __BADGECLASS__">__BADGE__</span></p>
<form id="f"><input id="q" placeholder="Ask about the documents, or give the agent a task" autofocus>
<button type="submit">Ask</button><button type="button" class="alt" id="ag">Agent</button></form>
<div id="out"></div>
<script>
const out=document.getElementById('out'),q=document.getElementById('q');
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
async function post(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify(body)});const j=await r.json();if(!r.ok)throw new Error(j.detail||r.status);return j}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;
const card=document.createElement('div');card.className='card';card.innerHTML='<div class="q">'+esc(text)+
'</div><div>Thinking on this machine...</div>';out.prepend(card);
try{const a=await post('/api/ask',{question:text});let h='<div class="q">'+esc(text)+'</div><div>'+esc(a.answer)+'</div>';
if(a.citations.length)h+='<div class="src">'+a.citations.map(c=>'<div>['+c.id+'] '+esc(c.label)+'</div>').join('')+'</div>';
if(a.status==='answered'&&!a.grounded)h+='<div class="warn">No valid source cited: check before relying on it.</div>';
h+='<div class="meta">'+esc(a.model)+' . retrieval '+a.retrieval_seconds.toFixed(2)+'s . answer '+
a.answer_seconds.toFixed(1)+'s . '+a.input_tokens+' tokens in, '+a.output_tokens+' out</div>';card.innerHTML=h}
catch(err){card.innerHTML+='<div class="warn">'+esc(err.message)+'</div>'}};
document.getElementById('ag').onclick=async()=>{const text=q.value.trim();if(!text)return;
const card=document.createElement('div');card.className='card';card.innerHTML='<div class="q">Agent: '+esc(text)+
'</div><div>Working...</div>';out.prepend(card);
try{const r=await post('/api/agent',{task:text});let h='<div class="q">Agent: '+esc(text)+'</div>';
h+=r.steps.map(s=>'<div class="step">-> '+esc(s.tool)+'('+esc(JSON.stringify(s.arguments))+')</div>').join('');
h+='<div style="margin-top:8px">'+esc(r.answer)+'</div><div class="meta">'+esc(r.model)+' . '+r.steps.length+
' tool calls . '+r.seconds.toFixed(1)+'s . '+esc(r.stopped)+'</div>';card.innerHTML=h}
catch(err){card.innerHTML+='<div class="warn">'+esc(err.message)+'</div>'}};
</script></body></html>"""


def create_app(app) -> FastAPI:
    api = FastAPI(title="Private AI assistant", docs_url=None, redoc_url=None)

    @api.middleware("http")
    async def local_only(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ALLOWED_HOSTS and host != "::1":
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": "this assistant only answers on localhost"}, status_code=403)
        return await call_next(request)

    @api.get("/", response_class=HTMLResponse)
    def page():
        a = app.assistant.answerer
        return (PAGE.replace("__COMPANY__", html.escape(app.settings.company))
                .replace("__MODEL__", html.escape(a.model))
                .replace("__BADGECLASS__", "local" if a.local else "cloud")
                .replace("__BADGE__", "on this machine, offline" if a.local else f"cloud: {html.escape(a.provider)}"))

    @api.get("/api/health")
    def health():
        try:
            version = app.ollama.version()
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True, "ollama": version, "passages": app.store.count(), "provider": app.settings.provider,
                "local": app.assistant.answerer.local}

    @api.post("/api/ask")
    def ask(body: AskIn):
        try:
            return app.assistant.ask(body.question, history=body.history).to_dict()
        except Exception as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}")

    @api.post("/api/agent")
    def agent(body: AgentIn):
        return app.agent.run(body.task).to_dict()

    return api
