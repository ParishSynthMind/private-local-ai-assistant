"""Proving "no data leaves the machine" instead of just saying it.

`offline_guard()` patches socket connections for the duration of a block: a connection to a
loopback address (127.0.0.0/8, ::1) is allowed, anything else raises OfflineViolation before a
single byte is sent, and every attempt is counted. The CLI wraps every local-provider command
in it, and the recorded live run prints the counts.

`check_local_host()` refuses an OLLAMA_HOST that is not on this machine, unless the person
has said a LAN server is intended (ALLOW_REMOTE_OLLAMA=1), because "local AI" pointed at a
remote box is not private any more.
"""
import ipaddress
import socket
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List
from urllib.parse import urlparse


class OfflineViolation(ConnectionError):
    pass


@dataclass
class NetLog:
    local: int = 0
    blocked: List[str] = field(default_factory=list)


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host.split("%")[0]).is_loopback
    except ValueError:
        return False


@contextmanager
def offline_guard(allow: tuple = ()):
    log = NetLog()
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(address):
        if isinstance(address, tuple) and address:
            host = str(address[0])
            if _is_loopback(host) or host in allow:
                log.local += 1
                return
            log.blocked.append(f"{host}:{address[1] if len(address) > 1 else ''}")
            raise OfflineViolation(f"blocked a network connection to {host}: this run is offline-only")
        # Unix sockets and anything that is not an internet address stay on the machine.

    def guarded_connect(sock, address):
        check(address)
        return real_connect(sock, address)

    def guarded_connect_ex(sock, address):
        check(address)
        return real_connect_ex(sock, address)

    def guarded_getaddrinfo(host, *args, **kwargs):
        # Refuse DNS for outside names too: a lookup alone tells a resolver what you are doing.
        if host is not None and not _is_loopback(str(host)) and str(host) not in allow:
            log.blocked.append(f"dns:{host}")
            raise OfflineViolation(f"blocked a DNS lookup for {host}: this run is offline-only")
        return real_getaddrinfo(host, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield log
    finally:
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        socket.getaddrinfo = real_getaddrinfo


def check_local_host(url: str, allow_remote: bool = False) -> str:
    host = urlparse(url if "://" in url else "http://" + url).hostname or ""
    if _is_loopback(host):
        return host
    if allow_remote:
        return host
    raise OfflineViolation(f"OLLAMA_HOST points at {host!r}, which is not this machine. Prompts and documents "
                           f"would travel over the network. Set ALLOW_REMOTE_OLLAMA=1 if that server is yours.")
