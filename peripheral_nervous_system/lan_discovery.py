"""
lan_discovery.py — find the desktop senses companion on the LAN when its IP has moved.

The desktop's DHCP lease can change between sessions, which breaks the hardcoded DESKTOP_IP in
start_stream_servers.bat (you'd see "<urlopen error timed out>" and Mira goes blind/deaf). Rather
than force a manual edit each time, the laptop rescans its own /24 for the companion — a host that
answers GET /health with "ok" on the senses port — and self-heals. Used by stream_vision (frames)
and game_audio (dialogue) when their configured URL stops responding.

Scan strategy: a fast TCP connect sweep of the /24 (cheap, parallel), then an HTTP /health verify
on just the hosts that had the port open — so another service on :8200 can't be mistaken for the
companion. Disable with MIRA_SENSES_AUTODISCOVER=0.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import socket
import urllib.request
from typing import Optional
from urllib.parse import urlparse, urlunparse


def enabled() -> bool:
    return os.environ.get("MIRA_SENSES_AUTODISCOVER", "1").strip().lower() not in ("0", "false", "no", "")


def local_ipv4() -> Optional[str]:
    """The laptop's primary LAN IPv4 — the interface that routes out (e.g. 192.168.12.116), NOT
    the WSL/loopback adapters. Uses a UDP socket's chosen source address (no traffic is sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def host_of(url: str) -> Optional[str]:
    try:
        return urlparse(url).hostname
    except Exception:
        return None


def port_of(url: str, default: int = 8200) -> int:
    try:
        return urlparse(url).port or default
    except Exception:
        return default


def repoint_url(url: str, new_host: str) -> str:
    """Swap the host in a URL, keeping scheme/port/path:
    http://OLD:8200/frame  ->  http://NEW:8200/frame."""
    p = urlparse(url)
    netloc = new_host + (f":{p.port}" if p.port else "")
    return urlunparse((p.scheme, netloc, p.path, p.params, p.query, p.fragment))


def _port_open(host: str, port: int, timeout: float) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


def _is_companion(host: str, port: int, timeout: float) -> bool:
    """Confirm it's really the senses companion (GET /health -> 'ok'), not some other :port."""
    try:
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never via a proxy on LAN
        with op.open(f"http://{host}:{port}/health", timeout=timeout) as r:
            return r.read(8).strip().lower() == b"ok"
    except Exception:
        return False


def find_companion(port: int = 8200, exclude: Optional[str] = None,
                   timeout: float = 0.5) -> Optional[str]:
    """Scan the laptop's /24 for the senses companion. Returns the host IP, or None.
    `exclude` (the known-dead IP) is verified last. Blocks ~1-2s."""
    me = local_ipv4()
    if not me or me.count(".") != 3:
        return None
    prefix = me.rsplit(".", 1)[0] + "."
    candidates = [prefix + str(i) for i in range(1, 255) if prefix + str(i) != me]
    open_hosts = []
    with cf.ThreadPoolExecutor(max_workers=128) as ex:
        for host, is_open in zip(candidates, ex.map(lambda h: _port_open(h, port, timeout), candidates)):
            if is_open:
                open_hosts.append(host)
    # verify /health; check non-excluded hosts before the known-dead one
    for host in sorted(open_hosts, key=lambda h: h == exclude):
        if _is_companion(host, port, max(timeout, 1.0)):
            return host
    return None
