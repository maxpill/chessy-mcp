"""Client-IP detection from ASGI scope.

Extracted from :mod:\`mcp_server.middleware.request_logger\`. Owns:

  * :func:\`is_trusted_proxy_peer\` — true for loopback / private IPs.
  * :func:\`effective_client_ip\` — picks the X-Forwarded-For value when
    the peer is a trusted proxy, falls back to the raw peer.
"""

from __future__ import annotations

import ipaddress


def is_trusted_proxy_peer(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def effective_client_ip(peer_ip: str, forwarded_for: str) -> str:
    peer = peer_ip.strip().strip("[]")
    if forwarded_for and is_trusted_proxy_peer(peer):
        candidate = forwarded_for.split(",", 1)[0].strip().strip("[]")
        return candidate or peer
    return peer


# Back-compat shims.
_is_trusted_proxy_peer = is_trusted_proxy_peer
_effective_client_ip = effective_client_ip
