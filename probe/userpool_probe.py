"""Read-only probe of the hosted user-pool MCP: confirm access and list the real persona
pools available to this account. Uses only free, read-only tools (no distribution simulation,
which is the metered one). Mirrors the MCP call pattern in socioverse/external_events.py.
"""
import json
import pathlib
import sys
import urllib.request

ENV = pathlib.Path("SocioVerse2/.env")
if not ENV.exists():
    ENV = pathlib.Path(".env")


def env(name):
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.startswith(name + "="):
            return line.split("=", 1)[1].strip()
    return None


URL = env("SV_USER_POOL_MCP_URL")
KEY = env("SV_USER_POOL_MCP_KEY")
_id = 0
SESSION = None           # Mcp-Session-Id, set by initialize
PROTO = "2024-11-05"


def _post(body, notify=False, timeout=60):
    global SESSION
    data = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream",
               "Authorization": f"Bearer {KEY}",
               "MCP-Protocol-Version": PROTO}
    if SESSION:
        headers["Mcp-Session-Id"] = SESSION
    req = urllib.request.Request(URL, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        sid = r.headers.get("Mcp-Session-Id") or r.headers.get("mcp-session-id")
        if sid:
            SESSION = sid
        raw = r.read().decode("utf-8")
    if notify:
        return None
    if raw.lstrip().startswith("data:"):
        raw = "\n".join(l[5:].strip() for l in raw.splitlines()
                        if l.strip().startswith("data:"))
    return json.loads(raw) if raw.strip() else {}


def handshake():
    global _id
    _id += 1
    init = _post({"jsonrpc": "2.0", "id": _id, "method": "initialize",
                  "params": {"protocolVersion": PROTO, "capabilities": {},
                             "clientInfo": {"name": "evolutics", "version": "0.1"}}})
    _post({"jsonrpc": "2.0", "method": "notifications/initialized"}, notify=True)
    return init


def rpc(method, params=None, timeout=60):
    global _id
    _id += 1
    body = {"jsonrpc": "2.0", "id": _id, "method": method, "params": params or {}}
    return _post(body, timeout=timeout)


def call_tool(name, arguments):
    resp = rpc("tools/call", {"name": name, "arguments": arguments})
    if resp.get("error"):
        raise RuntimeError(f"MCP error: {resp['error']}")
    result = resp.get("result") or {}
    content = result.get("content") or []
    text = content[0].get("text", "") if content else ""
    if result.get("isError"):
        raise RuntimeError(f"tool error: {text[:300]}")
    return json.loads(text) if text else result


def main():
    print(f"endpoint: {URL}")
    try:
        init = handshake()
        info = (init.get("result") or {}).get("serverInfo", {})
        print(f"handshake OK — server {info}  session={'yes' if SESSION else 'no'}")
    except Exception as e:
        print(f"handshake failed: {type(e).__name__}: {e}")
    # 1) list the tools the server exposes
    try:
        tl = rpc("tools/list")
        tools = [t.get("name") for t in (tl.get("result") or {}).get("tools", [])]
        print(f"tools exposed ({len(tools)}): {tools}")
    except Exception as e:
        print(f"tools/list failed: {type(e).__name__}: {e}")
        tools = []

    # 2) list registered persona platforms (read-only, free)
    for tool in ("list_user_pool_platforms", "get_user_pool_simulation_capabilities",
                 "get_user_pool_index_status"):
        if tools and tool not in tools:
            continue
        print(f"\n=== {tool} ===")
        try:
            out = call_tool(tool, {})
            print(json.dumps(out, ensure_ascii=False, indent=2)[:2500])
        except Exception as e:
            print(f"  {type(e).__name__}: {e}")


if __name__ == "__main__":
    if not URL or not KEY:
        sys.exit("SV_USER_POOL_MCP_URL / _KEY not found in .env")
    main()
