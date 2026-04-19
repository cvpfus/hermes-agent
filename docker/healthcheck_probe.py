#!/usr/bin/env python3
"""
Mode-aware Docker healthcheck probe for Hermes Agent.

Reads PID 1's argv to determine which Hermes mode is active, then applies the
appropriate liveness strategy:

  gateway run  -> read HERMES_HOME/gateway_state.json; healthy only when
                  gateway_state == "running" AND recorded pid == 1 AND PID 1
                  is not a zombie
  dashboard    -> HTTP GET to the local dashboard /api/status; healthy on 2xx
  _default     -> PID 1 exists and is not a zombie (conservative fallback)

Exits 0 (healthy) or 1 (unhealthy).  All decisions are logged to stderr for
operator visibility.
"""

import json
import os
import stat
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/opt/data"))
GATEWAY_STATE_FILE = HERMES_HOME / "gateway_state.json"
DASHBOARD_DEFAULT_PORT = 9119
DASHBOARD_DEFAULT_HOST = "127.0.0.1"
HEALTHCHECK_TIMEOUT_SECONDS = 10


# ---------------------------------------------------------------------------
# Low-level PID 1 inspection
# ---------------------------------------------------------------------------

def read_pid1_cmdline() -> Optional[list[str]]:
    """Return PID 1's argv as a list of strings, or None on error."""
    try:
        raw = Path("/proc/1/cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not raw:
        return None
    # /proc/<pid>/cmdline is null-separated; split and strip trailing empty
    parts = raw.replace(b"\x00", b"\n").decode("utf-8", errors="ignore").split("\n")
    # The last element is usually an empty string after the final \x00; drop it
    while parts and parts[-1] == "":
        parts.pop()
    return parts if parts else None


def pid1_alive_not_zombie() -> bool:
    """
    Return True when PID 1 exists and is not in a zombie state.

    We check the 'State' field in /proc/1/status.  A zombie has 'Z' as the
    first character.  This is a best-effort supplemental check — even if we
    misdetect, the gateway_state / HTTP probe will still fail and report
    unhealthy, which is the safe outcome.
    """
    try:
        status = Path("/proc/1/status").read_text(errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        return False
    for line in status.splitlines():
        if line.startswith("State:"):
            state_char = line.split()[1] if len(line.split()) > 1 else ""
            return state_char != "Z"
    return True  # unable to determine — assume alive


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def detect_mode(argv: Optional[list[str]]) -> str:
    """
    Classify the active Hermes mode from PID 1's argv.

    'gateway run' mode is identified by the adjacent token pair "gateway" then
    "run" anywhere in argv — this reliably distinguishes `hermes gateway run`
    from sibling commands like `hermes gateway status` or `hermes gateway stop`.

    'dashboard' mode is identified by the token "dashboard" anywhere in argv.

    All other invocations fall through to the generic PID-level check.
    """
    if not argv:
        return "other"

    # Normalise: strip leading path components so "python /opt/hermes/..." works
    tokens = []
    for tok in argv:
        tok = tok.strip()
        if not tok:
            continue
        # Keep only the basename of argv[0] (the executable) and full tokens
        # for argv[i>0] — the gateway/dashboard subcommand is always a full
        # token, never a path.
        tokens.append(tok)

    # Look for "gateway" followed by "run" as adjacent tokens
    for i, tok in enumerate(tokens):
        if tok == "gateway" and i + 1 < len(tokens) and tokens[i + 1] == "run":
            return "gateway_run"
        if tok == "dashboard":
            return "dashboard"

    return "other"


# ---------------------------------------------------------------------------
# Gateway-mode probe
# ---------------------------------------------------------------------------

def healthy_gateway() -> bool:
    """
    Probe the gateway runtime status file.

    Returns True only when ALL of the following hold:
      1. gateway_state.json exists and is readable
      2. It parses as a JSON object
      3. Its ``gateway_state`` field equals "running"
      4. Its recorded ``pid`` field equals 1  (validates we are the right host)
      5. PID 1 itself is not a zombie (supplemental safety)

    Any error — missing file, parse failure, wrong state, wrong pid — returns
    False (unhealthy).  This means the container will report unhealthy during
    ``starting``, ``startup_failed``, ``draining``, and ``stopped`` states,
    which is the intended behaviour so Docker restarts / Compose health
    dependencies resolve correctly.
    """
    if not pid1_alive_not_zombie():
        print("gateway: PID 1 is zombie or missing", file=sys.stderr)
        return False

    if not GATEWAY_STATE_FILE.exists():
        print("gateway: state file not found", file=sys.stderr)
        return False

    try:
        raw = GATEWAY_STATE_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"gateway: state file read error: {exc}", file=sys.stderr)
        return False

    if not raw:
        print("gateway: state file is empty", file=sys.stderr)
        return False

    try:
        state = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"gateway: state file parse error: {exc}", file=sys.stderr)
        return False

    if not isinstance(state, dict):
        print("gateway: state file is not a JSON object", file=sys.stderr)
        return False

    gateway_state = state.get("gateway_state")
    if gateway_state != "running":
        print(f"gateway: gateway_state={gateway_state!r} (want 'running')", file=sys.stderr)
        return False

    recorded_pid = state.get("pid")
    if recorded_pid != 1:
        print(f"gateway: recorded pid={recorded_pid!r} (want 1)", file=sys.stderr)
        return False

    print("gateway: healthy (running, pid=1)", file=sys.stderr)
    return True


# ---------------------------------------------------------------------------
# Dashboard-mode probe
# ---------------------------------------------------------------------------

def parse_host_port(argv: Optional[list[str]]) -> tuple[str, int]:
    """
    Extract --host and --port from the dashboard argv, with sensible defaults.

    If --host is 0.0.0.0 we still probe 127.0.0.1 because the dashboard binds
    to all interfaces but is only reachable from inside the container via
    loopback.  If HERMES_DASHBOARD_HEALTH_URL is set in the environment it
    takes precedence over the derived URL.
    """
    override = os.environ.get("HERMES_DASHBOARD_HEALTH_URL", "").strip()
    if override:
        return override, 0  # port=0 signals "use override as full URL"

    host = DASHBOARD_DEFAULT_HOST
    port = DASHBOARD_DEFAULT_PORT

    if not argv:
        return host, port

    i = 0
    tokens = list(argv)
    while i < len(tokens):
        tok = tokens[i]
        if tok == "--host" and i + 1 < len(tokens):
            host = tokens[i + 1]
            i += 2
        elif tok == "--port" and i + 1 < len(tokens):
            try:
                port = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
        elif tok.startswith("--host="):
            host = tok.split("=", 1)[1]
            i += 1
        elif tok.startswith("--port="):
            try:
                port = int(tok.split("=", 1)[1])
            except ValueError:
                pass
            i += 1
        else:
            i += 1

    # Normalise 0.0.0.0 -> 127.0.0.1 for loopback probing
    if host == "0.0.0.0":
        host = DASHBOARD_DEFAULT_HOST

    return host, port


def healthy_dashboard(argv: Optional[list[str]]) -> bool:
    """
    Probe the local dashboard's /api/status endpoint.

    Sends an HTTP GET to the derived (or environment-overridden) URL with a
    short timeout.  Returns True only on a 2xx response.  Any network error,
    timeout, or non-2xx status is treated as unhealthy.
    """
    host, port = parse_host_port(argv)
    override = os.environ.get("HERMES_DASHBOARD_HEALTH_URL", "")
    if host == "" or override:
        # Already a full URL — use it directly
        url = override
    else:
        url = f"http://{host}:{port}/api/status"

    print(f"dashboard: probing {url}", file=sys.stderr)

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=HEALTHCHECK_TIMEOUT_SECONDS) as resp:
            if 200 <= resp.status < 300:
                print("dashboard: healthy (2xx)", file=sys.stderr)
                return True
            print(f"dashboard: got HTTP {resp.status}", file=sys.stderr)
            return False
    except urllib.error.URLError as exc:
        print(f"dashboard: connection error: {exc}", file=sys.stderr)
        return False
    except OSError as exc:
        print(f"dashboard: OS error: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

def healthy_generic() -> bool:
    """Conservative fallback: PID 1 exists and is not a zombie."""
    alive = pid1_alive_not_zombie()
    if alive:
        print("generic: PID 1 alive", file=sys.stderr)
    else:
        print("generic: PID 1 missing or zombie", file=sys.stderr)
    return alive


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    argv = read_pid1_cmdline()
    mode = detect_mode(argv)

    if mode == "gateway_run":
        healthy = healthy_gateway()
    elif mode == "dashboard":
        healthy = healthy_dashboard(argv)
    else:
        healthy = healthy_generic()

    print(f"healthcheck: mode={mode} healthy={healthy}", file=sys.stderr)
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
