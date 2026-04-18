"""Tests for docker/healthcheck_probe.py — mode detection, gateway probe, dashboard probe."""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

# Make the probe importable without a /proc dependency at collection time.
PROBE = sys.modules["docker.healthcheck_probe"]


# ---------------------------------------------------------------------------
# detect_mode — unit tests
# ---------------------------------------------------------------------------

class TestDetectMode:
    @pytest.mark.parametrize("argv,expected", [
        # gateway run variants
        (["hermes", "gateway", "run"], "gateway_run"),
        (["python", "-m", "hermes_cli.main", "gateway", "run"], "gateway_run"),
        (["/opt/hermes/.venv/bin/python", "hermes", "gateway", "run"], "gateway_run"),
        (["hermes", "gateway", "run", "--profile", "test"], "gateway_run"),
        # dashboard variants
        (["hermes", "dashboard"], "dashboard"),
        (["python", "-m", "hermes_cli.main", "dashboard"], "dashboard"),
        (["python", "-m", "hermes_cli.main", "dashboard", "--port", "8080"], "dashboard"),
        (["hermes", "dashboard", "--host", "0.0.0.0", "--port", "9119"], "dashboard"),
        # other commands — must NOT match
        (["hermes", "gateway", "status"], "other"),
        (["hermes", "gateway", "stop"], "other"),
        (["hermes", "gateway", "restart"], "other"),
        (["hermes", "chat"], "other"),
        (["hermes", "setup"], "other"),
        (["hermes", "version"], "other"),
        (["hermes", "web"], "other"),
        # empty / None
        ([], "other"),
        (None, "other"),
    ])
    def test_detect_mode(self, argv, expected):
        assert PROBE.detect_mode(argv) == expected


# ---------------------------------------------------------------------------
# pid1_alive_not_zombie — patched
# ---------------------------------------------------------------------------

class TestPid1AliveNotZombie:
    def test_returns_true_when_alive(self):
        with mock.patch.object(PROBE.Path, "read_text", return_value="State:\tR (running)\n"):
            assert PROBE.pid1_alive_not_zombie() is True

    def test_returns_false_when_zombie(self):
        with mock.patch.object(PROBE.Path, "read_text", return_value="State:\tZ (zombie)\n"):
            assert PROBE.pid1_alive_not_zombie() is False

    def test_returns_false_when_status_file_missing(self):
        with mock.patch.object(PROBE.Path, "read_text", side_effect=FileNotFoundError):
            assert PROBE.pid1_alive_not_zombie() is False

    def test_returns_true_when_cannot_parse(self):
        """Best-effort: if we can't read the state, assume alive (generous default)."""
        with mock.patch.object(PROBE.Path, "read_text", return_value=""):
            assert PROBE.pid1_alive_not_zombie() is True


# ---------------------------------------------------------------------------
# healthy_gateway — patched PID + filesystem
# ---------------------------------------------------------------------------

class TestHealthyGateway:
    def _make_state(self, gateway_state: str, pid: int = 1) -> dict:
        return {"gateway_state": gateway_state, "pid": pid}

    def _patch(self, tmp_path, state: dict | None, pid1_alive: bool = True):
        state_file = tmp_path / "gateway_state.json"
        if state is not None:
            state_file.write_text(json.dumps(state), encoding="utf-8")

        healthcheck = sys.modules["docker.healthcheck_probe"]
        return mock.patch.multiple(
            healthcheck,
            GATEWAY_STATE_FILE=state_file,
            pid1_alive_not_zombie=lambda: pid1_alive,
        )

    def test_healthy_when_running_state_and_pid_1(self, tmp_path):
        state = self._make_state("running", pid=1)
        with self._patch(tmp_path, state):
            assert PROBE.healthy_gateway() is True

    def test_unhealthy_when_gateway_state_is_starting(self, tmp_path):
        state = self._make_state("starting", pid=1)
        with self._patch(tmp_path, state):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_gateway_state_is_startup_failed(self, tmp_path):
        state = self._make_state("startup_failed", pid=1)
        with self._patch(tmp_path, state):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_gateway_state_is_draining(self, tmp_path):
        state = self._make_state("draining", pid=1)
        with self._patch(tmp_path, state):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_gateway_state_is_stopped(self, tmp_path):
        state = self._make_state("stopped", pid=1)
        with self._patch(tmp_path, state):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_recorded_pid_is_not_1(self, tmp_path):
        state = self._make_state("running", pid=os.getpid())
        with self._patch(tmp_path, state):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_pid1_is_zombie(self, tmp_path):
        state = self._make_state("running", pid=1)
        with self._patch(tmp_path, state, pid1_alive=False):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_state_file_missing(self, tmp_path):
        with self._patch(tmp_path, None):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_state_file_empty(self, tmp_path):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text("", encoding="utf-8")
        with mock.patch.multiple(PROBE, GATEWAY_STATE_FILE=state_file, pid1_alive_not_zombie=lambda: True):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_state_file_not_valid_json(self, tmp_path):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text("not json {", encoding="utf-8")
        with mock.patch.multiple(PROBE, GATEWAY_STATE_FILE=state_file, pid1_alive_not_zombie=lambda: True):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_state_is_not_a_dict(self, tmp_path):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text('"just a string"', encoding="utf-8")
        with mock.patch.multiple(PROBE, GATEWAY_STATE_FILE=state_file, pid1_alive_not_zombie=lambda: True):
            assert PROBE.healthy_gateway() is False

    def test_unhealthy_when_pid_field_missing(self, tmp_path):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text(json.dumps({"gateway_state": "running"}), encoding="utf-8")
        with mock.patch.multiple(PROBE, GATEWAY_STATE_FILE=state_file, pid1_alive_not_zombie=lambda: True):
            assert PROBE.healthy_gateway() is False


# ---------------------------------------------------------------------------
# parse_host_port — no /proc needed
# ---------------------------------------------------------------------------

class TestParseHostPort:
    @pytest.mark.parametrize("argv,expected_host,expected_port", [
        # defaults
        (None, "127.0.0.1", 9119),
        ([], "127.0.0.1", 9119),
        # explicit --host / --port
        (["hermes", "dashboard", "--host", "0.0.0.0", "--port", "8080"], "127.0.0.1", 8080),
        (["hermes", "dashboard", "--port", "9999"], "127.0.0.1", 9999),
        (["hermes", "dashboard", "--host", "192.168.1.100"], "192.168.1.100", 9119),
        # equals-form
        (["hermes", "dashboard", "--host=0.0.0.0", "--port=7070"], "127.0.0.1", 7070),
        # mixed
        (["hermes", "dashboard", "--port", "5000", "--host", "127.0.0.1"], "127.0.0.1", 5000),
    ])
    def test_parse_host_port(self, argv, expected_host, expected_port):
        host, port = PROBE.parse_host_port(argv)
        assert host == expected_host
        assert port == expected_port


# ---------------------------------------------------------------------------
# healthy_generic — patched
# ---------------------------------------------------------------------------

class TestHealthyGeneric:
    def test_returns_true_when_pid1_alive(self):
        with mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=True):
            assert PROBE.healthy_generic() is True

    def test_returns_false_when_pid1_zombie(self):
        with mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=False):
            assert PROBE.healthy_generic() is False


# ---------------------------------------------------------------------------
# main() integration — end-to-end with all mocks wired
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_gateway_run_healthy(self, tmp_path, monkeypatch):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text(json.dumps({"gateway_state": "running", "pid": 1}), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        # Patch both probes so /proc is never touched
        with mock.patch.object(
            PROBE, "read_pid1_cmdline", return_value=["hermes", "gateway", "run"]
        ), mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=True):
            exit_code = PROBE.main()

        assert exit_code == 0

    def test_main_gateway_run_unhealthy_startup_failed(self, tmp_path, monkeypatch):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text(json.dumps({"gateway_state": "startup_failed", "pid": 1}), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with mock.patch.object(
            PROBE, "read_pid1_cmdline", return_value=["hermes", "gateway", "run"]
        ), mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=True):
            exit_code = PROBE.main()

        assert exit_code == 1

    def test_main_dashboard_healthy(self, tmp_path, monkeypatch):
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text("{}", encoding="utf-8")  # must exist for module import
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        with mock.patch.object(
            PROBE, "read_pid1_cmdline", return_value=["hermes", "dashboard", "--port", "9119"]
        ), mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=True), \
             mock.patch.object(
                 PROBE, "urllib.request.urlopen",
                 return_value=mock.MagicMock(status=200)
             ):
            exit_code = PROBE.main()

        assert exit_code == 0

    def test_main_other_command_pid1_alive(self, monkeypatch):
        with mock.patch.object(
            PROBE, "read_pid1_cmdline", return_value=["hermes", "chat"]
        ), mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=True):
            exit_code = PROBE.main()

        assert exit_code == 0

    def test_main_other_command_pid1_zombie(self, monkeypatch):
        with mock.patch.object(
            PROBE, "read_pid1_cmdline", return_value=["hermes", "chat"]
        ), mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=False):
            exit_code = PROBE.main()

        assert exit_code == 1

    def test_main_env_override_url(self, tmp_path, monkeypatch):
        """HERMES_DASHBOARD_HEALTH_URL takes precedence over parsed --port."""
        state_file = tmp_path / "gateway_state.json"
        state_file.write_text("{}", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_DASHBOARD_HEALTH_URL", "http://custom:9999/status")

        with mock.patch.object(
            PROBE, "read_pid1_cmdline", return_value=["hermes", "dashboard", "--port", "9119"]
        ), mock.patch.object(PROBE, "pid1_alive_not_zombie", return_value=True), \
             mock.patch.object(
                 PROBE, "urllib.request.urlopen",
                 return_value=mock.MagicMock(status=200)
             ) as mock_urlopen:
            PROBE.main()

        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        assert call_args[0][0].full_url == "http://custom:9999/status"
