#!/bin/bash
# Docker/Podman healthcheck wrapper.
#
# Docker HEALTHCHECK requires the CMD to be either:
#   a) an exec-form shell string:  CMD ["/path/to/script"]
#   b) a shell-form string:        CMD /path/to/script
#
# Both forms run the script WITHOUT a shell wrapper, so any `set -e` or
# environment differences between shell invocations do not affect behaviour.
# This wrapper simply exports HERMES_HOME and delegates to the Python probe.

set -u  # fail on unbound variables; harmless here

HERMES_HOME="${HERMES_HOME:-/opt/data}"
export HERMES_HOME

exec python3 /opt/hermes/docker/healthcheck_probe.py
