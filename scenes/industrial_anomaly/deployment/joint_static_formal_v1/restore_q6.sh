#!/usr/bin/env bash
set -euo pipefail
systemctl --user disable --now cloud-edge-edge-18101-joint-static-q5-v1.service || true
systemctl --user enable --now cloud-edge-edge-18101-51d796d-rollback.service
