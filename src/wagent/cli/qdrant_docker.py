"""Manage the Qdrant Docker container lifecycle.

Provides helpers to check, start, stop, and health-check the Qdrant container
used by wagent in server mode.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time

import urllib.request
import urllib.error

from wagent.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CONTAINER_NAME = "wagent-qdrant"
QDRANT_IMAGE = "qdrant/qdrant"
HOST_PORT = 6333
STORAGE_DIR = PROJECT_ROOT / "data" / "qdrant_storage"
HEALTH_URL = f"http://localhost:{HOST_PORT}/healthz"
HEALTH_TIMEOUT = 30


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _run(cmd: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


def container_running() -> bool:
    if not _docker_available():
        return False
    try:
        r = _run(["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME])
        return r.stdout.strip() == "true"
    except subprocess.CalledProcessError:
        return False


def container_exists() -> bool:
    if not _docker_available():
        return False
    try:
        _run(["docker", "inspect", CONTAINER_NAME])
        return True
    except subprocess.CalledProcessError:
        return False


def start_container() -> None:
    """Start the Qdrant Docker container, creating it if necessary."""
    if not _docker_available():
        raise RuntimeError(
            "Docker is not installed or not on PATH. "
            "Install Docker or set QDRANT_URL='' in .env to use local file mode."
        )

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    if container_running():
        logger.info("Qdrant container '%s' is already running", CONTAINER_NAME)
        return

    if container_exists():
        logger.info("Starting existing Qdrant container '%s'...", CONTAINER_NAME)
        _run(["docker", "start", CONTAINER_NAME])
    else:
        logger.info("Creating and starting Qdrant container '%s'...", CONTAINER_NAME)
        _run([
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{HOST_PORT}:6333",
            "-v", f"{STORAGE_DIR}:/qdrant/storage",
            "--restart", "unless-stopped",
            QDRANT_IMAGE,
        ])

    _wait_healthy()


def stop_container() -> None:
    if container_running():
        logger.info("Stopping Qdrant container '%s'...", CONTAINER_NAME)
        _run(["docker", "stop", CONTAINER_NAME], check=False)
    else:
        logger.info("Qdrant container '%s' is not running", CONTAINER_NAME)


def remove_container() -> None:
    stop_container()
    if container_exists():
        logger.info("Removing Qdrant container '%s'...", CONTAINER_NAME)
        _run(["docker", "rm", CONTAINER_NAME], check=False)


def _wait_healthy() -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    logger.info("Qdrant health check passed")
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError(
        f"Qdrant did not become healthy within {HEALTH_TIMEOUT}s. "
        f"Check: docker logs {CONTAINER_NAME}"
    )


def health_check() -> bool:
    try:
        req = urllib.request.Request(HEALTH_URL, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def status_info() -> dict[str, str]:
    """Return a dict with container status information."""
    info = {"container": CONTAINER_NAME, "running": "no", "healthy": "no"}
    if container_running():
        info["running"] = "yes"
        info["healthy"] = "yes" if health_check() else "no"
    return info
