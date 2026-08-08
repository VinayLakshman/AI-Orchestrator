from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)


class DockerError(RuntimeError):
    pass


@dataclass(slots=True)
class ContainerStatus:
    running: bool
    exists: bool
    name: str


class DockerRuntime:
    """Thin abstraction over Docker CLI interactions.

    The lifecycle manager depends on this interface instead of invoking
    `docker` directly. This keeps process-execution details isolated from
    orchestration logic.
    """

    def __init__(self, *, docker_cmd: str = "docker") -> None:
        self.docker_cmd = docker_cmd

    async def _run(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            self.docker_cmd,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return (
            proc.returncode or 0,
            stdout.decode("utf-8", errors="replace").strip(),
            stderr.decode("utf-8", errors="replace").strip(),
        )

    async def start(self, container_name: str) -> None:
        code, _, stderr = await self._run("start", container_name)
        if code != 0:
            raise DockerError(f"Failed to start container {container_name!r}: {stderr}")
        logger.info("docker_container_started container=%s", container_name)

    async def stop(self, container_name: str) -> None:
        code, _, stderr = await self._run("stop", container_name)
        if code != 0 and "not running" not in stderr.lower():
            raise DockerError(f"Failed to stop container {container_name!r}: {stderr}")
        logger.info("docker_container_stopped container=%s", container_name)

    async def status(self, container_name: str) -> ContainerStatus:
        code, stdout, stderr = await self._run("inspect", "--format", "{{.State.Running}}", container_name)
        if code != 0:
            return ContainerStatus(running=False, exists=False, name=container_name)
        running = stdout.strip().lower() == "true"
        return ContainerStatus(running=running, exists=True, name=container_name)

    async def exists(self, container_name: str) -> bool:
        status = await self.status(container_name)
        return status.exists

    async def wait_stopped(self, container_name: str, *, timeout_s: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            status = await self.status(container_name)
            if not status.exists or not status.running:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise DockerError(f"Timed out waiting for container {container_name!r} to stop")
            await asyncio.sleep(0.5)
