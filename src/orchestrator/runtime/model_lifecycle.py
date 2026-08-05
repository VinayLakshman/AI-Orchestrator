from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..logging import get_logger
from ..models.manager import ModelManager
from ..settings import Settings
from .docker_runtime import DockerRuntime, DockerError
from .model_catalog import CODER, CONTROLLER, REASONING, VISION, ModelPolicy

logger = get_logger(__name__)


class LifecycleError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelRuntimeState:

    role: str
    name: str

    # Explicit state machine:
    # UNLOADED -> STARTING -> WARM -> IDLE -> STOPPING -> UNLOADED

    status: str = "UNLOADED"

    # Timestamps
    last_used_at: float | None = None
    keep_warm_until: float | None = None

    # Active inference tracking.
    # Stopping/eviction is only allowed when this count is zero.
    active_inference_count: int = 0

    # Used for informational/logging.
    warm_invocations: int = 0
    touch_invocations: int = 0


def _parse_keep_alive_seconds(value: str) -> int:
    """Parse residency keep-alive values (e.g. "30m", "15s", "3600")."""
    if value is None:
        return 0
    text = str(value).strip().lower()
    if not text:
        return 0

    # pure integer seconds
    if text.isdigit():
        return int(text)

    if text.endswith("ms"):
        return max(0, int(int(text[:-2]) / 1000))
    if text.endswith("s"):
        return max(0, int(text[:-1]))
    if text.endswith("m"):
        return max(0, int(text[:-1]) * 60)
    if text.endswith("h"):
        return max(0, int(text[:-1]) * 3600)

    # unknown format => best effort
    try:
        return int(float(text))
    except Exception:
        return 0


class ModelLifecycle:
    """Model residency + warm-state manager backed by Docker containers.

    Important:
    - No inference/prompting beyond a minimal start/health cycle.
    - No orchestration logic.
    - Treats Docker as the source of truth; reconciles internal state.
    - Uses a per-model lifecycle lock so different models can start
      concurrently while a single model can only ever start once.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        models: ModelManager,
        docker: DockerRuntime,
        catalog_overrides: dict[str, ModelPolicy] | None = None,
        poll_interval_s: float = 60.0,
    ) -> None:
        self.settings = settings
        self.models = models
        self.docker = docker
        self._poll_interval_s = poll_interval_s

        self._catalog: dict[str, ModelPolicy] = {
            "controller": CONTROLLER,
            "reasoning": REASONING,
            "coder": CODER,
            "vision": VISION,
        }
        if catalog_overrides:
            self._catalog.update(catalog_overrides)

        # Apply keep-alive overrides from Settings without hardcoding.
        self._catalog["controller"] = ModelPolicy(
            role="controller",
            priority=self._catalog["controller"].priority,
            keep_alive_seconds=_parse_keep_alive_seconds(settings.controller_keep_alive),
            can_evict=self._catalog["controller"].can_evict,
            preload_enabled=self._catalog["controller"].preload_enabled,
        )
        self._catalog["reasoning"] = ModelPolicy(
            role="reasoning",
            priority=self._catalog["reasoning"].priority,
            keep_alive_seconds=_parse_keep_alive_seconds(settings.reasoning_keep_alive),
            can_evict=self._catalog["reasoning"].can_evict,
            preload_enabled=self._catalog["reasoning"].preload_enabled,
        )
        self._catalog["coder"] = ModelPolicy(
            role="coder",
            priority=self._catalog["coder"].priority,
            keep_alive_seconds=_parse_keep_alive_seconds(settings.coder_keep_alive),
            can_evict=self._catalog["coder"].can_evict,
            preload_enabled=self._catalog["coder"].preload_enabled,
        )
        self._catalog["vision"] = ModelPolicy(
            role="vision",
            priority=self._catalog["vision"].priority,
            keep_alive_seconds=_parse_keep_alive_seconds(settings.vision_keep_alive),
            can_evict=self._catalog["vision"].can_evict,
            preload_enabled=self._catalog["vision"].preload_enabled,
        )

        # Lock protecting the runtime-state dict and per-role state fields for
        # brief mutations (inference counters, touch, keep_warm, transitions).
        self._state_lock = asyncio.Lock()

        # Lock protecting the per-model lock registry itself (held only briefly
        # to look up or create a role's lock).
        self._registry_lock = asyncio.Lock()

        # Per-model lifecycle locks. Warming/stopping a specific model is
        # serialized under its own lock so only one startup / health poll /
        # transition occurs per model, while different models use independent
        # locks and can proceed concurrently.
        self._locks: dict[str, asyncio.Lock] = {}

        self._runtime: dict[str, ModelRuntimeState] = {}
        self._closing = False
        self._cleanup_task: asyncio.Task[None] | None = None

    async def _get_lock(self, role: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(role)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[role] = lock
            return lock

    def start_background_cleanup(self) -> None:
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def close(self) -> None:
        """Gracefully stop the lifecycle.

        Cancels the background cleanup loop and stops any containers that
        are still running so no orphaned Docker operations remain.
        """
        self._closing = True
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Stop any containers still believed to be warm/idle/starting/stopping.
        async with self._state_lock:
            roles = list(self._runtime.keys())
        for role in roles:
            lock = await self._get_lock(role)
            async with lock:
                async with self._state_lock:
                    st = self._runtime.get(role)
                    if st is None or st.status not in {"WARM", "IDLE", "STARTING", "STOPPING"}:
                        continue
                container_name = self.models.container_for_role(role)
                with contextlib.suppress(Exception):
                    await self.docker.stop(container_name)
                with contextlib.suppress(Exception):
                    await self.docker.wait_stopped(container_name, timeout_s=self.settings.health_timeout_s)
                async with self._state_lock:
                    state = self._runtime.get(role)
                    if state is not None:
                        state.status = "UNLOADED"
                        state.keep_warm_until = None
                logger.info("model_container_stopped_on_shutdown role=%s container=%s", role, container_name)

    def _ensure_state(self, role: str, name: str) -> ModelRuntimeState:
        existing = self._runtime.get(role)
        if existing is not None:
            return existing
        state = ModelRuntimeState(role=role, name=name)
        self._runtime[role] = state
        return state

    def _policy(self, role: str) -> ModelPolicy:
        return self._catalog[role]

    def _model_managed(self, role: str) -> Any:
        return self.models.__getattribute__(role)()

    async def _is_healthy(self, role: str) -> bool:
        """Check the container's /health endpoint with an explicit timeout."""
        endpoint = self.models.endpoint_for_role(role).rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=self.settings.health_timeout_s) as client:
                resp = await client.get(f"{endpoint}/health")
                return resp.status_code == 200
        except Exception:
            return False

    async def _start_container(self, role: str) -> None:
        container_name = self.models.container_for_role(role)
        await self.docker.start(container_name)

    async def _wait_healthy(self, role: str) -> None:
        """Poll /health until ready or timeout."""
        deadline = asyncio.get_running_loop().time() + self.settings.health_timeout_s
        interval = self.settings.health_poll_interval_s
        while True:
            if await self._is_healthy(role):
                logger.info("model_health_ok role=%s", role)
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise LifecycleError(f"Health check timed out for model role: {role}")
            await asyncio.sleep(interval)

    async def _reconcile(self, role: str) -> bool:
        """Return True if the container is running and healthy.

        Docker is the source of truth. This performs network/docker I/O and
        must not be called while holding self._state_lock.
        """
        container_name = self.models.container_for_role(role)
        status = await self.docker.status(container_name)
        if not status.exists:
            return False

        if not status.running:
            return False

        if not await self._is_healthy(role):
            return False

        return True

    async def ensure_warm(self, role: str) -> None:
        """Make sure the model is warm before use.

        Serialized per model via its own lifecycle lock: only one startup,
        one health-polling loop, and one transition per model can occur at a
        time. Different models use independent locks and can warm concurrently.
        """
        role = role.lower().strip()
        lock = await self._get_lock(role)
        async with lock:
            await self._ensure_warm_locked(role)

    async def _ensure_warm_locked(self, role: str) -> None:
        model = self._model_managed(role)
        policy = self._policy(role)

        # Fast path: if we believe the model is warm, reconcile against Docker
        # (the source of truth). Never trust stale in-memory state.
        async with self._state_lock:
            state = self._ensure_state(role, model.name)
            warm = (
                state.status in {"WARM", "IDLE"}
                and state.keep_warm_until is not None
                and state.keep_warm_until > time.time()
            )

        if warm:
            healthy = await self._reconcile(role)
            if healthy:
                async with self._state_lock:
                    state = self._runtime[role]
                    state.status = "WARM"
                    state.last_used_at = time.time()
                    state.touch_invocations += 1
                    state.keep_warm_until = time.time() + policy.keep_alive_seconds
                logger.info("model_already_warm role=%s model=%s status=%s", role, state.name, "WARM")
                return

            # Container is down; repair stale state and fall through to restart.
            async with self._state_lock:
                state = self._runtime[role]
                state.status = "UNLOADED"
                state.keep_warm_until = None
            logger.info("model_warm_stale_reconcile role=%s model=%s", role, state.name)

        # Acquire the STARTING transition (guarded by the per-model lock held
        # by the caller, plus the state lock for the dict mutation).
        async with self._state_lock:
            state = self._runtime[role]
            if state.status == "STARTING":
                logger.info("model_already_starting role=%s model=%s", role, state.name)
                return
            state.status = "STARTING"
            state.warm_invocations += 1
        logger.info("model_starting_started role=%s model=%s", role, state.name)

        container_name = self.models.container_for_role(role)
        try:
            await self._start_container(role)
            await self._wait_healthy(role)
        except Exception as exc:
            # Startup failed: stop the container and clear runtime state so no
            # partially-initialized model is ever left behind.
            logger.error("container_start_failed role=%s model=%s container=%s", role, state.name, container_name)
            with contextlib.suppress(Exception):
                await self.docker.stop(container_name)
            async with self._state_lock:
                st = self._runtime.get(role)
                if st is not None:
                    st.status = "UNLOADED"
            logger.exception("model_start_failed role=%s model=%s", role, state.name)
            raise LifecycleError(str(exc)) from exc

        async with self._state_lock:
            st = self._runtime.get(role)
            if st is None:
                st = self._ensure_state(role, model.name)
            st.status = "WARM"
            st.last_used_at = time.time()
            st.keep_warm_until = st.last_used_at + policy.keep_alive_seconds
            st.touch_invocations += 1
        logger.info("model_became_warm role=%s model=%s", role, model.name)

    def touch(self, role: str) -> None:
        """Extend keep-warm window based on last usage."""
        role = role.lower().strip()
        now = time.time()

        # touch is sync for ease of integration; schedule coroutine under the hood.
        async def _touch() -> None:
            async with self._state_lock:
                if role not in self._runtime:
                    return
                policy = self._policy(role)
                state = self._runtime[role]
                state.last_used_at = now
                state.keep_warm_until = now + policy.keep_alive_seconds
                state.status = "IDLE"
                state.touch_invocations += 1

        asyncio.create_task(_touch())

    async def keep_warm(self, role: str) -> None:
        """Mark model as IDLE but not evictable until keep_alive expires."""
        role = role.lower().strip()
        async with self._state_lock:
            if role not in self._runtime:
                return
            state = self._runtime[role]
            policy = self._policy(role)
            now = time.time()
            state.last_used_at = state.last_used_at or now
            state.keep_warm_until = now + policy.keep_alive_seconds
            state.status = "IDLE"

    def request_preload(self, role: str) -> None:
        """Schedule a background warm without awaiting."""
        role = role.lower().strip()
        policy = self._policy(role)
        if not policy.preload_enabled:
            return

        async def _preload() -> None:
            try:
                await self.ensure_warm(role)
                async with self._state_lock:
                    state = self._runtime[role]
                    state.status = "IDLE"
                logger.info("model_preload_finished role=%s model=%s", role, self._runtime[role].name)
            except Exception:
                logger.exception("model_preload_failed role=%s", role)

        asyncio.create_task(_preload())

    async def _begin_inference(self, role: str) -> None:
        role = role.lower().strip()
        lock = await self._get_lock(role)
        async with lock:
            async with self._state_lock:
                state = self._runtime.get(role)
                if state is None:
                    state = self._ensure_state(role, self._model_managed(role).name)
                state.active_inference_count += 1
                logger.info(
                    "model_inference_started role=%s model=%s active_inference_count=%d",
                    role,
                    state.name,
                    state.active_inference_count,
                )

    async def _end_inference(self, role: str) -> None:
        role = role.lower().strip()
        lock = await self._get_lock(role)
        async with lock:
            async with self._state_lock:
                state = self._runtime.get(role)
                if state is None:
                    return
                state.active_inference_count = max(0, state.active_inference_count - 1)
                logger.info(
                    "model_inference_finished role=%s model=%s active_inference_count=%d",
                    role,
                    state.name,
                    state.active_inference_count,
                )

    def active_inference(self, role: str):
        """Async context manager protecting against eviction during inference."""
        from .model_inference_guard import _InferenceGuard

        return _InferenceGuard(lifecycle=self, role=role)

    def is_loaded(self, role: str) -> bool:
        role = role.lower().strip()
        st = self._runtime.get(role)
        if st is None:
            return False
        return st.status in {"WARM", "IDLE", "STARTING"}

    async def evict_if_needed(self) -> None:
        """Evict models whose keep-warm window has expired.

        Candidates are collected under the state lock (briefly), then evicted
        one at a time under each model's own lifecycle lock. This avoids
        holding a global lock during container I/O and lets eviction and
        inference for the same model serialize safely.
        """
        now = time.time()
        candidates: list[str] = []

        async with self._state_lock:
            for role, state in self._runtime.items():
                if state.status not in {"IDLE", "WARM"}:
                    continue
                policy = self._policy(role)
                if not policy.can_evict:
                    continue
                if state.keep_warm_until is None or state.keep_warm_until > now:
                    continue
                if state.active_inference_count > 0:
                    logger.info(
                        "model_eviction_skipped_active_inference role=%s model=%s active_inference_count=%d",
                        role,
                        state.name,
                        state.active_inference_count,
                    )
                    continue
                candidates.append(role)

        # Evict lowest priority first
        candidates.sort(key=lambda role: self._policy(role).priority)

        for role in candidates:
            await self._evict(role=role)

    async def _evict(self, role: str) -> None:
        lock = await self._get_lock(role)
        async with lock:
            async with self._state_lock:
                state = self._runtime.get(role)
                if state is None:
                    return
                policy = self._policy(role)

                if not policy.can_evict:
                    return
                now = time.time()
                if state.keep_warm_until is not None and state.keep_warm_until > now:
                    logger.info("model_eviction_skipped role=%s model=%s still_warm", role, state.name)
                    return
                if state.status in {"STARTING", "STOPPING"}:
                    return
                if state.active_inference_count > 0:
                    logger.info(
                        "model_eviction_skipped_active_inference role=%s model=%s active_inference_count=%d",
                        role,
                        state.name,
                        state.active_inference_count,
                    )
                    return
                state.status = "STOPPING"
            logger.info("model_eviction_started role=%s model=%s", role, state.name)

            container_name = self.models.container_for_role(role)
            try:
                await self.docker.stop(container_name)
                await self.docker.wait_stopped(container_name, timeout_s=self.settings.health_timeout_s)
            except Exception:
                logger.exception("model_eviction_failed role=%s model=%s", role, state.name)
                async with self._state_lock:
                    st = self._runtime.get(role)
                    if st is not None:
                        st.status = "IDLE"
                return

            async with self._state_lock:
                st = self._runtime.get(role)
                if st is not None:
                    st.status = "UNLOADED"
                    st.keep_warm_until = None
                logger.info("model_evicted role=%s model=%s", role, st.name if st else "")

    async def _cleanup_loop(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(self._poll_interval_s)
                await self.evict_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("model_lifecycle_cleanup_loop_error")
