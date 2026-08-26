from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from ..logging import get_logger
from ..models.manager import ModelManager
from ..settings import Settings
from .docker_runtime import DockerRuntime
from .model_catalog import CODER, CONTROLLER, REASONING, VISION, ModelPolicy

logger = get_logger(__name__)


class LifecycleError(RuntimeError):
    pass


class LifecycleState(StrEnum):
    UNLOADED = "UNLOADED"
    STARTING = "STARTING"
    WARM = "WARM"
    IDLE = "IDLE"
    STOPPING = "STOPPING"
    # GPU ownership transition states
    TRANSITIONING_TO_COMFYUI = "TRANSITIONING_TO_COMFYUI"
    TRANSITIONING_TO_LLM = "TRANSITIONING_TO_LLM"


# Logical GPU-ownership sentinel values stored in ModelLifecycle._gpu_owner.
#
# These strings are NOT Docker container names and must NEVER be passed to
# any DockerRuntime / docker lifecycle operation:
#
#   "comfyui"                    -> image generation currently owns the GPU.
#   "transitioning_to_comfyui"   -> image generation has atomically RESERVED
#                                   the GPU and is preparing it (draining LLM
#                                   inference / unloading the resident model).
#   "TRANSITIONING_TO_LLM"       -> GPU is being handed back to the LLM pool.
#   None                         -> GPU is free.
#
# Any other non-None value is a physical llama.cpp container name that MAY be
# passed to Docker lifecycle operations.
GPU_OWNER_COMFYUI = "comfyui"
GPU_OWNER_TRANSITIONING_TO_COMFYUI = "transitioning_to_comfyui"
GPU_OWNER_TRANSITIONING_TO_LLM = LifecycleState.TRANSITIONING_TO_LLM.value

_IMAGE_GPU_SENTINELS = frozenset(
    {
        GPU_OWNER_COMFYUI,
        GPU_OWNER_TRANSITIONING_TO_COMFYUI,
        GPU_OWNER_TRANSITIONING_TO_LLM,
    }
)


def is_llm_container_owner(value: str | None) -> bool:
    """Return True only when ``value`` is a real llama.cpp container name.

    Sentinel values ("comfyui", "transitioning_to_comfyui",
    "TRANSITIONING_TO_LLM") and None are NOT containers and must never reach
    Docker lifecycle functions.
    """
    return value is not None and value not in _IMAGE_GPU_SENTINELS


@dataclass(slots=True)
class ModelRuntimeState:
    role: str
    name: str

    # Explicit state machine:
    # UNLOADED -> STARTING -> WARM -> IDLE -> STOPPING -> UNLOADED
    status: LifecycleState = LifecycleState.UNLOADED

    # Timestamps
    last_used_at: float | None = None
    keep_warm_until: float | None = None

    # Active inference tracking.
    # Stopping/eviction is only allowed when this count is zero.
    active_inference_count: int = 0

    # Used for informational/logging.
    warm_invocations: int = 0
    touch_invocations: int = 0


def _parse_keep_alive_seconds(value: str | None) -> int:
    """Parse residency keep-alive values (e.g. '30m', '15s', '3600')."""
    if value is None:
        return 0
    text = str(value).strip().lower()
    if not text:
        return 0

    if text.isdigit():
        return int(text)

    if text.endswith("ms"):
        try:
            return max(0, int(int(text[:-2]) / 1000))
        except Exception:
            return 0
    if text.endswith("s"):
        try:
            return max(0, int(text[:-1]))
        except Exception:
            return 0
    if text.endswith("m"):
        try:
            return max(0, int(text[:-1]) * 60)
        except Exception:
            return 0
    if text.endswith("h"):
        try:
            return max(0, int(text[:-1]) * 3600)
        except Exception:
            return 0

    try:
        return int(float(text))
    except Exception:
        return 0


class ModelLifecycle:
    """Model residency + warm-state manager backed by Docker containers.

    Important:
    - No inference/prompting beyond a minimal start/health cycle.
    - No orchestration logic.
    - Docker is treated as the source of truth; stale in-memory state is reconciled.
    - Per-model locks allow different models to warm concurrently while a single
      model can only transition once at a time.
    - Multiple logical roles may map to the same physical container. Runtime
      state remains tracked per logical role, while Docker operations
      (start/stop/health/reconcile/shutdown) are serialized once per unique
      container.
    - GPU residency is exclusive: exactly ONE llama.cpp container may own the
      GPU at a time. The lifecycle manager transfers GPU ownership between
      containers on demand (expert/vision/controller are transient, never
      simultaneously resident).
    """

    def __init__(
        self,
        *,
        settings: Settings,
        models: ModelManager,
        docker: DockerRuntime | None = None,
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

        # Protects runtime state mutations.
        self._state_lock = asyncio.Lock()

        # Tracks which physical llama.cpp container currently owns GPU residency.
        # Exactly one container may own the GPU at any point in time. Ownership is
        # keyed by physical container name (NOT logical role), so sibling roles
        # sharing a container (e.g. reasoning/coder -> llama-expert) share the same
        # owner and never trigger mutual stop/start.
        self._gpu_owner: str | None = None

        # Serializes all GPU residency transitions (stop current owner -> start new
        # owner). Only one ownership transition may execute at a time; this prevents
        # two threads from racing to start/stop different containers simultaneously.
        self._ownership_lock = asyncio.Lock()

        # Protects the per-role lock registry.
        self._registry_lock = asyncio.Lock()

        # Per-role lifecycle locks.
        self._locks: dict[str, asyncio.Lock] = {}

        # Protects the per-container lock registry.
        self._container_registry_lock = asyncio.Lock()

        # Per-container locks serialize Docker operations so multiple logical
        # roles mapping to the same physical container do not issue duplicate
        # start/stop/health/shutdown commands.
        self._container_locks: dict[str, asyncio.Lock] = {}

        # Runtime state for each model role.
        self._runtime: dict[str, ModelRuntimeState] = {}

        self._closing = False
        self._cleanup_task: asyncio.Task[None] | None = None
        self._health_client: httpx.AsyncClient | None = None

    async def _get_lock(self, role: str) -> asyncio.Lock:
        async with self._registry_lock:
            lock = self._locks.get(role)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[role] = lock
            return lock

    async def _get_container_lock(self, container_name: str) -> asyncio.Lock:
        async with self._container_registry_lock:
            lock = self._container_locks.get(container_name)
            if lock is None:
                lock = asyncio.Lock()
                self._container_locks[container_name] = lock
            return lock

    def start_background_cleanup(self) -> None:
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="model-lifecycle-cleanup")

    async def _close_health_client(self) -> None:
        client = self._health_client
        self._health_client = None
        if client is not None:
            await client.aclose()

    def _health_request_timeout_s(self) -> float:
        value = getattr(self.settings, "health_request_timeout_s", None)
        if value is not None:
            try:
                return max(1.0, float(value))
            except Exception:
                pass
        # Fallback: keep individual health requests short, regardless of total startup budget.
        try:
            total = float(self.settings.health_timeout_s)
        except Exception:
            total = 60.0
        return max(1.0, min(5.0, total / 6.0))

    async def _get_health_client(self) -> httpx.AsyncClient:
        if self._health_client is None:
            timeout_s = self._health_request_timeout_s()
            self._health_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=timeout_s,
                    read=timeout_s,
                    write=timeout_s,
                    pool=timeout_s,
                ),
                follow_redirects=False,
            )
        return self._health_client

    async def close(self) -> None:
        """Gracefully stop the lifecycle.

        Cancels the background cleanup loop and stops any containers that are
        still tracked as active so no orphaned Docker operations remain. Each
        unique physical container is stopped at most once, even when multiple
        logical roles share it. GPU ownership is reset to None.
        """
        self._closing = True

        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task

        # Stop any containers still believed to be active.
        async with self._state_lock:
            roles = list(self._runtime.keys())

        containers = {self.models.container_for_role(role) for role in roles}

        for container_name in containers:
            container_lock = await self._get_container_lock(container_name)
            async with container_lock:
                async with self._state_lock:
                    for role in roles:
                        if self.models.container_for_role(role) != container_name:
                            continue
                        st = self._runtime.get(role)
                        if st is None or st.status not in {
                            LifecycleState.WARM,
                            LifecycleState.IDLE,
                            LifecycleState.STARTING,
                            LifecycleState.STOPPING,
                        }:
                            continue
                        st.status = LifecycleState.STOPPING

                try:
                    await asyncio.wait_for(
                        self.docker.stop(container_name),
                        timeout=float(getattr(self.settings, "container_start_timeout_s", 30.0)),
                    )
                except Exception:
                    logger.exception("model_container_stop_failed_on_shutdown container=%s", container_name)

                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        self.docker.wait_stopped(
                            container_name,
                            timeout_s=float(getattr(self.settings, "container_start_timeout_s", 30.0)),
                        ),
                        timeout=float(getattr(self.settings, "container_start_timeout_s", 30.0)),
                    )

                async with self._state_lock:
                    for role in roles:
                        if self.models.container_for_role(role) != container_name:
                            continue
                        state = self._runtime.get(role)
                        if state is not None:
                            state.status = LifecycleState.UNLOADED
                            state.keep_warm_until = None

                if self._gpu_owner == container_name:
                    self._gpu_owner = None

                logger.info("model_container_stopped_on_shutdown container=%s", container_name)

        await self._close_health_client()

    def _ensure_state(self, role: str, name: str) -> ModelRuntimeState:
        existing = self._runtime.get(role)
        if existing is not None:
            return existing
        state = ModelRuntimeState(role=role, name=name)
        self._runtime[role] = state
        return state

    def _policy(self, role: str) -> ModelPolicy:
        return self._catalog[role]

    def _model_name(self, role: str) -> str:
        # Prefer an explicit name accessor if available.
        if hasattr(self.models, "model_name_for_role"):
            try:
                return str(self.models.model_name_for_role(role))
            except Exception:
                pass

        if hasattr(self.models, "model_for_role"):
            try:
                model = self.models.model_for_role(role)
                return getattr(model, "name", getattr(model, "model", role))
            except Exception:
                pass

        # Final fallback: use role name.
        return role

    async def _probe_health(self, role: str) -> bool:
        """Check the container's /health endpoint with a short request timeout."""
        endpoint = self.models.endpoint_for_role(role).rstrip("/")
        url = f"{endpoint}/health"
        client = await self._get_health_client()
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                return True

            logger.debug(
                "health_check_unhealthy role=%s endpoint=%s status=%s body=%s",
                role,
                url,
                resp.status_code,
                (resp.text or "")[:200],
            )
            return False
        except (httpx.HTTPError, asyncio.TimeoutError, OSError) as exc:
            logger.debug("health_check_failed role=%s endpoint=%s error=%r", role, url, exc)
            return False

    async def _stop_container_gracefully(self, container_name: str) -> None:
        """Stop a container and wait until it is fully stopped.

        Centralized stop helper reused by ownership transitions, eviction, and
        shutdown so container stop + wait_stopped remain in one place.
        """
        timeout_s = float(getattr(self.settings, "container_start_timeout_s", 30.0))
        try:
            await asyncio.wait_for(self.docker.stop(container_name), timeout=timeout_s)
        except Exception:
            logger.exception("model_container_stop_failed container=%s", container_name)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(
                self.docker.wait_stopped(container_name, timeout_s=timeout_s),
                timeout=timeout_s,
            )

    async def _start_container(self, role: str) -> None:
        container_name = self.models.container_for_role(role)

        status = await self.docker.status(container_name)
        if not status.exists:
            raise LifecycleError(
                f"Container {container_name!r} does not exist for role {role!r}. "
                "Create it once with `docker compose create` or `docker compose up` "
                "before relying on lifecycle-managed start/stop."
            )

        start_timeout_s = float(getattr(self.settings, "container_start_timeout_s", 30.0))

        # If it's already running but unhealthy/stale, stop it first so we can start
        # from a clean state.
        if status.running:
            logger.info("model_container_restart_needed role=%s container=%s", role, container_name)
            try:
                await asyncio.wait_for(self.docker.stop(container_name), timeout=start_timeout_s)
            except Exception as exc:
                raise LifecycleError(f"Failed to stop running container {container_name!r}: {exc}") from exc

            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    self.docker.wait_stopped(container_name, timeout_s=start_timeout_s),
                    timeout=start_timeout_s,
                )

        try:
            await asyncio.wait_for(self.docker.start(container_name), timeout=start_timeout_s)
        except Exception as exc:
            raise LifecycleError(f"Failed to start container {container_name!r}: {exc}") from exc

    async def _wait_healthy(self, role: str) -> None:
        """Poll /health until ready or timeout."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(self.settings.health_timeout_s)
        interval = max(0.1, float(self.settings.health_poll_interval_s))

        while loop.time() < deadline:
            if self._closing:
                raise LifecycleError(f"Lifecycle is shutting down while waiting for model role: {role}")

            if await self._probe_health(role):
                logger.info("model_health_ok role=%s", role)
                return

            remaining = deadline - loop.time()
            if remaining <= 0:
                break

            await asyncio.sleep(min(interval, remaining))

        raise LifecycleError(f"Health check timed out for model role: {role}")

    async def _reconcile(self, role: str) -> bool:
        """Return True if the container exists, is running, and is healthy.

        Docker is the source of truth. This performs network/docker I/O and
        must not be called while holding self._state_lock.
        """
        container_name = self.models.container_for_role(role)
        status = await self.docker.status(container_name)
        if not status.exists or not status.running:
            return False
        return await self._probe_health(role)

    async def _sibling_roles_active(self, container_name: str, *, exclude_role: str) -> bool:
        """Return True if any role other than exclude_role still maps to the
        given container and is in an active (keep-alive) runtime state.

        Must not be called while holding self._state_lock.
        """
        async with self._state_lock:
            for role, st in self._runtime.items():
                if role == exclude_role:
                    continue
                if self.models.container_for_role(role) != container_name:
                    continue
                if st.status in {
                    LifecycleState.WARM,
                    LifecycleState.IDLE,
                    LifecycleState.STARTING,
                }:
                    return True
        return False

    async def _wait_for_llm_gpu_availability(self, container_name: str) -> None:
        """Block until the GPU is free or already owned by ``container_name``.

        Image generation may own or have reserved the GPU (sentinel values
        "comfyui" / "transitioning_to_comfyui"). LLM requests must WAIT in
        that case — they must never evict ComfyUI, stop a sentinel-named
        "container", or clear ownership. Raises LifecycleError if the wait
        exceeds ``gpu_ownership_wait_timeout_s``.
        """
        timeout_s = float(getattr(self.settings, "gpu_ownership_wait_timeout_s", 1800.0))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s

        while True:
            async with self._ownership_lock:
                owner = self._gpu_owner

            if owner is None or owner == container_name:
                return

            if self._closing:
                raise LifecycleError(
                    "Lifecycle shutting down while waiting for GPU availability"
                )

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise LifecycleError(
                    f"Timed out waiting for GPU availability (owner={owner!r}); "
                    "image generation owns/reserves the GPU"
                )

            logger.info("llm_gpu_wait_for_availability owner=%s", owner)
            await asyncio.sleep(2)

    async def ensure_warm(self, role: str) -> None:
        """Make sure the model is warm before use."""
        role = role.lower().strip()
        lock = await self._get_lock(role)
        async with lock:
            await self._ensure_warm_locked(role)

    async def _ensure_warm_locked(self, role: str) -> None:
        model_name = self._model_name(role)
        policy = self._policy(role)
        container_name = self.models.container_for_role(role)

        async with self._state_lock:
            state = self._ensure_state(role, model_name)

        # Fast path: reconcile against Docker. If this role's container is already
        # running, healthy, AND is the current GPU owner (or no owner is recorded),
        # adopt it immediately without any stop/start or extra health polling.
        healthy = await self._reconcile(role)
        if healthy:
            async with self._ownership_lock:
                # Re-check owner under the ownership lock so we never adopt a
                # container that a concurrent transition is about to replace.
                if self._gpu_owner is None or self._gpu_owner == container_name:
                    now = time.time()
                    async with self._state_lock:
                        state = self._runtime[role]
                        state.status = LifecycleState.WARM
                        state.last_used_at = now
                        state.keep_warm_until = now + policy.keep_alive_seconds
                        state.touch_invocations += 1
                    self._gpu_owner = container_name
                    logger.info(
                        "model_already_warm role=%s model=%s owner=%s",
                        role,
                        state.name,
                        container_name,
                    )
                    return

        # Acquire the STARTING transition.
        async with self._state_lock:
            state = self._runtime[role]
            if state.status == LifecycleState.STARTING:
                logger.info("model_already_starting role=%s model=%s", role, state.name)
                return
            state.status = LifecycleState.STARTING
            state.warm_invocations += 1

        # GPU residency is exclusive: exactly one llama.cpp container may own the
        # GPU at a time. Serialize all ownership transitions so two threads can
        # never race stop/start across different containers (invariant preserved).
        #
        # Invariant: an LLM request can NEVER acquire the GPU while image
        # generation owns it ("comfyui") or has reserved it
        # ("transitioning_to_comfyui"). Wait outside the ownership lock until
        # the GPU is free instead of evicting a logical ownership sentinel.
        await self._wait_for_llm_gpu_availability(container_name)

        async with self._ownership_lock:
            try:
                # If a sibling logical role already brought this same physical
                # container up and it is healthy, simply adopt it.
                if await self._reconcile(role):
                    now = time.time()
                    async with self._state_lock:
                        st = self._runtime[role]
                        st.status = LifecycleState.WARM
                        st.last_used_at = now
                        st.keep_warm_until = now + policy.keep_alive_seconds
                        st.touch_invocations += 1
                    self._gpu_owner = container_name
                    logger.info(
                        "model_adopted_shared_container role=%s model=%s owner=%s",
                        role,
                        model_name,
                        container_name,
                    )
                    return

                # Stop whatever currently owns the GPU so the target container can
                # take exclusive residency. Only one owner may exist at a time.
                #
                # SAFETY: the previous owner is only ever stopped when it is a
                # REAL llama.cpp container name. Logical sentinels ("comfyui",
                # "transitioning_to_comfyui", "TRANSITIONING_TO_LLM") are never
                # passed to Docker. (The pre-lock wait above already blocks
                # while image generation owns/reserves the GPU; this guard is
                # defense in depth.)
                if (
                    is_llm_container_owner(self._gpu_owner)
                    and self._gpu_owner != container_name
                ):
                    previous_owner = self._gpu_owner
                    logger.info(
                        "gpu_ownership_transfer from=%s to=%s via_role=%s",
                        previous_owner,
                        container_name,
                        role,
                    )
                    container_lock = await self._get_container_lock(previous_owner)
                    async with container_lock:
                        await self._stop_container_gracefully(previous_owner)
                    # Mark all roles mapping to the previous owner as unloaded.
                    async with self._state_lock:
                        for other_role, st in self._runtime.items():
                            if self.models.container_for_role(other_role) == previous_owner:
                                if st.status != LifecycleState.STARTING:
                                    st.status = LifecycleState.UNLOADED
                                    st.keep_warm_until = None
                    self._gpu_owner = None

                # Start the target container and wait for health.
                container_lock = await self._get_container_lock(container_name)
                async with container_lock:
                    if not await self._reconcile(role):
                        await self._start_container(role)
                        await self._wait_healthy(role)

                now = time.time()
                async with self._state_lock:
                    st = self._runtime.get(role)
                    if st is None:
                        st = self._ensure_state(role, model_name)
                    st.status = LifecycleState.WARM
                    st.last_used_at = now
                    st.keep_warm_until = now + policy.keep_alive_seconds
                    st.touch_invocations += 1
                self._gpu_owner = container_name

                logger.info(
                    "model_became_warm role=%s model=%s owner=%s",
                    role,
                    model_name,
                    container_name,
                )
            except Exception as exc:
                # Startup failed: stop the container and clear runtime state so no
                # partially-initialized model is ever left behind.
                logger.error(
                    "container_start_failed role=%s model=%s container=%s",
                    role,
                    state.name,
                    container_name,
                )
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        self.docker.stop(container_name),
                        timeout=float(getattr(self.settings, "container_start_timeout_s", 30.0)),
                    )
                async with self._state_lock:
                    st = self._runtime.get(role)
                    if st is not None:
                        st.status = LifecycleState.UNLOADED
                        st.keep_warm_until = None
                if self._gpu_owner == container_name:
                    self._gpu_owner = None
                logger.exception("model_start_failed role=%s model=%s", role, state.name)
                raise LifecycleError(str(exc)) from exc

    def touch(self, role: str) -> None:
        """Extend keep-warm window based on last usage."""
        role = role.lower().strip()
        now = time.time()

        async def _touch() -> None:
            async with self._state_lock:
                if role not in self._runtime:
                    return
                policy = self._policy(role)
                state = self._runtime[role]
                state.last_used_at = now
                state.keep_warm_until = now + policy.keep_alive_seconds
                state.status = LifecycleState.IDLE
                state.touch_invocations += 1

        if not self._closing:
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
            state.status = LifecycleState.IDLE

    def request_preload(self, role: str) -> None:
        """Schedule a background warm without awaiting.

        Preload now means: "if no other container currently owns GPU residency,
        proactively start this model." It must never violate the single-owner
        invariant, so if another container owns the GPU it is skipped.
        """
        role = role.lower().strip()
        policy = self._policy(role)
        if not policy.preload_enabled or self._closing:
            return

        container_name = self.models.container_for_role(role)

        async def _preload() -> None:
            try:
                # Only preload when no other container owns the GPU. Holding the
                # ownership lock here guarantees the invariant is not violated.
                async with self._ownership_lock:
                    if self._gpu_owner is not None and self._gpu_owner != container_name:
                        logger.info(
                            "model_preload_skipped_owner_conflict role=%s owner=%s",
                            role,
                            self._gpu_owner,
                        )
                        return
                await self.ensure_warm(role)
                async with self._state_lock:
                    state = self._runtime.get(role)
                    if state is not None:
                        state.status = LifecycleState.IDLE
                        logger.info("model_preload_finished role=%s model=%s", role, state.name)
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
                    state = self._ensure_state(role, self._model_name(role))
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
        return st.status in {
            LifecycleState.WARM,
            LifecycleState.IDLE,
            LifecycleState.STARTING,
        }

    async def evict_if_needed(self) -> None:
        """Evict models whose keep-warm window has expired.

        Candidates are collected under the state lock briefly, then evicted one
        at a time under each model's own lifecycle lock. This avoids holding a
        global lock during container I/O and lets eviction and inference for
        the same model serialize safely.
        """
        now = time.time()
        candidates: list[str] = []

        async with self._state_lock:
            for role, state in self._runtime.items():
                if state.status not in {LifecycleState.IDLE, LifecycleState.WARM}:
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

        # Evict lowest priority first.
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
                if state.status in {LifecycleState.STARTING, LifecycleState.STOPPING}:
                    return
                if state.active_inference_count > 0:
                    logger.info(
                        "model_eviction_skipped_active_inference role=%s model=%s active_inference_count=%d",
                        role,
                        state.name,
                        state.active_inference_count,
                    )
                    return

                state.status = LifecycleState.STOPPING

            logger.info("model_eviction_started role=%s model=%s", role, state.name)

            container_name = self.models.container_for_role(role)

            # If a sibling logical role still needs this physical container,
            # keep it running and only unload this role's runtime state.
            if await self._sibling_roles_active(container_name, exclude_role=role):
                async with self._state_lock:
                    st = self._runtime.get(role)
                    if st is not None:
                        st.status = LifecycleState.UNLOADED
                        st.keep_warm_until = None
                logger.info(
                    "model_evicted_shared_container_kept role=%s model=%s container=%s",
                    role,
                    state.name,
                    container_name,
                )
                return

            container_lock = await self._get_container_lock(container_name)
            async with container_lock:
                # Re-check after acquiring the container lock in case a sibling
                # role warmed up the shared container while we were waiting.
                if await self._sibling_roles_active(container_name, exclude_role=role):
                    async with self._state_lock:
                        st = self._runtime.get(role)
                        if st is not None:
                            st.status = LifecycleState.UNLOADED
                            st.keep_warm_until = None
                    logger.info(
                        "model_evicted_shared_container_kept role=%s model=%s container=%s",
                        role,
                        state.name,
                        container_name,
                    )
                    return

                # GPU ownership transition for eviction: serialize so it never
                # races a concurrent start. Only clear the owner if this container
                # is in fact the current GPU owner.
                async with self._ownership_lock:
                    try:
                        await self._stop_container_gracefully(container_name)
                    finally:
                        if self._gpu_owner == container_name:
                            self._gpu_owner = None

            async with self._state_lock:
                st = self._runtime.get(role)
                if st is not None:
                    st.status = LifecycleState.UNLOADED
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
                
    async def acquire_comfyui_gpu(self) -> None:
        """Acquire exclusive GPU ownership for image generation.

        Ownership state machine (all transitions under ``_ownership_lock``):

            FREE (None) or LLM_OWNED (<container>)
                --atomic reservation--> TRANSITIONING_TO_COMFYUI
            [outside lock] wait for active LLM inference == 0
            [HTTP] unload resident llama-router model (POST /models/unload,
                   NOT /sleep) and verify via GET /v1/models
            TRANSITIONING_TO_COMFYUI --atomic finalize--> COMFYUI ("comfyui")

        Guarantees:
        - Concurrent image requests SERIALIZE: while another request holds
          "comfyui" or "transitioning_to_comfyui", this request WAITS for the
          owner to be released instead of borrowing its ownership token.
        - Once reserved, no LLM request may acquire or reclaim the GPU.
        - No Docker operations are used anywhere in this transition.
        - On any failure the reservation rolls back to its previous value so
          the GPU never wedges permanently in the transitioning state.
        """
        # Step 1: Atomically reserve the transition (from ANY prior state).
        previous_owner: str | None = None
        while True:
            occupied: str | None = None
            async with self._ownership_lock:
                owner = self._gpu_owner
                if owner in {GPU_OWNER_COMFYUI, GPU_OWNER_TRANSITIONING_TO_COMFYUI}:
                    # Another image generation owns/reserved the GPU: wait for
                    # it to be fully released rather than sharing ownership.
                    occupied = owner
                else:
                    previous_owner = owner
                    self._gpu_owner = GPU_OWNER_TRANSITIONING_TO_COMFYUI
                    logger.info(
                        "acquiring_comfyui_gpu: transition reserved previous_owner=%s",
                        previous_owner,
                    )
            if occupied is None:
                break
            logger.info("comfyui_gpu_wait_for_existing_generation owner=%s", occupied)
            await asyncio.sleep(2)

        reservation_held = True

        try:
            # Step 2: Wait for active LLM inference to drain (outside lock).
            roles_to_check = ["controller", "reasoning", "coder", "vision"]
            for role in roles_to_check:
                while True:
                    async with self._state_lock:
                        state = self._runtime.get(role)

                    if state is None or state.active_inference_count == 0:
                        break

                    logger.info(
                        "comfyui_gpu_wait_for_inference role=%s count=%d",
                        role,
                        state.active_inference_count,
                    )
                    await asyncio.sleep(2)

            # Step 3: Get actual resident model from llama-router.
            # This HTTP client is owned by this call (NOT the shared registry
            # client) and is closed when the context exits.
            async with await self._get_llm_client() as client:
                try:
                    resp = await client.get("/v1/models")
                    resp.raise_for_status()
                    models_data = resp.json()

                    # Find the actually loaded model
                    resident_model = None
                    for model in models_data.get("data", []):
                        status = model.get("status", {})
                        if status.get("value") in ("loaded", "loading", "sleeping"):
                            resident_model = model.get("id")
                            break

                    logger.info("comfyui_gpu: resident_model=%s", resident_model or "none")

                    # Step 4: Unload the resident model if any
                    if resident_model:
                        logger.info("unload_llm_model_for_comfyui model=%s", resident_model)
                        unload_resp = await client.post(
                            "/models/unload",
                            json={"model": resident_model}
                        )
                        unload_resp.raise_for_status()

                        # Step 5: Poll until confirmed unloaded
                        for _ in range(30):  # Wait up to 60 seconds
                            await asyncio.sleep(2)
                            check_resp = await client.get("/v1/models")
                            check_resp.raise_for_status()
                            models_data = check_resp.json()

                            for model in models_data.get("data", []):
                                if model.get("id") == resident_model:
                                    status = model.get("status", {}).get("value")
                                    if status == "unloaded":
                                        logger.info("llm_model_confirmed_unloaded model=%s", resident_model)
                                        resident_model = None
                                        break

                            if resident_model is None:
                                break

                    if resident_model is not None:
                        raise LifecycleError(f"Failed to unload llama-router model: {resident_model}")

                except httpx.HTTPError as e:
                    logger.error("llama_router_api_error error=%r", e)
                    raise LifecycleError(f"llama-router API error: {e}")

            # Step 6: Revalidate and atomically finalize COMFYUI_ACTIVE.
            async with self._ownership_lock:
                if self._gpu_owner != GPU_OWNER_TRANSITIONING_TO_COMFYUI:
                    raise LifecycleError(
                        "comfyui_gpu_reservation_lost "
                        f"(owner={self._gpu_owner!r}); acquisition aborted"
                    )
                self._gpu_owner = GPU_OWNER_COMFYUI
                logger.info("comfyui_gpu_acquired")
            reservation_held = False

        except BaseException:
            # Roll back the atomic reservation so the GPU never wedges in the
            # transitioning state after a failed preparation (unload error,
            # timeout, cancellation, ...). Only revert if we still hold it.
            if reservation_held:
                async with self._ownership_lock:
                    if self._gpu_owner == GPU_OWNER_TRANSITIONING_TO_COMFYUI:
                        self._gpu_owner = previous_owner
                        logger.info(
                            "comfyui_gpu_reservation_rolled_back restored_owner=%s",
                            previous_owner,
                        )
            raise

    async def release_comfyui_gpu(self) -> None:
        """Release GPU ownership after confirmed image-generation completion.

        The orchestrator no longer owns any direct ComfyUI workload: image
        generation is delegated to Open WebUI, whose synchronous API returns
        only after its generation has completed. Therefore no ComfyUI /free
        call is performed here — ComfyUI memory management belongs to Open
        WebUI. This method only transitions GPU ownership back to free.

        Callers must ONLY invoke this after confirmed completion/failure of
        the Open WebUI request. Unknown workload state must retain ownership.
        Does NOT use Docker for container lifecycle management.

        Verify-and-transition happen ATOMICALLY under ``_ownership_lock`` so a
        concurrent second image request can never observe a half-released
        state.
        """
        async with self._ownership_lock:
            if self._gpu_owner != GPU_OWNER_COMFYUI:
                logger.warning("comfyui_gpu_release_invalid owner=%s", self._gpu_owner)
                return

            self._gpu_owner = None
            logger.info("comfyui_gpu_released")

    async def _get_llm_client(self) -> httpx.AsyncClient:
        """Create a short-lived HTTP client for llama-router API calls.

        The returned client is owned by the caller and MUST be closed (use
        ``async with``). It is intentionally NOT a shared registry client.
        """
        base_url = self.settings.model_router_url
        if base_url.endswith("/v1"):
            base_url = base_url[: -len("/v1")]
        return httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(30.0))

            
