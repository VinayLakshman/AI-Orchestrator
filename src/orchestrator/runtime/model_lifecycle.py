from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional


from ..logging import get_logger
from ..models.chat import ChatMessage
from ..common.enums import ChatRole
from ..models.manager import ManagedModel, ModelManager
from ..settings import Settings
from .model_catalog import CODER, CONTROLLER, REASONING, VISION, ModelPolicy
from ..serialization import sanitize_for_json, validate_json_serializable, SerializationError

logger = get_logger(__name__)


class LifecycleError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelRuntimeState:

    role: str
    name: str

    # State machine
    # COLD -> LOADING -> WARM -> IDLE -> UNLOADED

    status: str = "COLD"

    # Timestamps
    last_used_at: float | None = None
    keep_warm_until: float | None = None

    # Async coordination
    pending_load_task: asyncio.Task[None] | None = None

    # Active inference tracking.
    # Eviction is only allowed when this count is zero.
    active_inference_count: int = 0

    # Used for informational/logging.
    warm_invocations: int = 0
    touch_invocations: int = 0



def _parse_keep_alive_seconds(value: str) -> int:
    """Parse ollama-like keep_alive values (e.g. "30m", "15s", "3600")."""
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
    """Model residency + warm-state manager.

    Important:
    - No inference/prompting beyond a minimal warm-up call.
    - No orchestration logic.
    - Avoids duplicate loads via pending_load_task.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        models: ModelManager,
        ollama_client: Any,
        catalog_overrides: dict[str, ModelPolicy] | None = None,
        poll_interval_s: float = 60.0,
    ) -> None:
        self.settings = settings
        self.models = models
        self.ollama = ollama_client
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
        # Coder/vision keep-alive are now explicit settings.
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

        self._lock = asyncio.Lock()
        self._runtime: dict[str, ModelRuntimeState] = {}
        self._closing = False
        self._cleanup_task: asyncio.Task[None] | None = None

    def start_background_cleanup(self) -> None:
        if self._cleanup_task is not None:
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def close(self) -> None:
        self._closing = True
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with asyncio.suppress(asyncio.CancelledError):
                await self._cleanup_task

    def _ensure_state(self, role: str, name: str) -> ModelRuntimeState:
        existing = self._runtime.get(role)
        if existing is not None:
            return existing
        state = ModelRuntimeState(role=role, name=name)
        self._runtime[role] = state
        return state

    def _policy(self, role: str) -> ModelPolicy:
        return self._catalog[role]

    def _model_managed(self, role: str) -> ManagedModel:
        if role == "controller":
            return self.models.controller()
        if role == "reasoning":
            return self.models.reasoning()
        if role == "coder":
            return self.models.coder()
        if role == "vision":
            return self.models.vision()
        raise KeyError(f"Unknown model role: {role}")

    async def ensure_warm(self, role: str) -> None:
        """Make sure the model is warm before use."""
        role = role.lower().strip()
        async with self._lock:
            model = self._model_managed(role)
            state = self._ensure_state(role, model.name)

            policy = self._policy(role)
            now = time.time()

            if state.status in {"WARM"} and state.keep_warm_until and state.keep_warm_until > now:

                logger.info("model_already_warm role=%s model=%s status=%s", role, state.name, state.status)
                state.status = "WARM"
                state.last_used_at = now

                state.touch_invocations += 1
                state.keep_warm_until = now + policy.keep_alive_seconds
                return

            if state.status == "LOADING" and state.pending_load_task is not None:
                logger.info("model_already_loading role=%s model=%s", role, state.name)
                task = state.pending_load_task
            else:
                state.status = "LOADING"
                state.pending_load_task = asyncio.create_task(self._warm_impl(role))
                state.warm_invocations += 1
                task = state.pending_load_task
                logger.info("model_loading_started role=%s model=%s", role, state.name)

        # Await outside lock.
        try:
            await task
        except Exception as exc:
            logger.exception("model_loading_failed role=%s model=%s", role, state.name)
            raise LifecycleError(str(exc)) from exc

        async with self._lock:
            # state should now be WARM
            state = self._runtime[role]
            policy = self._policy(role)
            state.status = "WARM"
            state.last_used_at = time.time()

            state.keep_warm_until = state.last_used_at + policy.keep_alive_seconds
            state.touch_invocations += 1

    async def _warm_impl(self, role: str) -> None:
        """Warm up the model.

        For now we rely on a minimal Ollama chat call.
        This is the only place lifecycle manager interacts with LLMs.
        """
        model = self._model_managed(role)
        policy = self._policy(role)

        # Warm-up prompt: keep it short and generic.
        # Controller warm-up is handled separately by existing warm_controller()
        # to preserve existing behavior exactly.
        if role == "controller":
            # Should never happen because controller is kept resident.
            await self.ollama.chat(
                model=model.name,
                messages=[
                    ChatMessage(role=ChatRole.SYSTEM, content="You are a resident controller. Reply with OK."),
                    ChatMessage(role=ChatRole.USER, content="Warm up and stay resident."),
                ],
                temperature=0.0,
                max_tokens=4,
                stream=False,
                keep_alive=self.settings.controller_keep_alive,
            )
        else:
            await self.ollama.chat(
                model=model.name,
                messages=[
                    ChatMessage(role=ChatRole.SYSTEM, content="Warm the model. Respond with OK."),
                    ChatMessage(role=ChatRole.USER, content="Warm up"),
                ],
                temperature=0.0,
                max_tokens=4,
                stream=False,
                # keep_alive handled by residency policy once ACTIVE/keep_warm_until is set.
                keep_alive=str(policy.keep_alive_seconds) + "s",
            )

        async with self._lock:
            state = self._runtime[role]
            state.status = "WARM"
            state.pending_load_task = None
            logger.info("model_became_warm role=%s model=%s", role, state.name)

    def touch(self, role: str) -> None:
        """Extend keep-warm window based on last usage."""
        role = role.lower().strip()
        now = time.time()

        # touch is sync for ease of integration; schedule coroutine under the hood.
        async def _touch() -> None:
            async with self._lock:
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
        async with self._lock:
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
                async with self._lock:
                    state = self._runtime[role]
                    # keep it warm but not ACTIVE
                    state.status = "IDLE"
                logger.info("model_preload_finished role=%s model=%s", role, self._runtime[role].name)
            except Exception:
                logger.exception("model_preload_failed role=%s", role)

        async def _kickoff() -> None:
            async with self._lock:
                model = self._model_managed(role)
                state = self._ensure_state(role, model.name)
                if state.status == "COLD":
                    state.status = "LOADING"
                    if state.pending_load_task is None:
                        logger.info("model_preload_started role=%s model=%s", role, state.name)
                        state.pending_load_task = asyncio.create_task(_preload())

        asyncio.create_task(_kickoff())

    async def _begin_inference(self, role: str) -> None:
        role = role.lower().strip()
        async with self._lock:
            state = self._runtime.get(role)
            if state is None:
                # Inference without warm: create state to keep counters consistent.
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
        async with self._lock:
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
        return st.status in {"WARM", "IDLE", "LOADING"}



    async def evict_if_needed(self) -> None:
        """Evict models whose keep-warm window has expired."""
        now = time.time()
        candidates: list[ModelRuntimeState] = []

        async with self._lock:
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
                candidates.append(state)


        # Evict lowest priority first
        candidates.sort(key=lambda s: self._policy(s.role).priority)

        for state in candidates:
            await self._evict(role=state.role)

    async def _evict(self, role: str) -> None:
        async with self._lock:
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
            if state.status == "LOADING" and state.pending_load_task is not None:
                # avoid interrupting load
                return
            state.status = "EVICTABLE"
            logger.info("model_eviction_started role=%s model=%s", role, state.name)

        # Best-effort: ask Ollama to unload model.
        # Ollama supports /api/unload; this client doesn't wrap it yet.
        # We call via a raw endpoint if available through client transport.
        try:
            await self._call_unload_model(state.name)
        except Exception:
            logger.exception("model_eviction_failed role=%s model=%s", role, state.name)
            async with self._lock:
                state = self._runtime.get(role)
                if state is not None:
                    state.status = "IDLE"
            return

        async with self._lock:
            state = self._runtime.get(role)
            if state is not None:
                state.status = "UNLOADED"
                state.pending_load_task = None
                state.keep_warm_until = None
            logger.info("model_evicted role=%s model=%s", role, state.name if state else "")

    async def _call_unload_model(self, model_name: str) -> None:
        """Attempt to unload a model.

        The current OllamaClient doesn't provide an explicit unload API.
        This lifecycle manager is isolated from orchestration; eviction is
        best-effort. If the endpoint isn't available, eviction becomes a
        no-op.
        """
        # Preferred: if underlying client exposes httpx AsyncClient at .client
        raw_client = getattr(self.ollama, "client", None)
        base_url = getattr(self.ollama.settings, "ollama_base_url", None) if hasattr(self.ollama, "settings") else None

        if raw_client is None or base_url is None:
            # No transport; no-op.
            return

        # Construct request directly.
        # Note: Ollama API: POST /api/unload {"name": "model"}
        payload = {"name": model_name}
        try:
            sanitized = sanitize_for_json(payload)
            validate_json_serializable(sanitized)
        except SerializationError:
            raise
        resp = await raw_client.post("/api/unload", json=sanitized)
        resp.raise_for_status()

    async def _cleanup_loop(self) -> None:
        while not self._closing:
            try:
                await asyncio.sleep(self._poll_interval_s)
                await self.evict_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("model_lifecycle_cleanup_loop_error")

