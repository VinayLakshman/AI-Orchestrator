from __future__ import annotations

import asyncio
from typing import Any

from ..clients.openwebui import (
    OpenWebUIAuthError,
    OpenWebUIClient,
    OpenWebUIError,
    OpenWebUIImageGenerationError,
    OpenWebUIUnreachableError,
)
from ..logging import get_logger
from ..models.state import OrchestratorState
from ..models.evidence import EvidenceLedger
from ..settings import Settings
from ..streaming.context import get_current_stream

logger = get_logger(__name__)


def make_image_generation_node(
    settings: Settings,
    *,
    model_lifecycle: Any,
    client_registry: Any,
):
    """Create an image-generation node that delegates to Open WebUI.

    Open WebUI is the single source of truth for all image-generation
    configuration (ComfyUI workflow, node mappings, model, size, steps).
    This node only:

    1. Acquires GPU ownership via the existing llama-router lifecycle.
    2. Calls Open WebUI's authenticated image-generation API.
    3. Returns the image results produced by Open WebUI.
    4. Releases GPU ownership only after confirmed completion.

    The endpoint is synchronous: a successful HTTP response means Open WebUI
    has fully completed the generation. A timeout/connection error leaves the
    workload state unknown, so GPU ownership is conservatively retained.

    ``model_lifecycle`` and ``client_registry`` are injected at construction
    time from the runtime (they are NOT reachable via ``execution.runtime``,
    which is per-request execution state).
    """

    if model_lifecycle is None:
        raise ValueError("ModelLifecycle is required for the image-generation node")

    async def image_generation_node(state: OrchestratorState) -> OrchestratorState:
        request = state.request
        response = state.response
        evidence: EvidenceLedger = state.evidence

        stream = get_current_stream()

        try:
            openwebui_client = client_registry.get("openwebui")
        except KeyError as exc:
            raise ValueError(
                "Open WebUI client is not registered in the client registry "
                "(required for image generation)"
            ) from exc
        if not openwebui_client or not isinstance(openwebui_client, OpenWebUIClient):
            raise ValueError("Open WebUI client not available in registry")

        lifecycle = model_lifecycle

        # Acquire GPU ownership (waits for LLM inference, evicts resident
        # models through the existing llama-router unload flow).
        logger.info("acquiring_gpu_for_image_generation")
        await lifecycle.acquire_comfyui_gpu()

        prompt = request.user_input or ""
        # Workload tracking for GPU-ownership safety:
        #   "not_started"  -> request not yet dispatched to Open WebUI
        #                     (release on failure is safe)
        #   "unknown"      -> request dispatched but outcome unconfirmed
        #                     (RETAIN ownership — never falsely declare free)
        #   "completed"    -> Open WebUI returned valid image URL(s)
        #                     (release is safe)
        workload_state = "not_started"

        try:
            if stream:
                await stream.image_generation_started(message="Generating image.")

            # Delegate entirely to Open WebUI. It reads its own current
            # persisted configuration on every request; no workflow or
            # generation parameters are constructed here.
            images = await openwebui_client.generate_image(prompt)
            workload_state = "completed"

            image_urls = [image.url for image in images]

            evidence.comfyui = {
                "provider": "open-webui",
                "prompt": prompt,
                "images": image_urls,
                "status": "COMPLETED",
            }

            response.metadata["image_urls"] = image_urls
            if len(image_urls) == 1:
                response.metadata["image_url"] = image_urls[0]
            response.metadata["route"] = "image_generation"

            if stream:
                await stream.image_generation_finished(
                    success=True,
                    image_url=image_urls[0],
                    message="Image generation completed.",
                )

            # The synchronous endpoint returned successfully: the workload is
            # confirmed complete, safe to release GPU ownership.
            logger.info("releasing_gpu_after_image_generation")
            await lifecycle.release_comfyui_gpu()
            return state

        except asyncio.CancelledError:
            # Cancellation while awaiting Open WebUI: the remote workload state
            # is unknown, so retain GPU ownership rather than falsely
            # declaring it free.
            logger.error(
                "image_generation_cancelled_gpu_retained provider=open-webui"
            )
            raise

        except (OpenWebUIAuthError, OpenWebUIImageGenerationError, OpenWebUIUnreachableError) as exc:
            # Definitive failures:
            # - AuthError: credentials missing/rejected; the request never
            #   started a generation.
            # - ImageGenerationError: Open WebUI's synchronous endpoint
            #   definitively failed (HTTP >= 400) or completed with zero
            #   usable images (HTTP success implies the downstream generation
            #   already finished).
            # - UnreachableError: the request never reached Open WebUI, so no
            #   workload was started.
            # No orchestrator-owned ComfyUI workload exists anymore, so
            # releasing GPU ownership is safe and required to avoid wedging.
            if stream:
                await stream.image_generation_finished(
                    success=False,
                    error=str(exc),
                    message="Image generation failed.",
                )
            logger.error("image_generation_failed_releasing_gpu error=%s", exc)
            await lifecycle.release_comfyui_gpu()
            raise ValueError(f"Image generation failed: {exc}") from exc

        except OpenWebUIError as exc:
            # Timeout / transport failure mid-flight / malformed: the actual
            # generation state is UNKNOWN. Conservative invariant: retain GPU
            # ownership instead of releasing while a workload may still be
            # executing. Ownership is retained until an operator restarts or
            # the deployment gains a verifiable job-status/cancellation API.
            logger.error(
                "image_generation_state_unknown_retaining_gpu_ownership "
                "error=%s",
                exc,
            )
            if stream:
                await stream.image_generation_finished(
                    success=False,
                    error=str(exc),
                    message="Image generation failed.",
                )
            raise RuntimeError(
                f"Image generation failed with unknown workload state: {exc}. "
                "GPU ownership retained for safety."
            ) from exc

        except Exception as exc:
            # Unexpected exceptions must never silently corrupt ownership:
            # - Before submission (workload_state == "not_started"): nothing
            #   was dispatched to Open WebUI -> safe to release.
            # - After submission with unconfirmed outcome: retain ownership
            #   (the invariant forbids declaring the GPU free while the
            #   workload state is unknown). NEVER use an unconditional
            #   ``finally: release`` here.
            if workload_state == "not_started":
                logger.exception(
                    "image_generation_pre_submission_failure_releasing_gpu"
                )
                await lifecycle.release_comfyui_gpu()
                raise RuntimeError(
                    f"Image generation failed before submission: {exc}"
                ) from exc

            logger.exception(
                "image_generation_unexpected_failure_retaining_gpu_ownership"
            )
            raise RuntimeError(
                f"Unexpected image-generation failure with unknown workload "
                f"state: {exc}. GPU ownership retained for safety."
            ) from exc

    return image_generation_node
