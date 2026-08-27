"""Client for Open WebUI's existing image-generation API.

Open WebUI is the single source of truth for all image-generation
configuration (engine, ComfyUI base URL, workflow, workflow node mappings,
image model, prompt model, size, steps). This client only performs an
authenticated call to Open WebUI's own ``/api/v1/images/generations``
endpoint so that Open WebUI reads its *current persisted* configuration on
every request. Nothing is cached or copied into the orchestrator.
"""

from __future__ import annotations

import httpx

from ..logging import get_logger
from ..settings import Settings

logger = get_logger(__name__)


class OpenWebUIError(RuntimeError):
    """Base error for Open WebUI image-generation delegation failures."""


class OpenWebUIAuthError(OpenWebUIError):
    """Authentication against Open WebUI failed."""


class OpenWebUIImageGenerationError(OpenWebUIError):
    """Open WebUI rejected or failed the image-generation request.

    Raised only when the request definitively failed (HTTP >= 400, or a
    successful response containing zero usable image URLs). Open WebUI's
    ``/api/v1/images/generations`` endpoint is synchronous: an HTTP success
    implies the downstream generation already completed, so an empty result
    is a definitive failure — not an ambiguous state.
    """


class OpenWebUIUnreachableError(OpenWebUIError):
    """Open WebUI could not be reached at connection time.

    The request never reached the server, so no image-generation workload
    was started. Callers may safely release GPU ownership.
    """


class GeneratedImage:
    """A single image result returned by Open WebUI."""

    __slots__ = ("url",)

    def __init__(self, url: str) -> None:
        self.url = url

    def __repr__(self) -> str:
        return f"GeneratedImage(url={self.url!r})"


class OpenWebUIClient:
    """Delegates image generation to Open WebUI's existing implementation.

    The endpoint is synchronous: when the HTTP response returns successfully,
    Open WebUI has already completed the generation (including its ComfyUI
    workflow execution) and uploaded the resulting images as Open WebUI file
    objects. That gives the orchestrator a deterministic completion signal for
    GPU ownership release.
    """

    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        # Ownership: when a client is injected (by the runtime/client
        # registry) it is SHARED and owned by the runtime — this class must
        # never close it. When ``client`` is None, this class creates its own
        # short-lived client per request and closes that one.
        self.client = client
        self.base_url = settings.openwebui_base_url.rstrip("/")
        # Service origins that may appear in Open WebUI-generated file URLs.
        # These are STRIPPED back to relative paths for browser delivery —
        # they are never used to prefix image results.
        legacy_public = (settings.openwebui_public_base_url or "").rstrip("/")
        self._service_origins: tuple[str, ...] = tuple(
            origin for origin in (self.base_url, legacy_public) if origin
        )

    def _create_owned_client(self) -> httpx.AsyncClient:
        """Create a NEW AsyncClient owned by the caller of generate_image."""
        timeout = httpx.Timeout(
            max(self.settings.image_generation_timeout, 60.0),
            connect=10.0,
        )
        return httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    def _image_result_url(self, url: str) -> str:
        """Return the BROWSER-FACING representation of an Open WebUI image URL.

        Generated images are rendered by the user's browser through Open WebUI,
        which resolves relative paths against its own public origin. The
        Docker-internal service hostname is NOT resolvable by the browser, so:

        - ``/api/v1/files/<id>/content`` is preserved EXACTLY as returned;
        - an absolute URL pointing at one of our configured Open WebUI service
          origins has that origin stripped back to its relative path
          (``http://open-webui:8080/api/v1/files/<id>/content``
          → ``/api/v1/files/<id>/content``);
        - any other external URL is left untouched.

        Query strings, fragments and percent-encoding are never modified.
        """
        cleaned = url.strip()
        if not cleaned:
            return ""
        for origin in sorted(self._service_origins, key=len, reverse=True):
            if cleaned.startswith(f"{origin}/"):
                remainder = cleaned[len(origin):]
                # Guarantee exactly one leading slash.
                return "/" + remainder.lstrip("/") if remainder else "/"
        return cleaned

    def _auth_headers(self) -> dict[str, str]:
        api_key = self.settings.openwebui_api_key
        if not api_key:
            raise OpenWebUIAuthError(
                "openwebui_api_key is not configured; cannot authenticate "
                "against Open WebUI for image generation"
            )
        return {"Authorization": f"Bearer {api_key}"}

    async def generate_image(
        self,
        prompt: str,
        *,
        model: str | None = None,
        n: int | None = None,
        size: str | None = None,
    ) -> list[GeneratedImage]:
        """Generate images via POST /api/v1/images/generations.

        Only the prompt (and optional explicit overrides) is sent. All other
        parameters are resolved by Open WebUI from its own live configuration.
        """
        payload: dict[str, object] = {"prompt": prompt}
        # Optional fields are only included when explicitly provided so that
        # Open WebUI's configured defaults always apply otherwise.
        if model:
            payload["model"] = model
        if n:
            payload["n"] = n
        if size:
            payload["size"] = size

        headers = self._auth_headers()
        logger.info(
            "openwebui_image_generation_request url=%s prompt_len=%d",
            f"{self.base_url}/api/v1/images/generations",
            len(prompt),
        )

        try:
            if self.client is not None:
                # Injected/shared client: use it directly. The runtime owns
                # its lifecycle — NEVER close it here (no ``async with``).
                response = await self.client.post(
                    "/api/v1/images/generations",
                    json=payload,
                    headers=headers,
                )
            else:
                # No injected client: create one owned by this call and close
                # it after the request completes.
                async with self._create_owned_client() as client:
                    response = await client.post(
                        "/api/v1/images/generations",
                        json=payload,
                        headers=headers,
                    )
        except httpx.ConnectError as exc:
            # Connection-level failure BEFORE the request reached the server:
            # no image-generation workload was started, so this is a
            # definitive failure (safe for callers to release GPU ownership).
            logger.error("openwebui_unreachable error=%r", exc)
            raise OpenWebUIUnreachableError(
                f"Open WebUI is unreachable: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            # Timeout while awaiting the synchronous generation: the workload
            # state is UNKNOWN and may still be running inside Open WebUI /
            # ComfyUI. Callers must retain GPU ownership.
            logger.error("openwebui_request_timeout error=%r", exc)
            raise OpenWebUIError(
                "Open WebUI image-generation request timed out; "
                "workload state unknown"
            ) from exc
        except httpx.HTTPError as exc:
            # Any other transport error mid-flight: the actual generation
            # state is unknown; treat conservatively.
            logger.error("openwebui_transport_error error=%r", exc)
            raise OpenWebUIError(
                f"Open WebUI transport failure; workload state unknown: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise OpenWebUIAuthError(
                f"Open WebUI rejected credentials (HTTP {response.status_code})"
            )
        if response.status_code >= 400:
            detail = response.text[:500]
            raise OpenWebUIImageGenerationError(
                f"Open WebUI image generation failed "
                f"(HTTP {response.status_code}): {detail}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise OpenWebUIError(
                "Malformed response from Open WebUI image generation"
            ) from exc

        return self._parse_images(body)

    def _parse_images(self, body: object) -> list[GeneratedImage]:
        """Parse Open WebUI's response: a JSON list of ``{"url": ...}`` items."""
        items: list[dict[str, object]]
        if isinstance(body, list):
            items = [item for item in body if isinstance(item, dict)]
        elif isinstance(body, dict):
            data = body.get("data")
            if isinstance(data, list):
                items = [item for item in data if isinstance(item, dict)]
            else:
                items = []
        else:
            items = []

        urls: list[GeneratedImage] = []
        for item in items:
            url = item.get("url")
            if isinstance(url, str) and url.strip():
                urls.append(
                    GeneratedImage(url=self._image_result_url(url.strip()))
                )

        if not urls:
            raise OpenWebUIImageGenerationError(
                "Open WebUI returned no image results"
            )

        logger.info("openwebui_image_generation_success count=%d", len(urls))
        return urls
