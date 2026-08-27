from __future__ import annotations

THREAD_ID_MAX_LENGTH = 255

# Shared canonical fallback emitted when a genuine (non-image) graph result
# finishes without any assistant text. Used by the finalizer node and the API
# response layer so the wording never diverges between surfaces.
FALLBACK_NO_ANSWER = (
    "I could not generate a complete answer for that request. "
    "Please try again with a little more detail."
)

