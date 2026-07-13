from __future__ import annotations

from ..models.vision import VisionAnalysis, VisionTaskType


_TASK_GUIDANCE: dict[VisionTaskType, str] = {
    VisionTaskType.OCR: (
        "Prioritize exact text transcription, preserving numbers, punctuation, casing, indentation, and line order."
    ),
    VisionTaskType.SCREENSHOT: (
        "Prioritize UI layout, control labels, panels, menu structure, and any visible technical content."
    ),
    VisionTaskType.TERMINAL: (
        "Prioritize command output, errors, stack traces, paths, ports, filenames, and exact terminal text."
    ),
    VisionTaskType.CHART: (
        "Prioritize axes, legends, units, trends, values, thresholds, and anomalies."
    ),
    VisionTaskType.DIAGRAM: (
        "Prioritize nodes, edges, labels, arrows, dependencies, ports, and system relationships."
    ),
    VisionTaskType.DOCUMENT: (
        "Prioritize structured text, headings, tables, paragraphs, and OCR fidelity."
    ),
    VisionTaskType.PHOTO: (
        "Prioritize objects, scene context, visible attributes, and anything technically relevant."
    ),
    VisionTaskType.MIXED: (
        "Balance OCR, layout, technical details, and any structured context that would help downstream reasoning."
    ),
}


def build_vision_system_prompt(task_type: VisionTaskType, image_count: int, user_text: str) -> str:
    task_guidance = _TASK_GUIDANCE.get(task_type, _TASK_GUIDANCE[VisionTaskType.MIXED])
    user_hint = f"\nUser text: {user_text.strip()}\n" if user_text.strip() else ""

    return f"""
You are a dedicated technical vision preprocessor for a homelab orchestrator.

Return STRICT JSON ONLY.
Do not wrap the output in markdown fences.
Do not add commentary outside the JSON object.

The JSON schema must be:

{{
  "task_type": "ocr|screenshot|terminal|chart|diagram|document|photo|mixed",
  "confidence": 0.0,
  "summary": "short overall description",
  "ocr": "exact visible text if any",
  "layout": "visible UI/layout structure if relevant",
  "metrics": "numbers, values, chart data, units, trends if relevant",
  "errors_warnings": "visible errors, warnings, exceptions, anomalies if relevant",
  "observations": "technical observations for downstream reasoning",
  "answer_context": "compact, high-signal summary another model can use"
}}

Rules:
- Preserve visible text exactly when possible.
- Keep numbers exact.
- If text is unclear, say so explicitly instead of guessing.
- If there are multiple images, compare them when helpful.
- Optimize for technical content: terminals, Grafana, Docker, Proxmox, Home Assistant, YAML, JSON, logs, code, diagrams, and screenshots.
- Do not answer the user's question directly. Only describe what is visible and provide context another model can use.

Task guidance:
{task_guidance}

There are {image_count} image(s) in this request.
{user_hint}
""".strip()


def render_vision_context(analysis: VisionAnalysis) -> str:
    parts: list[str] = [
        "# Vision Analysis",
        f"- Task type: `{analysis.task_type.value}`",
        f"- Confidence: `{analysis.confidence:.2f}`",
        f"- Images analyzed: `{analysis.image_count}`",
        "",
    ]

    def add_section(title: str, value: str) -> None:
        value = (value or "").strip()
        if not value:
            return
        parts.extend([f"## {title}", value, ""])

    add_section("Summary", analysis.summary)
    add_section("OCR", analysis.ocr)
    add_section("UI / Layout", analysis.layout)
    add_section("Metrics", analysis.metrics)
    add_section("Errors / Warnings", analysis.errors_warnings)
    add_section("Observations", analysis.observations)
    add_section("Answer Context", analysis.answer_context)

    return "\n".join(parts).strip()


def build_vision_injection_message(context_markdown: str, user_text: str = "") -> str:
    system_block = f"""
You have already analyzed the image(s) in this conversation.

Use the analysis below as authoritative context when answering questions about the image(s).

--- VISION ANALYSIS START ---
{context_markdown.strip()}
--- VISION ANALYSIS END ---

Important:
- Answer naturally and directly.
- Use the vision analysis directly.
- Do not say you cannot see the image.
- If the analysis includes OCR, metrics, layout, or errors, use those values exactly where relevant.
""".strip()

    if user_text.strip():
        system_block += f"\n\nThe user's original text was:\n{user_text.strip()}"

    return system_block