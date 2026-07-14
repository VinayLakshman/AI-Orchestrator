from __future__ import annotations

import json
import re
from typing import Any

from ..common.enums import ChatRole
from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse
from ..schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    NormalizedRequest,
    ToolResult,
)

logger = get_logger(__name__)

FINALIZE_CONTEXT_TOKEN_BUDGET = 700
FINALIZE_SOURCE_ORDER = {
    "knowledge": 0,
    "coder": 1,
    "vision": 2,
    "tool": 3,
    "reasoning": 4,
}


def estimate_text_tokens(text: str | None) -> int:
    value = (text or "").strip()
    if not value:
        return 0
    return max(1, len(re.findall(r"\S+", value)))


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _truncate(text: str, limit: int = 220) -> str:
    cleaned = _normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _dedupe_text_items(values: list[str]) -> tuple[list[str], int]:
    seen: set[str] = set()
    deduped: list[str] = []
    removed = 0
    for value in values:
        item = _normalize_text(value)
        if not item:
            continue
        key = item.lower()
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, removed


def _question_terms(question: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]+", question or "")
        if len(token) >= 3
    }
    stopwords = {
        "what",
        "when",
        "where",
        "which",
        "that",
        "this",
        "with",
        "from",
        "your",
        "about",
        "into",
        "how",
        "why",
        "are",
        "was",
        "the",
        "and",
        "for",
        "use",
        "used",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "please",
    }
    return {token for token in tokens if token not in stopwords}


def _content_relevance(question_terms: set[str], content: str) -> float:
    if not question_terms:
        return 1.0 if content.strip() else 0.0
    content_terms = {
        token.lower()
        for token in re.findall(r"[a-zA-Z0-9_][a-zA-Z0-9_\-]+", content or "")
        if len(token) >= 3
    }
    if not content_terms:
        return 0.0
    overlap = len(question_terms & content_terms)
    return overlap / max(1, len(question_terms))


def _collect_hit_evidence(
    hits: list[Any],
    *,
    question_terms: set[str],
    max_items: int,
    max_chars: int,
) -> tuple[list[str], list[str], list[str], int]:
    evidence: list[str] = []
    repositories: list[str] = []
    extra: list[str] = []
    ignored = 0

    for hit in hits[:max_items]:
        content = _truncate(getattr(hit, "content", "") or "", max_chars)
        if not content:
            continue
        relevance = _content_relevance(question_terms, content)
        if question_terms and relevance < 0.15:
            ignored += 1
            continue
        evidence.append(content)
        repository = _normalize_text(getattr(hit, "repository", "") or "")
        if repository:
            repositories.append(repository)
        extra.append(_truncate(content, max_chars))

    return evidence, repositories, extra, ignored


def _compact_lines(text: str | None, *, max_items: int, max_chars: int) -> list[str]:
    lines = [
        _truncate(line.strip(" -•\t"), max_chars)
        for line in str(text or "").splitlines()
        if line.strip()
    ]
    deduped, _ = _dedupe_text_items(lines)
    return deduped[:max_items]


def _source_text_items(source: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("evidence", "extended_evidence", "observations", "code_snippets"):
        value = source.get(key)
        if isinstance(value, list):
            items.extend(str(item) for item in value if str(item).strip())
    summary = source.get("summary")
    if isinstance(summary, str) and summary.strip():
        items.insert(0, summary)
    return items


def _source_token_count(source: dict[str, Any]) -> int:
    total = 0
    for item in _source_text_items(source):
        total += estimate_text_tokens(item)
    return total


def _trim_source_once(source: dict[str, Any]) -> bool:
    for key in ("extended_evidence", "evidence", "observations", "code_snippets"):
        value = source.get(key)
        if isinstance(value, list) and len(value) > 1:
            value.pop()
            source[key] = value
            return True
    for key in ("summary",):
        value = source.get(key)
        if isinstance(value, str) and len(value) > 80:
            source[key] = _truncate(value, max(40, len(value) // 2))
            return True
    return False


def _fit_context_to_budget(context: dict[str, Any], *, budget_tokens: int) -> tuple[dict[str, Any], int]:
    question = _normalize_text(str(context.get("question", "") or ""))
    sources = [dict(source) for source in context.get("sources", []) or []]

    def total_tokens() -> int:
        total = estimate_text_tokens(question)
        for source in sources:
            total += estimate_text_tokens(json.dumps(source, ensure_ascii=False, separators=(",", ":")))
        return total

    before = total_tokens()
    if before <= budget_tokens:
        return {"question": question, "sources": sources}, 0

    ordered_indices = sorted(
        range(len(sources)),
        key=lambda index: (
            FINALIZE_SOURCE_ORDER.get(str(sources[index].get("type", "")).lower(), 99),
            -_source_token_count(sources[index]),
        ),
    )

    changed = 0
    while total_tokens() > budget_tokens:
        progress = False
        for index in sorted(
            range(len(sources)),
            key=lambda idx: (-_source_token_count(sources[idx]), FINALIZE_SOURCE_ORDER.get(str(sources[idx].get("type", "")).lower(), 99)),
        ):
            if _trim_source_once(sources[index]):
                changed += 1
                progress = True
                break
        if progress:
            continue
        # Last-resort trim of the longest summary field when all sources are already minimal.
        longest_index = None
        longest_tokens = -1
        for index in ordered_indices:
            tokens = _source_token_count(sources[index])
            if tokens > longest_tokens:
                longest_tokens = tokens
                longest_index = index
        if longest_index is None:
            break
        if not _trim_source_once(sources[longest_index]):
            break
        changed += 1

    return {"question": question, "sources": sources}, changed


def _coerce_request(request: NormalizedRequest | dict[str, Any] | None) -> NormalizedRequest | None:
    if request is None:
        return None
    if isinstance(request, NormalizedRequest):
        return request
    if isinstance(request, dict):
        try:
            return NormalizedRequest.model_validate(request)
        except Exception:
            return None
    return None


def _coerce_knowledge(value: Any) -> KnowledgeRetrieveResponse | None:
    if value is None:
        return None
    if isinstance(value, KnowledgeRetrieveResponse):
        return value
    if isinstance(value, dict):
        try:
            return KnowledgeRetrieveResponse.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_coder(value: Any) -> CoderResult | None:
    if value is None:
        return None
    if isinstance(value, CoderResult):
        return value
    if isinstance(value, dict):
        try:
            return CoderResult.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_tool(value: Any) -> ToolResult | None:
    if value is None:
        return None
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        try:
            return ToolResult.model_validate(value)
        except Exception:
            return None
    return None


def _coerce_generation(value: Any) -> ModelGenerationResponse | None:
    if value is None:
        return None
    if isinstance(value, ModelGenerationResponse):
        return value
    if isinstance(value, dict):
        try:
            return ModelGenerationResponse.model_validate(value)
        except Exception:
            return None
    return None


def last_user_text(messages: list[dict[str, Any]] | None) -> str:
    if not messages:
        return ""

    for message in reversed(messages):
        if message.get("role") != ChatRole.USER.value:
            continue

        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []

            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and part.get("text")
                ):
                    parts.append(str(part["text"]).strip())

            return "\n".join(parts).strip()

        if content is not None:
            return str(content)

    return ""


def _state_messages_to_chat_messages(
    messages: list[dict[str, Any]] | None,
) -> list[ChatMessage]:
    if not messages:
        return []

    return [ChatMessage.model_validate(message) for message in messages]


def render_structured_context(
    *,
    vision_context: str = "",
    knowledge_result: KnowledgeRetrieveResponse | None = None,
    coder_result: CoderResult | None = None,
    tool_result: ToolResult | None = None,
    reasoning_result: ModelGenerationResponse | None = None,
    controller_plan: ControllerPlan | None = None,
    controller_validation: ControllerValidation | None = None,
) -> str:
    parts: list[str] = []

    def add(title: str, value: str) -> None:
        value = (value or "").strip()
        if value:
            parts.extend([f"## {title}", value, ""])

    if controller_plan:
        add("Controller Plan", controller_plan.model_dump_json(indent=2))
    if controller_validation and (
        not controller_plan
        or controller_validation.model_dump(exclude_none=True)
        != controller_plan.model_dump(exclude_none=True)
    ):
        add("Controller Validation", controller_validation.model_dump_json(indent=2))

    if knowledge_result and knowledge_result.context:
        add("Knowledge Context", knowledge_result.context)

    if vision_context:
        add("Vision Context", vision_context)

    if coder_result and (coder_result.summary or coder_result.code):
        add(
            "Coder Result",
            coder_result.model_dump_json(indent=2),
        )

    if tool_result and (tool_result.summary or tool_result.result):
        add(
            "Tool Result",
            tool_result.model_dump_json(indent=2),
        )

    if reasoning_result and reasoning_result.content:
        add("Reasoning Result", reasoning_result.content)

    return "\n".join(parts).strip()


def render_request_context(request: NormalizedRequest | dict[str, Any] | None) -> str:
    request = _coerce_request(request)
    if request is None:
        return ""

    payload = {
        "query": request.user_query or "",
        "metadata": {
            "message_count": int(request.metadata.get("message_count", 0) or 0),
            "images": int(request.metadata.get("image_count", 0) or 0),
            "files": int(len([item for item in request.attachments if item.attachment_type != "image"])),
            "urls": int(bool(request.metadata.get("contains_urls", False))),
            "code_blocks": int(bool(request.metadata.get("contains_code_blocks", False))),
            "estimated_tokens": int(request.metadata.get("estimated_prompt_tokens", 0) or 0),
        },
        "routing_hints": {
            "repository": round(float(request.routing_hints.repository_likelihood), 2),
            "code": round(float(request.routing_hints.code_likelihood), 2),
            "vision": round(float(request.routing_hints.vision_likelihood), 2),
        },
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def build_finalize_context(state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(state or {})
    request = _coerce_request(state.get("normalized_request"))
    question = _normalize_text(request.user_query if request else last_user_text(state.get("messages", [])))
    question_terms = _question_terms(question)

    sources: list[dict[str, Any]] = []
    raw_evidence_count = 0
    removed_duplicates = 0
    input_tokens = estimate_text_tokens(question)

    knowledge = _coerce_knowledge(state.get("knowledge_result"))
    if knowledge is not None:
        primary_hits = list(knowledge.primary_hits or [])
        extended_hits = list(knowledge.expanded_hits or [])
        primary_evidence, primary_repos, _, primary_ignored = _collect_hit_evidence(
            primary_hits,
            question_terms=question_terms,
            max_items=6,
            max_chars=240,
        )
        extended_evidence, extended_repos, _, extended_ignored = _collect_hit_evidence(
            extended_hits,
            question_terms=question_terms,
            max_items=4,
            max_chars=180,
        )
        raw_evidence_count += len(primary_evidence) + len(extended_evidence)
        removed_duplicates += primary_ignored + extended_ignored

        primary_evidence, removed = _dedupe_text_items(primary_evidence)
        removed_duplicates += removed
        extended_evidence, removed = _dedupe_text_items(extended_evidence)
        removed_duplicates += removed

        if not primary_evidence and extended_evidence:
            primary_evidence, extended_evidence = extended_evidence[:3], []

        if primary_evidence:
            sources.append(
                {
                    "type": "knowledge",
                    "repository": _truncate(
                        ", ".join(dict.fromkeys(primary_repos or extended_repos)),
                        80,
                    ),
                    "confidence": round(float(knowledge.confidence or 0.0), 2),
                    "evidence": primary_evidence[:5],
                    "extended_evidence": extended_evidence[:3],
                }
            )

    vision_context = _normalize_text(str(state.get("vision_context", "") or ""))
    vision = state.get("vision")
    vision_observations: list[str] = []
    if vision_context:
        vision_observations.extend(_compact_lines(vision_context, max_items=5, max_chars=180))
        input_tokens += estimate_text_tokens(vision_context)
        raw_evidence_count += len(vision_observations)
    if isinstance(vision, dict):
        for key in ("summary", "observations", "answer_context", "ocr", "layout", "metrics"):
            value = _normalize_text(str(vision.get(key, "") or ""))
            if value:
                vision_observations.append(_truncate(value, 180))
                input_tokens += estimate_text_tokens(value)
                raw_evidence_count += 1
    deduped_vision, removed = _dedupe_text_items(vision_observations)
    removed_duplicates += removed
    if deduped_vision:
        sources.append(
            {
                "type": "vision",
                "confidence": round(float((vision or {}).get("confidence", 0.0) if isinstance(vision, dict) else 0.0), 2),
                "observations": deduped_vision[:6],
            }
        )

    coder = _coerce_coder(state.get("coder_result"))
    if coder is not None:
        coder_evidence = []
        summary = _truncate(coder.summary, 180)
        code_snippet = _truncate(coder.code, 360)
        if summary:
            coder_evidence.append(summary)
            input_tokens += estimate_text_tokens(summary)
            raw_evidence_count += 1
        if code_snippet and code_snippet != summary:
            coder_evidence.append(code_snippet)
            input_tokens += estimate_text_tokens(code_snippet)
            raw_evidence_count += 1
        deduped_coder, removed = _dedupe_text_items(coder_evidence)
        removed_duplicates += removed
        if deduped_coder:
            payload: dict[str, Any] = {
                "type": "coder",
                "confidence": round(float(coder.confidence or 0.0), 2),
                "summary": deduped_coder[0],
            }
            if len(deduped_coder) > 1:
                payload["code_snippets"] = deduped_coder[1:3]
            sources.append(payload)

    tool = _coerce_tool(state.get("tool_result"))
    if tool is not None:
        tool_evidence: list[str] = []
        if tool.summary:
            tool_evidence.append(_truncate(tool.summary, 180))
            input_tokens += estimate_text_tokens(tool.summary)
            raw_evidence_count += 1
        if tool.result:
            compact_result = _truncate(json.dumps(tool.result, ensure_ascii=False, separators=(",", ":")), 220)
            if compact_result:
                tool_evidence.append(compact_result)
                input_tokens += estimate_text_tokens(compact_result)
                raw_evidence_count += 1
        deduped_tool, removed = _dedupe_text_items(tool_evidence)
        removed_duplicates += removed
        if deduped_tool:
            payload = {
                "type": "tool",
                "tool_name": _truncate(tool.tool_name or "tool", 80),
                "status": _truncate(tool.status or "ok", 40),
                "evidence": deduped_tool[:4],
            }
            sources.append(payload)

    reasoning = _coerce_generation(state.get("reasoning_result"))
    if reasoning is not None and reasoning.content:
        reasoning_items = [
            _truncate(part, 180)
            for part in re.split(r"[\n\r]+", reasoning.content)
            if part.strip()
        ]
        deduped_reasoning, removed = _dedupe_text_items(reasoning_items)
        removed_duplicates += removed
        raw_evidence_count += len(reasoning_items)
        input_tokens += estimate_text_tokens(reasoning.content)
        if deduped_reasoning:
            sources.append(
                {
                    "type": "reasoning",
                    "summary": deduped_reasoning[0],
                    "evidence": deduped_reasoning[1:4],
                }
            )

    context = {
        "question": question,
        "sources": sources,
    }
    context, trimmed_sources = _fit_context_to_budget(context, budget_tokens=FINALIZE_CONTEXT_TOKEN_BUDGET)
    rendered = json.dumps(context, separators=(",", ":"), ensure_ascii=False).strip()

    logger.debug(
        "finalize_context_built %s",
        json.dumps(
            {
                "context_builder_input_tokens": input_tokens,
                "context_builder_output_tokens": estimate_text_tokens(rendered),
                "evidence_count": raw_evidence_count,
                "duplicate_evidence_removed": removed_duplicates,
                "context_budget_tokens": FINALIZE_CONTEXT_TOKEN_BUDGET,
                "context_trimmed_sources": trimmed_sources,
            },
            sort_keys=True,
            default=str,
        ),
    )
    return context


def render_finalize_context(state: dict[str, Any] | None) -> str:
    return json.dumps(build_finalize_context(state), separators=(",", ":"), ensure_ascii=False).strip()


def build_controller_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]] | None = None,
    request_context: str = "",
    vision_context: str = "",
    knowledge_result: KnowledgeRetrieveResponse | None = None,
    coder_result: CoderResult | None = None,
    tool_result: ToolResult | None = None,
    reasoning_result: ModelGenerationResponse | None = None,
    controller_plan: ControllerPlan | None = None,
    controller_validation: ControllerValidation | None = None,
    latest_user_message: str | None = None,
) -> list[ChatMessage]:
    outgoing: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)]

    if request_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "normalized_request"},
                content=request_context,
            )
        )

    structured_context = render_structured_context(
        vision_context=vision_context,
        knowledge_result=knowledge_result,
        coder_result=coder_result,
        tool_result=tool_result,
        reasoning_result=reasoning_result,
        controller_plan=controller_plan,
        controller_validation=controller_validation,
    )
    if structured_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "structured_context"},
                content=structured_context,
            )
        )

    outgoing.extend(_state_messages_to_chat_messages(messages))

    if latest_user_message and last_user_text(messages) != latest_user_message:
        outgoing.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

    return outgoing
