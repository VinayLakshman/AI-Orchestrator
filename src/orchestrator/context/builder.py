from __future__ import annotations

import json
import re
from typing import Any

from ..common.enums import ChatRole
from ..logging import get_logger
from ..models.chat import ChatMessage
from ..models.knowledge import KnowledgeRetrieveResponse
from ..models.ollama import ModelGenerationResponse
from ..models.web import WebSearchResult
from ..schemas import (
    ControllerPlan,
    ControllerValidation,
    CoderResult,
    NormalizedRequest,
    ToolResult,
)

logger = get_logger(__name__)

FINALIZE_CONTEXT_TOKEN_BUDGET = 700
CONVERSATION_HISTORY_TOKEN_BUDGET = 2400
FINALIZE_SOURCE_ORDER = {
    "knowledge": 0,
    "knowledge_summary": 0,
    "web": 1,
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

def _collect_hit_evidence(
    hits: list[Any],
    *,
    max_items: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []

    for hit in hits[:max_items]:
        content = _truncate(getattr(hit, "content", "") or "", max_chars)
        if not content:
            continue

        score = getattr(hit, "score", None)
        source: dict[str, Any] = {
            "type": "knowledge",
            "repository": _truncate(getattr(hit, "repository", "") or "", 80),
            "branch": _truncate(getattr(hit, "branch", "") or "", 40),
            "commit": _truncate(getattr(hit, "commit", "") or "", 16),
            "path": _truncate(getattr(hit, "path", "") or "", 180),
            "language": _truncate(getattr(hit, "language", "") or "", 32),
            "chunk_index": getattr(hit, "chunk_index", None),
            "chunk_count": getattr(hit, "chunk_count", None),
            "score": round(float(score), 4) if score is not None else None,
            "evidence": [content],
        }
        sources.append(source)

    return sources


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


def _message_role(message: dict[str, Any] | ChatMessage) -> str:
    if isinstance(message, ChatMessage):
        return str(message.role.value)
    return str(message.get("role") or "")


def _message_content_text(message: dict[str, Any] | ChatMessage) -> str:
    content: Any
    if isinstance(message, ChatMessage):
        content = message.content
    else:
        content = message.get("content")

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                for key in ("text", "content", "url"):
                    value = part.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
        return "\n".join(parts).strip()
    if content is None:
        return ""
    return str(content).strip()


def _message_token_count(message: dict[str, Any] | ChatMessage) -> int:
    return estimate_text_tokens(_message_content_text(message))


def _message_sequence(messages: list[dict[str, Any] | ChatMessage]) -> list[str]:
    return [_message_role(message) for message in messages]


def _build_conversation_history(
    messages: list[dict[str, Any] | ChatMessage] | None,
) -> tuple[list[ChatMessage], str, dict[str, Any]]:
    raw_messages = list(messages or [])
    if not raw_messages:
        raise ValueError("Conversation is empty")

    role_sequence = _message_sequence(raw_messages)

    latest_user_index: int | None = None
    for index in range(len(raw_messages) - 1, -1, -1):
        message = raw_messages[index]
        role = message.role if isinstance(message, ChatMessage) else message.get("role")
        if role == ChatRole.USER.value:
            latest_user_index = index
            break

    if latest_user_index is None:
        raise ValueError("Conversation contains no user message")

    latest_user_text = _message_content_text(raw_messages[latest_user_index])

    # Only keep messages BEFORE the active user turn.
    history_messages = list(raw_messages[:latest_user_index])

    history_token_count = sum(
        _message_token_count(message) for message in history_messages
    )

    truncated = False
    while history_messages and history_token_count > CONVERSATION_HISTORY_TOKEN_BUDGET:
        history_messages.pop(0)
        history_token_count = sum(
            _message_token_count(message) for message in history_messages
        )
        truncated = True

    history_chat_messages = [
        ChatMessage.model_validate(message) for message in history_messages
    ]

    conversation_info = {
        "total_message_count": len(raw_messages),
        "role_sequence": role_sequence,
        "latest_user_index": latest_user_index,
        "latest_user_message_length": len(latest_user_text),
        "history_token_count": history_token_count,
        "truncation_occurred": truncated,
        "latest_user_survived_truncation": True,
        "history_message_count": len(history_chat_messages),
    }

    return history_chat_messages, latest_user_text, conversation_info


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
    web_search_result: WebSearchResult | None = None,
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

    if web_search_result and web_search_result.results:
        add("Web Evidence", json.dumps([item.model_dump(exclude_none=True) for item in web_search_result.results], ensure_ascii=False))

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
    question = _normalize_text(
        request.user_query if request else last_user_text(state.get("messages", []))
    )

    sources: list[dict[str, Any]] = []
    raw_evidence_count = 0
    removed_duplicates = 0
    input_tokens = estimate_text_tokens(question)

    knowledge = _coerce_knowledge(state.get("knowledge_result"))
    if knowledge is not None:
        primary_hits = list(knowledge.primary_hits or [])
        extended_hits = list(knowledge.expanded_hits or [])

        # Prefer the synthesized knowledge-service context first.
        knowledge_context = _normalize_text(str(knowledge.context or ""))
        if knowledge_context:
            summary = _truncate(knowledge_context, 1200)
            if summary:
                sources.append(
                    {
                        "type": "knowledge_summary",
                        "repository": _truncate(
                            ", ".join(
                                dict.fromkeys(
                                    [
                                        _normalize_text(getattr(hit, "repository", "") or "")
                                        for hit in (primary_hits + extended_hits)
                                        if _normalize_text(getattr(hit, "repository", "") or "")
                                    ]
                                )
                            ),
                            80,
                        ),
                        "confidence": round(float(knowledge.confidence or 0.0), 2),
                        "grounded": bool(knowledge.grounded),
                        "retrieval_reason": _truncate(str(knowledge.retrieval_reason or ""), 180),
                        "summary": summary,
                    }
                )
                raw_evidence_count += 1
                input_tokens += estimate_text_tokens(summary)

        primary_sources = _collect_hit_evidence(
            primary_hits,
            max_items=6,
            max_chars=240,
        )
        extended_sources = _collect_hit_evidence(
            extended_hits,
            max_items=4,
            max_chars=180,
        )

        # Dedupe by the actual text evidence only; metadata stays attached.
        primary_texts, removed = _dedupe_text_items(
            [item["evidence"][0] for item in primary_sources if item.get("evidence")]
        )
        removed_duplicates += removed

        extended_texts, removed = _dedupe_text_items(
            [item["evidence"][0] for item in extended_sources if item.get("evidence")]
        )
        removed_duplicates += removed

        if primary_texts:
            filtered_primary: list[dict[str, Any]] = []
            seen_texts: set[str] = set()
            for item in primary_sources:
                text = item["evidence"][0]
                key = _normalize_text(text).lower()
                if key in seen_texts or key not in {t.lower() for t in primary_texts}:
                    continue
                seen_texts.add(key)
                filtered_primary.append(item)
            sources.extend(filtered_primary[:5])
            raw_evidence_count += len(filtered_primary)

        if extended_texts:
            filtered_extended: list[dict[str, Any]] = []
            seen_texts = {
                _normalize_text(item["evidence"][0]).lower()
                for item in sources
                if isinstance(item.get("evidence"), list) and item["evidence"]
            }
            for item in extended_sources:
                text = item["evidence"][0]
                key = _normalize_text(text).lower()
                if key in seen_texts or key not in {t.lower() for t in extended_texts}:
                    continue
                seen_texts.add(key)
                filtered_extended.append(item)
            if filtered_extended:
                sources.extend(filtered_extended[:3])
            raw_evidence_count += len(filtered_extended)

    web_value = state.get("web_search_result")
    if web_value:
        try:
            web = web_value if isinstance(web_value, WebSearchResult) else WebSearchResult.model_validate(web_value)
        except Exception:
            web = None
        if web and web.results:
            evidence = []
            for item in web.results[:8]:
                evidence.append({
                    "title": _truncate(item.title, 140),
                    "url": _truncate(item.url, 240),
                    "snippet": _truncate(item.snippet, 280),
                    "engine": _truncate(item.engine, 60),
                })
            sources.append({"type": "web", "query": _truncate(web.query, 240), "evidence": evidence})
            raw_evidence_count += len(evidence)
            logger.info("results_used=%d source=web", len(evidence))

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
                "confidence": round(
                    float((vision or {}).get("confidence", 0.0) if isinstance(vision, dict) else 0.0),
                    2,
                ),
                "observations": deduped_vision[:6],
            }
        )

    coder = _coerce_coder(state.get("coder_result"))
    if coder is not None:
        coder_evidence: list[str] = []
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
            compact_result = _truncate(
                json.dumps(tool.result, ensure_ascii=False, separators=(",", ":")),
                220,
            )
            if compact_result:
                tool_evidence.append(compact_result)
                input_tokens += estimate_text_tokens(compact_result)
                raw_evidence_count += 1
        deduped_tool, removed = _dedupe_text_items(tool_evidence)
        removed_duplicates += removed
        if deduped_tool:
            sources.append(
                {
                    "type": "tool",
                    "tool_name": _truncate(tool.tool_name or "tool", 80),
                    "status": _truncate(tool.status or "ok", 40),
                    "evidence": deduped_tool[:4],
                }
            )

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
    context, trimmed_sources = _fit_context_to_budget(
        context,
        budget_tokens=FINALIZE_CONTEXT_TOKEN_BUDGET,
    )
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
    structured_context: str = "",
    additional_context: str = "",
) -> list[ChatMessage]:
    history_messages, latest_user_message, conversation_info = _build_conversation_history(messages)
    outgoing: list[ChatMessage] = [ChatMessage(role=ChatRole.SYSTEM, content=system_prompt)]

    if request_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "normalized_request"},
                content=request_context,
            )
        )

    if structured_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "structured_context"},
                content=structured_context,
            )
        )

    if additional_context:
        outgoing.append(
            ChatMessage(
                role=ChatRole.SYSTEM,
                metadata={"source": "additional_context"},
                content=additional_context,
            )
        )

    outgoing.extend(history_messages)
    outgoing.append(ChatMessage(role=ChatRole.USER, content=latest_user_message))

    logger.debug(
        "conversation_assembly %s",
        json.dumps(conversation_info, sort_keys=True, default=str),
    )

    return outgoing


def build_finalizer_messages(
    *,
    system_prompt: str,
    messages: list[dict[str, Any]] | None = None,
    evidence_context: str = "",
) -> list[ChatMessage]:
    return build_controller_messages(
        system_prompt=system_prompt,
        messages=messages,
        structured_context=evidence_context,
    )
