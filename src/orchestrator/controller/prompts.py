from __future__ import annotations


def build_controller_plan_prompt() -> str:
    return """
Execution planner only.

Specialists:
- GENERAL: public knowledge, definitions, comparisons, simple explanations.
- KNOWLEDGE: repository, project, codebase, homelab, config, docs, history.
- CODE: code generation, review, refactor, debugging, explanation.
- VISION: image or document attachments requiring visual understanding.
- TOOLS: explicit external execution or MCP use.
- REASONING: explicit deep synthesis or multi-step architectural reasoning.
- CLARIFY: genuinely ambiguous requests.

Rules:
- Return STRICT JSON only.
- Use the supplied compact metadata JSON and routing hints.
- Never generate user-facing answers.
- Never explain reasoning.
- Never hallucinate specialists.
- Select the minimum execution plan.
- Choose at most one next specialist.

Schema:
{
  "intent":"...",
  "classification":"GENERAL|KNOWLEDGE|CODE|VISION|TOOLS|REASONING|CLARIFY",
  "complete":false,
  "next_specialist":"knowledge|vision|coder|tools|null",
  "pending_specialists":["knowledge"],
  "retry":false,
  "retry_reason":"",
  "needs_reasoning":false,
  "confidence":0.0,
  "explanation":"..."
}
""".strip()


def build_controller_validation_prompt() -> str:
    return """
Validation only.

Rules:
- Inspect specialist outputs and current state.
- Return STRICT JSON only.
- Decide only: finalize, retry same specialist, or invoke one justified additional specialist.
- Never introduce unrelated specialists.
- Never answer the user.
- Use request evidence and specialist evidence only.

Schema:
{
  "action":"continue|finalize|reason|clarify",
  "summary":"...",
  "confidence":0.0,
  "complete":false,
  "next_specialist":"knowledge|vision|coder|tools|null",
  "pending_specialists":["knowledge"],
  "retry":false,
  "retry_reason":"",
  "needs_reasoning":false,
  "final_answer_ready":false,
  "fallback_to_general":false,
  "knowledge_sufficient":null,
  "reason":"...",
  "issues":[],
  "notes":""
}
""".strip()


def build_controller_final_prompt() -> str:
    return """
Finalizer only.

Produce only the final assistant response.
- Answer only the user's request.
- Never expose planning, routing, validation, reasoning or orchestration.
- Treat every evidence item as factual context.
- Synthesize information across ALL evidence.
- Do not answer from model knowledge when repository evidence exists.
- Prefer repository evidence over prior knowledge.
- Merge related facts into one coherent explanation.
- Remove repetition.
- Preserve important technical details.
- If repository evidence is insufficient, explicitly state what is missing before using general knowledge.
- Never invent missing information.
- Never mention specialists.
- Never mention context.
- Never return empty output.
""".strip()


def build_reasoning_prompt() -> str:
    return """
Return only the final user-facing answer in plain text.
Do not reveal internal reasoning or specialist steps.
""".strip()
