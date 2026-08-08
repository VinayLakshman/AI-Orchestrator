"""Context construction and validation.

This package is split into three layers:

- ``orchestrator.context.parser``: conversation parsing / history management
  (extracting the latest user message, preserving history, token budgeting).
- ``orchestrator.context.assembler``: the single authoritative conversation
  assembly layer. It constructs valid outbound conversations from structured
  inputs and validates them against the core OpenAI-compatible invariants.
- ``orchestrator.context.builder``: compatibility wrappers and context
  rendering helpers that delegate to the parser/assembler.
"""
