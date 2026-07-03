---
name: ml-inference
description: >-
  Phase 3 inference and deployment patterns: pipelines, batch generate, Spaces.
  Use after ml-research and ml-plan-validate when serving or running models.
---

# Inference (Phase 3)

## Flow

1. Research current inference API (pipeline, `generate`, vLLM, etc.)
2. Resolve model via Hub search + details (dtype, device map, gated)
3. Script or app: load model, run representative inputs, handle batching/errors
4. Run locally for dev; use Jobs or Spaces only with user approval

## Local GPU

- Match `device_map` / dtype to available VRAM
- Smoke test one batch before long runs

## Spaces (optional)

- Use HF MCP space tools when configured: search existing Spaces, avoid duplicating hosted demos
- `dynamic_space` only for MCP-enabled Spaces per Hub docs

## Deliverables

- How to run (command or Space URL)
- Expected latency / memory notes
- Input/output format for callers

## Done when

A reproducible inference path exists and user knows how to invoke it.
