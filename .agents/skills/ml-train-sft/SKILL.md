---
name: ml-train-sft
description: >-
  Phase 3 training and fine-tuning (SFT, DPO, etc.) with TRL/transformers, Trackio,
  Hub push, and optional HF Jobs. Use after ml-research and ml-plan-validate.
---

# Training and Fine-Tuning (Phase 3)

Implement only from **ml-research** findings and **ml-plan-validate** choices.

## Pre-flight checklist

- [ ] Research summary and plan are done
- [ ] Base model and dataset validated (including conversational format for SFT)
- [ ] User approved cloud job if not running locally

## Training script expectations

Use current APIs from research—not memorized snippets.

Include as applicable:

- Imports from verified TRL/transformers version
- Trackio (or agreed monitor): project, run name, config
- Model and tokenizer load
- Dataset load with verified columns
- Training args, including when targeting Hub:
  - `push_to_hub=True`
  - `hub_model_id` (user namespace / agreed name)
  - `report_to` including trackio when used
  - `output_dir`, epochs, batch size, LR, logging/save steps
  - `max_length` when relevant
- Trainer construction and `train()`
- `trainer.push_to_hub()` when Hub delivery is required
- Tracker cleanup (`finish()`)

Match the method (SFT, DPO, GRPO) to the dataset schema validated in Phase 2.

## Local run

- Use project tooling (`uv run`, existing `pyproject` deps)
- Start with a **smoke** step (tiny subset, 1 step) when possible
- Report command, env vars (e.g. `HF_TOKEN`), and log path

## HF Jobs (when used)

Confirm with user first. Job payload should include:

- Script with the checklist above
- Dependencies: e.g. `transformers`, `trl`, `torch`, `datasets`, `trackio` (adjust to research)
- `hardware_flavor` from plan
- **Timeout** examples:
  - Small (1–3B): `2h`–`4h`
  - Medium (7–13B): `4h`–`8h`
  - Large (30B+): `8h`–`24h`
- `HF_TOKEN` available in job environment for Hub operations

Job storage is **ephemeral**—artifacts must reach Hub or a persistent dataset.

After submit: verify job started (logs); give Trackio or Hub links when available.

## Common failures

Use **ml-errors** for full playbooks (timeout, Hub push, format, OOM, API errors).

Keep the active plan item in progress until training succeeds or the user accepts a documented partial result.

## Done when

User has: run instructions, artifact locations (Hub repo or local `output_dir`), and monitoring links if applicable.
