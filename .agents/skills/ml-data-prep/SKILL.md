---
name: ml-data-prep
description: >-
  Phase 3 dataset processing on Hugging Face: load, transform, validate schema,
  push to Hub or private storage. Use after ml-research and ml-plan-validate for
  data pipeline tasks.
---

# Data Preparation (Phase 3)

## Pre-flight

- [ ] Research defines target schema (e.g. conversational `messages` for downstream SFT)
- [ ] Source dataset discovered and inspected (columns, splits, size)
- [ ] Output repo or path agreed with user

## Implementation pattern

1. `load_dataset` (or streaming if huge)
2. Map / filter / merge per requirements
3. Validate a few rows manually (print schema + one example)
4. Persist:
   - `push_to_hub` to target dataset repo, or
   - private dataset / files per project conventions

## Jobs vs local

| Workload | Suggestion |
|----------|------------|
| Small / dev | Local `uv run` script |
| Large transforms | HF Job with `cpu-upgrade` or `cpu-performance`; timeout 1–4h typical |

Ask user before large cloud jobs.

## Quality gates

- Row counts and null rates sane after transform
- Column names match what **ml-train-sft** will expect
- No accidental leakage of eval split into train without user intent

## Done when

User has dataset id (or path), schema description, and reproduction command.
