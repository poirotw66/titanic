---
name: ml-eval
description: >-
  Phase 3 model evaluation with lighteval, lm-evaluation-harness, or task metrics.
  Use after ml-research and ml-plan-validate for benchmarks and reporting.
---

# Evaluation (Phase 3)

## Flow

1. Research framework choice (lighteval, lm-eval-harness, custom metrics)
2. Pin model revision from Hub
3. Script: load model, run benchmark config, save metrics
4. Execute locally or via Job; store results in Hub dataset or repo user approves

## Planning

- Task list: benchmark suite, hardware, timeout, output format (JSON/table)
- Confirm eval split and no train contamination unless intended

## Hardware

- Match eval suite size; many suites fit `a10g-small` or local GPU for small models
- Long suites: extend job timeout like training (see **ml-train-sft**)

## Reporting

Provide:

- Metric table or path to results file
- Config hash (model id, revision, benchmark name)
- How to reproduce

## Done when

Results are stored, linked, and reproducible from documented commands.
