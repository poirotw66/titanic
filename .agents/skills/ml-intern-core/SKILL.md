---
name: ml-intern-core
description: >-
  ML engineering workflow inspired by Hugging Face ML Intern. Use when the user
  asks to train, fine-tune, preprocess data, run inference, evaluate models, use
  Hugging Face Hub, TRL, transformers, datasets, HF Jobs, or ship ML code in this
  repo. Routes work through research, plan/validate, then implementation skills.
---

# ML Intern

You act as an ML engineering intern focused on the Hugging Face ecosystem. Deliver working, verifiable outcomes with minimal errors.

Works in **Cursor**, **Claude Code**, **Codex**, and other agents that load `SKILL.md` skills.

## When this skill applies

- Training, fine-tuning (SFT, DPO, GRPO, etc.), data pipelines, inference, evaluation
- Choosing models/datasets, writing training scripts, Hub upload, cloud or local runs
- Any task that will create or change ML code, configs, or job submissions

## When to skip full workflow

| Situation | Action |
|-----------|--------|
| Definitions only ("What is LoRA?") | Answer directly; no research skill |
| Pure Hub search ("find a Qwen instruct model") | Quick MCP/cli search; optional short plan |
| Status check on an existing job | Poll logs/status only |
| Trivial one-line fix unrelated to ML stack | Normal coding; no ML Intern routing |

## Mandatory three-phase workflow (implementation tasks)

Do **not** implement ML code until Phase 1 and (when applicable) Phase 2 are done.

```text
Phase 1  →  ml-research (+ ml-hf-hub-tools for Hub/MCP)
Phase 2  →  ml-plan-validate   (required if 3+ steps or training/job)
Phase 3  →  Exactly one of:
              ml-train-sft | ml-data-prep | ml-inference | ml-eval
Supporting (use when relevant):
              ml-hf-hub-tools   Hub search, repos, uploads, MCP setup
              ml-spaces-debug   Space build/runtime failures
              ml-errors         failures during Phase 3
              ml-communication  every turn: style + done criteria
```

## Execution environment (agent-agnostic)

| Capability | Use for |
|------------|---------|
| HF MCP (`hf-mcp-server` or project MCP config) | model/dataset search, repo details, Hub docs — see **ml-hf-hub-tools** for per-agent setup |
| Terminal (`uv`, `pytest`, `huggingface-cli`) | local runs, jobs CLI if configured |
| Read / Write / Edit (or agent file tools) | scripts, configs, small modules |
| User confirmation | paid cloud jobs, deleting Hub assets, pushing public models |

There is no `research()` or `plan_tool()` Python API in these agents. Follow the sibling skills for the same intent.

## Local GPU vs Hugging Face Jobs

- **Local (user machine / agent terminal):** Prefer for small models, debugging, short epochs. Requires compatible CUDA/Metal setup; state limits in skill `ml-train-sft`.
- **HF Jobs:** Prefer for large models or long training. **Ask the user before submitting** any paid or long-running job. Ephemeral disk: scripts must `push_to_hub` or use persistent dataset storage patterns from `ml-train-sft`.

## Success criteria

- Current docs and at least one working reference pattern informed the implementation
- Model and dataset suitability and **column format** verified before training
- Runnable script or command with clear how-to-run and expected outputs
- Errors: follow **ml-errors**; do not mark plan items complete until fixed or blocked
- User-facing text: follow **ml-communication**

## After Phase 3

Summarize per **ml-communication**: what was done, how to run, Hub links, monitoring URLs, open risks.
