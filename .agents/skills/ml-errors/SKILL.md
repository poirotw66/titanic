---
name: ml-errors
description: >-
  Recover from ML and Hugging Face Job failures: timeouts, Hub push, dataset format,
  OOM, API errors. Use when training, jobs, or Hub operations fail during ml-intern
  workflows.
---

# ML error recovery

When anything fails during **ml-train-sft**, **ml-data-prep**, **ml-inference**, or **ml-eval**:

1. Keep the active plan/todo item **in progress** (do not mark done).
2. Add a todo: diagnose → fix → retry.
3. Apply the matching playbook below, then retry with a **corrected** approach.
4. If stuck, state blocker and ask the user (gated model, quota, missing token).

## Job timeout exceeded

| | |
|--|--|
| **Symptom** | Job stops mid-run; incomplete checkpoints |
| **Cause** | `timeout` too short (e.g. default ~30m for training) |
| **Fix** | Increase timeout by model scale: 1–3B → `2h`–`4h`; 7–13B → `4h`–`8h`; 30B+ → `8h`–`24h`. Reduce data/steps only if user agrees. |

```text
WRONG:  timeout ~30m for full training
RIGHT:  timeout 4h+ for multi-hour SFT
```

## Model not on Hub after training

| | |
|--|--|
| **Symptom** | Job finished; no model repo |
| **Causes** | Missing `push_to_hub=True`, missing `hub_model_id`, missing/invalid `HF_TOKEN`, token lacks write |
| **Fix** | Set `push_to_hub=True`, `hub_model_id="user/model-name"`, ensure `HF_TOKEN` in job env; call `trainer.push_to_hub()`; verify with Hub URL |

## Dataset format mismatch

| | |
|--|--|
| **Symptom** | `KeyError`, column errors, trainer rejects batch |
| **Cause** | Columns do not match method |
| **Fix** | Re-run `hub_repo_details` / inspect schema; map data: SFT → `messages` or compatible text; DPO → `prompt`/`chosen`/`rejected`; GRPO → per current TRL docs; preprocess then retry |

## CUDA OOM

| | |
|--|--|
| **Symptom** | CUDA out of memory in logs |
| **Fix order** | 1) ↑ `gradient_accumulation_steps` 2) ↓ `per_device_train_batch_size` 3) `gradient_checkpointing=True` 4) ↓ `max_length` 5) larger GPU / Job flavor |

Do **not** switch training method (e.g. full SFT → LoRA) or slash `max_length` drastically without **user approval** — that changes what they asked for.

## API / syntax errors

- Re-run **ml-research** for the failing library version.
- Compare stack trace line to current docs example.
- Pin versions in job `dependencies` if drift is the issue.

## Configuration / job rejected

- Read full stderr from Job logs.
- Validate hardware flavor vs model size (see **ml-train-sft**).
- Confirm `HF_TOKEN` and gated model access.

## Hub / auth errors

- 401/403: token scope, OAuth re-login for MCP, or fine-grained token permissions.
- Gated model: user must accept license on Hub.

## After fix

- Retry once with a clear delta (“increased timeout to 8h”, “added messages column map”).
- End turn with **ml-communication** completion checklist if the user request is done.
