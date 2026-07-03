---
name: ml-plan-validate
description: >-
  Phase 2 planning and Hub resource validation for ML tasks: todos, model/dataset
  choice, format checks, hardware. Use after ml-research and before ml-train-sft,
  ml-data-prep, ml-inference, or ml-eval when the task has multiple steps or uses Hub assets.
---

# Plan and Validate (Phase 2)

Required when the task has **3+ steps**, submits a **job**, or **trains / fine-tunes**.

## Step 1: Execution plan

Use the todo tool (or an explicit numbered checklist in chat):

- Exactly **one** item `in_progress` at a time
- Mark **completed** immediately when done (do not batch)
- On failure: keep the failed item in progress; add a new item for the fix
- Do not mark complete if errors remain

Example structure:

1. Research (completed in Phase 1)
2. Validate base model on Hub
3. Validate dataset schema
4. Implement script
5. Run locally or submit job (with user approval if cloud)
6. Verify artifacts and report URLs

## Step 2: Discover resources

Use HF MCP or `huggingface-cli` when MCP is unavailable.

### Models

- Search with clear query (size, instruct/chat, license)
- Shortlist 3–5 candidates; pick one with rationale (size, downloads, license, task fit)
- `hub_repo_details` (or equivalent): architecture, size, license, gated access

### Datasets

- Search with tags if needed (`conversational`, task-specific)
- **Always** inspect schema before training

### Format validation (critical)

| Method | Required columns / shape |
|--------|---------------------------|
| SFT | `messages` (chat) or compatible `text` / prompt-completion |
| DPO | `prompt`, `chosen`, `rejected` |
| GRPO | `prompt` (per current TRL docs researched in Phase 1) |

If format does not match the method, **stop** and fix data or change method—do not start training.

## Step 3: Hardware choice

| Model scale (indicative) | HF Job flavor (if using Jobs) | Local |
|--------------------------|-------------------------------|--------|
| 1–3B | `t4-small` / `a10g-small` | Often feasible on one GPU |
| 7–13B | `a10g-large` | Usually Job or multi-GPU |
| 30B+ | `a100-large` / `h100` | Job |

Set job **timeout** by scale (never rely on ~30m default for training). See `ml-train-sft`.

## User checkpoints

Confirm with the user before:

- Submitting HF Jobs or other paid cloud runs
- Creating **public** Hub repos or pushing under their namespace
- Using datasets that may be gated or non-commercial licenses

## Phase 2 complete when

- [ ] Plan exists and reflects current state
- [ ] Model and dataset IDs chosen and validated
- [ ] Format matches training/eval method
- [ ] Local vs Job path decided and user aligned if Job

Proceed to the matching Phase 3 skill.
