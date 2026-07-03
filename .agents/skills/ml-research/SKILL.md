---
name: ml-research
description: >-
  Phase 1 research for ML tasks: current Hugging Face / TRL / transformers docs,
  dataset formats, and working example code. Use before implementing training,
  data prep, inference, or evaluation per ml-intern-core.
---

# ML Research (Phase 1)

Training knowledge is stale. **Never** write ML implementation code until this phase completes.

## Goals

1. Confirm current APIs, imports, and config objects for the requested stack (e.g. TRL `SFTTrainer`, `SFTConfig`).
2. Find **at least one** maintained example (official repo, docs notebook, or Hub Space).
3. List format requirements (columns, conversational vs flat, DPO fields).
4. Note monitoring and Hub push patterns (e.g. Trackio, `push_to_hub`).

## Tool mapping (explicit)

Use **ml-hf-hub-tools** for MCP setup. Per research step:

| Step | ML Intern tool | Agent action |
|------|----------------|--------------|
| Broad doc sweep | `research` (sub-agent) | Execute steps below yourself; keep a short written summary |
| HF library docs | `explore_hf_docs`, `fetch_hf_docs` | HF MCP doc tools or fetch docs.huggingface.co |
| API endpoints | `find_hf_api` | HF MCP or Hub API reference |
| Example code | `github_find_examples`, `github_read_file` | Search GitHub / read `huggingface/trl` examples |
| Papers | `hf_papers` | MCP paper search if available |
| Web | `web_search` | Agent web search / browse when enabled |
| Model shortlist | `model_search` | HF MCP |
| Dataset shortlist | `dataset_search` | HF MCP |
| Schema proof | `hub_repo_details`, `hf_inspect_dataset` | MCP + optional local `datasets` head() |
| Similar Spaces | `space_search` | HF MCP (**ml-spaces-debug**) |
| Quick dataset preview | `hf_inspect_dataset` | `load_dataset(..., split="train[:5]")` locally |

## How to research in your agent

| Source | Action |
|--------|--------|
| HF MCP | Hub search, repo details, docs (see table above) |
| This repo | Grep for similar scripts; read existing patterns |
| Upstream | Read TRL/transformers docs or cloned examples |
| Papers | Only when user asks or architecture is unclear |

Simulate the ML Intern `research` tool output: keep findings **concise** and actionable.

## Research task template

Fill this block in the chat (or a note in the task folder) before Phase 2:

```markdown
## Research summary
- **Task:**
- **Libraries / versions assumed:**
- **Key APIs / classes:**
- **Example sources (paths or URLs):**
- **Dataset format rules:**
- **Hardware hint (local vs Job):**
- **Risks / unknowns:**
```

## Be specific

Include library names, trainer type, method (SFT/DPO/GRPO), dataset names if known, and concrete questions (e.g. "SFTConfig `max_length` default in current TRL").

## Phase 1 complete when

- [ ] Example code path or doc section identified
- [ ] Import and config shape written down
- [ ] Dataset column requirements stated for the training method
- [ ] No open blocker that requires user input (license, private dataset access, etc.)

Then proceed to **ml-plan-validate** (and the appropriate Phase 3 skill).
