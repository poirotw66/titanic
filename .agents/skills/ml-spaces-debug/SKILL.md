---
name: ml-spaces-debug
description: >-
  Debug Hugging Face Spaces: startup failures, missing dependencies, build logs,
  Gradio/streamlit apps. Use when a Space crashes, fails to build, or behaves
  wrong at runtime. Works with ml-hf-hub-tools and ml-research.
---

# Hugging Face Spaces debugging

Follow **ml-intern-core** routing: quick **ml-research** if the fix needs current SDK docs, then fix and verify.

## Workflow (from ML Intern Space example)

1. **Plan** (2+ steps): inspect repo → read logs → identify error → fix → verify running.
2. **Inspect Space repo**
   - `hub_repo_details` / MCP for Space metadata, runtime, hardware, SDK type.
   - List files: `app.py` (or entry), `requirements.txt`, `Dockerfile`, `README.md`, `.env` secrets (never commit secrets).
3. **Get logs**
   - Use MCP `find_hf_api` or Hub docs for **Space build/runtime logs** if MCP exposes them.
   - User may paste build log from the Space “Logs” tab — treat as source of truth.
4. **Diagnose common failures**

| Symptom | Likely cause | Fix |
|---------|----------------|-----|
| ImportError on startup | Missing package in `requirements.txt` | Add pinned version; rebuild |
| `ModuleNotFoundError` for local module | Wrong file layout or missing `packages.txt` | Fix structure per Space SDK docs |
| Gradio version mismatch | API changed | Pin `gradio==...` from current docs |
| CUDA OOM on free CPU Space | Model too large for hardware | Smaller model, CPU weights, or upgrade Space hardware |
| 503 / sleeping | Free tier idle | Redeploy or user refresh; document cold start |
| Secret/env missing | `HF_TOKEN` or API keys not in Space secrets | Add in Space settings (user action) |
| Wrong SDK | Using Gradio settings on Docker Space | Align SDK type with file layout |

5. **Research docs**
   - `explore_hf_docs` / fetch: `gradio`, `streamlit`, `docker`, `spaces-config` as needed.
6. **Apply fix**
   - Edit files via git push to Space repo or Hub upload (user approves writes).
   - Prefer minimal diff: one dependency or one import path at a time.
7. **Verify**
   - Confirm Space shows **Running** (user or MCP status).
   - Smoke-test main UI path if possible.

## Space discovery (optional)

- **space_search**: find similar working Spaces before inventing layout.
- **use_space**: send user a link to duplicate or open a template Space.
- **dynamic_space**: only for MCP-enabled Spaces; do not assume all Spaces support it.

## Deliverables

- Root cause in one sentence
- Files changed (paths)
- Space URL and whether build/runtime succeeded
- Follow-up if user must set secrets or upgrade hardware

## Related skills

- **ml-hf-hub-tools** — MCP, repo files, uploads
- **ml-errors** — if failures overlap training/job issues
- **ml-communication** — how to report status to the user
