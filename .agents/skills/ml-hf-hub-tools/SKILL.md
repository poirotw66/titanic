---
name: ml-hf-hub-tools
description: >-
  Hugging Face Hub operations for coding agents: enable HF MCP, model/dataset/paper
  search, repo details, Space discovery, persistent uploads, and CLI fallbacks. Use
  during ml-research, ml-plan-validate, ml-spaces-debug, or any Hub/repo task.
---

# Hugging Face Hub tools

Maps ML Intern built-in/MCP capabilities to what you can run in **Cursor**, **Claude Code**, **Codex**, and similar agents.

## 1. Enable HF MCP (required for parity)

### Cursor

Project: `.cursor/mcp.json` or user MCP settings. Example:

```json
{
  "mcpServers": {
    "hf-mcp-server": {
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

### Claude Code

Project: `.mcp.json` in repo root, or user `~/.claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "hf-mcp-server": {
      "type": "http",
      "url": "https://huggingface.co/mcp?login"
    }
  }
}
```

Or: `claude mcp add --transport http hf-mcp-server https://huggingface.co/mcp?login`

### Codex

`~/.codex/config.toml` or project `.codex/config.toml`:

```toml
[mcp_servers.hf-mcp-server]
url = "https://huggingface.co/mcp?login"
```

Or: `codex mcp add hf-mcp-server --url https://huggingface.co/mcp?login`

Authenticate when prompted on first use. Without MCP, fall back to `huggingface-cli` and `HfApi` scripts in the terminal.

Reference: `configs/cli_agent_config.json`, `configs/frontend_agent_config.json`.

## 2. Tool mapping (ML Intern → agent)

| ML Intern tool | Use in agent |
|----------------|---------------|
| `model_search` | HF MCP model search (or Hub web + `huggingface-cli repo info`) |
| `dataset_search` | HF MCP dataset search |
| `paper_search` | HF MCP / `hf_papers` equivalent if exposed |
| `hub_repo_details` | HF MCP repo details — **always before training** |
| `explore_hf_docs` / `fetch_hf_docs` | MCP doc tools or fetch official docs URLs |
| `find_hf_api` | MCP if available; else Hub API docs / `curl` Hub API |
| `hf_inspect_dataset` | `hub_repo_details` + local `datasets` preview script |
| `hf_repo_files` | MCP list files; or `huggingface-cli download` / `hf_hub_download` |
| `hf_repo_git` | Terminal: clone repo, branch, commit, push (user token); confirm before push |
| `hf_private_repos` | See section 3 (persistent storage) |
| `hf_jobs` | `ml-train-sft` + Jobs CLI/API; user approval first |
| `space_search` | HF MCP Space search |
| `use_space` | Share Space URL with user; MCP grant-access if available |
| `dynamic_space` | Only MCP-enabled Spaces per Hub docs |
| `github_find_examples` / `github_read_file` | Grep/GitHub MCP / `gh api` |
| `web_search` | Agent web search when enabled |

## 3. Persistent storage (`hf_private_repos` behavior)

Job filesystems are **ephemeral**. Anything that must survive job end:

1. **`push_to_hub`** from training script (models/datasets), or  
2. **Private Hub repo** (dataset/model) for logs, scripts, artifacts that cannot use `push_to_hub`.

Rules (match ML Intern):

- Pass **file contents** (strings/bytes) in upload APIs — do not assume the remote job can read your laptop paths.
- After upload, give the user a **direct Hub URL**.
- Create **private** repos for sensitive outputs unless the user wants public.
- Never log or paste `HF_TOKEN`.

**Agent pattern:** small files → `HfApi.upload_file` in a short `uv run` script; large artifacts → script inside a Job that calls `push_to_hub`.

## 4. Repo file operations

**Inspect (read-only):**

- Prefer MCP / `hub_repo_details` for metadata and config.
- List files: MCP or `HfApi.list_repo_tree`.
- Read single file: `hf_hub_download` with `repo_type` model/dataset/space.

**Write (git-style):**

- Confirm repo id, branch, and visibility with the user.
- Clone → edit → commit → push with `git` + credential helper, or Hub API commits for small files.
- For Spaces: also check `README.md`, `app.py`, `requirements.txt`, Docker/`sdk` settings.

## 5. Links and discovery

When referencing Hub assets, include full URLs:

- Model: `https://huggingface.co/{repo_id}`
- Dataset: `https://huggingface.co/datasets/{repo_id}`
- Space: `https://huggingface.co/spaces/{repo_id}`
- Job: use Jobs UI/CLI link after submit

## 6. When to load this skill

- Any Hub search, gated model access, repo upload, or Space URL
- Phase 2 validation in **ml-plan-validate**
- **ml-spaces-debug** for Space repos
