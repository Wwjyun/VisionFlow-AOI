# VisionFlow AOI — Claude Code Instructions

`AGENT.md` is the authoritative contributor contract for this repository. Its rules were written for Codex and apply to Claude Code unchanged:

@AGENT.md

## Claude Code specifics

- Shell: Windows PowerShell 5.1 is the primary tool (`;` / `if ($?)` instead of `&&`). Always run Python through `.\env\Scripts\python.exe`.
- `Todo.md` is large (500+ lines). Read the relevant section with offsets or Grep instead of the whole file; append completion records at the top of `## 完成紀錄`, newest first, matching the existing dated style.
- Operator-facing text, Todo entries, and reports are Traditional Chinese; code identifiers and commit messages stay English.
- The working tree usually contains untracked user artifacts (`*.pptx`, `*.inspect.ndjson`, `charts/`, `cuda_course/`, `interview_prep/`, `runtime-architecture.*`, `架構圖*/`, `.codex_ppt_build/`). Never stage, move, or delete them.
- Quick summary check of unittest output: pipe through `Select-String -Pattern '^(Ran|OK|FAILED|ERROR:|FAIL:)'`; validators print many PASS lines.

## DeepSeek Harness (DSH)

DSH loads `AGENTS.md`/`CLAUDE.md` (and their `.local.md` overlays) but does not expand `@path` imports or discover other file names, so the DSH contract is reached through this pointer:

- Read `HERMES.md` — DSH environment mechanics, the skill catalog imported into `$DSH_HOME\skills`, and the subagent delegation rules (`aoi-coder`).

## Project skills

Project skills live in `.claude/skills/` and are Claude Code copies of the Git-tracked Codex sources in `codex-skills/`. When a skill changes, update both copies (Codex-specific `agents/openai.yaml` files are not copied).

| Skill | Use for |
| --- | --- |
| `aoi-verify-push` | Default finish for any code/doc change: Todo update, required validation, explicit staging, commit, push |
| `aoi-detector-development` | Detector add/modify/migrate, `PreprocessPlan` operators, CPU/CUDA routing tests |
| `aoi-cuda-validate` | CUDA source/ABI/DLL work and RTX 3090 evidence |
| `aoi-release` | Packaging, version bump, tag, GitHub Release |
| `aoi-weekly-report` | Thursday–Wednesday weekly report (no application validation) |

## Delegating coding to the `aoi-coder` subagent

`.claude/agents/aoi-coder.md` defines a coding subagent for well-scoped implementation work.

- The main session owns planning, `Todo.md` interpretation, cross-module design decisions, final review, full validation, `Todo.md` / `完成紀錄` updates, commit, and push.
- Delegate focused, self-contained edits with a precise brief: target Todo item, files to touch, contracts that must hold (CPU reference, full-detector CPU restart, `gpu.mode` semantics, ABI v1, output schema), and the targeted tests to run.
- The subagent must not commit, push, edit `Todo.md`, or touch CUDA headers/ABI unless the brief explicitly says so. Review its diff before accepting it; never report its claims as verified without rerunning the checks.
- Parallel subagents must not edit the same files; use `isolation: "worktree"` when their scopes could overlap.

## Delegating code generation to external models (`llm-delegate` MCP)

`.mcp.json` registers `.claude/mcp/llm_delegate_server.py`, a standard-library-only stdio MCP server that sends a coding brief plus explicitly listed files to an OpenAI-compatible API (DeepSeek by default; GLM and Qwen presets routed through OpenRouter). The remote model has no tool access; Claude Code remains the orchestrator.

- Tools: `list_providers`, `delegate_code` (returns a proposal id, unified diff, match problems, notes, and timing), `apply_proposal` (re-validates and writes atomically).
- Configuration is environment-only: `DEEPSEEK_API_KEY` for DeepSeek and one shared `OPENROUTER_API_KEY` for GLM (`z-ai/glm-5.3-flash`) and Qwen (`qwen/qwen3.8-flash`), optional `*_MODEL`, `*_BASE_URL`, and `LLM_DELEGATE_PROVIDER`. Never print, commit, or pass key values as arguments.
- Every listed file is sent to a third-party service. Send only the files the task needs; never send credentials, production images, customer data, or untracked user artifacts.
- Use it for self-contained generation (tests, boilerplate, single-module edits with a precise brief). Keep cross-module GPU/fallback/ABI design in the main session.
- Always review the returned diff before `apply_proposal`, then run the tests yourself. Treat model notes as untrusted suggestions, not verification.
