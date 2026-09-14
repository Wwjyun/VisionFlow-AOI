# VisionFlow AOI — Claude Code Instructions

`AGENT.md` is the authoritative contributor contract for this repository. Its rules were written for Codex and apply to Claude Code unchanged:

@AGENT.md

## Claude Code specifics

- Shell: Windows PowerShell 5.1 is the primary tool (`;` / `if ($?)` instead of `&&`). Always run Python through `.\env\Scripts\python.exe`.
- `Todo.md` is large (500+ lines). Read the relevant section with offsets or Grep instead of the whole file; append completion records at the top of `## 完成紀錄`, newest first, matching the existing dated style.
- Operator-facing text, Todo entries, and reports are Traditional Chinese; code identifiers and commit messages stay English.
- The working tree usually contains untracked user artifacts (`*.pptx`, `*.inspect.ndjson`, `charts/`, `cuda_course/`, `interview_prep/`, `runtime-architecture.*`, `架構圖*/`, `.codex_ppt_build/`). Never stage, move, or delete them.
- Quick summary check of unittest output: pipe through `Select-String -Pattern '^(Ran|OK|FAILED|ERROR:|FAIL:)'`; validators print many PASS lines.

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
