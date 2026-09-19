# HERMES.md — DeepSeek Harness (DSH) Contributor Contract

**摘要**：本檔是這個 repository 的 DeepSeek Harness 代理契約，與 Codex 的 `AGENT.md`、Claude Code 的 `CLAUDE.md` 並列。`AGENT.md` 仍是唯一權威的貢獻者契約，本檔只補上 DSH 的載入規則、環境機制，以及已匯入本機 DSH 的 6 個 skill 與 subagent 委派規則。

## Scope and precedence

- `AGENT.md` — the authoritative contributor contract (CPU/GPU architecture, detector parameters, GUI rules, required validation, git policy). Everything there applies to DSH unchanged.
- `CLAUDE.md` — Claude Code specifics plus the repository map.
- `HERMES.md` (this file) — DSH specifics: instruction loading, shell/tool mechanics, the imported skill catalog, and subagent delegation.
- On conflict, `AGENT.md` wins. This file adds mechanics; it never relaxes a project invariant.

### How DSH loads instructions (verified, 2026-09-14)

DSH's instruction loader reads candidates from the project root down to the session working directory:

1. `$DSH_HOME/AGENTS.md` (here `C:\Users\user\.dsh\AGENTS.md` — currently absent).
2. For each directory from the nearest `.git` ancestor to the cwd: `AGENTS.md`, then `CLAUDE.md`, then the `AGENTS.local.md` / `CLAUDE.local.md` overlays.
3. Sibling files whose content is byte-identical after trimming are collapsed to the earliest candidate.

Consequences worth knowing:

- `HERMES.md` is **not** a discovered candidate name, and `@path` imports are **not** expanded (`@AGENT.md` inside `CLAUDE.md` stays literal text). This file therefore reaches the model only through the one-line pointer added to `CLAUDE.md`. To make DSH auto-load it, either register the name in the `agent-instructions` config or create `AGENTS.md` with a pointer.
- Because `AGENTS.md` is absent, `CLAUDE.md` is the baseline this session starts from. Keep both in sync when project policy changes.

## Environment mechanics

- Shell: the harness shell tool executes **Windows PowerShell 5.1** (`powershell.exe`) on this machine — `$PSVersionTable.PSEdition` is `Desktop` and no `pwsh.exe` is installed, so write PowerShell 5.1-compatible code (`;` / `if ($?)`, never `&&`). **Every call is a fresh process** — cwd, variables, and functions do not persist. Pass a working directory per call instead of relying on `cd`.
- Text encoding: this machine's ANSI code page is 950 (Big5), so `Get-Content` / `Set-Content` without an explicit UTF-8 encoding silently mangle Traditional Chinese. Read with `-Encoding utf8` (or `[System.IO.File]::ReadAllText($p, [System.Text.Encoding]::UTF8)`), write with `[System.IO.File]::WriteAllText($p, $t, (New-Object System.Text.UTF8Encoding($false)))` to avoid a BOM, and prefer the file tools for anything containing non-ASCII.
- Python: always `.\env\Scripts\python.exe`. The system interpreter is not the project environment.
- Repository root: `C:\Users\user\Desktop\AOI_CVBased` on this machine, but never hard-code it in committed text — resolve from the directory containing `AGENT.md` and `Todo.md`.
- File permissions: this session runs `danger-full-access` with approval prompts disabled, so a sandbox denial is final policy, not a bug — report it instead of retrying another way.
- Long-running work: `run_in_background: true`, then read with `job_output` and stop with `job_kill`. Do not busy-poll.
- Planning: `todo_write` for multi-step work; plan mode only when the user asks for a plan first. `Todo.md` remains the durable project roadmap — `todo_write` is session-scoped and never replaces it.

### Validation (identical to `AGENT.md`; run before finishing)

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py gui_launcher.py tools contour_preprocess_tool core detectors devices gui gpu
.\env\Scripts\python.exe gpu\preflight_cuda_build.py
git diff --check
```

GUI changes add the offscreen `MainWindow` smoke; pipeline/detector/CUDA/packaging surfaces add the extra commands and evidence rules that `AGENT.md` lists. Summarize long unittest output with `Select-String -Pattern '^(Ran|OK|FAILED|ERROR:|FAIL:)'`.

### Git and artifacts (unchanged from `AGENT.md`)

- `main` → `origin/main`. Stage explicit paths only; never `git add .` in this workspace.
- Never stage or delete untracked user artifacts. They are grouped as `簡報/` (投影片＋`.inspect.ndjson`) and `架構圖/` (with `runtime-architecture/`, `charts/`, `_backup_20260911/`), plus `cuda_course/`, `interview_prep/`, `docs/reports/*.md` drafts, and `.codex_ppt_build/`. `ARTIFACTS.md` owns the map and the move rules; moving a whole group requires updating every reference in the same change.
- `Todo.md` is the only task list: mark only genuinely complete items, append the dated `完成紀錄` entry newest-first, and leave hardware-dependent items unchecked until they run on the RTX 3090 machine.

## Skills imported into DSH

Installed at `$DSH_HOME/skills` (`C:\Users\user\.dsh\skills`), so they are available in every DSH session on this machine. DSH discovers `<root>/<name>/SKILL.md` and `<root>/<name>.md` one level deep, parses YAML frontmatter (`name`, `description`, optional `whenToUse`/`metadata`/invocation flags), and requires kebab-case names.

| Skill | Load it for |
| --- | --- |
| `aoi-verify-push` | Default finish for any code/doc change: Todo update, required validation, explicit staging, commit, push. |
| `aoi-detector-development` | Detector add/modify/register/migrate, `PreprocessPlan` operators, CPU/CUDA routing and equivalence tests. |
| `aoi-cuda-validate` | CUDA `.cu`/header/C ABI work, `visionflow_cuda.dll` and `test_cuda_api.exe` builds, RTX 3090 evidence. |
| `aoi-release` | Packaging, version bump, tag, GitHub Release assets, PyInstaller verification. |
| `aoi-weekly-report` | Thursday–Wednesday weekly report; commits the report and runs **no** application validation. |
| `aoi-coder` | Delegating one well-scoped implementation task to a DSH subagent (role contract + delegation mechanics). |

Notes:

- The catalog the model sees carries only `name` + `description`. Load the full body with the `skill` tool using the exact name; the file is re-read on every load, so body edits need no restart. Adding or removing a skill directory is picked up by the watcher without restarting DSH.
- Provenance and sync rule: `aoi-verify-push`, `aoi-detector-development`, `aoi-cuda-validate`, `aoi-release`, `aoi-weekly-report` are byte-identical copies of the Git-tracked `codex-skills/<name>/` sources (including `aoi-release/scripts/publish_github_release.ps1`), with exactly **one** deliberate difference: the `DSH_HOME` path block in `aoi-release/SKILL.md` replaces the `CODEX_HOME` block. When a Codex skill changes, re-copy the directory and re-apply that one edit.
- `aoi-coder` is DSH-only. It is derived from `.claude/agents/aoi-coder.md` because DSH has no file-based agent registry; that Claude file remains the role contract's source of truth.
- A project-scoped alternative exists: `<repo>/.dsh/skills` (rank 100) loads ahead of the user root (rank 400). It is intentionally **not** used here, to keep the working tree free of untracked tooling; use it only if the skills must travel with the repository instead of the machine.

## Subagent delegation in DSH

DSH spawns subagents as ordinary child sessions through the `subagent` and `subagent_fork` tools; there is no `agentType` selector and no `.dsh/agents/*.md` definition file. The reusable role contract therefore lives in the `aoi-coder` skill.

- `subagent` — fresh child with no parent context. `subagent_fork` — child seeded with this conversation's completed turns. Either way the child never sees the in-flight turn.
- Background-first: `run_in_background: true` (the default) and start independent delegations together in one assistant message. Use the foreground only when the next action depends on the result.
- The prompt must be self-contained: repository root, the exact `Todo.md` item, files to touch, contracts that must hold, and the exact test commands.
- Children inherit the parent's sandbox scope and have their approval policy pinned to `never`; a child facing a denied operation reports the limitation instead of escalating. Keep every brief inside the current permission.
- Delegation depth is capped at 3. Children must not commit, push, edit `Todo.md`, or touch CUDA headers/ABI unless the brief says so.
- Continuable children are managed with `list_agents`, `send_message`, and `interrupt_agent` (interrupt stops the current turn, not the child).
- Parallel children must not edit the same files; serialize overlapping scopes or use the `workflow` tool, which fans out across many subagents with phases and structured results. `ralph` is only for an explicitly requested fresh-agent loop.
- The main session keeps planning, `Todo.md` interpretation, cross-module design, final review, the full validation set, the `完成紀錄` entry, commit, and push. Always review a child's diff and rerun its checks — never report its claims as verified.

## Import record

- 2026-09-14: created this contract; imported the five AOI Codex skills into `$DSH_HOME\skills` (verified live in the session catalog); added the DSH-only `aoi-coder` subagent skill; added the `HERMES.md` pointer to `CLAUDE.md`.
- 2026-09-14: consolidated the untracked artifacts — 8 `.pptx` (+ sidecars) and the slide exports into `簡報/`, and the loose `runtime-architecture.*`, `charts/`, and the old diagram backup into `架構圖/`; created `ARTIFACTS.md` as the map, updated the 4 deck scripts' output paths, and fixed the two Phase2 reports' broken diagram links. Recorded here: the harness shell is Windows PowerShell 5.1 (not `pwsh`), and ANSI code page 950 will mangle Chinese written through plain `Set-Content`.
