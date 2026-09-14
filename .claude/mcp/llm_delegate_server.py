"""Stdio MCP server that delegates scoped code generation to OpenAI-compatible LLM APIs.

Claude Code stays the orchestrator: it chooses which repository files are sent,
reviews the proposed diff, and decides whether to apply it. The remote model has
no tool access and cannot read files on its own.

Standard library only. API keys are read from environment variables and are never
returned, logged, or written to disk.
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SERVER_NAME = "llm-delegate"
SERVER_VERSION = "1.1.0"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCKED_TOP_LEVEL = {".git", "env", ".venv", "venv", "build", "dist", "release_artifacts"}
MAX_FILE_BYTES = 200_000
MAX_CONTEXT_BYTES = 600_000
MAX_PROPOSALS = 20

PROVIDERS = {
    "deepseek": {
        "base_url_env": "DEEPSEEK_BASE_URL",
        "default_base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-flash",
    },
    # GLM and Qwen are routed through OpenRouter and share one key.
    "glm": {
        "base_url_env": "GLM_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "GLM_MODEL",
        "default_model": "z-ai/glm-5.3-flash",
    },
    "qwen": {
        "base_url_env": "QWEN_BASE_URL",
        "default_base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model_env": "QWEN_MODEL",
        "default_model": "qwen/qwen3.8-flash",
    },
}

# Built from repeated characters so git does not flag this source as containing merge-conflict markers.
SEARCH_MARKER = "<" * 7 + " SEARCH"
DIVIDER_MARKER = "=" * 7
REPLACE_MARKER = ">" * 7 + " REPLACE"

SYSTEM_PROMPT = """You are a precise coding assistant working on VisionFlow AOI, a recipe-driven OpenCV \
inspection system (Python 3.12, PySide6 GUI, optional CUDA DLL backend via ctypes).

Hard contracts:
- CPU execution is the correctness reference. Do not change recipe semantics, PASS/NG, coordinates, \
defect metadata, output formats, or ordering unless the task says so.
- A failed GPU step restarts the entire detector on CPU; never continue from partial GPU results. \
gpu.mode: cpu never loads CUDA, auto may fall back, cuda fails explicitly.
- Preserve CUDA ABI v1 and optional-export probing. Detectors declare cached immutable PreprocessPlan \
objects with shared typed operators; no detector-specific CUDA workflows.
- No mutable module globals for detector/recipe/image/GPU state. Match surrounding style and naming.
- Operator-facing GUI text is Traditional Chinese (PASS, NG, ERROR, CPU, CUDA, ROI, DLL stay English).
- Add or update unittest tests for behavior changes.

Output format (strict). Emit only edit blocks followed by a NOTES section:

FILE: repo/relative/path.py
{search}
exact existing lines copied verbatim, enough to be unique in the file
{divider}
replacement lines
{replace}

Rules: SEARCH must match the current file exactly, including indentation, and must be unique. \
Use several small blocks rather than one huge block. To create a new file, use an empty SEARCH section \
and put the full file in the replacement. Only edit files listed as editable. \
After all blocks write a line `NOTES:` followed by brief design notes, assumptions, and risks.""".format(
    search=SEARCH_MARKER, divider=DIVIDER_MARKER, replace=REPLACE_MARKER,
)


class ToolError(Exception):
    pass


def log(message: str) -> None:
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr, flush=True)


def provider_config(name: str | None, model_override: str | None = None) -> dict:
    name = (name or os.environ.get("LLM_DELEGATE_PROVIDER") or "deepseek").lower()
    if name not in PROVIDERS:
        raise ToolError(f"Unknown provider '{name}'. Choose one of: {', '.join(PROVIDERS)}")
    spec = PROVIDERS[name]
    model = model_override or os.environ.get(spec["model_env"]) or spec["default_model"]
    if not model:
        raise ToolError(f"Provider '{name}' has no model; set {spec['model_env']} or pass `model`.")
    api_key = os.environ.get(spec["key_env"])
    if not api_key:
        raise ToolError(f"Provider '{name}' needs the {spec['key_env']} environment variable.")
    base_url = (os.environ.get(spec["base_url_env"]) or spec["default_base_url"]).rstrip("/")
    return {"name": name, "model": model, "api_key": api_key, "base_url": base_url}


def resolve_repo_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ToolError("File paths must be non-empty strings")
    candidate = (REPO_ROOT / relative.strip()).resolve()
    try:
        parts = candidate.relative_to(REPO_ROOT).parts
    except ValueError:
        raise ToolError(f"Path is outside the repository: {relative}") from None
    if not parts or parts[0] in BLOCKED_TOP_LEVEL or candidate.name.startswith(".env"):
        raise ToolError(f"Path is not allowed: {relative}")
    return candidate


def repo_key(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def build_context(editable: list[str], reference: list[str]) -> tuple[str, dict[str, str]]:
    sections = []
    snapshots: dict[str, str] = {}
    total = 0
    for label, paths in (("EDITABLE", editable), ("REFERENCE (read-only)", reference)):
        for relative in paths:
            path = resolve_repo_path(relative)
            key = repo_key(path)
            if not path.exists():
                if label == "EDITABLE":
                    snapshots[key] = ""
                    sections.append(f"=== {label} FILE: {key} (does not exist yet) ===")
                    continue
                raise ToolError(f"Reference file not found: {relative}")
            size = path.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ToolError(f"{key} is {size} bytes; limit is {MAX_FILE_BYTES}. Pass a smaller file.")
            total += size
            if total > MAX_CONTEXT_BYTES:
                raise ToolError(f"Context exceeds {MAX_CONTEXT_BYTES} bytes; send fewer files.")
            text = read_text(path)
            if label == "EDITABLE":
                snapshots[key] = text
            sections.append(f"=== {label} FILE: {key} ===\n{text}\n=== END FILE: {key} ===")
    return "\n\n".join(sections), snapshots


def call_chat_completion(provider: dict, messages: list[dict], max_tokens: int, temperature: float) -> dict:
    body = json.dumps({
        "model": provider["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{provider['base_url']}/chat/completions",
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {provider['api_key']}",
        },
    )
    started = time.perf_counter()
    first_token = None
    content_parts: list[str] = []
    reasoning_chars = 0
    usage = None
    finish_reason = None
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    # DeepSeek streams `reasoning_content`; OpenRouter streams `reasoning`.
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if (text or reasoning) and first_token is None:
                        first_token = time.perf_counter()
                    reasoning_chars += len(reasoning)
                    if text:
                        content_parts.append(text)
                    finish_reason = choice.get("finish_reason") or finish_reason
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ToolError(f"{provider['name']} API HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ToolError(f"{provider['name']} API request failed: {exc}") from None
    finished = time.perf_counter()
    content = "".join(content_parts)
    generation_s = finished - (first_token or started)
    completion_tokens = (usage or {}).get("completion_tokens")
    return {
        "content": content,
        "finish_reason": finish_reason,
        "usage": usage,
        "timing": {
            "time_to_first_token_s": round((first_token or finished) - started, 3),
            "total_s": round(finished - started, 3),
            "output_chars": len(content),
            "reasoning_chars": reasoning_chars,
            "output_tokens_per_s": (
                round(completion_tokens / generation_s, 1)
                if completion_tokens and generation_s > 0 else None
            ),
        },
    }


def parse_edit_blocks(text: str) -> tuple[list[dict], str]:
    blocks: list[dict] = []
    current_file = None
    state = "idle"
    search: list[str] = []
    replace: list[str] = []
    notes = ""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if state == "idle":
            if stripped.startswith("NOTES:"):
                notes = "\n".join([stripped[6:].strip(), *lines[index + 1:]]).strip()
                break
            match = re.match(r"^\**FILE:\s*`?([^`*]+?)`?\**$", stripped)
            if match:
                current_file = match.group(1).strip()
            elif stripped == SEARCH_MARKER:
                if current_file is None:
                    raise ToolError("Model output has a SEARCH block without a preceding FILE line")
                state, search, replace = "search", [], []
        elif state == "search":
            if stripped == DIVIDER_MARKER:
                state = "replace"
            else:
                search.append(line)
        elif state == "replace":
            if stripped == REPLACE_MARKER:
                blocks.append({"file": current_file, "search": search, "replace": replace})
                state = "idle"
            else:
                replace.append(line)
    if state != "idle":
        raise ToolError("Model output ended inside an unterminated edit block")
    return blocks, notes


def plan_edits(blocks: list[dict], editable: set[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return {path: (original, updated)} and per-block problems; nothing is written."""
    originals: dict[str, str] = {}
    updated: dict[str, str] = {}
    problems: list[str] = []
    for number, block in enumerate(blocks, start=1):
        try:
            path = resolve_repo_path(block["file"])
        except ToolError as exc:
            problems.append(f"block {number}: {exc}")
            continue
        key = repo_key(path)
        if key not in editable:
            problems.append(f"block {number}: {key} was not listed as editable")
            continue
        if key not in updated:
            originals[key] = read_text(path) if path.exists() else ""
            updated[key] = originals[key]
        content = updated[key]
        newline = "\r\n" if "\r\n" in content else "\n"
        search = newline.join(block["search"])
        replacement = newline.join(block["replace"])
        if not search.strip():
            if content.strip():
                problems.append(f"block {number}: empty SEARCH but {key} already has content")
                continue
            updated[key] = replacement + newline if replacement else ""
            continue
        occurrences = content.count(search)
        if occurrences != 1:
            problems.append(f"block {number}: SEARCH text found {occurrences} times in {key}")
            continue
        updated[key] = content.replace(search, replacement, 1)
    return {key: (originals[key], updated[key]) for key in updated}, problems


def unified_diff(changes: dict[str, tuple[str, str]]) -> str:
    chunks = []
    for key, (before, after) in changes.items():
        chunks.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{key}", tofile=f"b/{key}",
        ))
    return "".join(chunk if chunk.endswith("\n") else chunk + "\n" for chunk in chunks)


class DelegateServer:
    def __init__(self) -> None:
        self.proposals: dict[str, dict] = {}

    def tools(self) -> list[dict]:
        return [
            {
                "name": "list_providers",
                "description": "Show configured LLM providers, base URLs, models, and whether each API key "
                               "environment variable is set (values are never shown).",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "delegate_code",
                "description": "Send a scoped coding task plus selected repository files to an external "
                               "OpenAI-compatible model (DeepSeek, or GLM/Qwen via OpenRouter). Returns a proposal id, a unified "
                               "diff of the proposed edits, per-block match problems, model notes, and timing "
                               "(time to first token, tokens/s). Nothing is written until apply_proposal. "
                               "Only the listed files leave this machine.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Precise implementation brief: goal, contracts, "
                                 "expected tests."},
                        "files": {"type": "array", "items": {"type": "string"},
                                  "description": "Repo-relative files the model may edit or create."},
                        "reference_files": {"type": "array", "items": {"type": "string"},
                                            "description": "Repo-relative read-only context files."},
                        "provider": {"type": "string", "enum": list(PROVIDERS)},
                        "model": {"type": "string", "description": "Override the provider's model id."},
                        "max_output_tokens": {"type": "integer", "minimum": 256, "maximum": 32000},
                        "temperature": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["task", "files"],
                },
            },
            {
                "name": "apply_proposal",
                "description": "Write a previously returned proposal to disk. Re-validates every SEARCH block "
                               "against current file contents and writes nothing if any block no longer matches.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"proposal_id": {"type": "string"}},
                    "required": ["proposal_id"],
                },
            },
        ]

    def list_providers(self, _arguments: dict) -> str:
        rows = []
        default = (os.environ.get("LLM_DELEGATE_PROVIDER") or "deepseek").lower()
        for name, spec in PROVIDERS.items():
            rows.append({
                "provider": name,
                "default": name == default,
                "base_url": os.environ.get(spec["base_url_env"]) or spec["default_base_url"],
                "model": os.environ.get(spec["model_env"]) or spec["default_model"],
                "model_env": spec["model_env"],
                "key_env": spec["key_env"],
                "key_present": bool(os.environ.get(spec["key_env"])),
            })
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def delegate_code(self, arguments: dict) -> str:
        task = str(arguments.get("task") or "").strip()
        files = list(arguments.get("files") or [])
        if not task or not files:
            raise ToolError("`task` and at least one entry in `files` are required")
        provider = provider_config(arguments.get("provider"), arguments.get("model"))
        context, snapshots = build_context(files, list(arguments.get("reference_files") or []))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{context}\n\n=== TASK ===\n{task}"},
        ]
        log(f"delegate_code -> {provider['name']}/{provider['model']} ({len(context)} context chars)")
        result = call_chat_completion(
            provider, messages,
            int(arguments.get("max_output_tokens") or 8000),
            float(arguments.get("temperature") if arguments.get("temperature") is not None else 0.0),
        )
        try:
            blocks, notes = parse_edit_blocks(result["content"])
            changes, problems = plan_edits(blocks, set(snapshots))
        except ToolError as exc:
            blocks, notes, changes, problems = [], "", {}, [str(exc)]
        proposal_id = uuid.uuid4().hex[:12]
        self.proposals[proposal_id] = {"blocks": blocks, "editable": set(snapshots)}
        while len(self.proposals) > MAX_PROPOSALS:
            self.proposals.pop(next(iter(self.proposals)))
        summary = {
            "proposal_id": proposal_id,
            "provider": provider["name"],
            "model": provider["model"],
            "finish_reason": result["finish_reason"],
            "usage": result["usage"],
            "timing": result["timing"],
            "edit_blocks": len(blocks),
            "files_changed": sorted(changes),
            "problems": problems,
            "applicable": bool(blocks) and not problems,
        }
        parts = [json.dumps(summary, ensure_ascii=False, indent=2)]
        if notes:
            parts.append(f"MODEL NOTES:\n{notes}")
        diff = unified_diff(changes)
        parts.append(f"PROPOSED DIFF:\n{diff}" if diff else "PROPOSED DIFF: (none)")
        if not blocks:
            parts.append(f"RAW MODEL OUTPUT (no parsable edit blocks):\n{result['content'][:20000]}")
        return "\n\n".join(parts)

    def apply_proposal(self, arguments: dict) -> str:
        proposal = self.proposals.get(str(arguments.get("proposal_id") or ""))
        if proposal is None:
            raise ToolError("Unknown or expired proposal_id")
        changes, problems = plan_edits(proposal["blocks"], proposal["editable"])
        if problems or not changes:
            raise ToolError("Nothing written. " + ("; ".join(problems) or "Proposal has no edits."))
        for key, (_before, after) in changes.items():
            path = resolve_repo_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="") as handle:
                handle.write(after)
        return json.dumps({"written": sorted(changes)}, ensure_ascii=False)

    def handle(self, message: dict) -> dict | None:
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            return None  # notifications need no response
        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion")
            return self._result(request_id, {
                "protocolVersion": requested if requested in SUPPORTED_PROTOCOLS else SUPPORTED_PROTOCOLS[0],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            })
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": self.tools()})
        if method == "tools/call":
            params = message.get("params") or {}
            handler = {
                "list_providers": self.list_providers,
                "delegate_code": self.delegate_code,
                "apply_proposal": self.apply_proposal,
            }.get(params.get("name"))
            if handler is None:
                return self._error(request_id, -32602, f"Unknown tool: {params.get('name')}")
            try:
                text, is_error = handler(params.get("arguments") or {}), False
            except ToolError as exc:
                text, is_error = str(exc), True
            except Exception as exc:  # report unexpected failures to the client instead of crashing
                log(f"unexpected error: {exc!r}")
                text, is_error = f"Internal error: {exc!r}", True
            return self._result(request_id, {"content": [{"type": "text", "text": text}], "isError": is_error})
        return self._error(request_id, -32601, f"Method not found: {method}")

    @staticmethod
    def _result(request_id, result: dict) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id, code: int, message: str) -> dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> None:
    server = DelegateServer()
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    for raw in stdin:
        if not raw.strip():
            continue
        try:
            message = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        else:
            response = server.handle(message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
            stdout.flush()


if __name__ == "__main__":
    main()
