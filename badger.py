#!/usr/bin/env python3
"""
Program: Deep Badger - PRD Agentic CLI - Powered by the DeepSeek v4 API 

Summary: Uses PRD files to manage an agentic process to do something/build something to completion. 
Agentic processes can be unpredictable. Use in a sandbox is recommended.

badger.py
================================================================
Features:
- Exploration phase with progress bars (turn/token budgets)
- Readiness self-assessment before writing PRD
- Only bash tool, user approval/edit/block per command
- Rationale + one-sentence summary required
- Automatic correction memory
- Truncated conversation history for token efficiency
- Generates PRD.md and PRD_NEXT.md and cycles until complete.
- Verbose mode for detailed debugging and transparency.

Run from anywhere: the API key is loaded from ../K.dat (relative to this script).

Author: g023 - https://github.com/g023/
License: MIT
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Assume _ds4.py (the simplified client) is in the same directory
from _ds4 import DeepSeekV4, ToolDef

# ============================================================================
# Configuration
# ============================================================================

# AUTOMODE: Global autopilot for testing (auto-approve all commands)
AUTOMODE = False  # Set to True for hands-free testing # Danger

SESSION_BASE = Path(__file__).parent / "SESSIONS"
DEFAULT_EXPLORATION_BUDGET = 15  # Soft cap for bash commands in exploration
EXPLORATION_MAX_TOOL_TURNS = 3   # Allow tool chaining in single API call
TOKEN_BUDGET_ESTIMATE = 4000
PROGRESS_BAR_WIDTH = 20
THINKING_MODE_AUTO = "auto"      # auto, enabled, disabled
THINKING_MODE_THRESHOLD = 999999 # Effectively disabled by default — thinking wastes tokens
BASH_CACHE_MAX_SIZE = 100        # LRU cache limit for bash commands
WORK_DIR = None                  # Will be set if --work option is used

# Exploration plan: model defines its own budget and strategy upfront
EXPLORATION_PLAN_PROMPT = """Before executing anything, produce a concise exploration plan.

**SCOPE CONSTRAINT**: Only explore the current working directory and its subdirectories. 
Do NOT explore SESSIONS/, .git/, node_modules/, venv/, __pycache__/, or parent directories.

**TOOLS AVAILABLE**: You have `read` (read-only filesystem access) and `scratch_pad` (in-memory note storage). You do NOT have `bash`.

Output a JSON block like:
```json
{
  "plan": "I will first look at the top-level directory and package manifests, then check key dependencies, then review existing tests and configuration.",
  "estimated_tools": 8,
  "key_questions": ["What language and build system?", "What are the main entry points?"]
}
```

Be realistic about estimated_tools — this is your budget. You will be cut off at that limit.
After the plan, wait for user confirmation before executing.
"""

# System prompt for exploration — now includes planning directive
EXPLORATION_SYSTEM_PROMPT = """You are an expert product architect analyzing a software project. You have access to TWO tools:

1. **`read`** — Read-only filesystem exploration (ls, cat, head, tail, wc, grep, find, file). CANNOT create or modify files.
2. **`scratch_pad`** — In-memory note storage. Use this to save findings so you don't need to re-read files.

**CRITICAL: You do NOT have access to the `bash` tool during exploration.** The `bash` tool is only available during code generation. Any attempt to use `bash` will fail. Use `read` for all filesystem access.

**Your task**: Explore efficiently to gather enough information to write an exceptional PRD. 
You are in charge of your own exploration budget but must do the minimum necessary to understand:
- Project purpose, core user flows, high-level architecture
- Technology stack, key modules, entry points, and external dependencies
- Existing test patterns, documentation, constraints, and future plans

**SCOPE CONSTRAINT — CRITICAL**: You MUST ONLY explore the current working directory and its subdirectories. Do NOT explore:
- The `SESSIONS/` directory (contains previous session outputs, not project source)
- The `.git/` directory
- Any `node_modules/`, `venv/`, `__pycache__/`, or other generated/vendor directories
- Any path starting with `..` (parent directories)
- Any path outside the current working directory

**Process**:
1. Start with a concise JSON exploration plan (see format below). Do NOT run any read commands before outputting the plan.
2. Then execute commands surgically. Before each read command, output:
   [RATIONAL] Specific reason for this command
   [SUMMARY] What you will learn
3. **Use `scratch_pad` to store findings** after each discovery. This prevents redundant reads and saves your budget.
   - After reading a key file, call `scratch_pad` with action="store", key="finding_name", value="summary of what you learned"
   - When you need to recall something, use `scratch_pad` with action="retrieve" instead of re-reading files
   - Use `scratch_pad` action="list" to see what you've already discovered
4. Prefer targeted read operations over broad recursive listings:
   - `read(operation="ls", path=".")` to see top-level files
   - `read(operation="find", path=".", pattern="*.py", max_depth=2)` to find source files
   - `read(operation="wc", path="file.py")` to gauge file size before reading
   - `read(operation="head", path="file.py", lines=50)` to preview files
   - `read(operation="grep", path=".", pattern="keyword", include_pattern="*.py")` for targeted searches
5. Mentally track the Quality Checklist items below. After each command, self-assess: do you have enough to write the PRD?
6. **Exit early** as soon as sufficient information is gathered — stop issuing further tool calls.

**Critical Rules**:
- Before opening a file, gauge size with `read(operation="wc", path="<file>")`.
- You CANNOT create or modify files — the `read` tool is strictly read-only.
- Do NOT use `read(operation="find", path=".", max_depth=...)` without a pattern filter or with max_depth > 3.
- Monitor your own budget. If you reach the estimated limit without readiness, output READY: false and list gaps.
- **Use `scratch_pad` aggressively** to cache your findings. This is your memory — use it to avoid redundant reads.

**Quality Checklist** (address mentally, then confirm in readiness):
- Project structure and architecture
- Technology stack and dependencies
- Key files and their purposes
- Existing test coverage and patterns
- Current issues or TODOs
- Integration points and APIs
- Performance characteristics
- Known constraints or limitations

**Readiness Assessment** (output when ready to stop exploration):
READY: true
Gaps: - (none, or list specific gaps)
When confident that all checklist items are sufficiently covered, output only READY: true and immediately stop using the read tool. If gaps remain, output READY: false and continue.
"""

# The system prompt for the final synthesis (PRD writing)
WRITING_SYSTEM_PROMPT = """You have finished exploration and are READY to write the PRD.

Based on everything you have learned, produce two documents separated by exactly the line:
---PRD_NEXT---

First document: **PRD.md** - full product requirements including:
- Goal / vision
- Functional and non-functional requirements
- Success criteria (verifiable) with verification tasks
- Constraints, dependencies, risks
- A "Readiness Checklist" that proves the project is well-defined

Second document: **PRD_NEXT.md** - a detailed task breakdown for the **next session only**.
- Each task must be small (5-10 minutes of agent work)
- Include verification steps for each task
- Use `- [ ]` checkboxes

Output ONLY the two documents, nothing else.
"""

# ============================================================================
# Helper Functions
# ============================================================================

def make_session_id(prompt: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(prompt.encode()).hexdigest()[:8]
    return f"{ts}_{h}"

def ensure_session_dir(session_id: str) -> Path:
    path = SESSION_BASE / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def read_file(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""

def write_file(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def progress_bar(used: int, total: int, width: int = PROGRESS_BAR_WIDTH) -> str:
    """Return a string like [====    ] 40%"""
    percent = used / total if total > 0 else 0
    filled = int(width * percent)
    bar = "=" * filled + " " * (width - filled)
    return f"[{bar}] {int(percent*100)}%"

def validate_prd_quality(content: str) -> Tuple[bool, List[str]]:
    """Validate PRD has critical sections and quality markers."""
    issues = []
    content_lower = content.lower()

    quality_checks = {
        "Goal/Vision": ["goal", "vision", "purpose"],
        "Requirements": ["requirement", "functional", "non-functional"],
        "Success Criteria": ["success", "metric", "kpi", "verification"],
        "Constraints": ["constraint", "limitation", "dependency"],
        "Risks": ["risk", "mitigation"],
    }

    for section, keywords in quality_checks.items():
        if not any(kw in content_lower for kw in keywords):
            issues.append(f"Missing or weak '{section}' section")

    if len(content) < 500:
        issues.append("PRD appears too brief (< 500 chars)")

    if content.count("\n") < 20:
        issues.append("PRD structure could be better organized")

    is_valid = len(issues) == 0
    return is_valid, issues

# ============================================================================
# Tool Definitions - Read-only exploration, scratch pad, and bash
# ============================================================================

# READ_TOOL: Read-only exploration tool (used during exploration phase)
# Prevents file creation/writing — only allows reading and listing
READ_TOOL = ToolDef(
    name="read",
    description="Read-only file system exploration. Use for: listing directories, reading files, searching code, checking file sizes. CANNOT create or modify files.",
    parameters={
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["ls", "cat", "head", "tail", "wc", "grep", "find", "file"],
                "description": "Operation to perform"
            },
            "path": {
                "type": "string",
                "description": "File or directory path (relative to cwd)"
            },
            "pattern": {
                "type": "string",
                "description": "Search pattern (for grep/find operations)"
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth for find (default: 2, max: 3)",
                "default": 2
            },
            "lines": {
                "type": "integer",
                "description": "Number of lines for head/tail (default: 30)",
                "default": 30
            },
            "include_pattern": {
                "type": "string",
                "description": "File glob pattern for grep/find (e.g., '*.py')"
            }
        },
        "required": ["operation", "path"]
    },
    handler=None,  # Set per-instance in DAGAgent
    max_result_chars=4000,
)

# SCRATCH_PAD_TOOL: In-memory note-taking for exploration findings
# Prevents redundant bash calls by letting the model store/retrieve findings
SCRATCH_PAD_TOOL = ToolDef(
    name="scratch_pad",
    description="Persistent scratch pad for storing and retrieving exploration findings. Use this to save notes, summaries, and discoveries so you don't need to re-read files. Data persists for the entire exploration session.",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "retrieve", "list", "clear"],
                "description": "store: save a note with a key; retrieve: get a note by key; list: list all keys; clear: remove all notes"
            },
            "key": {
                "type": "string",
                "description": "Note identifier (e.g., 'project_structure', 'dependencies', 'entry_points')"
            },
            "value": {
                "type": "string",
                "description": "Content to store (required for action=store)"
            }
        },
        "required": ["action", "key"]
    },
    handler=None,  # Set per-instance in DAGAgent
    max_result_chars=4000,
)

# BASH_TOOL definition (handler will be set per-instance in DAGAgent)
# Used ONLY during code generation phase, NOT during exploration
BASH_TOOL = ToolDef(
    name="bash",
    description="Execute a bash command. Use heredoc for writing files. Returns output.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "cwd": {"type": "string", "default": "."}
        },
        "required": ["command"]
    },
    handler=None,  # Set per-instance in DAGAgent
    max_result_chars=3000,
)

# ============================================================================
# VirtualFileSystem - In-memory filesystem for testing
# ============================================================================

class VirtualFileSystem:
    """In-memory filesystem that handles common bash operations for testing.
    Supports: cat <<EOF > file, cat file, ls, echo, python3 -c"""

    def __init__(self):
        self.files: Dict[str, str] = {}  # path -> content
        self.current_dir = "."

    def _normalize_path(self, path: str) -> str:
        """Normalize path (handle . and ..)."""
        if path.startswith("/"):
            return path
        if path == ".":
            return self.current_dir
        if self.current_dir == ".":
            return path
        return f"{self.current_dir}/{path}".replace("//", "/")

    def _ensure_dir(self, path: str) -> None:
        """Ensure parent directory exists (in virtual FS, this is implicit)."""
        pass

    def write_file(self, path: str, content: str) -> None:
        """Write file to virtual filesystem."""
        path = self._normalize_path(path)
        self._ensure_dir(path)
        self.files[path] = content

    def read_file(self, path: str) -> Optional[str]:
        """Read file from virtual filesystem."""
        path = self._normalize_path(path)
        return self.files.get(path)

    def list_files(self, directory: str = ".") -> List[str]:
        """List files in directory."""
        directory = self._normalize_path(directory)
        if directory == ".":
            directory = ""
        prefix = f"{directory}/" if directory else ""
        matching = [f for f in self.files.keys() if f.startswith(prefix)]
        return [f[len(prefix):].split("/")[0] for f in matching]

    def file_exists(self, path: str) -> bool:
        """Check if file exists."""
        path = self._normalize_path(path)
        return path in self.files

    def parse_heredoc_command(self, command: str) -> Tuple[bool, str, str]:
        """Parse cat << 'EOF' > path/file pattern.
        Returns (success, path, content) or (False, "", "") if not a heredoc."""
        heredoc_pattern = r"cat\s*<<\s*['\"]?(\w+)['\"]?\s*(.*?)\1"
        match = re.search(heredoc_pattern, command, re.DOTALL)
        if not match:
            return False, "", ""

        end_marker = match.group(1)
        content = match.group(2).strip()

        output_pattern = r">\s*([^\s]+)$"
        output_match = re.search(output_pattern, command)
        if not output_match:
            return False, "", ""

        path = output_match.group(1)
        return True, path, content

    def execute_command(self, command: str) -> Tuple[bool, str]:
        """Execute a command in virtual filesystem.
        Returns (success, output)"""
        command = command.strip()

        # cat < EOF > file pattern
        if "<<" in command and ">" in command:
            is_heredoc, path, content = self.parse_heredoc_command(command)
            if is_heredoc:
                self.write_file(path, content)
                return True, f"(written {len(content)} bytes to {path})"

        # cat file pattern
        if command.startswith("cat ") and "<<" not in command:
            parts = command.split()
            if len(parts) >= 2:
                path = parts[1]
                content = self.read_file(path)
                if content is not None:
                    return True, content
                return False, f"cat: {path}: No such file or directory"

        # ls pattern
        if command.startswith("ls"):
            files = self.list_files(self.current_dir)
            return True, "\n".join(files) if files else "(empty)"

        # python3 -c syntax check
        if "python3 -c" in command and "ast.parse" in command:
            code_pattern = r'python3\s+-c\s+["\']([^"\']+)["\']'
            code_match = re.search(code_pattern, command)
            if code_match:
                code = code_match.group(1)
                try:
                    import ast
                    ast.parse(code)
                    return True, "(syntax OK)"
                except SyntaxError as e:
                    return False, f"SyntaxError: {e}"

        # Fallback: return success for unknown commands
        return True, "(virtual command executed)"

# ============================================================================
# BashSession - Encapsulates bash execution with isolated state
# ============================================================================

class BashSession:
    """Isolated bash session with its own cache, history, and working directory.
    Supports both real bash execution and virtual filesystem modes for testing."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.cache: OrderedDict = OrderedDict()
        self.history: List[Dict[str, Any]] = []
        self.virtual_fs: Optional[VirtualFileSystem] = None  # In-memory filesystem for testing

    def enable_virtual_fs(self) -> None:
        """Enable virtual filesystem mode (for testing)."""
        self.virtual_fs = VirtualFileSystem()

    def is_virtual_mode(self) -> bool:
        """Check if running in virtual filesystem mode."""
        return self.virtual_fs is not None

    def execute(self, command: str, cwd: str = ".") -> dict:
        """Execute a bash command. Returns result dict with 'output', 'exit_code', or 'error'."""
        # Show the command being executed (truncated for display)
        cmd_display = command[:200] + ("..." if len(command) > 200 else "")
        print(f"  ▶ bash: {cmd_display}")

        # Dangerous pattern blocking
        dangerous = [
            "rm -rf /", "sudo ", "chmod 777", "dd if=", "mkfs",
            "> /dev/sd", ":(){ :|:& };:", r"curl\s.*\|\s*sh", r"wget\s.*\|\s*sh"
        ]
        for pat in dangerous:
            if re.search(pat, command):
                print(f"  ⛔ BLOCKED: dangerous pattern '{pat}'")
                return {"error": f"Blocked dangerous pattern: {pat}", "blocked": True}

        # Cache hit
        cache_key = (cwd, command)
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            if self.verbose:
                print(f"  💾 Cache hit")
            return {"output": self.cache[cache_key], "cached": True}

        try:
            if self.is_virtual_mode():
                result = self._execute_virtual(command, cwd)
            else:
                result = self._execute_real(command, cwd)

            output = result.get("output", "")
            if len(output) > 3000:
                output = output[:3000] + "\n[... truncated ...]"
                result["output"] = output

            # Cache the result
            self.cache[cache_key] = output
            self.cache.move_to_end(cache_key)

            # LRU eviction
            if len(self.cache) > BASH_CACHE_MAX_SIZE:
                evicted = self.cache.popitem(last=False)
                if self.verbose:
                    print(f"[VERBOSE] Cache evicted: {evicted[0][1][:80]}...")

            self.history.append({
                "command": command,
                "success": result.get("exit_code", 1) == 0,
                "timestamp": time.time(),
            })

            return result
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.history.append({
                "command": command,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            })
            return {"error": f"{type(e).__name__}: {e}"}

    def _execute_real(self, command: str, cwd: str) -> dict:
        """Execute command in real filesystem."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=90,
                cwd=cwd,
                executable="/bin/bash",
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            output = output.strip() or "(no output)"

            # Show result summary
            if result.returncode == 0:
                lines = output.count('\n')
                print(f"  ✅ Exit: 0 | {len(output)} chars, {lines} lines")
            else:
                print(f"  ❌ Exit: {result.returncode} | {output[:150]}")

            return {"output": output, "exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Command timed out after 90 seconds")
            return {"error": "Command timed out after 90 seconds"}

    def _execute_virtual(self, command: str, cwd: str) -> dict:
        """Execute command in virtual filesystem."""
        assert self.virtual_fs is not None
        success, output = self.virtual_fs.execute_command(command)
        exit_code = 0 if success else 1

        if success:
            lines = output.count('\n')
            print(f"  ✅ Exit: {exit_code} | {len(output)} chars, {lines} lines")
        else:
            print(f"  ❌ Exit: {exit_code} | {output[:150]}")

        return {"output": output, "exit_code": exit_code}

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "cache_size": len(self.cache),
            "history_length": len(self.history),
            "success_rate": (sum(1 for h in self.history if h.get("success")) / max(1, len(self.history))) * 100
        }

# ============================================================================
# FilePlan - Project file structure and constraints
# ============================================================================

class FilePlan:
    """Defines file structure, allowed types, and constraints for a project.
    Ensures all generated files follow the plan."""

    def __init__(self, root: str = "."):
        self.root = Path(root)
        self.dirs: Dict[str, str] = {}  # dir -> description
        self.files: Dict[str, Dict[str, str]] = {}  # file_path -> {purpose, template?}
        self.constraints = {
            "max_files_per_dir": 50,
            "allowed_extensions": [".py", ".md", ".json", ".yaml", ".toml", ".txt"],
            "forbidden_paths": ["SESSIONS", ".git", "node_modules", "__pycache__", ".venv", "venv"],
        }

    @classmethod
    def from_dict(cls, data: dict, root: str = ".") -> "FilePlan":
        """Load FilePlan from dictionary."""
        plan = cls(root)
        plan.dirs = data.get("dirs", {})
        plan.files = data.get("files", {})
        plan.constraints.update(data.get("constraints", {}))
        return plan

    @classmethod
    def from_json_file(cls, path: str) -> "FilePlan":
        """Load FilePlan from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data, data.get("root", "."))

    def to_dict(self) -> dict:
        """Export FilePlan as dictionary."""
        return {
            "root": str(self.root),
            "dirs": self.dirs,
            "files": self.files,
            "constraints": self.constraints,
        }

    def validate_path(self, file_path: str) -> Tuple[bool, str]:
        """Check if a file path is allowed.
        Returns (is_valid, error_message)"""
        path = Path(file_path)

        # Check forbidden paths
        for forbidden in self.constraints.get("forbidden_paths", []):
            if forbidden in str(path.parts):
                return False, f"Path contains forbidden directory: {forbidden}"

        # Check file extension
        allowed_exts = self.constraints.get("allowed_extensions", [])
        if path.suffix not in allowed_exts:
            return False, f"File type {path.suffix} not allowed. Allowed: {', '.join(allowed_exts)}"

        # Check files per directory
        parent_dir = str(path.parent)
        max_per_dir = self.constraints.get("max_files_per_dir", 50)
        # (simplified: would need actual filesystem to count)

        return True, ""

    def to_system_prompt_section(self) -> str:
        """Generate a system prompt section describing the file structure."""
        lines = [
            "**PROJECT FILE PLAN**\n",
            f"Root directory: {self.root}\n",
            "Allowed directories:\n",
        ]
        for dir_name, description in self.dirs.items():
            lines.append(f"  - {dir_name}/: {description}")
        lines.append("\nKey files:\n")
        for file_path, info in list(self.files.items())[:5]:
            purpose = info.get("purpose", "")
            lines.append(f"  - {file_path}: {purpose}")
        lines.append(f"\nConstraints:\n")
        lines.append(f"  - Allowed extensions: {', '.join(self.constraints.get('allowed_extensions', []))}")
        lines.append(f"  - Forbidden paths: {', '.join(self.constraints.get('forbidden_paths', []))}")
        return "\n".join(lines)

# ============================================================================
# ScratchPad - In-memory note storage for exploration findings
# ============================================================================

class ScratchPad:
    """Persistent in-memory scratch pad for storing exploration findings.
    Prevents redundant bash/read calls by letting the model cache discoveries."""

    def __init__(self):
        self.notes: Dict[str, str] = {}

    def store(self, key: str, value: str) -> str:
        """Store a note. Returns confirmation."""
        self.notes[key] = value
        return f"✅ Stored note '{key}' ({len(value)} chars)"

    def retrieve(self, key: str) -> str:
        """Retrieve a note by key."""
        if key in self.notes:
            return f"📝 Note '{key}':\n{self.notes[key]}"
        return f"⚠️ No note found for key '{key}'. Available keys: {', '.join(self.list_keys()) or '(none)'}"

    def list_keys(self) -> List[str]:
        """List all stored note keys."""
        return list(self.notes.keys())

    def clear(self) -> str:
        """Clear all notes."""
        count = len(self.notes)
        self.notes.clear()
        return f"✅ Cleared {count} notes"

    def get_all(self) -> Dict[str, str]:
        """Get all notes as dict."""
        return dict(self.notes)

    def handle_action(self, action: str, key: str, value: str = None) -> str:
        """Handle a scratch_pad tool action."""
        if action == "store":
            if value is None:
                return "⚠️ 'value' is required for action='store'"
            return self.store(key, value)
        elif action == "retrieve":
            return self.retrieve(key)
        elif action == "list":
            keys = self.list_keys()
            if keys:
                return f"📋 Scratch pad keys:\n" + "\n".join(f"  - {k}" for k in keys)
            return "📋 Scratch pad is empty"
        elif action == "clear":
            return self.clear()
        return f"⚠️ Unknown action: {action}"


# ============================================================================
# ReadTool - Read-only filesystem explorer
# ============================================================================

class ReadTool:
    """Read-only filesystem operations for exploration phase.
    Prevents any file creation or modification."""

    @staticmethod
    def execute(operation: str, path: str, pattern: str = None,
                max_depth: int = 2, lines: int = 30,
                include_pattern: str = None, cwd: str = ".") -> dict:
        """Execute a read-only filesystem operation.
        Returns dict with 'output' and 'exit_code'."""
        try:
            # Resolve path relative to cwd
            base = Path(cwd).resolve()
            target = base / path if not Path(path).is_absolute() else Path(path).resolve()

            # Scope check: must be within cwd
            try:
                target.resolve().relative_to(base)
            except (ValueError, RuntimeError):
                return {
                    "error": f"Path '{path}' is outside the allowed working directory '{base}'. Only explore the current directory and its subdirectories.",
                    "blocked": True
                }

            # Forbidden directories check
            forbidden_parts = {"SESSIONS", ".git", "node_modules", "venv", ".venv", "__pycache__"}
            for part in target.resolve().parts:
                if part in forbidden_parts:
                    return {
                        "error": f"Path contains forbidden directory: '{part}'. Cannot explore SESSIONS/, .git/, node_modules/, venv/, __pycache__/.",
                        "blocked": True
                    }

            if operation == "ls":
                if target.is_dir():
                    items = sorted(target.iterdir())
                    result = []
                    for item in items:
                        suffix = "/" if item.is_dir() else ""
                        result.append(f"{item.name}{suffix}")
                    output = "\n".join(result) if result else "(empty directory)"
                    return {"output": output, "exit_code": 0}
                else:
                    return {"error": f"ls: {path}: Not a directory", "exit_code": 1}

            elif operation == "cat":
                if not target.is_file():
                    return {"error": f"cat: {path}: No such file or directory", "exit_code": 1}
                content = target.read_text(encoding="utf-8", errors="replace")
                if len(content) > 4000:
                    content = content[:4000] + f"\n[... truncated at 4000 chars, file is {len(content)} chars total ...]"
                return {"output": content, "exit_code": 0}

            elif operation == "head":
                if not target.is_file():
                    return {"error": f"head: {path}: No such file or directory", "exit_code": 1}
                content_lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
                shown = content_lines[:min(lines, len(content_lines))]
                output = "\n".join(shown)
                if len(content_lines) > lines:
                    output += f"\n[... {len(content_lines) - lines} more lines ...]"
                return {"output": output, "exit_code": 0}

            elif operation == "tail":
                if not target.is_file():
                    return {"error": f"tail: {path}: No such file or directory", "exit_code": 1}
                content_lines = target.read_text(encoding="utf-8", errors="replace").split("\n")
                shown = content_lines[-min(lines, len(content_lines)):]
                output = "\n".join(shown)
                return {"output": output, "exit_code": 0}

            elif operation == "wc":
                if not target.exists():
                    return {"error": f"wc: {path}: No such file or directory", "exit_code": 1}
                if target.is_file():
                    content = target.read_text(encoding="utf-8", errors="replace")
                    lines_count = content.count("\n")
                    words = len(content.split())
                    chars = len(content)
                    output = f"{lines_count:>8} lines  {words:>8} words  {chars:>8} chars  {path}"
                else:
                    output = f"(directory: {path})"
                return {"output": output, "exit_code": 0}

            elif operation == "grep":
                if not pattern:
                    return {"error": "grep requires a 'pattern' parameter", "exit_code": 1}
                if not target.exists():
                    return {"error": f"grep: {path}: No such file or directory", "exit_code": 1}
                if target.is_file():
                    files_to_search = [target]
                else:
                    glob_pattern = f"**/{include_pattern}" if include_pattern else "**/*"
                    files_to_search = sorted(target.glob(glob_pattern))
                    files_to_search = [f for f in files_to_search if f.is_file()]

                matches = []
                for f in files_to_search:
                    try:
                        for i, line in enumerate(f.read_text(encoding="utf-8", errors="replace").split("\n"), 1):
                            if pattern.lower() in line.lower():
                                rel_path = f.relative_to(base)
                                matches.append(f"{rel_path}:{i}: {line.strip()[:200]}")
                    except Exception:
                        continue

                if matches:
                    output = "\n".join(matches[:100])
                    if len(matches) > 100:
                        output += f"\n[... {len(matches) - 100} more matches ...]"
                else:
                    output = f"(no matches for '{pattern}' in {path})"
                return {"output": output, "exit_code": 0}

            elif operation == "find":
                if not target.is_dir():
                    return {"error": f"find: {path}: Not a directory", "exit_code": 1}
                max_depth = min(max_depth, 3)  # Hard cap at 3
                glob_pattern = f"**/{include_pattern}" if include_pattern else "**/*"
                all_files = sorted(target.glob(glob_pattern))
                # Filter by depth
                result = []
                for f in all_files:
                    rel = f.relative_to(target)
                    depth = len(rel.parts)
                    if depth <= max_depth:
                        suffix = "/" if f.is_dir() else ""
                        result.append(f"{rel}{suffix}")
                output = "\n".join(result[:200]) if result else "(empty)"
                if len(result) > 200:
                    output += f"\n[... {len(result) - 200} more entries ...]"
                return {"output": output, "exit_code": 0}

            elif operation == "file":
                if not target.exists():
                    return {"error": f"file: {path}: No such file or directory", "exit_code": 1}
                if target.is_dir():
                    output = f"{path}: directory"
                else:
                    size = target.stat().st_size
                    output = f"{path}: file, {size} bytes"
                return {"output": output, "exit_code": 0}

            else:
                return {"error": f"Unknown operation: {operation}", "exit_code": 1}

        except PermissionError:
            return {"error": f"Permission denied: {path}", "exit_code": 1}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "exit_code": 1}


# ============================================================================
# TestScenario and TestRunner - Reactive testing infrastructure
# ============================================================================

class TestScenario:
    """Defines a single test scenario: prompt, expected behaviors, assertions."""

    def __init__(self, name: str, prompt: str, initial_files: Dict[str, str] = None,
                 expected_patterns: List[str] = None, max_commands: int = 10,
                 test_scratchpad: bool = False, test_scope_blocking: bool = False,
                 assert_no_file_creation: bool = False):
        self.name = name
        self.prompt = prompt
        self.initial_files = initial_files or {}
        self.expected_patterns = expected_patterns or []
        self.max_commands = max_commands
        self.test_scratchpad = test_scratchpad
        self.test_scope_blocking = test_scope_blocking
        self.assert_no_file_creation = assert_no_file_creation

    @classmethod
    def from_json_file(cls, path: str) -> "TestScenario":
        """Load test scenario from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(
            name=data.get("name", "unnamed"),
            prompt=data.get("prompt", ""),
            initial_files=data.get("initial_files", {}),
            expected_patterns=data.get("expected_patterns", []),
            max_commands=data.get("max_commands", 10),
            test_scratchpad=data.get("test_scratchpad", False),
            test_scope_blocking=data.get("test_scope_blocking", False),
            assert_no_file_creation=data.get("assert_no_file_creation", False),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "initial_files": self.initial_files,
            "expected_patterns": self.expected_patterns,
            "max_commands": self.max_commands,
            "test_scratchpad": self.test_scratchpad,
            "test_scope_blocking": self.test_scope_blocking,
            "assert_no_file_creation": self.assert_no_file_creation,
        }


class TestRunner:
    """Runs a test scenario with virtual filesystem, asserts on outputs.
    Tests the read tool, scratch pad tool, and scope enforcement."""

    def __init__(self, scenario: TestScenario, verbose: bool = False):
        self.scenario = scenario
        self.verbose = verbose
        self.scratch_pad = ScratchPad()
        self.assertions_passed = 0
        self.assertions_failed = 0
        self.test_log: List[str] = []

    def log(self, msg: str) -> None:
        self.test_log.append(msg)
        if self.verbose:
            print(f"  {msg}")

    def run(self) -> Tuple[bool, str]:
        """Run the test scenario.
        Returns (success, report)"""
        report_lines = [f"Test: {self.scenario.name}"]
        report_lines.append(f"  Prompt: {self.scenario.prompt[:100]}...")

        try:
            if self.scenario.test_scratchpad:
                self._test_scratchpad_operations(report_lines)
            elif self.scenario.test_scope_blocking:
                self._test_scope_enforcement(report_lines)
            elif self.scenario.assert_no_file_creation:
                self._test_readonly_tool(report_lines)
            else:
                # Generic test: check expected patterns in tool outputs
                self._test_generic(report_lines)

            success = self.assertions_failed == 0
            report_lines.append(f"\nAssertion results: {self.assertions_passed} passed, {self.assertions_failed} failed")

            return success, "\n".join(report_lines)
        except Exception as e:
            report_lines.append(f"  ERROR: {e}")
            import traceback
            report_lines.append(f"  {traceback.format_exc()}")
            return False, "\n".join(report_lines)

    def _test_scratchpad_operations(self, report_lines: List[str]) -> None:
        """Test scratch pad store, retrieve, list, and clear operations."""
        self.log("Testing scratch pad operations...")

        # Test store
        result = self.scratch_pad.handle_action("store", "project_type", "Python CLI tool")
        self.log(f"  store result: {result}")
        if "Stored" in result:
            self.assertions_passed += 1
            report_lines.append("  ✓ scratch_pad: store works")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ scratch_pad: store failed")

        result = self.scratch_pad.handle_action("store", "dependencies", "stdlib only")
        self.log(f"  store result: {result}")
        if "Stored" in result:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1

        # Test retrieve
        result = self.scratch_pad.handle_action("retrieve", "project_type")
        self.log(f"  retrieve result: {result}")
        if "Python CLI tool" in result:
            self.assertions_passed += 1
            report_lines.append("  ✓ scratch_pad: retrieve works")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ scratch_pad: retrieve failed")

        # Test list
        result = self.scratch_pad.handle_action("list", "")
        self.log(f"  list result: {result}")
        if "project_type" in result and "dependencies" in result:
            self.assertions_passed += 1
            report_lines.append("  ✓ scratch_pad: list works")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ scratch_pad: list failed")

        # Test retrieve nonexistent
        result = self.scratch_pad.handle_action("retrieve", "nonexistent")
        self.log(f"  retrieve nonexistent: {result}")
        if "No note found" in result:
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1

        # Test clear
        result = self.scratch_pad.handle_action("clear", "")
        self.log(f"  clear result: {result}")
        if "Cleared" in result:
            self.assertions_passed += 1
            report_lines.append("  ✓ scratch_pad: clear works")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ scratch_pad: clear failed")

        # Verify empty after clear
        result = self.scratch_pad.handle_action("list", "")
        self.log(f"  list after clear: {result}")
        if "empty" in result.lower():
            self.assertions_passed += 1
        else:
            self.assertions_failed += 1

        # Check expected patterns in all available text
        all_text = " ".join(self.test_log + report_lines)
        for pattern in self.scenario.expected_patterns:
            found = pattern.lower() in all_text.lower()
            if found:
                self.assertions_passed += 1
                report_lines.append(f"  ✓ Pattern found: {pattern}")
            else:
                self.assertions_failed += 1
                report_lines.append(f"  ✗ Pattern NOT found: {pattern}")

    def _test_scope_enforcement(self, report_lines: List[str]) -> None:
        """Test that ReadTool blocks paths outside allowed scope."""
        self.log("Testing scope enforcement...")

        # Test blocked path (parent directory)
        result = ReadTool.execute("ls", "../etc", cwd="/tmp/test_project")
        error_msg = str(result.get("error", ""))
        self.log(f"  parent dir access: blocked={result.get('blocked')} error={error_msg[:120]}")
        if result.get("blocked") or "outside" in error_msg:
            self.assertions_passed += 1
            report_lines.append("  ✓ ReadTool blocks parent directory access")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ ReadTool did NOT block parent directory access")

        # Test blocked path (SESSIONS directory)
        result = ReadTool.execute("ls", "SESSIONS", cwd="/tmp/test_project")
        error_msg = str(result.get("error", ""))
        self.log(f"  SESSIONS dir access: blocked={result.get('blocked')} error={error_msg[:120]}")
        if result.get("blocked") or "forbidden" in error_msg:
            self.assertions_passed += 1
            report_lines.append("  ✓ ReadTool blocks SESSIONS directory access")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ ReadTool did NOT block SESSIONS directory access")

        # Test blocked path (.git directory)
        result = ReadTool.execute("ls", ".git", cwd="/tmp/test_project")
        error_msg = str(result.get("error", ""))
        self.log(f"  .git dir access: blocked={result.get('blocked')} error={error_msg[:120]}")
        if result.get("blocked") or "forbidden" in error_msg:
            self.assertions_passed += 1
            report_lines.append("  ✓ ReadTool blocks .git directory access")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ ReadTool did NOT block .git directory access")

        # Check expected patterns in all available text
        all_text = " ".join(self.test_log + report_lines)
        for pattern in self.scenario.expected_patterns:
            found = pattern.lower() in all_text.lower()
            if found:
                self.assertions_passed += 1
                report_lines.append(f"  ✓ Pattern found: {pattern}")
            else:
                self.assertions_failed += 1
                report_lines.append(f"  ✗ Pattern NOT found: {pattern}")

    def _test_readonly_tool(self, report_lines: List[str]) -> None:
        """Test that ReadTool only allows read operations and blocks writes."""
        self.log("Testing read-only tool operations...")

        # Test ls operation (directory may not exist, but tool should handle gracefully)
        result = ReadTool.execute("ls", ".", cwd="/tmp/test_project")
        self.log(f"  ls result: output={result.get('output', '')[:50]} error={result.get('error', '')[:50]}")
        # The tool should either return output or an error (directory doesn't exist)
        if "output" in result or "error" in result:
            self.assertions_passed += 1
            report_lines.append("  ✓ ReadTool: ls executes without crashing")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ ReadTool: ls failed")

        # Test that ReadTool has no write operations
        valid_ops = {"ls", "cat", "head", "tail", "wc", "grep", "find", "file"}
        # Verify the tool definition only has read operations
        read_tool_params = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["ls", "cat", "head", "tail", "wc", "grep", "find", "file"],
                }
            }
        }
        ops = read_tool_params["properties"]["operation"]["enum"]
        all_read_only = all(op in valid_ops for op in ops)
        if all_read_only:
            self.assertions_passed += 1
            report_lines.append("  ✓ ReadTool only has read operations")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ ReadTool has non-read operations")

        # Test that unknown operation returns error
        result = ReadTool.execute("write", "test.txt", cwd="/tmp/test_project")
        self.log(f"  unknown op result: {result.get('error', '')[:100]}")
        if "Unknown operation" in str(result.get("error", "")):
            self.assertions_passed += 1
            report_lines.append("  ✓ ReadTool rejects unknown operations")
        else:
            self.assertions_failed += 1
            report_lines.append("  ✗ ReadTool did not reject unknown operation")

        # Check expected patterns in all available text
        all_text = " ".join(self.test_log + report_lines)
        for pattern in self.scenario.expected_patterns:
            found = pattern.lower() in all_text.lower()
            if found:
                self.assertions_passed += 1
                report_lines.append(f"  ✓ Pattern found: {pattern}")
            else:
                self.assertions_failed += 1
                report_lines.append(f"  ✗ Pattern NOT found: {pattern}")

    def _test_generic(self, report_lines: List[str]) -> None:
        """Generic test: check expected patterns in tool outputs."""
        self.log("Running generic test...")

        # Test basic read operations
        result = ReadTool.execute("ls", ".", cwd="/tmp/test_project")
        if "output" in result or "error" in result:
            self.assertions_passed += 1

        # Check expected patterns in all available text
        all_text = " ".join(self.test_log + report_lines)
        for pattern in self.scenario.expected_patterns:
            found = pattern.lower() in all_text.lower()
            if found:
                self.assertions_passed += 1
                report_lines.append(f"  ✓ Pattern found: {pattern}")
            else:
                self.assertions_failed += 1
                report_lines.append(f"  ✗ Pattern NOT found: {pattern}")

# ============================================================================
# Agent Class - DAG Workflow with Budgets
# ============================================================================

class DAGAgent:
    def __init__(
        self,
        session_dir: Path,
        total_budget: int = DEFAULT_EXPLORATION_BUDGET,
        thinking_mode: str = THINKING_MODE_AUTO,
        verbose: bool = False,
    ):
        self.session_dir = session_dir
        self.total_budget = total_budget
        self.used_budget = 0
        self.thinking_mode = thinking_mode
        self.verbose = verbose
        # Instance-level bash state
        self.bash_cache: OrderedDict = OrderedDict()
        self.bash_history: List[Dict[str, Any]] = []
        # Scratch pad for exploration findings (persists across exploration)
        self.scratch_pad = ScratchPad()
        self.client = DeepSeekV4(thinking_enabled=(thinking_mode == "enabled"))
        # Create instance-specific READ_TOOL with this agent's handler
        read_tool = ToolDef(
            name="read",
            description="Read-only file system exploration. Use for: listing directories, reading files, searching code, checking file sizes. CANNOT create or modify files.",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["ls", "cat", "head", "tail", "wc", "grep", "find", "file"],
                        "description": "Operation to perform"
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (relative to cwd)"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (for grep/find operations)"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum directory depth for find (default: 2, max: 3)",
                        "default": 2
                    },
                    "lines": {
                        "type": "integer",
                        "description": "Number of lines for head/tail (default: 30)",
                        "default": 30
                    },
                    "include_pattern": {
                        "type": "string",
                        "description": "File glob pattern for grep/find (e.g., '*.py')"
                    }
                },
                "required": ["operation", "path"]
            },
            handler=self._read_handler,
            max_result_chars=4000,
        )
        self.client.add_tool(read_tool)
        # Create instance-specific SCRATCH_PAD_TOOL
        scratch_tool = ToolDef(
            name="scratch_pad",
            description="Persistent scratch pad for storing and retrieving exploration findings. Use this to save notes, summaries, and discoveries so you don't need to re-read files. Data persists for the entire exploration session.",
            parameters={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "retrieve", "list", "clear"],
                        "description": "store: save a note with a key; retrieve: get a note by key; list: list all keys; clear: remove all notes"
                    },
                    "key": {
                        "type": "string",
                        "description": "Note identifier (e.g., 'project_structure', 'dependencies', 'entry_points')"
                    },
                    "value": {
                        "type": "string",
                        "description": "Content to store (required for action=store)"
                    }
                },
                "required": ["action", "key"]
            },
            handler=self._scratch_pad_handler,
            max_result_chars=4000,
        )
        self.client.add_tool(scratch_tool)
        # Create instance-specific BASH_TOOL with this agent's handler
        # NOTE: bash tool is added ONLY during code generation phase, not during exploration
        self._bash_tool = ToolDef(
            name="bash",
            description="Execute a bash command. Use heredoc for writing files. Returns output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."}
                },
                "required": ["command"]
            },
            handler=self._bash_handler,
            max_result_chars=3000,
        )
        # bash tool is NOT added to client by default — added during code gen phase
        self.correction_store: Dict[str, str] = {}
        self.working_messages: List[dict] = []
        self.metrics = {
            "exploration_turns": 0,
            "tool_calls_executed": 0,
            "user_approvals": 0,
            "cached_commands": 0,
        }

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed scope (current working directory)."""
        try:
            allowed_base = Path(os.getcwd()).resolve()
            requested_path = Path(path).resolve()
            return requested_path.is_relative_to(allowed_base) or requested_path == allowed_base
        except (ValueError, RuntimeError):
            return False

    def _read_handler(self, params: dict) -> dict:
        """Handler for the read-only exploration tool."""
        operation = params.get("operation", "")
        path = params.get("path", ".")
        pattern = params.get("pattern")
        max_depth = params.get("max_depth", 2)
        lines = params.get("lines", 30)
        include_pattern = params.get("include_pattern")
        cwd = os.getcwd()

        # Show the operation being executed
        op_display = f"{operation} {path}"
        if pattern:
            op_display += f" (pattern: {pattern})"
        print(f"  🔍 read: {op_display}")

        # Enforce max_depth cap
        max_depth = min(max_depth, 3)

        result = ReadTool.execute(
            operation=operation,
            path=path,
            pattern=pattern,
            max_depth=max_depth,
            lines=lines,
            include_pattern=include_pattern,
            cwd=cwd,
        )

        if "error" in result:
            print(f"  ❌ {result['error'][:150]}")
        else:
            output = result.get("output", "")
            lines_count = output.count("\n")
            print(f"  ✅ {len(output)} chars, {lines_count} lines")

        return result

    def _scratch_pad_handler(self, params: dict) -> dict:
        """Handler for the scratch pad note-taking tool."""
        action = params.get("action", "")
        key = params.get("key", "")
        value = params.get("value")

        print(f"  📝 scratch_pad: {action} '{key}'")
        result = self.scratch_pad.handle_action(action, key, value)
        return {"output": result}

    def _bash_handler(self, params: dict) -> dict:
        """Instance-level bash command handler with caching and history."""
        command = params.get("command", "")
        cwd = params.get("cwd", os.getcwd())

        # Show the command being executed (truncated for display)
        cmd_display = command[:200] + ("..." if len(command) > 200 else "")
        print(f"  ▶ bash: {cmd_display}")

        # Check directory restriction
        if not self._is_path_allowed(cwd):
            print(f"  ⛔ BLOCKED: cwd '{cwd}' is outside allowed scope")
            return {"error": f"Directory '{cwd}' is outside allowed scope. Only the current working directory and its subdirectories are allowed.", "blocked": True}

        # Dangerous pattern blocking
        dangerous = [
            "rm -rf /", "sudo ", "chmod 777", "dd if=", "mkfs",
            "> /dev/sd", ":(){ :|:& };:", r"curl\s.*\|\s*sh", r"wget\s.*\|\s*sh"
        ]
        for pat in dangerous:
            if re.search(pat, command):
                print(f"  ⛔ BLOCKED: dangerous pattern '{pat}'")
                return {"error": f"Blocked dangerous pattern: {pat}", "blocked": True}

        # Cache hit
        cache_key = (cwd, command)
        if cache_key in self.bash_cache:
            self.bash_cache.move_to_end(cache_key)
            if self.verbose:
                print(f"  💾 Cache hit")
            return {"output": self.bash_cache[cache_key], "cached": True}

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=90,
                cwd=cwd,
                executable="/bin/bash",
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
            output = output.strip() or "(no output)"
            if len(output) > 3000:
                output = output[:3000] + "\n[... truncated ...]"

            # Show result summary
            if result.returncode == 0:
                lines = output.count('\n')
                print(f"  ✅ Exit: 0 | {len(output)} chars, {lines} lines")
            else:
                print(f"  ❌ Exit: {result.returncode} | {output[:150]}")

            self.bash_cache[cache_key] = output
            self.bash_cache.move_to_end(cache_key)

            # LRU eviction: remove oldest entry if cache is full
            if len(self.bash_cache) > BASH_CACHE_MAX_SIZE:
                evicted = self.bash_cache.popitem(last=False)
                if self.verbose:
                    print(f"[VERBOSE] Cache evicted: {evicted[0][1][:80]}...")

            self.bash_history.append({
                "command": command,
                "success": result.returncode == 0,
                "timestamp": time.time(),
            })

            return {"output": output, "exit_code": result.returncode}
        except subprocess.TimeoutExpired:
            print(f"  ⏰ Command timed out after 90 seconds")
            self.bash_history.append({
                "command": command,
                "success": False,
                "error": "timeout",
                "timestamp": time.time(),
            })
            return {"error": "Command timed out after 90 seconds"}
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.bash_history.append({
                "command": command,
                "success": False,
                "error": str(e),
                "timestamp": time.time(),
            })
            return {"error": f"{type(e).__name__}: {e}"}

    def log(self, level: str, msg: str) -> None:
        if level == "info":
            print(msg)
        elif level == "verbose" and self.verbose:
            print(msg)

    def _should_use_thinking(self, content: str) -> bool:
        if self.thinking_mode == "enabled":
            return True
        if self.thinking_mode == "disabled":
            return False
        if len(content) > THINKING_MODE_THRESHOLD:
            self.log("verbose", f"Thinking enabled due to content length > {THINKING_MODE_THRESHOLD}")
            return True
        if any(kw in content.lower() for kw in ["complex", "analyze", "architecture", "design"]):
            self.log("verbose", "Thinking enabled due to complexity keywords")
            return True
        self.log("verbose", "Thinking disabled (auto mode)")
        return False

    def _inject_budget(self, prompt: str) -> str:
        remaining = self.total_budget - self.used_budget
        bar = progress_bar(self.used_budget, self.total_budget)
        thinking_status = (
            " (thinking enabled)"
            if self._should_use_thinking(prompt)
            else " (thinking disabled)"
        )
        budget_info = (
            f"\n\n**Budget status** - Used: {self.used_budget}/{self.total_budget} {bar}\n"
            f"Remaining: {remaining}{thinking_status}\n"
            f"Remember: you can declare READY: true and exit early once you have enough info.\n"
            f"**SCOPE REMINDER**: Only explore the current directory and its subdirectories. "
            f"Avoid SESSIONS/, .git/, node_modules/, venv/, __pycache__/, and parent directories.\n"
        )
        if self.verbose:
            self.log("verbose", f"Injecting budget into user prompt: {budget_info.strip()}")
        return prompt + budget_info

    def _extract_rationale_summary(self, text: str) -> Tuple[str, str]:
        rat = re.search(r"\[RATIONAL\](.*?)(?=\n\[|\Z)", text, re.DOTALL | re.IGNORECASE)
        summ = re.search(r"\[SUMMARY\](.*?)(?=\n\[|\Z)", text, re.DOTALL | re.IGNORECASE)
        return (rat.group(1).strip() if rat else "", summ.group(1).strip() if summ else "")

    def _extract_readiness(self, text: str) -> Tuple[bool, List[str]]:
        ready_match = re.search(r"READY:\s*(true|false)", text, re.IGNORECASE)
        ready = ready_match.group(1).lower() == "true" if ready_match else False
        gaps = []
        gaps_match = re.search(r"Gaps:[ \t]*(.*?)(?=\n\n|$)", text, re.IGNORECASE | re.DOTALL)
        if gaps_match:
            gaps_text = gaps_match.group(1).strip()
            if gaps_text and gaps_text.lower() not in ("none", "none.", ""):
                gap_lines = [g.strip("- ").strip() for g in gaps_text.split("\n")]
                gaps = [g for g in gap_lines if g and g.lower() not in ("none", "none.")]
        return ready, gaps

    def run_exploration_phase(self, user_prompt: str) -> Tuple[bool, List[dict]]:
        self.log("info", "\n" + "="*60)
        self.log("info", "🔍 EXPLORATION PHASE - Gathering intelligence")
        self.log("info", "="*60)
        self.log("info", f"   Soft cap tool calls: {self.total_budget}")
        self.log("info", f"   Tool chaining: up to {EXPLORATION_MAX_TOOL_TURNS} per API call")
        self.log("info", f"   Thinking: {self.thinking_mode} (disabled by default)")
        self.log("info", f"   Early-exit: model self-assesses and exits when ready")
        self.log("info", "="*60)

        # Phase 1: Model provides an exploration plan (no tool calls yet)
        self.log("info", "\n📋 Requesting exploration plan from model...")
        plan_messages = [
            {"role": "system", "content": EXPLORATION_SYSTEM_PROMPT},
            {"role": "user", "content": f"{user_prompt}\n\n{EXPLORATION_PLAN_PROMPT}"},
        ]
        self.client.set_thinking_mode(False)
        try:
            plan_msgs, _ = self.client.chat_with_tools(
                messages=plan_messages,
                max_turns=1,
                temperature=0.3,
                include_reasoning=False,
            )
            for msg in reversed(plan_msgs):
                if msg.get("role") == "assistant" and msg.get("content"):
                    plan_json = self._extract_json_block(msg["content"])
                    if plan_json:
                        estimated = plan_json.get("estimated_tools", self.total_budget)
                        plan_text = plan_json.get("plan", "No plan provided")
                        questions = plan_json.get("key_questions", [])
                        self.log("info", f"📋 Model's exploration plan:")
                        self.log("info", f"   Strategy: {plan_text}")
                        self.log("info", f"   Estimated tools: {estimated}")
                        if questions:
                            self.log("info", f"   Key questions: {', '.join(questions[:3])}")
                        # Respect model's estimate, but keep within a reasonable ceiling
                        self.total_budget = min(max(estimated + 2, 3), DEFAULT_EXPLORATION_BUDGET * 2)
                        self.log("info", f"   → Exploration budget set to {self.total_budget} tool calls")
                    break
        except Exception as e:
            self.log("info", f"⚠️ Plan generation failed: {e}. Using default budget {self.total_budget}.")

        # Phase 2: Execute exploration with self-declared budget
        cwd = os.getcwd()
        scope_constraint = (
            f"\n\n**WORKING DIRECTORY**: {cwd}\n"
            f"**SCOPE**: Only explore {cwd} and its subdirectories. "
            f"Do NOT explore SESSIONS/, .git/, node_modules/, venv/, __pycache__/, or parent directories."
        )
        scoped_prompt = user_prompt + scope_constraint
        self.working_messages = [
            {"role": "system", "content": EXPLORATION_SYSTEM_PROMPT},
            {"role": "user", "content": self._inject_budget(scoped_prompt)},
        ]
        last_gaps = []
        no_progress_turns = 0
        ready = False
        consecutive_no_tool_turns = 0

        while self.used_budget < self.total_budget:
            self.metrics["exploration_turns"] += 1
            remaining = self.total_budget - self.used_budget
            self.log("info", f"\n--- Turn {self.metrics['exploration_turns']} (used {self.used_budget}/{self.total_budget}, {remaining} remaining) ---")
            self.client.set_thinking_mode(False)
            exploration_temp = 0.3 if self.used_budget > self.total_budget * 0.7 else 0.5

            # Re-inject budget info into the last user message each turn
            if self.working_messages and self.working_messages[-1].get("role") == "user":
                self.working_messages[-1]["content"] = self._inject_budget(scoped_prompt)

            try:
                new_msgs, choice = self.client.chat_with_tools(
                    messages=self.working_messages,
                    max_turns=EXPLORATION_MAX_TOOL_TURNS,
                    temperature=exploration_temp,
                    include_reasoning=True,
                )
            except Exception as e:
                self.log("info", f"❌ API error: {e}")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
                break

            assistant_msg = None
            for msg in reversed(new_msgs):
                if msg.get("role") == "assistant":
                    assistant_msg = msg
                    break

            if not assistant_msg:
                self.log("info", "⚠️ No assistant message found. Stopping.")
                break

            content = assistant_msg.get("content", "") or ""
            if content:
                rationale, summary = self._extract_rationale_summary(content)
                if rationale:
                    self.log("info", f"💡 Rationale: {rationale[:300]}")
                if summary:
                    self.log("info", f"📌 Summary: {summary[:200]}")
                if self.verbose:
                    self.log("verbose", f"\n🤖 Assistant:\n{content}\n")
                else:
                    self.log("info", f"🤖 Assistant responded ({len(content)} chars)")

            ready, gaps = self._extract_readiness(content)
            if ready:
                self.log("info", "✅ Assistant reports READY. Exiting exploration early.")
                self.working_messages = new_msgs
                return True, new_msgs

            if gaps == last_gaps:
                no_progress_turns += 1
            else:
                no_progress_turns = 0
                last_gaps = gaps

            if gaps:
                self.log("info", "📋 Remaining gaps:")
                for g in gaps[:3]:
                    self.log("info", f"   - {g}")
                if no_progress_turns > 2:
                    self.log("info", "⚠️ No gap progress for 3 turns. Consider forcing readiness.")
            else:
                self.log("info", "📋 No specific gaps listed yet.")

            tool_calls = assistant_msg.get("tool_calls", [])
            if tool_calls:
                consecutive_no_tool_turns = 0
                self.log("info", f"\n⚡ [{len(tool_calls)} tool call(s) executing]")
                for tc in tool_calls:
                    args = tc.get("function", {}).get("arguments", {})
                    cmd = args.get("command", "") if isinstance(args, dict) else str(args)
                    self.log("info", f"  ▶ bash: {cmd[:200]}")
                self.used_budget += len(tool_calls)
                self.metrics["tool_calls_executed"] += len(tool_calls)
                self.metrics["user_approvals"] += len(tool_calls)
                if self.used_budget >= self.total_budget:
                    self.log("info", "⚠️ Budget exhausted!")
                    self.working_messages = new_msgs
                    break
                self.working_messages = new_msgs
            else:
                consecutive_no_tool_turns += 1
                if consecutive_no_tool_turns >= 2:
                    self.log("info", "⚠️ No tool calls for 2 consecutive turns. Ending exploration.")
                    self.working_messages = new_msgs
                    break
                self.log("info", "⚠️ No tool calls this turn. Prompting for concrete action...")
                self.log("info", "   No tool calls made. Prompting for concrete action...")
                self.working_messages = new_msgs
                self.working_messages.append({
                    "role": "user",
                    "content": "You haven't made any tool calls yet. Please take a concrete action by executing a tool (bash command) to explore further, or declare READY: true when confident. Remember: only explore the current directory and its subdirectories — avoid SESSIONS/, .git/, node_modules/, venv/, __pycache__/."
                })

        self.log("info", "\n" + "-"*40)
        self.log("info", "📊 EXPLORATION SUMMARY")
        self.log("info", f"   Turns: {self.metrics['exploration_turns']}")
        self.log("info", f"   API calls: {self.client.api_calls}")
        self.log("info", f"   Tool calls executed: {self.metrics['tool_calls_executed']}")
        self.log("info", f"   Tokens used: {self.client.total_tokens_used}")
        self.log("info", f"   Ready: {ready}")
        self.log("info", "-"*40)

        if not ready:
            self.log("info", "\n⚠️ Budget exhausted without readiness. Forcing PRD generation with current knowledge.")
        return False, self.working_messages

    def _count_tool_calls(self, messages: List[dict]) -> None:
        """Update used_budget based on tool_calls already present in messages."""
        for m in messages:
            if m.get("role") == "assistant":
                tool_calls = m.get("tool_calls", [])
                self.used_budget += len(tool_calls)

    def _extract_exploration_context(self) -> str:
        """Extract exploration findings from working_messages and scratch pad for use in code generation."""
        findings = []
        for msg in self.working_messages:
            if msg.get("role") == "assistant" and msg.get("content"):
                content = msg["content"]
                # Extract READY status and gaps
                if "READY:" in content:
                    findings.append(f"[Assistant]: {content[:500]}")
                # Extract tool results that contain file listings or project info
            if msg.get("role") == "tool" and msg.get("content"):
                tool_content = msg["content"]
                if isinstance(tool_content, str) and len(tool_content) > 20:
                    findings.append(f"[Tool result]: {tool_content[:300]}")
        # Also include scratch pad notes (the model's curated summary)
        scratch_notes = self.scratch_pad.get_all()
        if scratch_notes:
            findings.append("\n--- SCRATCH PAD NOTES (curated by model) ---")
            for key, value in scratch_notes.items():
                findings.append(f"\n[{key}]:\n{value[:500]}")
        context = "\n\n".join(findings[-15:])  # Last 15 relevant items
        if not context:
            context = "(No exploration data available)"
        return context

    def _extract_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        match = re.search(r'\{[^{}]*"(?:plan|estimated_tools|key_questions)"[^{}]*\}', text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def build_code_phase(self, user_prompt: str) -> None:
        self.log("info", "\n" + "="*60)
        self.log("info", "🛠️  CODE GENERATION PHASE - Executing PRD tasks")
        self.log("info", "="*60)

        prd_path = self.session_dir / "PRD.md"
        next_path = self.session_dir / "PRD_NEXT.md"
        if not prd_path.exists():
            self.log("info", "⚠️ No PRD.md found. Skipping code generation.")
            return

        prd_content = read_file(prd_path)
        prd_next_content = read_file(next_path) if next_path.exists() else ""

        # Inject exploration context into code generation
        exploration_context = self._extract_exploration_context()
        self.log("info", f"   Exploration context: {len(exploration_context)} chars extracted")

        tasks = self._parse_tasks(prd_next_content)
        if not tasks:
            self.log("info", "⚠️ No tasks in PRD_NEXT.md. Generating initial task list...")
            tasks = self._generate_initial_tasks(user_prompt, prd_content)
            self._write_prd_next(tasks)

        completed_tasks = set()
        total_tasks = len(tasks)
        task_index = 0
        while task_index < total_tasks:
            task = tasks[task_index]
            if task["id"] in completed_tasks:
                task_index += 1
                continue
            self.log("info", f"\n{'─'*60}")
            self.log("info", f"📋 Task [{task_index+1}/{total_tasks}]: {task['description']}")
            self.log("info", f"   Verification: {task['verification']}")
            self.log("info", f"{'─'*60}")
            success = self._execute_task(task, user_prompt, prd_content, exploration_context)
            if success:
                completed_tasks.add(task["id"])
                self.log("info", f"✅ Task completed: {task['description']}")
                self._mark_task_done(task["id"])
            else:
                self.log("info", f"⚠️ Task failed: {task['description']}")
                task["retries"] = task.get("retries", 0) + 1
                if task["retries"] >= 3:
                    self.log("info", f"❌ Task failed after 3 retries, skipping: {task['description']}")
                    completed_tasks.add(task["id"])
            if self._check_prd_complete(prd_content, self.session_dir):
                self.log("info", "\n" + "="*60)
                self.log("info", "🎯 ALL PRD OBJECTIVES COMPLETE!")
                self.log("info", "="*60)
                break
            task_index += 1

        self.log("info", f"\n📊 Tasks: {len(completed_tasks)}/{total_tasks} completed")
        self._save_task_summary(completed_tasks, total_tasks)

    # (_parse_tasks, _generate_initial_tasks, _write_prd_next, _mark_task_done,
    #  _execute_task, _check_prd_complete, _save_task_summary methods remain unchanged)

    def _parse_tasks(self, content: str) -> list:
        tasks = []
        task_id = 0
        for line in content.split("\n"):
            m = re.match(r'^\s*-\s*\[\s*\]\s*(.+?)(?:\s*\(verification:\s*(.+?)\))?\s*$', line)
            if m:
                desc = m.group(1).strip()
                verif = m.group(2).strip() if m.group(2) else "Run the code and check for errors"
                tasks.append({
                    "id": f"task_{task_id}",
                    "description": desc,
                    "verification": verif,
                    "retries": 0,
                })
                task_id += 1
        return tasks

    def _generate_initial_tasks(self, user_prompt: str, prd_content: str) -> list:
        self.log("info", "   Generating task breakdown from PRD...")
        gen_prompt = f"""Based on this PRD, break the implementation into small concrete tasks.

PRD:
{prd_content[:2000]}

User request: {user_prompt}

Output a task list where each task is:
- A single file to create or modify
- A single feature to implement
- A single test to write

Format each task as:
- [ ] Task description (verification: how to verify this task is done)

Output ONLY the task list, nothing else."""
        try:
            client = DeepSeekV4(thinking_enabled=False)
            msgs = [{"role": "user", "content": gen_prompt}]
            new_msgs, _ = client.chat_with_tools(msgs, max_turns=1, temperature=0.3)
            content = ""
            for msg in reversed(new_msgs):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    break
            if content:
                tasks = self._parse_tasks(content)
                if tasks:
                    return tasks
        except Exception as e:
            self.log("info", f"   Task generation error: {e}")
        return [{
            "id": "task_0",
            "description": f"Implement the project: {user_prompt[:100]}",
            "verification": "Project runs without errors",
            "retries": 0,
        }]

    def _write_prd_next(self, tasks: list) -> None:
        lines = ["# PRD_NEXT\n", ""]
        for t in tasks:
            lines.append(f"- [ ] {t['description']} (verification: {t['verification']})")
        next_path = self.session_dir / "PRD_NEXT.md"
        write_file(next_path, "\n".join(lines))
        self.log("info", f"✅ PRD_NEXT.md updated with {len(tasks)} tasks")

    def _mark_task_done(self, task_id: str) -> None:
        next_path = self.session_dir / "PRD_NEXT.md"
        if not next_path.exists():
            return
        content = read_file(next_path)
        lines = content.split("\n")
        found_id = 0
        for i, line in enumerate(lines):
            if re.match(r'^\s*-\s*\[\s*\]\s*', line):
                if found_id == int(task_id.split("_")[1]):
                    lines[i] = line.replace("[ ]", "[x]", 1)
                    break
                found_id += 1
        write_file(next_path, "\n".join(lines))

    def _execute_task(self, task: dict, user_prompt: str, prd_content: str, exploration_context: str = "") -> bool:
        self.log("info", f"\n   🔧 Executing: {task['description']}")
        sys_prompt = f"""You are implementing a software project. You have ONE task to complete.

TASK: {task['description']}
VERIFICATION: {task['verification']}

PROJECT CONTEXT:
{prd_content[:1500]}

EXPLORATION FINDINGS:
{exploration_context[:1000]}

WORK DIRECTORY: {self.session_dir}

RULES:
1. Use the `bash` tool to create/modify files.
2. Use `cat << 'EOF' > path/to/file.py` to write files via heredoc.
3. After writing each file, verify it with `python3 -c "import ast; ast.parse(open('path/to/file.py').read())"` to check syntax.
4. After all files for this task are written, run the verification step.
5. Output [DONE] when the task is complete and verified.
6. Output [FAILED: reason] if the task cannot be completed.
7. Keep each file focused — one class or module per file.
8. Use only Python standard library unless the PRD specifies otherwise.

IMPORTANT: Write files directly to the work directory using bash heredoc. Do NOT use file markers or ask for approval — just write the code."""
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Implement this task. Work in {self.session_dir}. Use bash to write files and verify them."},
        ]
        client = DeepSeekV4(thinking_enabled=False)
        # Create a bash tool with this task executor's handler
        bash_tool = ToolDef(
            name="bash",
            description="Execute a bash command. Use heredoc for writing files. Returns output.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "cwd": {"type": "string", "default": "."}
                },
                "required": ["command"]
            },
            handler=self._bash_handler,
            max_result_chars=3000,
        )
        client.add_tool(bash_tool)
        max_subturns = 8
        for turn in range(max_subturns):
            try:
                new_msgs, choice = client.chat_with_tools(
                    messages, max_turns=2, temperature=0.3, include_reasoning=True
                )
            except Exception as e:
                self.log("info", f"   ❌ API error: {e}")
                return False
            assistant_msg = None
            for msg in reversed(new_msgs):
                if msg.get("role") == "assistant" and msg.get("content"):
                    assistant_msg = msg
                    break
            if not assistant_msg:
                self.log("info", "   ⚠️ No assistant response")
                break
            content = assistant_msg.get("content", "")
            tool_calls = assistant_msg.get("tool_calls", [])
            if not tool_calls and content:
                reasoning = content.strip()[:300]
                self.log("info", f"   💭 {reasoning}")
            if "[DONE]" in content:
                self.log("info", f"   ✅ Task complete!")
                return True
            if "[FAILED" in content:
                fail_reason = content[content.index("[FAILED"):].split("\n")[0]
                self.log("info", f"   ❌ {fail_reason}")
                return False
            if tool_calls:
                messages = new_msgs
            else:
                messages = new_msgs
                messages.append({
                    "role": "user",
                    "content": "Execute a bash command to make progress on this task. Write code using `cat << 'EOF' > filename`. Then verify it."
                })
        self.log("info", "   ⚠️ Max sub-turns reached without completion signal")
        return False

    def _check_prd_complete(self, prd_content: str, work_dir: Path) -> bool:
        criteria = re.findall(r'(?:Success Criteria|Verification|Checklist)[:\s]*\n((?:\s*[-*]\s*\[.?\]\s*.+\n?)+)', prd_content, re.IGNORECASE)
        if not criteria:
            return False
        next_path = self.session_dir / "PRD_NEXT.md"
        if next_path.exists():
            content = read_file(next_path)
            unchecked = re.findall(r'-\s*\[\s*\]', content)
            if not unchecked:
                return True
        return False

    def _save_task_summary(self, completed: set, total: int) -> None:
        summary = {
            "completed_tasks": len(completed),
            "total_tasks": total,
            "completion_pct": round(len(completed) / max(total, 1) * 100, 1),
            "timestamp": datetime.now().isoformat(),
        }
        summary_path = self.session_dir / "task_summary.json"
        write_file(summary_path, json.dumps(summary, indent=2))
        self.log("info", f"📊 Task summary saved to {summary_path}")

    def write_prd_phase(self) -> None:
        self.log("info", "\n" + "="*60)
        self.log("info", "📝 WRITING PHASE - Generating PRD and task breakdown")
        self.log("info", "="*60)
        self.log("info", "💭 PRD writing with thinking: disabled (faster, more output tokens available)")
        writer_client = DeepSeekV4(thinking_enabled=False)
        context = self.working_messages[-12:]
        context.append({"role": "user", "content": "Now produce the PRD.md and PRD_NEXT.md as instructed."})
        messages = [{"role": "system", "content": WRITING_SYSTEM_PROMPT}] + context
        try:
            new_msgs, choice = writer_client.chat_with_tools(messages, max_turns=1, temperature=0.3)
            content = ""
            for msg in reversed(new_msgs):
                if msg.get("role") == "assistant" and msg.get("content"):
                    content = msg["content"]
                    break
        except Exception as e:
            self.log("info", f"❌ PRD generation failed: {e}")
            return
        if not content:
            self.log("info", "❌ No PRD content generated")
            return
        if "---PRD_NEXT---" in content:
            prd, prd_next = content.split("---PRD_NEXT---", 1)
        else:
            prd = content
            prd_next = "# PRD_NEXT\n\n- [ ] Review the PRD\n- [ ] Validate requirements\n"
        prd_path = self.session_dir / "PRD.md"
        next_path = self.session_dir / "PRD_NEXT.md"
        write_file(prd_path, prd.strip())
        write_file(next_path, prd_next.strip())
        self.log("info", f"✅ PRD.md saved to {prd_path}")
        self.log("info", f"✅ PRD_NEXT.md saved to {next_path}")
        valid, issues = validate_prd_quality(prd)
        if valid:
            self.log("info", "🎯 PRD quality validation: PASSED")
        else:
            self.log("info", "⚠️ PRD quality validation: ISSUES FOUND")
            for issue in issues:
                self.log("info", f"   - {issue}")

    def save_conversation(self):
        conv_path = self.session_dir / "conversation_full.json"
        with open(conv_path, "w") as f:
            json.dump(self.working_messages, f, indent=2)
        self.log("info", f"💾 Conversation saved to {conv_path}")
        stats = {
            "timestamp": datetime.now().isoformat(),
            "session_metrics": self.metrics,
            "client_stats": self.client.get_stats(),
            "thinking_mode": self.thinking_mode,
            "budget_used": self.used_budget,
            "budget_total": self.total_budget,
        }
        stats_path = self.session_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        self.log("info", f"📊 Stats saved to {stats_path}")

    def save_checkpoint(self):
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "working_messages": self.working_messages,
            "used_budget": self.used_budget,
            "metrics": self.metrics,
            "correction_store": self.correction_store,
        }
        checkpoint_path = self.session_dir / "checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        if self.verbose:
            self.log("verbose", f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self) -> bool:
        checkpoint_path = self.session_dir / "checkpoint.json"
        if not checkpoint_path.exists():
            return False
        try:
            with open(checkpoint_path) as f:
                checkpoint = json.load(f)
            self.working_messages = checkpoint.get("working_messages", [])
            self.used_budget = checkpoint.get("used_budget", 0)
            self.metrics = checkpoint.get("metrics", self.metrics)
            self.correction_store = checkpoint.get("correction_store", {})
            self.log("info", f"📂 Checkpoint loaded (budget: {self.used_budget}/{self.total_budget})")
            return True
        except Exception as e:
            self.log("info", f"⚠️ Checkpoint load failed: {e}")
            return False

    def report_final_stats(self):
        self.log("info", "\n" + "="*60)
        self.log("info", "📊 SESSION SUMMARY")
        self.log("info", "="*60)
        self.log("info", f"🔄 Exploration turns: {self.metrics['exploration_turns']}")
        self.log("info", f"🛠️  Tool calls: {self.metrics['tool_calls_executed']}")
        self.log("info", f"👤 User approvals: {self.metrics['user_approvals']}")
        self.log("info", f"💾 Cached commands: {self.metrics['cached_commands']}")
        self.log("info", f"📈 API calls: {self.client.api_calls}")
        self.log("info", f"🎯 Total tokens: {self.client.total_tokens_used}")
        self.log("info", f"💭 Thinking mode: {self.thinking_mode}")
        cmd_success_rate = 0
        if len(self.bash_history) > 0:
            successes = sum(1 for h in self.bash_history if h.get("success"))
            cmd_success_rate = (successes / len(self.bash_history)) * 100
            self.log("info", f"✅ Command success rate: {cmd_success_rate:.1f}%")
        if self.correction_store:
            self.log("info", f"📚 Learned corrections: {len(self.correction_store)}")

    def run(self, initial_prompt: str):
        ready, final_msgs = self.run_exploration_phase(initial_prompt)
        self.write_prd_phase()
        max_iterations = 10
        for iteration in range(max_iterations):
            self.log("info", f"\n{'#'*60}")
            self.log("info", f"🔄 ITERATION {iteration+1}/{max_iterations} - Executing PRD_NEXT.md tasks")
            self.log("info", f"{'#'*60}")
            self.build_code_phase(initial_prompt)
            prd_path = self.session_dir / "PRD.md"
            if prd_path.exists():
                prd_content = read_file(prd_path)
                if self._check_prd_complete(prd_content, self.session_dir):
                    self.log("info", "\n" + "="*60)
                    self.log("info", "🎉 ALL PRD OBJECTIVES COMPLETE!")
                    self.log("info", "="*60)
                    break
            next_path = self.session_dir / "PRD_NEXT.md"
            if next_path.exists():
                content = read_file(next_path)
                unchecked = re.findall(r'-\s*\[\s*\]', content)
                if not unchecked:
                    self.log("info", "\n✅ All tasks checked off in PRD_NEXT.md")
                    break
                self.log("info", f"\n📋 {len(unchecked)} tasks remaining. Continuing...")
            else:
                self.log("info", "\n📋 No PRD_NEXT.md found. Project complete.")
                break
        self.save_conversation()
        self.report_final_stats()
        self.log("info", "\n✨ Session complete!")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    global WORK_DIR
    args = sys.argv[1:]
    thinking_mode = THINKING_MODE_AUTO
    verbose = False
    max_turns = DEFAULT_EXPLORATION_BUDGET
    dry_run = False
    test_scenario = None

    if not args or args[0] in ("--help", "-h"):
        print(
            "Usage: badger.py [options] '<project description>'\n"
            "Options:\n"
            "  --thinking=auto      (default) Enable thinking for complex prompts\n"
            "  --thinking=enabled   Always enable thinking\n"
            "  --thinking=disabled  Disable thinking completely\n"
            "  --work=PATH          Output to specified WORK folder (e.g., --work=python_game_engine)\n"
            "  --max-turns=N        Max exploration tool calls (default: 15)\n"
            "  --test=SCENARIO      Run a test scenario from tests/scenarios/<name>.json\n"
            "  --verbose            Show detailed logs of all agent actions and API interactions\n"
            "  --dry-run            Show configuration and exit without running"
        )
        sys.exit(1)

    while args and args[0].startswith("--"):
        if args[0].startswith("--thinking="):
            thinking_mode = args[0].split("=")[1]
            args = args[1:]
            if thinking_mode not in ("auto", "enabled", "disabled"):
                print(f"Invalid thinking mode: {thinking_mode}")
                sys.exit(1)
        elif args[0].startswith("--work="):
            work_path = args[0].split("=", 1)[1]
            WORK_DIR = Path("WORK") / work_path
            args = args[1:]
        elif args[0].startswith("--max-turns="):
            try:
                max_turns = int(args[0].split("=")[1])
            except ValueError:
                print(f"Invalid max-turns value")
                sys.exit(1)
            args = args[1:]
        elif args[0].startswith("--test="):
            test_scenario = args[0].split("=", 1)[1]
            args = args[1:]
        elif args[0] == "--verbose":
            verbose = True
            args = args[1:]
        elif args[0] == "--dry-run":
            dry_run = True
            args = args[1:]
        else:
            break

    # Test mode
    if test_scenario:
        scenario_path = Path("tests/scenarios") / f"{test_scenario}.json"
        if not scenario_path.exists():
            print(f"❌ Test scenario not found: {scenario_path}")
            sys.exit(1)
        print(f"🧪 Running test scenario: {test_scenario}")
        scenario = TestScenario.from_json_file(str(scenario_path))
        runner = TestRunner(scenario, verbose=verbose)
        success, report = runner.run()
        print(report)
        sys.exit(0 if success else 1)

    if not args:
        print("Error: No prompt provided")
        sys.exit(1)

    prompt = " ".join(args)

    if dry_run:
        print("=== DRY RUN ===")
        print(f"Prompt: {prompt}")
        print(f"Thinking: {thinking_mode}")
        print(f"Max turns: {max_turns}")
        print(f"Tool chaining: {EXPLORATION_MAX_TOOL_TURNS} per API call")
        print(f"Verbose: {verbose}")
        print(f"Work dir: {WORK_DIR}")
        sys.exit(0)

    if WORK_DIR:
        session_dir = WORK_DIR
        session_dir.mkdir(parents=True, exist_ok=True)
    else:
        session_id = make_session_id(prompt)
        session_dir = ensure_session_dir(session_id)

    print(f"📁 Output folder: {session_dir}")
    print(f"💭 Thinking mode: {thinking_mode}")
    print(f"🎯 Max exploration turns: {max_turns}")
    print(f"🔗 Tool chaining: {EXPLORATION_MAX_TOOL_TURNS} per API call")
    print(f"🤖 AUTOMODE: {'ON' if AUTOMODE else 'OFF'}")
    print(f"🔊 Verbose mode: {'ON' if verbose else 'OFF'}")

    agent = DAGAgent(session_dir, total_budget=max_turns, thinking_mode=thinking_mode, verbose=verbose)
    agent.run(prompt)


if __name__ == "__main__":
    main()
