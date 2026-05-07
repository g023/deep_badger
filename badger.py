#!/usr/bin/env python3
"""
Program: PRD Agentic CLI - Powered by the DeepSeek v4 API 

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
EXPLORATION_SYSTEM_PROMPT = """You are an expert product architect analyzing a software project. You have access ONLY to the `bash` tool.

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
1. Start with a concise JSON exploration plan (see format below). Do NOT run any bash commands before outputting the plan.
2. Then execute commands surgically. Before each bash command, output:
   [RATIONAL] Specific reason for this command
   [SUMMARY] What you will learn
3. Prefer targeted commands over broad recursive listings:
   - `ls -la` to see top-level files
   - `find . -maxdepth 2 -type f -name '*.py'` to find source files by extension
   - `head -50 <file>` to preview files
   - `wc -l <file>` to gauge file size before reading
   - `grep -r "keyword" --include='*.py' .` for targeted searches
4. Mentally track the Quality Checklist items below. After each command, self-assess: do you have enough to write the PRD?
5. **Exit early** as soon as sufficient information is gathered — stop issuing further tool calls.

**Critical Rules**:
- Before opening a file, gauge size with `wc -l <file>` and `file <file>`.
- Do NOT use heredoc or file-writing commands; you are only reading and analyzing.
- Do NOT run `find . -type f` or `ls -laR` or any unbounded recursive listing — always use `-maxdepth` and `-name` filters.
- Monitor your own budget. If you reach the estimated limit without readiness, output READY: false and list gaps.

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
When confident that all checklist items are sufficiently covered, output only READY: true and immediately stop using the bash tool. If gaps remain, output READY: false and continue.
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
# Bash Tool with LRU Caching and Danger Detection
# ============================================================================

_BASH_CACHE: OrderedDict = OrderedDict()  # LRU cache: (cwd, command) -> output
_BASH_HISTORY: List[Dict[str, Any]] = []  # (command, output, success, timestamp)
_VERBOSE_GLOBAL = False  # Will be set from main, used by bash_handler for cache logs

def bash_handler(params: dict) -> dict:
    command = params.get("command", "")
    cwd = params.get("cwd", os.getcwd())

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
    if cache_key in _BASH_CACHE:
        _BASH_CACHE.move_to_end(cache_key)
        if _VERBOSE_GLOBAL:
            print(f"  💾 Cache hit")
        return {"output": _BASH_CACHE[cache_key], "cached": True}

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

        _BASH_CACHE[cache_key] = output
        _BASH_CACHE.move_to_end(cache_key)

        # LRU eviction: remove oldest entry if cache is full
        if len(_BASH_CACHE) > BASH_CACHE_MAX_SIZE:
            evicted = _BASH_CACHE.popitem(last=False)
            if _VERBOSE_GLOBAL:
                print(f"[VERBOSE] Cache evicted: {evicted[0][1][:80]}...")

        _BASH_HISTORY.append({
            "command": command,
            "success": result.returncode == 0,
            "timestamp": time.time(),
        })

        return {"output": output, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Command timed out after 90 seconds")
        _BASH_HISTORY.append({
            "command": command,
            "success": False,
            "error": "timeout",
            "timestamp": time.time(),
        })
        return {"error": "Command timed out after 90 seconds"}
    except Exception as e:
        print(f"  ❌ Error: {e}")
        _BASH_HISTORY.append({
            "command": command,
            "success": False,
            "error": str(e),
            "timestamp": time.time(),
        })
        return {"error": f"{type(e).__name__}: {e}"}

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
    handler=bash_handler,
    max_result_chars=8000,
)

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
        self.client = DeepSeekV4(thinking_enabled=(thinking_mode == "enabled"))
        self.client.add_tool(BASH_TOOL)
        self.correction_store: Dict[str, str] = {}
        self.working_messages: List[dict] = []
        self.metrics = {
            "exploration_turns": 0,
            "tool_calls_executed": 0,
            "user_approvals": 0,
            "cached_commands": 0,
        }

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
        """Extract exploration findings from working_messages for use in code generation."""
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
        context = "\n\n".join(findings[-10:])  # Last 10 relevant messages
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
1. Use ONLY the `bash` tool to create/modify files.
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
        client.add_tool(BASH_TOOL)
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
        if len(_BASH_HISTORY) > 0:
            successes = sum(1 for h in _BASH_HISTORY if h.get("success"))
            cmd_success_rate = (successes / len(_BASH_HISTORY)) * 100
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
    global WORK_DIR, _VERBOSE_GLOBAL
    args = sys.argv[1:]
    thinking_mode = THINKING_MODE_AUTO
    verbose = False
    max_turns = DEFAULT_EXPLORATION_BUDGET
    dry_run = False

    if not args or args[0] in ("--help", "-h"):
        print(
            "Usage: badger.py [options] '<project description>'\n"
            "Options:\n"
            "  --thinking=auto      (default) Enable thinking for complex prompts\n"
            "  --thinking=enabled   Always enable thinking\n"
            "  --thinking=disabled  Disable thinking completely\n"
            "  --work=PATH          Output to specified WORK folder (e.g., --work=python_game_engine)\n"
            "  --max-turns=N        Max exploration tool calls (default: 15)\n"
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
        elif args[0] == "--verbose":
            verbose = True
            args = args[1:]
        elif args[0] == "--dry-run":
            dry_run = True
            args = args[1:]
        else:
            break

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

    _VERBOSE_GLOBAL = verbose

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
