# 🦡 Deep Badger — AI-Powered PRD Agent & Code Generator

**Turn any project folder into a battle-tested PRD — then build it, iteratively.**

Deep Badger is an intelligent agentic CLI that explores your codebase, asks the *right* questions, writes professional Product Requirement Documents (PRD), and then executes the tasks to build the project — all powered by [DeepSeek V4](https://www.deepseek.com/).

Author: g023
License: MIT

---

## 🎯 What It Does

1. **🔍 Explores** your project structure, code, tests, and docs autonomously
2. **🧠 Self-assesses readiness** — identifies missing information *before* writing
3. **📝 Generates two documents**:
   - **PRD.md** — Full vision, requirements, success criteria, and risks
   - **PRD_NEXT.md** — Actionable task breakdown for the next iteration
4. **🛠️ Executes tasks** — iterates through PRD_NEXT.md, writing code and verifying each task
5. **🔄 Iterates** — loops until all PRD objectives are complete

No fluff. No hallucinations. Just a structured, verifiable plan — and the code to match.

---

## ⚡ Quick Start

### 1. Set Up Your API Key

Place your DeepSeek API key in a file **one directory above** where the script lives (e.g., `../K.dat` relative to `badger.py`):

```bash
# If badger.py is in /home/user/projects/deep-badger/badger.py, create:
/home/user/projects/K.dat

# Add your key:
sk-your-deepseek-api-key-here
```

Or use an environment variable:

```bash
export DEEPSEEK_API_KEY="sk-your-key"
```

### 2. Run It

```bash
# From anywhere, analyze a folder
cd /tmp/some-random-folder
python3 /path/to/deep-badger/badger.py "A Python package for real-time data streaming"

# Or with a specific output folder
python3 badger.py --work=my_project "Describe what you want to build"
```

### 3. Watch It Explore

```
🔍 EXPLORATION PHASE - Gathering intelligence
============================================================

--- Turn 1/15 (used 0/15, 15 remaining) ---

💡 Rationale: Need to understand the project structure before diving into requirements
📌 Summary: Will map out directory structure, identify tech stack, and list key files

✅ [1 tool call(s) executing]

📋 Remaining gaps:
   - Test coverage and patterns
   - Integration points and APIs
   - Performance characteristics
```

The agent thinks out loud. Every command has a reason. Every discovery moves you closer to a solid PRD.

---

## 🎨 Real-World Examples

### Example 1: Python Game Engine

```bash
python3 badger.py --work=game_engine \
  "A 2D roguelike game engine with turn-based combat and procedural dungeon generation"
```

**Output:**
- `WORK/game_engine/PRD.md` — Full feature spec, success metrics, risk analysis
- `WORK/game_engine/PRD_NEXT.md` — Sprint-ready task list with checkboxes
- `WORK/game_engine/conversation_full.json` — Complete chat transcript
- `WORK/game_engine/stats.json` — Token usage, timing, metrics
- `WORK/game_engine/checkpoint.json` — Resumable session state

### Example 2: API Refactor

```bash
python3 badger.py --work=api_refactor \
  "Refactor the legacy authentication system to OAuth2 with JWT tokens. Current system uses session cookies."
```

The agent explores:
- Existing auth code and database schema
- Dependency tree (what breaks if we change it?)
- Test coverage (do we have safety nets?)
- Compliance requirements

Then writes a PRD with migration strategy and rollback plan, and executes the refactor tasks.

### Example 3: Run from Anywhere

```bash
# You're in /tmp/lunch-break
python3 ~/dev/deep-badger/badger.py "Fast API for multiplayer tic-tac-toe"

# Outputs to SESSIONS/[timestamp]/ wherever you are
# No context pollution, no temp files left behind
```

---

## 🧪 Features in Action

### 🧠 Exploration Plan (Self-Budgeting)

Before executing any commands, the model produces a JSON exploration plan:

```json
{
  "plan": "I will first look at the top-level directory and package manifests, then check key dependencies, then review existing tests and configuration.",
  "estimated_tools": 8,
  "key_questions": ["What language and build system?", "What are the main entry points?"]
}
```

The agent respects its own budget and exits early when confident — no wasted API calls.

### 🔄 Thinking Mode

By default, thinking is **disabled** to save tokens. You can control it:

```bash
# Force thinking on for architecture reviews
python3 badger.py --thinking=enabled "Design a scalable microservice mesh"

# Turn it off completely for speed
python3 badger.py --thinking=disabled "Fix typo in README"

# Auto mode (default) — enables thinking for complex/long prompts
python3 badger.py --thinking=auto "Add a login button to the homepage"
```

### ✅ Automatic Readiness Checks

The agent refuses to write until it's sure. Watch it work through the quality checklist:

```
📋 Remaining gaps:
   - Need to understand error handling patterns
   - Missing performance profiling data

READY: false
```

Once all gaps are closed: **READY: true** → PRD generation begins.

### 💾 Zero External Dependencies

Both files are pure Python 3.9+ with **zero pip dependencies**:
- `badger.py` — The orchestrator (DAG agent, budget management, exploration loops, code generation)
- `_ds4.py` — A minimal DeepSeek client (no `requests`, no `httpx`, no `aiohttp` — just `urllib`)

Just run it. It works.

### 🛡️ Safe by Default

```bash
# The agent has ONLY the bash tool (no file writes, no deletions without heredoc)
# Dangerous patterns are blocked:
rm -rf /           ❌ Blocked
sudo chmod 777     ❌ Blocked
curl ... | sh      ❌ Blocked
```

Optional autopilot for testing (edit `AUTOMODE = True` in `badger.py`):

```python
AUTOMODE = True  # Set to True for hands-free testing — use only in sandboxed environments!
```

### 💾 Bash LRU Caching

Repeated commands (e.g., `ls -la`) are cached with an LRU eviction policy (default: 100 entries), saving tokens and time.

### 📊 Detailed Session Metrics

After exploration, you get a summary:

```
📊 SESSION SUMMARY
============================================================
🔄 Exploration turns: 8
🛠️  Tool calls: 23
👤 User approvals: 23
💾 Cached commands: 3
📈 API calls: 12
🎯 Total tokens: 14,832
💭 Thinking mode: auto
✅ Command success rate: 100.0%
📚 Learned corrections: 0
```

### 🔗 Tool Chaining

The agent can execute up to **3 tool calls per API turn** (configurable via `EXPLORATION_MAX_TOOL_TURNS`), reducing latency and token overhead.

### 📂 Checkpoint & Resume

If interrupted, the session saves a `checkpoint.json` so you can resume where you left off.

---

## 📋 Flag Reference

```bash
badger.py [OPTIONS] "Project description"

OPTIONS:
  --thinking={auto|enabled|disabled}
    auto      (default) Enable thinking for complex/long prompts only
    enabled   Always enable thinking (slower, more thorough)
    disabled  Never use thinking (faster, cheaper)

  --work=PATH
    Output to WORK/<PATH>/ instead of SESSIONS/[timestamp]/
    Example: --work=auth_refactor creates WORK/auth_refactor/

  --max-turns=N
    Max exploration tool calls (default: 15)
    Example: --max-turns=30 for deep analysis

  --verbose
    Show detailed logs of all agent actions and API interactions

  --dry-run
    Show configuration and exit without running anything
```

---

## 🗂️ What Gets Generated

### Default Output (no `--work` flag)

```
SESSIONS/20250506_142830_a1b2c3d4/
├── PRD.md                    # Full product specification
├── PRD_NEXT.md               # Next iteration task breakdown
├── conversation_full.json    # Complete chat transcript
├── stats.json                # Token usage, timing, metrics
└── checkpoint.json           # Resumable session state (if interrupted)
```

### Custom Output (with `--work=my_project`)

```
WORK/my_project/
├── PRD.md
├── PRD_NEXT.md
├── conversation_full.json
├── stats.json
├── checkpoint.json
└── ... (generated code files)
```

### PRD.md Structure

```markdown
# Project Vision

## Goal / Vision

## Functional Requirements
- Feature 1: ...
- Feature 2: ...

## Non-Functional Requirements
- Performance: ...
- Security: ...

## Success Criteria (Verifiable)
- [ ] Metric 1: Deploy <2% latency regression
- [ ] Metric 2: 99.95% uptime
- [ ] Verification: Run load test against baseline

## Constraints & Dependencies
- Must run on Python 3.9+
- Depends on PostgreSQL 12+

## Risks & Mitigation
- Risk: Migration downtime
  Mitigation: Blue-green deployment with canary rollout

## Readiness Checklist
- [x] Architecture documented
- [x] Dependencies mapped
- [x] Test strategy defined
```

### PRD_NEXT.md Structure

```markdown
## Next Sprint (5-10 min tasks each)

- [ ] **Task 1** - Refactor authentication layer
  Verification: All auth tests pass, no regressions

- [ ] **Task 2** - Add metrics collection
  Verification: Prometheus endpoint returns valid metrics

- [ ] **Task 3** - Document API schema
  Verification: Schema validates against OpenAPI 3.0
```

---

## 🔄 The Full Workflow

Deep Badger runs in a **multi-iteration loop**:

1. **Exploration Phase** — Agent explores the project, self-assesses readiness, exits early when confident
2. **Writing Phase** — Generates `PRD.md` and `PRD_NEXT.md` with quality validation
3. **Code Generation Phase** — Iterates through tasks in `PRD_NEXT.md`, writing and verifying code
4. **Completion Check** — Verifies all success criteria are met; if not, loops back to step 3

Up to **10 iterations** are attempted before declaring completion.

---

## 🎯 Who Should Use This?

✅ **Product Managers** — Auto-generate specs for engineering teams  
✅ **Architects** — Explore legacy codebases quickly  
✅ **Developers** — Document your own projects (before refactoring)  
✅ **CTOs** — Plan large features with risk analysis built-in  
✅ **Students** — Learn what good PRDs look like  
✅ **Nerds** — Tinker with agent thinking, budgets, and truncation logic

---

## 🔧 Advanced: Customization

Edit `badger.py` to tweak behavior:

```python
# Exploration budget
DEFAULT_EXPLORATION_BUDGET = 15    # Soft cap for bash commands in exploration
EXPLORATION_MAX_TOOL_TURNS = 3     # Tool chaining per API call

# Token management
TOKEN_BUDGET_ESTIMATE = 4000       # Context window estimate
BASH_CACHE_MAX_SIZE = 100          # LRU cache limit for bash commands

# Progress display
PROGRESS_BAR_WIDTH = 20            # Width of [====    ] bar

# Thinking threshold (auto mode)
THINKING_MODE_THRESHOLD = 999999   # Content length to auto-enable thinking
```

### Conversation Truncation

The `_ds4.py` client automatically truncates long conversations:
- Keeps last **20 messages** max
- Truncates tool results to **2000 chars**
- Truncates user/assistant content to **3000 chars**
- Removes orphaned tool messages

### Rate Limiting

Built-in token bucket rate limiter:
- **5 requests/second** sustained
- **10 burst** capacity
- Automatic backoff on 429/5xx errors (up to 5 retries)

### Use a Different DeepSeek Model

Edit `_ds4.py`:
```python
DEFAULT_MODEL = "deepseek-v4-flash"  # or "deepseek-v4" for more power
```

---

## 🧠 Architecture

```
┌─────────────────────────────────────────────────┐
│                   badger.py                      │
│  ┌─────────────┐   ┌─────────────────────────┐  │
│  │ DAGAgent     │   │  bash_handler()         │  │
│  │  - explore   │──▶│  - LRU cache            │  │
│  │  - write PRD │   │  - danger detection     │  │
│  │  - build code│   │  - timeout (90s)        │  │
│  │  - iterate   │   └─────────────────────────┘  │
│  └──────┬──────┘                                  │
│         │ uses                                    │
│         ▼                                         │
│  ┌─────────────────────────────────────────────┐  │
│  │              _ds4.py                         │  │
│  │  DeepSeekV4 client                          │  │
│  │  - Streaming support                        │  │
│  │  - Tool execution loop                      │  │
│  │  - Conversation truncation                  │  │
│  │  - Retry with backoff                       │  │
│  │  - Rate limiting                            │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## 🐛 Troubleshooting

### "K.dat not found"
Make sure your API key file is one directory *above* the script:
```
/home/user/K.dat                  ← Put it here
/home/user/projects/
  └── deep-badger/
      └── badger.py               ← Script is here
```

Or set the env var:
```bash
export DEEPSEEK_API_KEY="sk-..."
python3 badger.py "Your prompt"
```

### "DeepSeek HTTP 429: Rate Limited"
The client has built-in exponential backoff. Just wait — it'll retry automatically.

If it keeps happening, you've hit the API rate limit. Wait 30 seconds and try again.

### "Thinking mode is too slow"
Use `--thinking=disabled` for faster results, or adjust `THINKING_MODE_THRESHOLD` in `badger.py`.

### Agent got stuck or ran out of budget
Session state is saved to `checkpoint.json`. You can resume later (future feature).

---

## 📈 How It Works (Technical Deep Dive)

### The DAGAgent Lifecycle

1. **Initialization**
   - Creates a session directory (timestamped or custom)
   - Loads API key from `K.dat` or env
   - Initializes DeepSeek client with optional thinking

2. **Exploration Phase**
   - System prompt injected with budget display
   - Agent uses only `bash` tool
   - Every command requires [RATIONAL] and [SUMMARY]
   - Readiness checked after each turn
   - Conversation truncated to last 12 messages (token efficiency)

3. **Writing Phase**
   - Context frozen from exploration phase
   - New client instance for clean PRD generation
   - Optional thinking for quality
   - Output split by `---PRD_NEXT---` marker

4. **Metrics & Cleanup**
   - Token usage tracked
   - Command success rate logged
   - Full transcript saved (JSON)
   - Session ready for review or further iteration

### Why Zero Dependencies?

```python
# _ds4.py uses only:
import json          # stdlib
import urllib        # stdlib
import threading     # stdlib
import time          # stdlib
from pathlib import Path  # stdlib
```

No `requests`. No `httpx`. No `aiohttp`. This means:
- ✅ Drop the script anywhere, it works
- ✅ Lightweight CI/CD integration
- ✅ Minimal attack surface
- ✅ Easy to audit (read the code)

---

## 🚦 Thinking Mode Explained

Extended thinking is a DeepSeek feature where the model can "reason in private" before responding:

```
User: "Design a microservice architecture"

[Model thinks privately for 5-10 seconds...]
💭 Internal reasoning (not shown to user):
   - What are the requirements?
   - What are the tradeoffs?
   - What could go wrong?
   - What's the simplest design?
[/thinking]

🤖 Output: Clear, well-reasoned architecture PRD
```

**When it's enabled:**
- Exploration of complex projects (`--thinking=auto` detects this)
- PRD synthesis (always uses thinking for quality)
- Architecture reviews

**When it's disabled:**
- Simple features ("add a button")
- Budget-conscious runs (`--thinking=disabled`)
- Faster turnaround when thinking isn't needed

---

## 🎓 Learning Resources

Want to understand how Deep Badger works?

1. **Read `badger.py`** (740 lines)
   - Agent orchestration logic
   - Exploration vs. writing phases
   - Readiness self-assessment

2. **Read `_ds4.py`** (413 lines)
   - DeepSeek API client
   - Streaming buffer
   - Tool execution with timeout
   - Rate limiting & retry logic

3. **Inspect `PRD.md` output**
   - Example of what the agent produces
   - Understand PRD structure
   - See how success criteria are written

---

## 🎉 Examples You Can Run Today

### 1. Analyze This Repo
```bash
cd /wherever/deep_badger/lives
python3 badger.py "Analyze the deep_badger codebase and document its architecture"
```

### 2. Document Your Own Project
```bash
cd /home/you/your-cool-project
python3 /path/to/deep_badger/badger.py "Describe what this project does"
```

### 3. Plan a Feature
```bash
python3 badger.py --thinking=enabled --work=oauth_sprint \
  "I need to add OAuth2 authentication to my Flask app. Current state: basic session auth with SQLite. Target: OAuth2 with JWT."
```

### 4. Reverse-Engineer Legacy Code
```bash
cd /path/to/legacy/monster-app
python3 badger.py "What does this codebase do? Map all the modules and their interactions."
```

---

## 📜 License

MIT — use it, fork it, break it, learn from it.

---

## 🙋 Questions?

- 📖 Read the code (it's the best documentation)
- 🔗 Check out [DeepSeek Docs](https://api-docs.deepseek.com/)
- 💡 Tinker with the thinking budget and token limits
- 🚀 Build something cool and share it

Happy PRD-ing! 🎉

---

*Made with 🧠 by [g023](https://github.com/g023/)*
