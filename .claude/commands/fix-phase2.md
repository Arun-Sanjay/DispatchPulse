---
description: Targeted playbook for fixing a specific Phase 2 check failure
argument-hint: "<check_name>  e.g. docker | inference | parsing | tasks | llm"
---

I'll tell you which of the 5 Phase 2 checks failed in `$ARGUMENTS`. Follow the matching playbook in `.claude/skills/dispatchpulse/SKILL.md` section 2.

First, re-read `CLAUDE.md` and the skill file to make sure you're synced on the current state.

Then:
1. Identify which section of SKILL.md section 2 applies based on $ARGUMENTS:
   - `docker` → section 2.3
   - `inference` → section 2.4
   - `parsing` → section 2.5
   - `tasks` → section 2.6
   - `llm` → section 2.7
2. Run the debug checklist for that section FIRST before proposing any fix.
3. Propose the MINIMAL fix — do not bundle multiple changes.
4. Show me the exact code diff before applying it.
5. After applying, run `.venv/bin/python tests/test_reward.py && .venv/bin/python tests/test_simulation.py`.
6. Prepare the commit locally with a descriptive message.
7. Give me the two `git push` commands to run (with token placeholders). Do NOT push yourself.

IMPORTANT: fix ONLY the specified check. Do not touch anything else. Do not refactor working code. The rest of the submission is passing and must stay that way.
