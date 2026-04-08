---
description: Load full DispatchPulse context and report current state
---

Read these files in order:
1. `CLAUDE.md` at the project root
2. `.claude/skills/dispatchpulse/SKILL.md`

Then run these verification commands in parallel and report the results:

```bash
git log --oneline | head -10
git status
git remote -v
```

```bash
curl -sf --max-time 10 https://arun-sanjay-dispatchpulse.hf.space/health
```

```bash
curl -sf --max-time 10 https://huggingface.co/api/spaces/Arun-Sanjay/dispatchpulse \
  | python3 -c "import sys,json; d=json.load(sys.stdin); rt=d.get('runtime',{}); print('stage:', rt.get('stage'), 'sha:', rt.get('sha','')[:7], 'lastModified:', d.get('lastModified'))"
```

```bash
curl -sf --max-time 10 https://arun-sanjay-dispatchpulse.hf.space/tasks \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('tasks count:', d.get('count'), ' all graded:', all(t.get('has_grader') for t in d.get('tasks', [])))"
```

After running those, give me a one-paragraph summary:
- Is the HF Space live and serving?
- Is the latest commit on both remotes?
- What's the current Round 1 / Phase 2 status (ask me if unsure)?
- What should we work on next?

Do NOT make any code changes yet. Just report the state and wait for my instructions.
