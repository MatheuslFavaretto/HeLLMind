# Contributing

Thanks for your interest in HeLLMind!

## Standard
- **English everywhere** — code, comments, docstrings, prompts, notes, commits, docs.
- Python 3.12. Keep the code style consistent with the surrounding files.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Before opening a PR
```bash
python -m pytest -q        # all tests must pass (no ViZDoom/Ollama needed)
```
- Add tests for pure logic (see `tests/` and the `make_doom_info` fixture in
  `conftest.py`) — these run without ViZDoom or Ollama.
- Keep the LLM **out of the training loop**: anything that calls Ollama must run in
  post-processing (`writer/`), never inside a callback's `_on_step`.

## Architecture in one line
Training collects snapshots to `.cache/pending_runs/*.jsonl` (fast, no LLM); after
training, `writer.process_run` turns them into Obsidian notes. See [README](README.md)
for the full picture and [TODO.md](TODO.md) for the backlog.
