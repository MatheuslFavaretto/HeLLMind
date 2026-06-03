"""HeLLMind unified CLI — one pretty entrypoint for the whole project.

Run `doom-cli` (menu) or `doom-cli -h` (full help). Every command is explained below
and has its own `doom-cli <command> -h` for options.
"""
import argparse
import glob
import os
import subprocess
import sys

from rich.align import Align
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()
PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
EMBER = ["#ffd000", "#ff9500", "#ff5a00", "#ff2d00", "#c41200"]

BANNER = r"""
 ██╗  ██╗███████╗██╗     ██╗     ███╗   ███╗██╗███╗   ██╗██████╗
 ██║  ██║██╔════╝██║     ██║     ████╗ ████║██║████╗  ██║██╔══██╗
 ███████║█████╗  ██║     ██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
 ██╔══██║██╔══╝  ██║     ██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
 ██║  ██║███████╗███████╗███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
 ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
"""

# (group, name, one-liner, longer explanation, example)
COMMANDS = [
    ("▶ Run", "diagnose", "Full diagnostic: eval + behavior flags + next step recommendation",
     "Runs 10 eval episodes, detects behavior patterns (circling/passive/low-exploration) "
     "from telemetry, and prints the exact next test to run. Start here every session.",
     "doom-cli diagnose"),
    ("▶ Run", "train", "Train the agent (auto-resumes this vault's brain)",
     "Runs PPO on ViZDoom. By default it RESUMES the brain stored in the vault "
     "(same vault = keeps learning). Use --fresh to start over, --spatial for the "
     "memory channel, --lstm for a RecurrentPPO/LSTM policy, --no-docs to skip notes.",
     "doom-cli train --map MAP02 --steps 200000"),
    ("▶ Run", "auto", "Autonomous loop: train → eval → self-adjust the reward",
     "The supervisor trains a chunk, evaluates it against the GOAL (explore + finish "
     "+ fight), nudges the reward weights toward the weakest metric, and reverts any "
     "change that makes things worse. Add --llm to let the LLM refine the combat knobs, "
     "--lstm for a recurrent policy. Logs its own progress to the vault.",
     "doom-cli auto --iterations 6 --steps 100000 --map MAP02 --llm"),
    ("▶ Run", "watch", "Watch the brain play in a real-time Doom window",
     "Opens the Doom window and plays N deterministic episodes at real speed so you "
     "can actually see what the agent does.",
     "doom-cli watch --episodes 3"),

    ("📊 Measure", "eval", "Deterministic metrics (kills, accuracy, coverage%, exit-rate)",
     "Runs the brain with no exploration and reports clean numbers — the honest way to "
     "judge it. Add --json for machine-readable output.",
     "doom-cli eval --episodes 10"),
    ("📊 Measure", "maps", "Probe maps for combat / exploration density",
     "Boots each map and spams turn+shoot to measure how many enemies are near spawn "
     "(dense vs sparse) and whether the map hangs ViZDoom. Picks good training maps.",
     "doom-cli maps MAP01 MAP02 MAP07"),
    ("📊 Measure", "progress", "Prove it's learning: eval across checkpoints",
     "Deterministically evaluates several saved checkpoints and shows kills/accuracy/"
     "exploration rising over training — the honest proof the policy improves (training "
     "curves are noisy; argmax eval is the truth).",
     "doom-cli progress --points 5"),
    ("📊 Measure", "status", "Brain + memory + reward config at a glance",
     "A dashboard: which brain is in the vault, how many checkpoints, the persistent "
     "cognitive memory (events/deaths per map), and the current reward weights.",
     "doom-cli status"),
    ("📊 Measure", "config", "Show the resolved configuration (.env + defaults)",
     "Prints every effective setting so you know exactly what a run will use.",
     "doom-cli config"),

    ("🧠 Cognition", "notes", "Regenerate the Obsidian notes (LLM) from the last run",
     "Re-runs the offline writer over the saved snapshots: checkpoint notes, concepts, "
     "minimap, synthesis, lessons, reward suggestions, knowledge graph. Needs Ollama.",
     "doom-cli notes --model qwen2.5:7b"),
    ("🧠 Cognition", "lessons", "Show the cross-run lessons the LLM extracted",
     "Prints 60-lessons/Lessons.md — reusable insights mined from the episode memory "
     "(e.g. 'most deaths happen at low HP').",
     "doom-cli lessons"),
    ("🧠 Cognition", "suggest", "Show the LLM's reward-tweak suggestions",
     "Prints 00-index/Reward Suggestions.md — bounded reward changes the agent proposes "
     "to improve (human-approved: you apply them in .env).",
     "doom-cli suggest"),
    ("🧠 Cognition", "log", "Show the autonomy log (the agent improving itself)",
     "Prints 00-index/Autonomy Log.md — every supervisor iteration, its score, and "
     "whether the reward tweak was kept or reverted.",
     "doom-cli log"),
    ("🧠 Cognition", "bestiary", "Show the factual monster bestiary (world model)",
     "Prints 70-bestiary/Bestiary.md — per-monster facts from ViZDoom telemetry: attack "
     "style (ranged/melee), weapon used against it, threat, where it's seen.",
     "doom-cli bestiary"),
    ("🧠 Cognition", "compare", "Compare two runs (A/B) into a note",
     "Runs writer.compare_runs to produce a side-by-side verdict of two runs' metrics.",
     "doom-cli compare runA runB"),
    ("🧠 Cognition", "recall", "Query episodic memory by keyword, enemy or region",
     "doom-cli recall deaths MAP01  |  doom-cli recall --enemy DoomImp  |  doom-cli recall --region 1x2",
     "doom-cli recall MAP01"),
    ("🧠 Cognition", "behavior", "Detect bad behavior patterns from telemetry",
     "Flags shoot-spam, circling, low-exploration, passivity, route-repetition — "
     "each with a confidence score and a recommendation. Writes 80-recommendations/Behavior.md.",
     "doom-cli behavior"),
    ("🧠 Cognition", "hypothesize", "Generate falsifiable hypotheses from behavior flags",
     "Turns behavior flags into hypotheses (metric, direction, config delta). "
     "Writes 70-hypotheses/Hypotheses.md and saves to the SQLite memory.",
     "doom-cli hypothesize"),
    ("🧠 Cognition", "experiment", "Run a hypothesis-driven A/B experiment",
     "Takes a hypothesis ID, trains control+experimental branches (multi-seed), "
     "judges the result honestly, and records it. Use --list to see open hypotheses.",
     "doom-cli experiment --list / doom-cli experiment --hypothesis 1 --steps 200000"),
    ("🧠 Cognition", "db", "Build/query the SQLite cognitive memory",
     "Rebuilds hellmind.db from the JSONL stores or queries events/lessons/experiments.",
     "doom-cli db build / doom-cli db query MAP01"),
    ("🧠 Cognition", "curriculum", "Show map difficulty scores and forgetting alerts",
     "Computes per-map difficulty (deaths + timeouts + coverage + kills) and detects "
     "skill regression vs historical peak. Writes 40-maps/Curriculum.md.",
     "doom-cli curriculum"),
    ("🧠 Cognition", "research", "Run the full autonomous research loop",
     "Chains: behavior detection → hypothesis → experiment → curriculum → training. "
     "Logs everything to the vault. The cognitive loop in one command.",
     "doom-cli research --iterations 3 --steps 200000 --map MAP01"),

    ("🧠 Cognition", "perception", "Write the agent perception model note to the vault",
     "Creates 20-concepts/Concept - Agent Perception.md explaining how the agent sees "
     "the world: pixels, game vars, objects_info, what it does/doesn't know.",
     "doom-cli perception"),
    ("🛠 Tools", "gif", "Render a gameplay GIF + screenshots from the brain",
     "Builds an animated GIF straight from the observation tensor (agent view + spatial "
     "memory, no screen recording) plus a few PNG stills for the README.",
     "doom-cli gif"),
    ("🛠 Tools", "tb", "Launch TensorBoard on the training logs",
     "Opens TensorBoard at ./tb so you can watch the raw learning curves.",
     "doom-cli tb"),
    ("🛠 Tools", "tests", "Run the test suite (pytest)",
     "Runs all unit tests — fast, no training.",
     "doom-cli tests"),
    ("🛠 Tools", "clean", "Clean caches (and optionally the brain)",
     "Removes ./.cache experiment dirs. Add --brain to also wipe the vault's brain and "
     "--memory to wipe the cognitive memory (asks for confirmation).",
     "doom-cli clean"),
]

GROUP_ORDER = ["▶ Run", "📊 Measure", "🧠 Cognition", "🛠 Tools"]


def banner() -> None:
    text = Text()
    for i, line in enumerate(BANNER.strip("\n").splitlines()):
        text.append(line + "\n", style=f"bold {EMBER[min(i, len(EMBER) - 1)]}")
    console.print(Align.center(text))
    console.print(Align.center(
        Text("Hell + LLM + Mind — an RL agent that explores Doom and documents "
             "its own learning", style="italic #ff9500")))
    console.print()


def menu(full: bool = False) -> None:
    """Pretty help. full=True (from -h) also prints the longer explanation per command."""
    banner()
    for group in GROUP_ORDER:
        rows = [c for c in COMMANDS if c[0] == group]
        t = Table(show_header=True, header_style="bold #ff5a00",
                  border_style="#7a0a00", expand=True, title=f"[bold #ffd000]{group}[/bold #ffd000]",
                  title_justify="left")
        t.add_column("command", style="bold #ffd000", no_wrap=True)
        t.add_column("what it does", style="white")
        t.add_column("example", style="dim italic", no_wrap=True)
        for _, name, short, long, ex in rows:
            t.add_row(name, (short + ("\n[dim]" + long + "[/dim]" if full else "")), ex)
        console.print(t)
    console.print(Align.center(Text(
        "doom-cli <command> -h   for that command's options    ·    "
        "-h / --help   for this screen", style="dim")))


# --------------------------------------------------------------------------- #
def run(cmd: list, env: dict | None = None, title: str = "") -> int:
    if title:
        console.print(Panel(title, style=f"bold {EMBER[0]}", border_style=EMBER[3]))
    return subprocess.run(cmd, env={**os.environ, **(env or {})}, cwd=ROOT).returncode


def _show_md(rel_path: str, missing: str) -> int:
    from config import Config
    path = os.path.join(Config().vault_path, rel_path)
    if not os.path.exists(path):
        console.print(Panel(missing, border_style=EMBER[3], style="yellow"))
        return 1
    with open(path, encoding="utf-8") as f:
        console.print(Panel(Markdown(f.read()), title=rel_path, border_style=EMBER[2],
                            title_align="left"))
    return 0


def cmd_diagnose(a) -> int:
    import glob
    from config import Config
    from writer.memory_store import MemoryStore
    from writer.behavior import detect
    from writer.snapshot_log import SnapshotLog, log_path_for

    cfg = Config()

    # --- Brain state ---
    zips = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "*_steps.zip")),
                  key=os.path.getmtime)
    brain_ok = bool(zips)
    console.rule("[bold #ffd000]🔥 HeLLMind Diagnostic[/bold #ffd000]")

    # --- Eval (only if brain exists) ---
    exploration = 0.0
    kills = 0.0
    exit_rate = 0.0
    accuracy = 0.0

    if brain_ok:
        console.print(f"[dim]Brain: {os.path.basename(zips[-1])}  ({len(zips)} checkpoints)[/dim]")
        console.print("[dim]Running 10 deterministic episodes...[/dim]")
        try:
            import json as _json
            out = subprocess.run(
                [PY, "-m", "rl.eval", "--episodes", "10", "--json"],
                cwd=ROOT, capture_output=True, text=True,
                env={**os.environ, "DOCS_ENABLED": "0", "MEMORY_ENABLED": "0"}
            )
            for line in out.stdout.splitlines():
                if line.startswith("METRICS_JSON "):
                    m = _json.loads(line[len("METRICS_JSON "):])
                    exploration = m.get("explored_fraction", 0.0)
                    kills = m.get("kills_per_episode", 0.0)
                    exit_rate = m.get("exit_rate", 0.0)
                    accuracy = m.get("shooting_accuracy", 0.0)
                    break
        except Exception as e:
            console.print(f"[yellow]Eval failed: {e}[/yellow]")
    else:
        console.print("[yellow]No brain found — train first.[/yellow]")

    # --- Behavior flags ---
    events = MemoryStore.read_events(cfg.memory_dir)
    snap_path = log_path_for(cfg.pending_dir, cfg.run_name)
    snaps = SnapshotLog.read_all(snap_path)
    flags = detect(events, snaps)

    # --- Dashboard ---
    t = Table(header_style="bold #ff5a00", border_style="#7a0a00", show_edge=False)
    t.add_column("metric", style="bold #ffd000")
    t.add_column("value", justify="right")
    t.add_column("target", justify="right", style="dim")
    t.add_column("status", justify="center")

    def _row(label, val, target, ok):
        status = "[green]✅[/green]" if ok else "[red]❌[/red]"
        return label, val, target, status

    t.add_row(*_row("exploration", f"{exploration:.0%}", "> 20%", exploration >= 0.20))
    t.add_row(*_row("kills/ep",    f"{kills:.1f}",       "> 1.0", kills >= 1.0))
    t.add_row(*_row("exit_rate",   f"{exit_rate:.0%}",   "> 0%",  exit_rate > 0.0))
    t.add_row(*_row("accuracy",    f"{accuracy:.0%}",    "> 5%",  accuracy >= 0.05))
    console.print(Panel(t, title="📊 Deterministic eval", border_style=EMBER[2], title_align="left"))

    # --- Behavior flags ---
    if flags:
        ft = Table(header_style="bold #ff5a00", border_style="#7a0a00", show_edge=False)
        ft.add_column("flag", style="bold red")
        ft.add_column("confidence", justify="right")
        ft.add_column("fix")
        for f in sorted(flags, key=lambda x: -x.confidence):
            icon = "🔴" if f.confidence >= 0.7 else "🟡"
            ft.add_row(f"{icon} {f.name}", f"{f.confidence:.0%}", f.recommendation[:60] + "…")
        console.print(Panel(ft, title="🚩 Behavior flags", border_style=EMBER[3], title_align="left"))
    else:
        console.print(Panel("[green]No behavior flags detected.[/green]",
                            title="🚩 Behavior", border_style=EMBER[2], title_align="left"))

    # --- Recommendation ---
    if not brain_ok:
        next_cmd = "doom-cli train --map MAP01 --steps 400000 --fresh"
        reason = "No brain — start training from zero."
    elif exploration < 0.20 and exit_rate == 0.0 and kills < 1.0:
        use_rnd = cfg.use_rnd
        if use_rnd:
            next_cmd = "doom-cli train --map MAP01 --steps 400000 --fresh --rnd"
            reason = "RND already on — train fresh to test intrinsic curiosity."
        else:
            next_cmd = "doom-cli train --map MAP01 --steps 400000 --fresh"
            reason = "Agent passive/circling. Test corrected .env (FRONTIER+MOVE_REWARD floor)."
    elif exploration < 0.20:
        next_cmd = "doom-cli train --map MAP01 --steps 400000 --fresh --rnd"
        reason = "Exploration stuck. Enable RND (USE_RND=1 in .env) — intrinsic curiosity."
    elif exit_rate == 0.0:
        next_cmd = "doom-cli train --map MAP01 --steps 800000 --resume"
        reason = "Exploring OK but never finds exit. Train longer + EXIT_REWARD=1000."
    else:
        next_cmd = "doom-cli research --iterations 3 --steps 200000 --map MAP01"
        reason = "Agent works! Run the cognitive research loop."

    console.print(Panel(
        f"[bold #ffd000]{reason}[/bold #ffd000]\n\n[white]{next_cmd}[/white]",
        title="👉 Next step", border_style=EMBER[0], title_align="left"
    ))
    return 0


def cmd_train(a) -> int:
    env = {"DOCS_ENABLED": "0" if a.no_docs else "1"}
    if a.envs:
        env["N_ENVS"] = str(a.envs)
    if a.spatial:
        env["SPATIAL_MEMORY"] = "1"
    if getattr(a, "depth", False):
        env["DEPTH_PERCEPTION"] = "1"
    if a.lstm:
        env["USE_LSTM"] = "1"
    if a.rnd:
        env["USE_RND"] = "1"
    cmd = [PY, "-m", "rl.train", "--timesteps", str(a.steps)]
    if a.map:
        cmd += ["--maps", a.map]
    if a.envs:
        cmd += ["--n-envs", str(a.envs)]
    cmd.append("--fresh" if a.fresh else "--resume")
    return run(cmd, env, f"🔥 Training {a.map or 'campaign'} · {a.steps:,} steps · "
                         f"{'fresh' if a.fresh else 'resume'} · docs {'off' if a.no_docs else 'on'}"
                         f"{' · LSTM' if a.lstm else ''}")


def cmd_watch(a) -> int:
    cmd = [PY, "-m", "rl.eval", "--render", "--episodes", str(a.episodes)]
    if a.path:
        cmd += ["--path", a.path]
    env = {"USE_LSTM": "1"} if a.lstm else None
    return run(cmd, env, title=f"🎮 Watching the brain play · {a.episodes} episodes")


def cmd_eval(a) -> int:
    cmd = [PY, "-m", "rl.eval", "--episodes", str(a.episodes)]
    if a.path:
        cmd += ["--path", a.path]
    if a.json:
        cmd.append("--json")
    if a.stochastic:
        cmd.append("--stochastic")
    env = {"USE_LSTM": "1"} if a.lstm else None
    mode = "stochastic" if a.stochastic else "deterministic"
    return run(cmd, env, title=f"📊 Evaluating · {a.episodes} {mode} episodes")


def cmd_auto(a) -> int:
    cmd = [PY, "-m", "rl.autonomous", "--iterations", str(a.iterations),
           "--steps", str(a.steps)]
    if a.map:
        cmd += ["--map", a.map]
    if a.fresh:
        cmd.append("--fresh")
    if getattr(a, "spatial", False):
        cmd.append("--spatial")
    if getattr(a, "rnd", False):
        cmd.append("--rnd")
    if getattr(a, "goexplore", False):
        cmd.append("--goexplore")
    if getattr(a, "depth", False):
        cmd.append("--depth")
    if getattr(a, "resume", False):
        cmd.append("--resume")
    if a.llm:
        cmd.append("--llm")
    env = {"USE_LSTM": "1"} if a.lstm else None
    return run(cmd, env, title=f"🤖 Autonomous loop · {a.iterations} iters × {a.steps:,} steps"
                               f"{' · LLM' if a.llm else ''}{' · LSTM' if a.lstm else ''}")


def cmd_notes(a) -> int:
    cmd = [PY, "-m", "writer.process_run"]
    if a.run:
        cmd += ["--run", a.run]
    if a.model:
        cmd += ["--model", a.model]
    return run(cmd, title="📝 Regenerating Obsidian notes (LLM)")


def cmd_compare(a) -> int:
    return run([PY, "-m", "writer.compare_runs", *a.runs],
               title="⚖️  Comparing runs")


def cmd_gif(a) -> int:
    cmd = [PY, os.path.join(ROOT, "scripts", "make_gif.py"), "--steps", str(a.steps)]
    if a.path:
        cmd += ["--path", a.path]
    if a.out:
        cmd += ["--out", a.out]
    return run(cmd, title="🎞️  Rendering gameplay GIF + screenshots")


def cmd_tb(a) -> int:
    console.print(Panel("📈 TensorBoard at http://localhost:6006  (Ctrl-C to stop)",
                        style=f"bold {EMBER[0]}", border_style=EMBER[3]))
    return run([PY, "-m", "tensorboard.main", "--logdir", "./tb"])


def cmd_tests(a) -> int:
    return run([PY, "-m", "pytest", "-q"], title="🧪 Running tests")


def cmd_lessons(a) -> int:
    return _show_md("60-lessons/Lessons.md", "No lessons yet — train with memory enabled.")


def cmd_suggest(a) -> int:
    return _show_md("00-index/Reward Suggestions.md", "No suggestions yet — run a documented training.")


def cmd_log(a) -> int:
    return _show_md("00-index/Autonomy Log.md", "No autonomy log yet — run `doom-cli auto`.")


def cmd_progress(a) -> int:
    return run([PY, "-m", "rl.progress", "--episodes", str(a.episodes), "--points", str(a.points)],
               title="📈 Deterministic-eval progression across checkpoints")


def cmd_bestiary(a) -> int:
    return _show_md("70-bestiary/Bestiary.md",
                    "No bestiary yet — train in campaign mode (monsters are recorded then).")


def cmd_behavior(a) -> int:
    return run([PY, "-m", "writer.behavior"], title="🔍 Detecting behavioral patterns")


def cmd_hypothesize(a) -> int:
    cmd = [PY, "-m", "writer.hypothesize"]
    if a.json:
        cmd.append("--json")
    return run(cmd, title="💡 Generating hypotheses from behavior flags")


def cmd_experiment(a) -> int:
    cmd = [PY, "-m", "rl.experiment"]
    if a.list:
        cmd.append("--list")
        return run(cmd, title="📋 Listing open hypotheses")
    if a.hypothesis is None:
        console.print(Panel(
            "Use --hypothesis <id> to run an experiment, or --list to see open hypotheses.",
            border_style=EMBER[3], style="yellow"
        ))
        return 1
    cmd += ["--hypothesis", str(a.hypothesis), "--steps", str(a.steps),
            "--seeds", a.seeds, "--episodes", str(a.episodes)]
    if a.map:
        cmd += ["--map", a.map]
    if a.dry_run:
        cmd.append("--dry-run")
    return run(cmd, title=f"🧪 Running experiment H{a.hypothesis} · {a.steps:,} steps/arm")


def cmd_curriculum(a) -> int:
    cmd = [PY, "-m", "rl.curriculum"]
    if a.note:
        cmd.append("--note")
    return run(cmd, title="📚 Curriculum: difficulty scores + forgetting alerts")


def cmd_research(a) -> int:
    cmd = [PY, "-m", "rl.research_agent",
           "--iterations", str(a.iterations),
           "--steps", str(a.steps),
           "--episodes", str(a.episodes)]
    if a.map:
        cmd += ["--map", a.map]
    if a.fresh:
        cmd.append("--fresh")
    if a.dry_run:
        cmd.append("--dry-run")
    return run(cmd, title=f"🤖 Research Agent · {a.iterations} iter × {a.steps:,} steps"
                          f"{' [DRY RUN]' if a.dry_run else ''}")


def cmd_db(a) -> int:
    cmd = [PY, "-m", "writer.db", a.db_cmd or "build"]
    if a.keyword:
        cmd.append(a.keyword)
    if a.event_type:
        cmd += ["--type", a.event_type]
    if a.map_name:
        cmd += ["--map", a.map_name]
    if a.lessons:
        cmd.append("--lessons")
    return run(cmd, title=f"🗄️  SQLite memory: {a.db_cmd or 'build'}")


def cmd_eureka(a) -> int:
    cmd = [PY, "-m", "rl.eureka", "--generations", str(a.generations),
           "--pop", str(a.pop), "--steps", str(a.steps), "--episodes", str(a.episodes)]
    if a.map:
        cmd += ["--map", a.map]
    return run(cmd, title="🧬 Eureka — evolving the reward design (LLM-guided)")


def cmd_audit(a) -> int:
    cmd = [PY, "-m", "rl.audit"]
    if a.run:
        cmd += ["--run", a.run]
    if a.json:
        cmd.append("--json")
    if a.plot:
        cmd.append("--plot")
    return run(cmd, title="🔬 RL quality audit — is the agent genuinely learning?")


def cmd_recall(a) -> int:
    from config import Config
    from writer import db as _db
    from writer.recall import recall, recall_enemy, recall_region

    cfg = Config()
    _db.build(cfg.memory_dir)

    if a.enemy:
        from writer.recall import recall_enemy as _re
        rows = _re(a.enemy, memory_dir=cfg.memory_dir)
        console.print(Panel(f"Episodes near [bold]{a.enemy}[/bold]: {len(rows)}",
                            title="🔍 recall enemy", border_style=EMBER[2]))
        for r in rows[:20]:
            console.print(f"  [{r.get('type','?')}] {r.get('map','?')} · "
                          f"hp={r.get('health')} kills={r.get('kills')} "
                          f"region={r.get('region','')} cov={r.get('coverage')}")
        return 0

    if a.region:
        rows = recall_region(a.region, memory_dir=cfg.memory_dir)
        console.print(Panel(f"Episodes ending in region [bold]{a.region}[/bold]: {len(rows)}",
                            title="🗺️  recall region", border_style=EMBER[2]))
        for r in rows[:20]:
            console.print(f"  [{r.get('type','?')}] {r.get('map','?')} · "
                          f"hp={r.get('health')} kills={r.get('kills')} enemy={r.get('nearest_enemy','')}")
        return 0

    # General keyword recall
    query = " ".join(a.query) if a.query else "MAP01"
    results = recall(query, memory_dir=cfg.memory_dir, top_k=a.top_k)
    console.print(Panel(f"Query: [bold]{query}[/bold] → {len(results)} results",
                        title="💭 recall", border_style=EMBER[2]))
    for r in results:
        icon = "📖" if r["source"] == "lesson" else "🎮"
        console.print(f"  {icon} [bold]{r['title']}[/bold]")
        console.print(f"     {r['body'][:120]}")
    return 0


def cmd_clean(a) -> int:
    import shutil

    from config import Config
    cfg = Config()
    targets = [os.path.join(ROOT, ".cache")]
    if a.brain:
        targets.append(cfg.checkpoint_dir)
    if a.memory:
        targets.append(cfg.memory_dir)
    console.print(Panel("Will delete:\n" + "\n".join(f"  • {t}" for t in targets),
                        title="🧹 clean", border_style=EMBER[3], style="yellow"))
    if input("Proceed? [y/N] ").strip().lower() != "y":
        console.print("[dim]aborted[/dim]")
        return 1
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
        console.print(f"[green]removed[/green] {t}")
    return 0


def cmd_config(a) -> int:
    from config import Config
    cfg = Config()
    t = Table(header_style="bold #ff5a00", border_style="#7a0a00", show_edge=False)
    t.add_column("setting", style="bold #ffd000")
    t.add_column("value")
    fields = [
        ("vault", cfg.vault_path), ("campaign", cfg.campaign), ("maps", ",".join(cfg.maps)),
        ("episode_timeout", cfg.episode_timeout), ("n_envs", cfg.n_envs),
        ("spatial_memory", cfg.spatial_memory), ("coverage_reward", cfg.coverage_reward),
        ("exit_reward", cfg.exit_reward), ("hit_reward", cfg.hit_reward),
        ("miss_penalty", cfg.miss_penalty), ("death_penalty", cfg.death_penalty),
        ("weapon_variety_reward", cfg.weapon_variety_reward), ("use_lstm", cfg.use_lstm),
        ("docs_enabled", cfg.docs_enabled), ("llm_model", cfg.llm_model),
        ("write_every_steps", cfg.write_every_steps), ("checkpoint_dir", cfg.checkpoint_dir),
    ]
    for k, v in fields:
        t.add_row(k, str(v))
    banner()
    console.print(Panel(t, title="⚙️  resolved config", border_style=EMBER[2], title_align="left"))
    return 0


def cmd_maps(a) -> int:
    console.print(Panel("🗺️  Probing map density (random turn+shoot, ~30s each)",
                        style=f"bold {EMBER[0]}", border_style=EMBER[3]))
    t = Table(header_style="bold #ff5a00", border_style="#7a0a00")
    for col in ("map", "hits", "kills", "dmg taken", "verdict"):
        t.add_column(col)
    for m in a.maps:
        try:
            out = subprocess.run([PY, os.path.join(ROOT, "scripts", "probe_map.py"), m], cwd=ROOT, timeout=45,
                                 capture_output=True, text=True).stdout
            line = next((l for l in out.splitlines() if l.startswith("MAP=")), "")
            d = dict(p.split("=") for p in line.split() if "=" in p)
            k = int(d.get("kills", 0))
            verdict = ("[green]dense ✅[/green]" if k >= 3
                       else "[yellow]sparse[/yellow]" if d else "[red]?[/red]")
            t.add_row(m, d.get("hits", "-"), d.get("kills", "-"),
                      d.get("dmg_taken", "-"), verdict)
        except subprocess.TimeoutExpired:
            t.add_row(m, "-", "-", "-", "[red]HANGS ✗[/red]")
    console.print(t)
    return 0


def cmd_status(a) -> int:
    from collections import Counter

    from config import Config
    from writer.memory_store import MemoryStore
    cfg = Config()
    banner()

    zips = sorted(glob.glob(os.path.join(cfg.checkpoint_dir, "*_steps.zip")),
                  key=os.path.getmtime)
    brain = Table.grid(padding=(0, 2))
    brain.add_column(style="bold #ffd000")
    brain.add_column()
    brain.add_row("vault", cfg.vault_path)
    brain.add_row("brain", os.path.basename(zips[-1]) if zips else "[dim]none yet[/dim]")
    brain.add_row("checkpoints", str(len(zips)))
    brain.add_row("spatial memory", "on" if cfg.spatial_memory else "off")
    console.print(Panel(brain, title="🧠 brain", border_style=EMBER[2], title_align="left"))

    events = MemoryStore.read_events(cfg.memory_dir)
    by_type = Counter(e.get("type") for e in events)
    by_map = Counter(e.get("map") for e in events if e.get("map"))
    mem = Table.grid(padding=(0, 2))
    mem.add_column(style="bold #ffd000")
    mem.add_column()
    mem.add_row("events", str(len(events)))
    mem.add_row("endings", ", ".join(f"{k}={v}" for k, v in by_type.items()) or "[dim]—[/dim]")
    mem.add_row("per map", ", ".join(f"{k}:{v}" for k, v in by_map.most_common(5)) or "[dim]—[/dim]")
    console.print(Panel(mem, title="💾 memory (persists across runs)",
                        border_style=EMBER[2], title_align="left"))

    rw = cfg.reward_weights()
    rt = Table(header_style="bold #ff5a00", border_style="#7a0a00", show_edge=False)
    rt.add_column("knob", style="bold #ffd000")
    rt.add_column("value", justify="right")
    for k in ("hit_reward", "miss_penalty", "death_penalty", "move_reward",
              "coverage_reward", "exit_reward", "living_reward"):
        rt.add_row(k, str(rw.get(k)))
    console.print(Panel(rt, title="🎯 reward shaping", border_style=EMBER[2], title_align="left"))
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="doom-cli", add_help=False)
    p.add_argument("-h", "--help", action="store_true")
    sub = p.add_subparsers(dest="command")

    t = sub.add_parser("train", add_help=True)
    t.add_argument("--map"); t.add_argument("--steps", type=int, default=200000)
    t.add_argument("--envs", type=int); t.add_argument("--fresh", action="store_true")
    t.add_argument("--spatial", action="store_true"); t.add_argument("--no-docs", action="store_true")
    t.add_argument("--depth", action="store_true", help="Depth-perception obs channel (forces --fresh).")
    t.add_argument("--lstm", action="store_true", help="RecurrentPPO/LSTM policy (USE_LSTM).")
    t.add_argument("--rnd", action="store_true", help="Enable RND intrinsic curiosity (USE_RND=1).")
    t.set_defaults(fn=cmd_train)

    w = sub.add_parser("watch"); w.add_argument("--episodes", type=int, default=3)
    w.add_argument("--path"); w.add_argument("--lstm", action="store_true")
    w.set_defaults(fn=cmd_watch)

    e = sub.add_parser("eval"); e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--path"); e.add_argument("--json", action="store_true")
    e.add_argument("--lstm", action="store_true", help="Evaluate an LSTM brain (USE_LSTM).")
    e.add_argument("--stochastic", action="store_true",
                   help="Sample the policy (vs argmax) — for unconverged brains.")
    e.set_defaults(fn=cmd_eval)

    au = sub.add_parser("auto"); au.add_argument("--iterations", type=int, default=5)
    au.add_argument("--steps", type=int, default=100000); au.add_argument("--map")
    au.add_argument("--fresh", action="store_true")
    au.add_argument("--spatial", action="store_true", help="Spatial memory obs (forces --fresh).")
    au.add_argument("--rnd", action="store_true", help="RND intrinsic curiosity.")
    au.add_argument("--goexplore", action="store_true", help="Go-Explore frontier-goal resets.")
    au.add_argument("--depth", action="store_true", help="Depth-perception obs channel.")
    au.add_argument("--resume", action="store_true", help="Continue a prior auto session.")
    au.add_argument("--llm", action="store_true", help="LLM-refined reward proposals.")
    au.add_argument("--lstm", action="store_true", help="RecurrentPPO/LSTM policy.")
    au.set_defaults(fn=cmd_auto)

    n = sub.add_parser("notes"); n.add_argument("--run"); n.add_argument("--model")
    n.set_defaults(fn=cmd_notes)

    c = sub.add_parser("compare"); c.add_argument("runs", nargs="*"); c.set_defaults(fn=cmd_compare)

    g = sub.add_parser("gif"); g.add_argument("--path"); g.add_argument("--steps", type=int, default=320)
    g.add_argument("--out"); g.set_defaults(fn=cmd_gif)

    sub.add_parser("tb").set_defaults(fn=cmd_tb)
    sub.add_parser("tests").set_defaults(fn=cmd_tests)
    sub.add_parser("lessons").set_defaults(fn=cmd_lessons)
    sub.add_parser("suggest").set_defaults(fn=cmd_suggest)
    sub.add_parser("log").set_defaults(fn=cmd_log)
    sub.add_parser("bestiary").set_defaults(fn=cmd_bestiary)
    pr = sub.add_parser("progress"); pr.add_argument("--episodes", type=int, default=8)
    pr.add_argument("--points", type=int, default=5); pr.set_defaults(fn=cmd_progress)
    sub.add_parser("config").set_defaults(fn=cmd_config)
    sub.add_parser("status").set_defaults(fn=cmd_status)

    au_p = sub.add_parser("audit", help="RL quality audit (explained variance, entropy, KL)")
    au_p.add_argument("--run", default=None, help="TensorBoard run dir (default: latest)")
    au_p.add_argument("--json", action="store_true", dest="json")
    au_p.add_argument("--plot", action="store_true", help="Show matplotlib charts")
    au_p.set_defaults(fn=cmd_audit)

    eu = sub.add_parser("eureka", help="LLM-guided evolutionary reward search")
    eu.add_argument("--generations", type=int, default=3)
    eu.add_argument("--pop", type=int, default=4, help="Candidates per generation")
    eu.add_argument("--steps", type=int, default=50000, help="Train steps per candidate")
    eu.add_argument("--episodes", type=int, default=10)
    eu.add_argument("--map", default=None)
    eu.set_defaults(fn=cmd_eureka)

    cl = sub.add_parser("clean"); cl.add_argument("--brain", action="store_true")
    cl.add_argument("--memory", action="store_true"); cl.set_defaults(fn=cmd_clean)

    m = sub.add_parser("maps"); m.add_argument("maps", nargs="+"); m.set_defaults(fn=cmd_maps)

    sub.add_parser("diagnose").set_defaults(fn=cmd_diagnose)
    sub.add_parser("perception").set_defaults(fn=lambda a: run(
        [PY, "-m", "writer.perception_note"],
        title="🧠 Writing agent perception note to vault"
    ))
    sub.add_parser("behavior").set_defaults(fn=cmd_behavior)

    rc = sub.add_parser("recall", help="Query episodic memory")
    rc.add_argument("query", nargs="*", help="Free-text query")
    rc.add_argument("--enemy", default=None, help="Filter by nearest_enemy (partial match)")
    rc.add_argument("--region", default=None, help="Filter by map region (e.g. '1x2')")
    rc.add_argument("--top-k", type=int, default=10, dest="top_k")
    rc.set_defaults(fn=cmd_recall)

    hyp = sub.add_parser("hypothesize")
    hyp.add_argument("--json", action="store_true"); hyp.set_defaults(fn=cmd_hypothesize)

    ex = sub.add_parser("experiment")
    ex.add_argument("--hypothesis", type=int, default=None)
    ex.add_argument("--steps", type=int, default=200000)
    ex.add_argument("--seeds", default="42,123")
    ex.add_argument("--episodes", type=int, default=15)
    ex.add_argument("--map", default=None)
    ex.add_argument("--list", action="store_true")
    ex.add_argument("--dry-run", action="store_true")
    ex.set_defaults(fn=cmd_experiment)

    db_p = sub.add_parser("db")
    db_p.add_argument("db_cmd", nargs="?", default="build", choices=["build", "query"])
    db_p.add_argument("keyword", nargs="?", default=None)
    db_p.add_argument("--type", dest="event_type", default=None)
    db_p.add_argument("--map", dest="map_name", default=None)
    db_p.add_argument("--lessons", action="store_true")
    db_p.set_defaults(fn=cmd_db)

    cur = sub.add_parser("curriculum")
    cur.add_argument("--note", action="store_true", help="Write vault note.")
    cur.set_defaults(fn=cmd_curriculum)

    res = sub.add_parser("research")
    res.add_argument("--iterations", type=int, default=3)
    res.add_argument("--steps", type=int, default=200000)
    res.add_argument("--episodes", type=int, default=15)
    res.add_argument("--map", default=None)
    res.add_argument("--fresh", action="store_true")
    res.add_argument("--dry-run", action="store_true")
    res.set_defaults(fn=cmd_research)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "help", False) or not getattr(args, "command", None):
        menu(full=getattr(args, "help", False))
        return
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
