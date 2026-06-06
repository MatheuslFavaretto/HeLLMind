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

# Claude-Code-style refined shell palette: ONE warm accent + muted warm greys, lots of
# whitespace. The loud fire palette (EMBER) stays for the `-h` menu; the interactive shell
# uses these calmer tones so it reads clean and modern like Claude Code.
ACCENT      = "#ff7a45"   # the single accent (a warm coral-ember)
ACCENT_DIM  = "#b85c33"   # dimmed accent for borders/secondary marks
SHELL_TEXT  = "#e8dcc8"   # warm off-white body text
SHELL_MUTED = "#988b78"   # muted warm grey (hints, labels)
SHELL_BORDER = "#534637"  # subtle warm-brown border

BANNER = r"""
 ██╗  ██╗███████╗██╗     ██╗     ███╗   ███╗██╗███╗   ██╗██████╗
 ██║  ██║██╔════╝██║     ██║     ████╗ ████║██║████╗  ██║██╔══██╗
 ███████║█████╗  ██║     ██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
 ██╔══██║██╔══╝  ██║     ██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
 ██║  ██║███████╗███████╗███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
 ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
"""

# For the REAL in-game Doomguy face, drop a PNG at assets/doomguy.png — it renders inline as
# a small mark on iTerm2/Warp/WezTerm (the shell stays text-clean everywhere else).
DOOMGUY_IMG = "assets/doomguy.png"

# (group, name, one-liner, longer explanation, example)
COMMANDS = [
    ("▶ Run", "diagnose", "Full diagnostic: eval + behavior flags + next step recommendation",
     "Runs 10 eval episodes, detects behavior patterns (circling/passive/low-exploration) "
     "from telemetry, and prints the exact next test to run. Start here every session.",
     "doom-cli diagnose"),
    ("▶ Run", "dqn", "Train with QR-DQN — off-policy, replay buffer (V2 engine)",
     "Sample-efficient alternative to PPO: a replay buffer lets the agent learn from past "
     "experiences, not just fresh rollouts. Much faster to reach non-zero exit-rate.",
     "doom-cli dqn --map MAP01 --steps 500000"),
    ("▶ Run", "train", "Train the agent with PPO (auto-resumes this vault's brain)",
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
     "judges the result honestly, and records it. Run 'experiment --list' first to see IDs.",
     "doom-cli experiment --hypothesis 1 --steps 200000"),
    ("🧠 Cognition", "db", "Build/query the SQLite cognitive memory",
     "Rebuilds hellmind.db, or queries it. Run each as a SEPARATE command: "
     "'db build', then 'db query MAP01', or 'db query --runs'.",
     "doom-cli db query --runs"),
    ("🧠 Cognition", "timeline", "Show the agent's evolution across auto iterations",
     "Reads the runs table (explored/exit/kills/score per iteration) and prints the trend — "
     "the honest 'is it actually improving?' view over time.",
     "doom-cli timeline"),
    ("🧠 Cognition", "rollback", "Structured rollback audit trail (never degrade permanently)",
     "Every auto adjustment is logged as before/change/after/result + kept|reverted — the "
     "safety net that rolls back any regression. `doom-cli rollback` shows the trail.",
     "doom-cli rollback"),
    ("🧠 Cognition", "knowledge", "Long-term knowledge in 3 tiers (facts/hypotheses/validated)",
     "Aggregates the bestiary (facts), open hypotheses, and proven experiments/learned_config "
     "(validated) into the three certainty tiers — what the agent KNOWS vs suspects vs proved.",
     "doom-cli knowledge"),
    ("📊 Measure", "intel", "Intelligence report: NN architecture, params, training, memory, disk",
     "Proves a real neural network exists: layer-by-layer architecture, parameter count, depth, "
     "best run, key metrics, and disk usage — the 'is this real?' dashboard.",
     "doom-cli intel"),
    ("📊 Measure", "audit", "Is it REALLY learning? (entropy, KL, value loss, grad norm)",
     "Reads the training logs and reports the learning signals — entropy (exploration), "
     "approx-KL (update size), value loss, explained variance — to confirm real learning.",
     "doom-cli audit"),
    ("🧠 Cognition", "learned", "Reward knobs the agent has PROVEN help (persisted)",
     "Shows the validated reward changes in learned_config — knobs an experiment proved help, "
     "re-applied on every train/auto boot so wins accumulate across runs.",
     "doom-cli learned"),
    ("▶ Run", "bc", "Behavioral cloning from your recorded human demos",
     "Trains the policy to imitate your recorded play (record_demo → bc) as a starting point "
     "for RL — the strongest lever for reaching the exit.",
     "doom-cli bc --epochs 10"),
    ("🔬 Research", "benchmark", "Ablation: prove each layer (RND/memory/full) adds value",
     "Trains baseline/rnd/memory/full across seeds with the SAME budget, evaluates honestly, "
     "and writes results/ (csv+json+md) with mean±std so wins aren't luck.",
     "doom-cli benchmark"),
    ("🧠 Cognition", "curriculum", "Show map difficulty scores and forgetting alerts",
     "Computes per-map difficulty (deaths + timeouts + coverage + kills) and detects "
     "skill regression vs historical peak. Writes 40-maps/Curriculum.md.",
     "doom-cli curriculum"),
    ("🧠 Cognition", "perception", "Write the agent perception model note to the vault",
     "Creates 20-concepts/Concept - Agent Perception.md explaining how the agent sees "
     "the world: pixels, game vars, objects_info, what it does/doesn't know.",
     "doom-cli perception"),
    ("▶ Run", "curriculum2", "Progressive curriculum: my_way_home → deadly_corridor → MAP01",
     "The V2 curriculum trains the individual skills first (find-exit, then survive+navigate) "
     "before combining them on full maps — the approach that gets exit-rate off zero.",
     "doom-cli curriculum2 --steps 150000"),
    ("▶ Run", "shell", "Interactive chat-style REPL with the Doomguy backdrop",
     "Starts a chat-like prompt: type /command to run it, /help for the menu, /exit to leave. "
     "Unknown commands get suggestions. The whole CLI, one slash away.",
     "doom-cli shell"),
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
    ("🧠 Cognition", "semantic", "Semantic memory (vector DB): search past episodes by meaning",
     "Embeds episodes as vectors (Ollama nomic-embed-text or TF-IDF fallback) and retrieves "
     "the most similar past situations. 'recall deaths near corridor' > keyword search.",
     "doom-cli semantic recall deaths near corridor"),
]

GROUP_ORDER = ["▶ Run", "📊 Measure", "🔬 Research", "🧠 Cognition", "🛠 Tools"]


def banner() -> None:
    text = Text()
    for i, line in enumerate(BANNER.strip("\n").splitlines()):
        text.append(line + "\n", style=f"bold {EMBER[min(i, len(EMBER) - 1)]}")
    console.print(Align.center(text))
    console.print(Align.center(
        Text("Hell + LLM + Mind — an RL agent that explores Doom and documents "
             "its own learning", style="italic #ff9500")))
    console.print()


# One-line "what this group is for", shown under each section header.
GROUP_TAGLINES = {
    "▶ Run": "train the agent, watch it play, learn from your own demos",
    "📊 Measure": "honest metrics, the neural-net proof, learning signals",
    "🔬 Research": "prove each layer adds value (reproducible ablation)",
    "🧠 Cognition": "memory, hypotheses, experiments, knowledge & rollback",
    "🛠 Tools": "gifs, tensorboard, tests, housekeeping",
}


def menu(full: bool = False) -> None:
    """Pretty help. full=True (from -h) also prints the longer explanation per command."""
    from rich import box
    banner()
    for group in GROUP_ORDER:
        rows = [c for c in COMMANDS if c[0] == group]
        if not rows:
            continue
        tagline = GROUP_TAGLINES.get(group, "")
        title = f"[bold #ffd000]{group}[/bold #ffd000]"
        if tagline:
            title += f"   [italic #ff9500]— {tagline}[/italic #ff9500]"
        t = Table(show_header=True, header_style="bold #ff5a00", box=box.SIMPLE_HEAVY,
                  border_style="#7a0a00", expand=True, title=title, title_justify="left",
                  padding=(0, 1))
        t.add_column("command", style="bold #ffd000", no_wrap=True)
        t.add_column("what it does", style="white")
        t.add_column("example", style="dim italic", no_wrap=True)
        for _, name, short, long, ex in rows:
            t.add_row(name, (short + ("\n[dim]" + long + "[/dim]" if full else "")), ex)
        console.print(t)
        console.print()  # breathing room between sections
    console.print(Align.center(Text.from_markup(
        "[#ffd000]▶ quick start:[/#ffd000]  [bold]doom-cli auto[/bold]  "
        "[dim]→ train + self-improve  ·[/dim]  [bold]doom-cli watch[/bold]  "
        "[dim]→ see it play  ·[/dim]  [bold]doom-cli intel[/bold]  [dim]→ the proof[/dim]")))
    console.print(Align.center(Text(
        "doom-cli <command> -h   for that command's options    ·    "
        "-h / --help   for this screen", style="dim")))


def resolve_slash(token: str, known: set):
    """Resolve a typed slash-command token. Returns (kind, payload):
      ('builtin', name)  for shell built-ins (help/exit/clear/palette)
      ('command', name)  for a real doom-cli command (exact OR unique prefix)
      ('suggest', [..])  for an ambiguous/unknown token (candidates, possibly empty)
    Matching order: builtins → exact → unique prefix (type less) → fuzzy. Pure + testable."""
    import difflib
    t = token.lstrip("/").strip().lower()
    if t in ("help", "h", "?", "menu"):
        return "builtin", "help"
    if t in ("exit", "quit", "q"):
        return "builtin", "exit"
    if t in ("clear", "cls"):
        return "builtin", "clear"
    if t in ("", "commands", "ls"):
        return "builtin", "palette"
    if t in known:
        return "command", t
    prefix = sorted(c for c in known if c.startswith(t))
    if len(prefix) == 1:
        return "command", prefix[0]          # /bench -> benchmark
    if len(prefix) > 1:
        return "suggest", prefix[:5]          # ambiguous prefix -> show the options
    return "suggest", difflib.get_close_matches(t, known, n=3)


# Terminals that speak the iTerm2 inline-image protocol (so the real PNG can render).
_IMG_TERMINALS = {"iTerm.app", "WarpTerminal", "WezTerm", "mintty"}


def _render_iterm_image(path: str, height_rows: int = 16) -> bool:
    """Render a real image inline using the iTerm2 image protocol. Returns True on success.
    Lets the shell show the ACTUAL in-game Doomguy face when a PNG is provided + the terminal
    supports it (iTerm2 / Warp / WezTerm). Other terminals fall back to the ASCII art."""
    import base64
    if (os.environ.get("TERM_PROGRAM") not in _IMG_TERMINALS
            or not sys.stdout.isatty()           # don't dump base64 into pipes/logs
            or not os.path.exists(path)):
        return False
    try:
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        # ESC ]1337;File=...:base64 BEL  — centred-ish via a leading indent.
        sys.stdout.write(
            f"\033]1337;File=inline=1;height={height_rows};preserveAspectRatio=1;size="
            f"{len(data)}:{data}\a\n")
        sys.stdout.flush()
        return True
    except OSError:
        return False


def _doom_backdrop() -> None:
    """Minimal backdrop (Claude-Code-clean). On iTerm2/Warp the real Doomguy PNG renders
    inline as a small mark; everywhere else we stay text-only and let the welcome card carry
    the brand — no giant ASCII art (keeps the shell calm and modern)."""
    console.print()
    _render_iterm_image(os.path.join(ROOT, DOOMGUY_IMG), height_rows=8)


def _shell_welcome() -> None:
    """A Claude-Code-style welcome card: a calm rounded box, one accent, muted hints."""
    from rich import box
    from config import Config
    cfg = Config()
    maps = ", ".join(cfg.maps[:3]) + ("…" if len(cfg.maps) > 3 else "")
    # Which brain is in the vault right now (a small "you're set up" signal). Show the brain
    # FAMILY (strip the _<steps>_steps / _final suffix) so it stays short and on one line.
    brain = "—"
    try:
        import glob as _g
        import re as _re
        cks = _g.glob(os.path.join(cfg.checkpoint_dir, "*.zip"))
        if cks:
            name = os.path.basename(max(cks, key=os.path.getmtime)).replace(".zip", "")
            brain = _re.sub(r"_(\d+_steps|final)$", "", name)
    except Exception:
        pass

    body = Text.from_markup(
        f"[bold {ACCENT}]✻[/bold {ACCENT}] [bold {SHELL_TEXT}]HeLLMind[/bold {SHELL_TEXT}]"
        f"  [{SHELL_MUTED}]· an RL Doom agent that documents its own learning[/{SHELL_MUTED}]\n\n"
        f"[{SHELL_MUTED}]  Type[/{SHELL_MUTED}] [bold {ACCENT}]/[/bold {ACCENT}] "
        f"[{SHELL_MUTED}]for the command palette, or a slash command directly.[/{SHELL_MUTED}]\n\n"
        f"  [{ACCENT}]/diagnose[/{ACCENT}] [{SHELL_MUTED}]where to start[/{SHELL_MUTED}]"
        f"      [{ACCENT}]/auto[/{ACCENT}] [{SHELL_MUTED}]train + improve[/{SHELL_MUTED}]\n"
        f"  [{ACCENT}]/help[/{ACCENT}]     [{SHELL_MUTED}]all commands [/{SHELL_MUTED}]"
        f"     [{ACCENT}]/exit[/{ACCENT}] [{SHELL_MUTED}]quit[/{SHELL_MUTED}]")
    console.print(Panel(body, box=box.ROUNDED, border_style=SHELL_BORDER,
                        padding=(1, 2), expand=True))
    # A subtle context line under the card (cwd-style, like Claude shows the working dir).
    console.print(
        f"  [{SHELL_MUTED}]vault[/{SHELL_MUTED}] [{SHELL_TEXT}]{cfg.vault_path}[/{SHELL_TEXT}]"
        f"   [{SHELL_MUTED}]·[/{SHELL_MUTED}]   [{SHELL_MUTED}]maps[/{SHELL_MUTED}] {maps}"
        f"   [{SHELL_MUTED}]·[/{SHELL_MUTED}]   [{SHELL_MUTED}]brain[/{SHELL_MUTED}] {brain}")
    console.print()


def _shell_palette(filter_text: str = "") -> None:
    """A Claude-style slash menu: each command with its description, grouped. An optional
    filter narrows it (so typing /be then the menu shows only matching commands)."""
    from rich import box
    f = filter_text.lstrip("/").strip().lower()
    shown = 0
    for group in GROUP_ORDER:
        rows = [(c[1], c[2]) for c in COMMANDS
                if c[0] == group and c[1] != "shell" and (not f or c[1].startswith(f))]
        if not rows:
            continue
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 1, 0, 2),
                  title=f"[{SHELL_MUTED}]{group}[/{SHELL_MUTED}]", title_justify="left")
        t.add_column(style=f"bold {ACCENT}", no_wrap=True)
        t.add_column(style=SHELL_TEXT)
        for name, desc in rows:
            t.add_row(f"/{name}", desc)
            shown += 1
        console.print(t)
    if not shown:
        console.print(f"  [{SHELL_MUTED}]no command starts with[/{SHELL_MUTED}] "
                      f"[{ACCENT}]/{f}[/{ACCENT}]\n")
        return
    console.print(f"  [{SHELL_MUTED}]type the start of any name —[/{SHELL_MUTED}] "
                  f"[{ACCENT}]/bench[/{ACCENT}] [{SHELL_MUTED}]runs[/{SHELL_MUTED}] "
                  f"[{ACCENT}]/benchmark[/{ACCENT}]\n")


# Rotating one-liners shown under the input (a little personality, like Claude's tips).
_SHELL_TIPS = [
    "/watch shows the agent play (tempered — the real policy, not the frozen argmax)",
    "/benchmark proves each layer adds value — it runs a finite matrix and stops",
    "/auto --fast uses all your CPU cores without turning any perception off",
    "/knowledge shows what the agent KNOWS in 3 tiers: facts / hypotheses / validated",
    "/rollback is the safety net — every reward change + its keep/revert verdict",
    "type just / to see the whole command palette",
]


def _shell_prompt(tip: str) -> str:
    """A Claude-Code-style boxed input (rounded border + `>` prompt). Returns the stripped
    line (raises on EOF). Used when prompt_toolkit isn't available."""
    width = min(console.width, 100)
    bar = "─" * (width - 2)
    console.print(f"[{SHELL_BORDER}]╭{bar}╮[/{SHELL_BORDER}]")
    line = console.input(f"[{SHELL_BORDER}]│[/{SHELL_BORDER}] [bold {ACCENT}]>[/bold {ACCENT}] ")
    console.print(f"[{SHELL_BORDER}]╰{bar}╯[/{SHELL_BORDER}]")
    console.print(f"  [{SHELL_MUTED}]? /help · {tip}[/{SHELL_MUTED}]")
    return line.strip()


def _make_slash_reader(tips):
    """A live Claude-CLI-style input: pressing / pops a dropdown of commands that filters as
    you type, each with its description. Up-arrow recalls past commands (persisted across
    sessions), and a ghost auto-suggestion completes from history. Returns a `read(tip)->str`
    callable, or None if prompt_toolkit isn't available / there's no TTY (boxed fallback)."""
    if not sys.stdin.isatty():
        return None
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.formatted_text import HTML
        from prompt_toolkit.styles import Style
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    except ImportError:
        return None

    cmds = [(c[1], c[2]) for c in COMMANDS if c[1] != "shell"]

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/") or " " in text:   # only complete the command token
                return
            word = text[1:].lower()
            for name, desc in cmds:
                if name.startswith(word):
                    yield Completion(name, start_position=-len(word),
                                     display=HTML(f"<b>/{name}</b>"), display_meta=desc)

    style = Style.from_dict({
        "prompt": f"{ACCENT} bold",
        "completion-menu.completion": "bg:#2a211a #cdbfa8",
        "completion-menu.completion.current": f"bg:{ACCENT} #1a140e bold",
        "completion-menu.meta.completion": "bg:#211a14 #8f8275",
        "completion-menu.meta.completion.current": f"bg:{ACCENT_DIM} #ffffff",
        "bottom-toolbar": f"{SHELL_MUTED} bg:#1c1611",
    })
    # Persist history under the memory dir so up-arrow survives across shell sessions.
    try:
        from config import Config
        hist_dir = Config().memory_dir
        os.makedirs(hist_dir, exist_ok=True)
        history = FileHistory(os.path.join(hist_dir, "shell_history"))
    except Exception:
        history = None
    session = PromptSession(completer=SlashCompleter(), complete_while_typing=True,
                            style=style, history=history,
                            auto_suggest=AutoSuggestFromHistory())

    def read(tip: str) -> str:
        # Claude-style: a clean `> ` prompt with a calm status bar underneath. The live `/`
        # dropdown (the completer above) is the signature interaction. Inline fg colours keep
        # the toolbar robust across prompt_toolkit versions (no class-resolution surprises).
        a = ACCENT
        toolbar = HTML(
            f"  <style fg='{a}'>/</style> commands   ·   ↑ history   ·   "
            f"<style fg='{a}'>/help</style> shortcuts   ·   "
            f"<style fg='{a}'>/exit</style> quit"
            f"     <style fg='#6e6457'>·  {tip}</style>")
        return session.prompt([("class:prompt", "> ")], bottom_toolbar=toolbar)
    return read


def cmd_shell(a) -> int:
    """A Claude-Code-style REPL: type /command to run it, /help for the menu, /exit to leave."""
    import itertools
    import shlex
    known = {c[1] for c in COMMANDS} - {"shell"}  # you're already in it
    console.clear()
    _doom_backdrop()
    _shell_welcome()
    tips = itertools.cycle(_SHELL_TIPS)
    reader = _make_slash_reader(tips)  # live dropdown (prompt_toolkit) or None -> boxed fallback
    if reader is None:
        console.print(f"  [{SHELL_MUTED}](tip: `pip install prompt_toolkit` for the live "
                      f"/ dropdown)[/{SHELL_MUTED}]\n")
    while True:
        try:
            line = (reader(next(tips)) if reader else _shell_prompt(next(tips))).strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n  [{SHELL_MUTED}]rip and tear… until it is done. 👋[/{SHELL_MUTED}]")
            return 0
        if not line:
            continue
        if not line.startswith("/"):
            console.print(f"  [{SHELL_MUTED}]commands start with[/{SHELL_MUTED}] "
                          f"[{ACCENT}]/[/{ACCENT}] [{SHELL_MUTED}]— try[/{SHELL_MUTED}] "
                          f"[{ACCENT}]/help[/{ACCENT}] [{SHELL_MUTED}]or just[/{SHELL_MUTED}] "
                          f"[{ACCENT}]/[/{ACCENT}]\n")
            continue
        try:
            parts = shlex.split(line[1:])
        except ValueError:
            console.print(f"  [{ACCENT}]couldn't parse that line[/{ACCENT}]\n")
            continue
        cmd = parts[0] if parts else ""   # bare "/" -> palette
        rest = parts[1:]
        kind, payload = resolve_slash(cmd, known)
        if kind == "builtin" and payload == "exit":
            console.print(f"  [{SHELL_MUTED}]rip and tear… until it is done. 👋[/{SHELL_MUTED}]")
            return 0
        if kind == "builtin" and payload == "help":
            console.print(); menu(full=False); console.print(); continue
        if kind == "builtin" and payload == "palette":
            console.print(); _shell_palette(); continue
        if kind == "builtin" and payload == "clear":
            console.clear(); _doom_backdrop(); _shell_welcome(); continue
        if kind == "suggest":
            if len(payload) > 1:
                # Ambiguous prefix (e.g. /be) -> show the matching commands WITH descriptions.
                console.print(); _shell_palette(cmd); continue
            hint = (f"  [{SHELL_MUTED}]did you mean[/{SHELL_MUTED}] [{ACCENT}]/{payload[0]}[/{ACCENT}]?"
                    if payload else f"  [{SHELL_MUTED}]— type / to see them all[/{SHELL_MUTED}]")
            console.print(f"  [{SHELL_MUTED}]unknown:[/{SHELL_MUTED}] "
                          f"[{ACCENT}]/{cmd}[/{ACCENT}]{hint}\n")
            continue
        # A real command: run it through the full CLI (subprocess = clean isolation).
        console.rule(f"[{ACCENT}]/{cmd}[/{ACCENT}]", style=SHELL_BORDER)
        try:
            subprocess.run([PY, os.path.abspath(__file__), payload, *rest], cwd=ROOT)
        except KeyboardInterrupt:
            # Ctrl-C cancels the running command and returns to the shell — it must NOT
            # crash the REPL with a traceback (the child already got the SIGINT).
            console.print("\n  [dim]↩ command interrupted — back to the shell[/dim]")
        console.print()


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
        next_cmd = "doom-cli auto --iterations 6 --steps 150000 --map MAP01 --algo dqn"
        reason = "Agent works! Run the self-improving loop (QR-DQN, sample-efficient)."

    console.print(Panel(
        f"[bold #ffd000]{reason}[/bold #ffd000]\n\n[white]{next_cmd}[/white]",
        title="👉 Next step", border_style=EMBER[0], title_align="left"
    ))
    return 0


def cmd_dqn(a) -> int:
    """Train with QR-DQN (off-policy, replay buffer — V2 sample-efficient engine)."""
    cmd = [PY, "-m", "rl.train_dqn", "--timesteps", str(a.steps)]
    if a.map:
        cmd += ["--map", a.map]
    if getattr(a, "n_envs", None) is not None:
        cmd += ["--n-envs", str(a.n_envs)]  # explicit override only
    if getattr(a, "fresh", False):
        cmd.append("--fresh")
    label = f"🧠 QR-DQN · {a.steps:,} steps · map {a.map or 'default'}"
    return run(cmd, title=label)


def cmd_train(a) -> int:
    env = {"DOCS_ENABLED": "0" if a.no_docs else "1"}
    if a.envs:
        env["N_ENVS"] = str(a.envs)
    if a.spatial:
        env["SPATIAL_MEMORY"] = "1"
    if getattr(a, "depth", False):
        env["DEPTH_PERCEPTION"] = "1"
    if getattr(a, "strafe", False):
        env["STRAFE"] = "1"
    if getattr(a, "game_vars", False):
        env["GAME_VARS"] = "1"
    if getattr(a, "automap", False):
        env["AUTOMAP"] = "1"
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
    # Default to tempered sampling: this agent's pure-argmax policy collapses to passive
    # (it looks "dead" — ignores enemies, won't shoot). T=0.5 shows the REAL learned behavior.
    # Pass --temperature 0 to watch the raw argmax.
    if a.temperature and a.temperature > 0:
        cmd += ["--temperature", str(a.temperature)]
    if getattr(a, "overlay", False):
        cmd.append("--overlay")
    env = {"USE_LSTM": "1"} if a.lstm else None
    label = "argmax" if (a.temperature == 0) else f"tempered T={a.temperature}"
    overlay_note = " + overlay" if getattr(a, "overlay", False) else ""
    return run(cmd, env, title=f"🎮 Watching · {a.episodes} eps · {label}{overlay_note}")


def cmd_eval(a) -> int:
    cmd = [PY, "-m", "rl.eval", "--episodes", str(a.episodes)]
    if a.path:
        cmd += ["--path", a.path]
    if a.json:
        cmd.append("--json")
    if a.stochastic:
        cmd.append("--stochastic")
    if getattr(a, "temperature", None) is not None:
        cmd += ["--temperature", str(a.temperature)]
    env = {"USE_LSTM": "1"} if a.lstm else None
    if getattr(a, "temperature", None) is not None:
        mode = f"tempered T={a.temperature}"
    else:
        mode = "stochastic" if a.stochastic else "deterministic"
    return run(cmd, env, title=f"📊 Evaluating · {a.episodes} {mode} episodes")


def cmd_auto(a) -> int:
    cmd = [PY, "-m", "rl.autonomous", "--iterations", str(a.iterations),
           "--steps", str(a.steps)]
    if a.map:
        cmd += ["--map", a.map]
    # Resume is the DEFAULT; --fresh/--clear starts over.
    if a.fresh or getattr(a, "clear", False):
        cmd.append("--fresh")
    if getattr(a, "spatial", False):
        cmd.append("--spatial")
    if getattr(a, "rnd", False):
        cmd.append("--rnd")
    if getattr(a, "goexplore", False):
        cmd.append("--goexplore")
    if getattr(a, "depth", False):
        cmd.append("--depth")
    if getattr(a, "strafe", False):
        cmd.append("--strafe")
    if getattr(a, "game_vars", False):
        cmd.append("--game-vars")
    if getattr(a, "automap", False):
        cmd.append("--automap")
    # --resume is now the DEFAULT in rl.autonomous; no need to pass it.
    if getattr(a, "fast", False):
        cmd.append("--fast")
    if getattr(a, "graph", False):
        cmd.append("--graph")
    if a.llm:
        cmd.append("--llm")
    _algo = getattr(a, "algo", "ppo")
    if _algo != "ppo":
        cmd += ["--algo", _algo]
    env = {"USE_LSTM": "1"} if a.lstm else None
    _algo_tag = f" · {_algo.upper()}" if _algo != "ppo" else ""
    return run(cmd, env, title=f"🤖 Autonomous loop · {a.iterations} iters × {a.steps:,} steps"
                               f"{_algo_tag}{' · LLM' if a.llm else ''}{' · LSTM' if a.lstm else ''}")


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
    if getattr(a, "runs", False):
        cmd.append("--runs")
    return run(cmd, title=f"🗄️  SQLite memory: {a.db_cmd or 'build'}")


def cmd_intel(a) -> int:
    from config import Config
    from rl.algo import brain_prefix
    from rl.introspect import (
        best_run, brain_report, cognition_stats, disk_usage, training_stats,
    )
    cfg = Config()

    # Resolve the current brain family + its latest checkpoint.
    try:
        from doom.campaign import campaign_metadata
        meta = campaign_metadata(cfg.wad_path, cfg.maps[0], strafe=cfg.strafe)
        n_actions = meta["num_actions"]
    except Exception:
        n_actions = 11
    name_prefix = brain_prefix("campaign", n_actions, cfg.use_lstm,
                               cfg.spatial_memory, cfg.depth_perception, cfg.automap,
                               cfg.frame_stack, cfg.game_vars,
                               getattr(cfg, "semantic_channel", False))
    from rl.train import _latest_checkpoint
    brain_path = _latest_checkpoint(cfg, name_prefix)

    console.print(Panel("🧠  HeLLMind — Intelligence Report", border_style=EMBER[1], style="bold"))

    # ---- Neural network (the proof) ----
    rep = brain_report(brain_path) if brain_path else {"exists": False}
    if not rep.get("exists"):
        console.print(Panel("No trained brain found in this vault yet — run `doom-cli train`.",
                            title="🔬 Neural network", border_style=EMBER[3]))
    elif rep.get("error"):
        console.print(Panel(f"Brain present but unreadable: {rep['error']}",
                            title="🔬 Neural network", border_style=EMBER[3]))
    else:
        t = Table(title="🔬 Neural network — proof it's a real CNN", border_style=EMBER[2])
        t.add_column("property"); t.add_column("value", style="bold")
        t.add_row("policy", rep["policy_class"])
        t.add_row("total parameters", f"{rep['total_params']:,}")
        t.add_row("trainable parameters", f"{rep['trainable_params']:,}")
        t.add_row("weight-bearing layers (depth)", str(rep["depth"]))
        t.add_row("input (observation)", str(rep["obs_shape"]))
        t.add_row("output (actions)", str(rep["n_actions"]))
        t.add_row("device", rep["device"])
        t.add_row("file size", f"{rep['size_mb']} MB")
        console.print(t)
        arch = Table(title="Architecture (the layers)", border_style=EMBER[3])
        arch.add_column("#"); arch.add_column("layer"); arch.add_column("params", justify="right")
        for i, lay in enumerate(rep["weight_layers"]):
            arch.add_row(str(i), lay["desc"], f"{lay['params']:,}")
        console.print(arch)

    # ---- Training ----
    ts = training_stats(cfg.checkpoint_dir, name_prefix)
    br = best_run(cfg.memory_dir)
    tr = Table(title="📈 Training", border_style=EMBER[2])
    tr.add_column("metric"); tr.add_column("value", style="bold")
    tr.add_row("total steps trained", f"{ts['total_steps']:,}")
    tr.add_row("checkpoints saved", str(ts["checkpoints"]))
    if br:
        m = br.get("metrics", {})
        tr.add_row("best run score", f"{br['score']:.3f} ({br['source']}, iter {br.get('iter')})")
        if m:
            tr.add_row("  └ at that run", f"explored={m.get('explored_fraction',0):.0%} "
                       f"kills={m.get('kills_per_episode',0):.2f} exit={m.get('exit_rate',0):.0%}")
    console.print(tr)

    # ---- Cognitive memory ----
    cg = cognition_stats(cfg.memory_dir)
    cm = Table(title="🧩 Cognitive memory (accumulated)", border_style=EMBER[2])
    cm.add_column("store"); cm.add_column("count", style="bold")
    cm.add_row("episode events", f"{cg['events']:,}")
    cm.add_row("lessons", str(cg["lessons"]))
    cm.add_row("hypotheses", f"{cg['hypotheses']} ({cg['confirmed_hypotheses']} confirmed)")
    cm.add_row("experiments", str(cg["experiments"]))
    cm.add_row("learned (proven) knobs", str(cg["learned_knobs"]))
    cm.add_row("frontier cells (Go-Explore)", str(cg["frontier_cells"]))
    cm.add_row("maps with known exit", str(cg["exits_known"]))
    console.print(cm)

    # ---- Disk ----
    du = disk_usage(cfg)
    dk = Table(title="💾 Disk usage", border_style=EMBER[2])
    dk.add_column("component"); dk.add_column("size", style="bold")
    dk.add_row("brains (checkpoints)", f"{du['checkpoints_mb']} MB")
    dk.add_row("cognitive memory", f"{du['memory_mb']} MB")
    dk.add_row("vault (everything)", f"{du['vault_total_mb']} MB")
    console.print(dk)
    return 0


def cmd_learned(a) -> int:
    from config import Config
    from writer.learned_config import LearnedConfig
    from writer.memory_policy import adopt_improved_experiments
    cfg = Config()
    adopt_improved_experiments(cfg.memory_dir)  # pull in any new "improved" verdicts
    rec = LearnedConfig(cfg.memory_dir).load()
    if not rec:
        console.print(Panel("No proven reward knobs yet — run experiments that get an "
                            "'improved' verdict (doom-cli experiment).",
                            title="🧠 learned config", border_style=EMBER[3]))
        return 0
    console.print(Panel(f"{len(rec)} reward knob(s) the agent has PROVEN help — applied on "
                        f"every train/auto boot.", title="🧠 learned config",
                        border_style=EMBER[2]))
    for k, v in rec.items():
        console.print(f"  [bold]{k}[/bold] = {v.get('value')}   "
                      f"[dim]({v.get('source')}, {v.get('verdict')}, "
                      f"conf={v.get('confidence')})[/dim]")
    return 0


def cmd_rollback(a) -> int:
    """Show the structured rollback audit trail: every auto adjustment + its verdict."""
    from config import Config
    from writer.rollback import RollbackLog
    hist = RollbackLog(Config().memory_dir).history()
    if not hist:
        console.print(Panel("No adjustments logged yet — run `doom-cli auto`.",
                            title="↩ rollback log", border_style=EMBER[3]))
        return 0
    n_rev = sum(1 for r in hist if not r.get("kept", True))
    table = Table(title=f"↩ Rollback log — {len(hist)} adjustments, {n_rev} reverted",
                  title_style=f"bold {EMBER[1]}", border_style=EMBER[3])
    table.add_column("iter"); table.add_column("change"); table.add_column("score", justify="right")
    table.add_column("verdict", justify="center")
    for r in hist[-30:]:
        change = "; ".join(f"{k}: {v[0]}→{v[1]}" for k, v in r.get("change", {}).items())
        verdict = "[green]kept[/green]" if r.get("kept") else "[red]↩ reverted[/red]"
        score = r.get("result", {}).get("score", "")
        table.add_row(str(r.get("iter")), change[:60], str(score), verdict)
    console.print(table)
    console.print("[dim]The safety net: any change that regressed was rolled back automatically.[/dim]")
    return 0


def cmd_knowledge(a) -> int:
    """Show the agent's long-term knowledge in 3 tiers: facts / hypotheses / validated."""
    from config import Config
    from writer.knowledge import knowledge_tiers
    cfg = Config()
    tiers = knowledge_tiers(cfg.memory_dir)
    titles = {"facts": ("📚 FACTS (measured)", EMBER[1]),
              "hypotheses": ("❓ HYPOTHESES (open)", EMBER[2]),
              "validated": ("✅ VALIDATED (proven)", EMBER[0])}
    any_shown = False
    for key in ("facts", "validated", "hypotheses"):
        items = tiers.get(key, [])
        title, color = titles[key]
        body = []
        for it in items[:20]:
            line = f"• {it['text']}"
            if it.get("evidence"):
                line += f"  [dim]({it['evidence']})[/dim]"
            body.append(line)
        if not body:
            body = ["[dim](nothing yet — train + run experiments to populate this tier)[/dim]"]
        else:
            any_shown = True
        console.print(Panel("\n".join(body), title=f"{title}  ·  {len(items)}",
                            border_style=color))
    if not any_shown:
        console.print("[dim]Tip: `doom-cli auto` fills facts; `doom-cli experiment` validates.[/dim]")
    return 0


def cmd_benchmark(a) -> int:
    cmd = [PY, "-m", "rl.benchmark", "--map", a.map, "--steps", str(a.steps),
           "--seeds", a.seeds, "--episodes", str(a.episodes), "--n-envs", str(a.n_envs)]
    if a.configs:
        cmd += ["--configs", a.configs]
    if getattr(a, "algo", "ppo") != "ppo":
        cmd += ["--algo", a.algo]
    algo_label = "QR-DQN" if getattr(a, "algo", "ppo") == "dqn" else "PPO"
    return run(cmd, title=f"📊 Ablation benchmark [{algo_label}]: does each layer add value?")


def cmd_timeline(a) -> int:
    """Evolution report: explored / exit-rate / kills / score per auto-loop iteration.
    Reads the SQLite `runs` table (mirrored from autonomy.jsonl) so you can SEE whether
    the agent is actually improving over time, not just trust a single eval."""
    import json as _json

    from config import Config
    from writer import db
    cfg = Config()
    db.build(cfg.memory_dir)                      # always show the freshest view
    runs = db.query_runs(cfg.memory_dir, limit=getattr(a, "limit", 50))
    if not runs:
        console.print(Panel("No auto-loop runs yet — run `doom-cli auto` first.",
                            title="📈 timeline", border_style=EMBER[3]))
        return 0

    runs = sorted(runs, key=lambda r: r["name"])  # chronological (iter-001 → N)
    rows = []
    for r in runs:
        c = _json.loads(r["config_json"] or "{}")
        m = c.get("metrics", {})
        rows.append({
            "name": r["name"], "map": r["maps"] or "?",
            "explored": m.get("explored_fraction"), "exit": m.get("exit_rate"),
            "exit_prog": m.get("exit_progress"),
            "kills": m.get("kills_per_episode"), "score": c.get("score"),
            "kept": c.get("kept"),
        })

    best = max(rows, key=lambda x: x["score"] if x["score"] is not None else -1)
    first_exp = rows[0]["explored"]
    last_exp = rows[-1]["explored"]

    table = Table(title="📈 Agent evolution (per auto iteration)",
                  title_style=f"bold {EMBER[1]}", border_style=EMBER[3])
    table.add_column("iter"); table.add_column("map")
    table.add_column("explored", justify="right"); table.add_column("exit%", justify="right")
    table.add_column("→exit", justify="right")  # dense progress toward the exit
    table.add_column("kills", justify="right"); table.add_column("score", justify="right")
    table.add_column("kept", justify="center")

    def pct(v):
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) else "–"

    def num(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "–"

    for r in rows:
        star = " ⭐" if r["name"] == best["name"] else ""
        table.add_row(
            r["name"].replace("iter-", "#"), str(r["map"]),
            pct(r["explored"]), pct(r["exit"]), pct(r["exit_prog"]),
            num(r["kills"]), num(r["score"]) + star,
            "[green]✓[/green]" if r["kept"] else "[dim]·[/dim]",
        )
    console.print(table)

    # One honest sentence on the trend.
    if isinstance(first_exp, (int, float)) and isinstance(last_exp, (int, float)):
        delta = (last_exp - first_exp) * 100
        arrow = "↑" if delta > 0.5 else ("↓" if delta < -0.5 else "→")
        console.print(f"  exploration {arrow} {first_exp*100:.1f}% → {last_exp*100:.1f}% "
                      f"over {len(rows)} iters · best score {best['score']:.3f} ({best['name']})")
    any_exit = any(isinstance(r["exit"], (int, float)) and r["exit"] > 0 for r in rows)
    console.print(f"  exit reached: {'[green]YES[/green]' if any_exit else '[red]not yet (0%)[/red]'}")
    return 0


def cmd_bc(a) -> int:
    cmd = [PY, "-m", "rl.bc", "--epochs", str(a.epochs)]
    if a.demos:
        cmd += ["--demos", a.demos]
    if getattr(a, "only_success", False):
        cmd.append("--only-success")
    return run(cmd, title="🎓 Behavioral cloning — learning from human demos")


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
    from writer.recall import recall, recall_region

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


def cmd_semantic(a) -> int:
    subcmd = getattr(a, "subcmd", None) or "stats"
    cmd = [PY, "-m", "writer.semantic_memory", subcmd]
    if subcmd == "recall":
        cmd += (a.query or [])
        if getattr(a, "top_k", None):
            cmd += ["--top-k", str(a.top_k)]
    return run(cmd, title=f"🔎 Semantic memory · {subcmd}")


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
    t.add_argument("--strafe", action="store_true", help="Add strafe actions (forces --fresh).")
    t.add_argument("--game-vars", dest="game_vars", action="store_true", help="Feed HEALTH/AMMO into the policy (forces --fresh).")
    t.add_argument("--automap", action="store_true", help="Native top-down automap channel (forces --fresh).")
    t.add_argument("--lstm", action="store_true", help="RecurrentPPO/LSTM policy (USE_LSTM).")
    t.add_argument("--rnd", action="store_true", help="Enable RND intrinsic curiosity (USE_RND=1).")
    t.set_defaults(fn=cmd_train)

    dqn_p = sub.add_parser("dqn", help="Train with QR-DQN (off-policy, replay buffer — V2 engine)")
    dqn_p.add_argument("--map", default=None)
    dqn_p.add_argument("--steps", type=int, default=500_000)
    dqn_p.add_argument("--n-envs", dest="n_envs", type=int, default=None,
                       help="Parallel envs (default: N_ENVS from .env / config).")
    dqn_p.add_argument("--fresh", action="store_true")
    dqn_p.set_defaults(fn=cmd_dqn)

    w = sub.add_parser("watch"); w.add_argument("--episodes", type=int, default=3)
    w.add_argument("--path"); w.add_argument("--lstm", action="store_true")
    w.add_argument("--temperature", type=float, default=0.5,
                   help="Tempered sampling for watching (default 0.5). Use 0 for raw argmax.")
    w.add_argument("--overlay", action="store_true",
                   help="Show HUD (health/ammo bars) + minimap overlay (needs opencv-python).")
    w.set_defaults(fn=cmd_watch)

    e = sub.add_parser("eval"); e.add_argument("--episodes", type=int, default=10)
    e.add_argument("--path"); e.add_argument("--json", action="store_true")
    e.add_argument("--lstm", action="store_true", help="Evaluate an LSTM brain (USE_LSTM).")
    e.add_argument("--stochastic", action="store_true",
                   help="Sample the policy (vs argmax) — for unconverged brains.")
    e.add_argument("--temperature", type=float, default=None,
                   help="Tempered sampling (e.g. 0.5): act on the learned distribution "
                        "without the argmax-collapse. Overrides --stochastic.")
    e.set_defaults(fn=cmd_eval)

    au = sub.add_parser("auto"); au.add_argument("--iterations", type=int, default=5)
    au.add_argument("--steps", type=int, default=100000); au.add_argument("--map")
    au.add_argument("--fresh", "--clear", dest="fresh", action="store_true",
                    help="Start over from zero (fresh brain + cleared history). Default = resume.")
    au.add_argument("--spatial", action="store_true", help="Spatial memory obs (forces --fresh).")
    au.add_argument("--rnd", action="store_true", help="RND intrinsic curiosity.")
    au.add_argument("--goexplore", action="store_true", help="Go-Explore frontier-goal resets.")
    au.add_argument("--depth", action="store_true", help="Depth-perception obs channel.")
    au.add_argument("--strafe", action="store_true", help="Add strafe actions (forces --fresh).")
    au.add_argument("--game-vars", dest="game_vars", action="store_true", help="Feed HEALTH/AMMO into the policy.")
    au.add_argument("--automap", action="store_true", help="Native top-down automap channel (forces --fresh).")
    au.add_argument("--resume", action="store_true", help="(default now) Continue prior session.")
    au.add_argument("--fast", action="store_true",
                    help="Throughput: scale parallel envs to your CPU cores. Disables NOTHING.")
    au.add_argument("--llm", action="store_true", help="LLM-refined reward proposals.")
    au.add_argument("--lstm", action="store_true", help="RecurrentPPO/LSTM policy.")
    au.add_argument("--graph", action="store_true",
                    help="LangGraph coach (V2): observe→diagnose→hypothesize→propose graph.")
    au.add_argument("--algo", default="ppo", choices=["ppo", "dqn"],
                    help="RL algorithm: ppo (default) or dqn (QR-DQN, off-policy + replay buffer).")
    au.set_defaults(fn=cmd_auto)

    sub.add_parser("shell", help="Interactive chat-style REPL (type /command to run)").set_defaults(fn=cmd_shell)

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

    sub.add_parser("intel", help="Intelligence report: NN architecture/params/depth, training, memory, disk").set_defaults(fn=cmd_intel)
    sub.add_parser("learned", help="Show reward knobs the agent has PROVEN help").set_defaults(fn=cmd_learned)
    tl = sub.add_parser("timeline", help="Evolution report: explored/exit/kills/score per auto iteration")
    tl.add_argument("--limit", type=int, default=50)
    tl.set_defaults(fn=cmd_timeline)
    sub.add_parser("knowledge", help="Long-term knowledge in 3 tiers: facts / hypotheses / validated").set_defaults(fn=cmd_knowledge)
    sub.add_parser("rollback", help="Structured rollback audit trail (before/change/after/result per adjustment)").set_defaults(fn=cmd_rollback)
    bm = sub.add_parser("benchmark", help="Ablation: train baseline/rnd/memory/full × seeds, prove each layer adds value")
    bm.add_argument("--map", default="MAP01"); bm.add_argument("--steps", type=int, default=50000)
    bm.add_argument("--seeds", default="42,123"); bm.add_argument("--episodes", type=int, default=20)
    bm.add_argument("--n-envs", dest="n_envs", type=int, default=4)
    bm.add_argument("--configs", default=None)
    bm.add_argument("--algo", default="ppo", choices=["ppo", "dqn"],
                    help="Algorithm to benchmark (ppo=default, dqn=QR-DQN V2 engine).")
    bm.set_defaults(fn=cmd_benchmark)

    bc_p = sub.add_parser("bc", help="Behavioral cloning from human SPECTATOR demos")
    bc_p.add_argument("--demos", default=None, help="Demos dir (default: <memory>/demos)")
    bc_p.add_argument("--epochs", type=int, default=10)
    bc_p.add_argument("--only-success", dest="only_success", action="store_true",
                      help="Clone ONLY demos that reached the exit (recommended — BC's premise).")
    bc_p.set_defaults(fn=cmd_bc)

    # eureka and research are removed from the parser (Phase 0 cut — still runnable
    # directly via python -m rl.eureka / python -m rl.research_agent).

    cl = sub.add_parser("clean"); cl.add_argument("--brain", action="store_true")
    cl.add_argument("--memory", action="store_true"); cl.set_defaults(fn=cmd_clean)

    sem = sub.add_parser("semantic",
                         help="Semantic memory (vector DB): embed events, search by meaning")
    sem.add_argument("subcmd", nargs="?", default="stats",
                     choices=["recall", "index", "stats"],
                     help="recall QUERY | index | stats")
    sem.add_argument("query", nargs="*", help="Free-text query (for recall)")
    sem.add_argument("--top-k", type=int, default=5, dest="top_k")
    sem.set_defaults(fn=cmd_semantic)

    cur2 = sub.add_parser("curriculum2",
                           help="Progressive curriculum: my_way_home → deadly_corridor → MAP01 (V2 Phase 2)")
    cur2.add_argument("--map", default="MAP01")
    cur2.add_argument("--steps", type=int, default=150000, help="Steps per stage.")
    cur2.add_argument("--algo", default="ppo", choices=["ppo", "dqn"])
    cur2.add_argument("--stages", default="mywh,corridor,navigate,full",
                      help="Stages: mywh,corridor (built-in scenarios) + navigate,survive,full (campaign).")
    cur2.set_defaults(fn=lambda a: __import__("subprocess").run(
        [__import__("sys").executable, "-m", "rl.progressive_curriculum",
         "--map", a.map, "--steps-per-stage", str(a.steps),
         "--algo", a.algo, "--stages", a.stages],
        cwd=ROOT).returncode)

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
    db_p.add_argument("--runs", action="store_true")
    db_p.set_defaults(fn=cmd_db)

    cur = sub.add_parser("curriculum")
    cur.add_argument("--note", action="store_true", help="Write vault note.")
    cur.set_defaults(fn=cmd_curriculum)

    # research removed from parser (Phase 0 cut — run via python -m rl.research_agent).

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
