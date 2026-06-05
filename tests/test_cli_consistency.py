"""Guard: every doom-cli command registered in the parser must have a help card (and vice
versa), so the CLI help never drifts out of sync with what the CLI actually accepts."""
import re
from pathlib import Path

import doom_cli


def _parser_commands() -> set:
    src = Path(doom_cli.__file__).read_text(encoding="utf-8")
    return set(re.findall(r'add_parser\(\s*"([a-z_0-9]+)"', src))


def test_every_command_has_a_help_card():
    parser_cmds = _parser_commands()
    card_cmds = {c[1] for c in doom_cli.COMMANDS}
    missing_card = parser_cmds - card_cmds
    assert not missing_card, f"commands without a help card: {sorted(missing_card)}"


def test_no_help_card_points_to_a_missing_command():
    parser_cmds = _parser_commands()
    card_cmds = {c[1] for c in doom_cli.COMMANDS}
    orphan_cards = card_cmds - parser_cmds
    assert not orphan_cards, f"help cards with no command: {sorted(orphan_cards)}"


def test_every_card_group_renders_in_the_menu():
    # A card whose group isn't in GROUP_ORDER is invisible in the help menu (silent bug).
    bad = {c[1]: c[0] for c in doom_cli.COMMANDS if c[0] not in doom_cli.GROUP_ORDER}
    assert not bad, f"commands in non-rendering groups: {bad}"


def test_help_cards_are_well_formed():
    # Each card is (group, name, short, long, example) and the example must invoke the command.
    for card in doom_cli.COMMANDS:
        assert len(card) == 5, f"malformed card: {card}"
        group, name, short, long, example = card
        assert name in example, f"{name}: example '{example}' doesn't run the command"


# --------------------------- interactive shell dispatch ---------------------------
def test_resolve_slash_builtins():
    known = {c[1] for c in doom_cli.COMMANDS}
    assert doom_cli.resolve_slash("/help", known) == ("builtin", "help")
    assert doom_cli.resolve_slash("/exit", known) == ("builtin", "exit")
    assert doom_cli.resolve_slash("/q", known) == ("builtin", "exit")
    assert doom_cli.resolve_slash("/clear", known) == ("builtin", "clear")


def test_resolve_slash_real_command():
    known = {c[1] for c in doom_cli.COMMANDS}
    assert doom_cli.resolve_slash("/benchmark", known) == ("command", "benchmark")
    assert doom_cli.resolve_slash("watch", known) == ("command", "watch")  # leading / optional


def test_resolve_slash_suggests_on_typo():
    known = {c[1] for c in doom_cli.COMMANDS}
    kind, payload = doom_cli.resolve_slash("/benchmrk", known)
    assert kind == "suggest" and "benchmark" in payload
    kind, payload = doom_cli.resolve_slash("/zzzzz", known)
    assert kind == "suggest" and payload == []


def test_resolve_slash_unique_prefix_runs_command():
    known = {c[1] for c in doom_cli.COMMANDS}
    assert doom_cli.resolve_slash("/bench", known) == ("command", "benchmark")
    assert doom_cli.resolve_slash("/know", known) == ("command", "knowledge")


def test_resolve_slash_ambiguous_prefix_suggests():
    known = {"auto", "audit", "watch"}
    kind, payload = doom_cli.resolve_slash("/au", known)
    assert kind == "suggest" and set(payload) == {"auto", "audit"}


def test_resolve_slash_bare_slash_opens_palette():
    known = {c[1] for c in doom_cli.COMMANDS}
    assert doom_cli.resolve_slash("/", known) == ("builtin", "palette")
    assert doom_cli.resolve_slash("/commands", known) == ("builtin", "palette")
