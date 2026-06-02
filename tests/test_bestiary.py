"""Factual bestiary: entity roster, persistent store, callback folding, note, chart."""
import os

import cv2

from doom.entities import is_monster, is_projectile, PROJECTILE_CASTER
from rl.enemy_callback import EnemyMemoryCallback
from writer.bestiary import BestiaryStore, display_name, write_bestiary
from writer.bestiary_chart import render_bestiary_chart


def test_entity_classification():
    assert is_monster("DoomImp") and is_monster("Demon")
    assert not is_monster("Medikit") and not is_monster("DoomPlayer")
    assert is_projectile("DoomImpBall")
    assert PROJECTILE_CASTER["DoomImpBall"] == "DoomImp"


def test_display_name():
    assert display_name("DoomImp") == "Imp"
    assert display_name("ShotgunGuy") == "Shotgun Guy"


def test_store_merges_across_runs(tmp_path):
    store = BestiaryStore(str(tmp_path))
    store.merge({"DoomImp": {"encounters": 2, "total": 21, "killed": 3, "killed_agent": 1,
                             "ranged": True, "dist_min": 120.0, "kill_weapon": {"3": 3},
                             "outcomes": {"death": 1}, "maps": {"MAP02": 2}}})
    s = store.merge({"DoomImp": {"encounters": 1, "total": 21, "killed": 2, "killed_agent": 0,
                                 "ranged": False, "dist_min": 80.0, "kill_weapon": {"3": 1, "2": 1},
                                 "outcomes": {"timeout": 1}, "maps": {"MAP02": 1}}})
    imp = s["DoomImp"]
    assert imp["encounters"] == 3 and imp["killed"] == 5 and imp["killed_agent"] == 1
    assert imp["total"] == 21                      # max, not sum (spawn count)
    assert imp["ranged"] is True                   # sticky once seen ranged
    assert imp["dist_min"] == 80.0                 # closest across runs
    assert imp["kill_weapon"]["3"] == 4 and imp["kill_weapon"]["2"] == 1


def _ep_info(enemies, terminal="death", map_name="MAP02"):
    return {"map": map_name, "episode": {"r": 1.0, "l": 100},
            "doom": {"terminal": terminal, "map": map_name, "enemies": enemies}}


def test_callback_folds_episodes_and_persists(tmp_path):
    store = BestiaryStore(str(tmp_path))
    cb = EnemyMemoryCallback(store)
    cb.locals = {"infos": [_ep_info({"Demon": {"total": 10, "killed": 2, "killed_agent": 1,
                                               "seen": 20, "approach": 18, "dist_min": 40.0,
                                               "kill_weapon": {"3": 2}}}, terminal="death")]}
    cb._on_step()
    cb.locals = {"infos": [_ep_info({"Demon": {"total": 10, "killed": 1, "killed_agent": 0,
                                               "seen": 10, "approach": 9, "dist_min": 55.0,
                                               "kill_weapon": {"3": 1}}}, terminal="exit")]}
    cb._on_step()
    assert store.load() == {}            # nothing on disk until training ends
    cb.on_training_end()
    s = store.load()["Demon"]
    assert s["encounters"] == 2 and s["killed"] == 3 and s["killed_agent"] == 1
    assert s["outcomes"]["death"] == 1 and s["outcomes"]["exit"] == 1
    assert s["maps"]["MAP02"] == 2


def _demon_imp_store(memory_dir):
    store = BestiaryStore(memory_dir)
    store.merge({
        "Demon": {"encounters": 30, "total": 10, "killed": 25, "killed_agent": 12,
                  "seen": 300, "approach": 270, "ranged": False, "dist_min": 30.0,
                  "kill_weapon": {"3": 25}, "outcomes": {"death": 12}, "maps": {"MAP02": 30}},
        "DoomImp": {"encounters": 8, "total": 21, "killed": 4, "killed_agent": 1,
                    "seen": 100, "approach": 20, "ranged": True, "dist_min": 200.0,
                    "kill_weapon": {"2": 4}, "outcomes": {"death": 1}, "maps": {"MAP02": 8}},
    })
    return store


def test_write_bestiary_note(tmp_path):
    class Cfg:
        vault_path = str(tmp_path)
        memory_dir = str(tmp_path / ".memory")
        dir_attachments = "attachments"
    _demon_imp_store(Cfg.memory_dir)
    body = open(write_bestiary(Cfg())).read()
    assert "# Bestiary" in body
    assert "Killed by the agent:** 25" in body         # exact kills
    assert "Killed the agent:** 12 time" in body        # who killed me
    assert "charges the player" in body                 # Demon melee
    assert "ranged (throws projectiles)" in body        # Imp
    assert "40%" in body                                # Demon threat 12/30


def test_hitscan_shooters_marked_ranged(tmp_path):
    class Cfg:
        vault_path = str(tmp_path)
        memory_dir = str(tmp_path / ".memory")
        dir_attachments = "attachments"
    BestiaryStore(Cfg.memory_dir).merge({
        "Zombieman": {"encounters": 10, "total": 10, "killed": 8, "killed_agent": 2,
                      "seen": 100, "approach": 20, "ranged": False, "dist_min": 50.0,
                      "kill_weapon": {"2": 8}, "outcomes": {"death": 2}, "maps": {"MAP02": 10}}})
    assert "hitscan — fires bullets" in open(write_bestiary(Cfg())).read()


def test_threat_multipliers_scale_with_who_kills_the_agent(tmp_path):
    from writer.bestiary import threat_multipliers
    store = _demon_imp_store(str(tmp_path / ".memory")).load()
    m = threat_multipliers(store)
    # Demon killed the agent 12× vs Imp's 1× -> Demon's share is far higher -> worth more.
    assert m["Demon"] > m["DoomImp"]
    assert m["Demon"] == 3.0          # share 12/13 -> 1+3·0.92 = 3.77, clamped to cap 3.0
    assert 1.0 < m["DoomImp"] < 1.5   # share 1/13 -> ~1.23


def test_threat_multipliers_ignore_low_confidence():
    from writer.bestiary import threat_multipliers
    # only 3 encounters -> too noisy -> dropped (and no killed_agent anyway) -> {}.
    assert threat_multipliers({"Demon": {"encounters": 3, "killed_agent": 3}}) == {}


def test_threat_multipliers_uniform_when_no_killer_data():
    from writer.bestiary import threat_multipliers
    # map-wide deaths but nobody attributed as killer -> no signal -> {} (all stay 1.0).
    assert threat_multipliers({"Demon": {"encounters": 50, "outcomes": {"death": 40}}}) == {}


def test_bestiary_chart_renders(tmp_path):
    store = _demon_imp_store(str(tmp_path / ".memory")).load()
    out = os.path.join(tmp_path, "bestiary.png")
    assert render_bestiary_chart(store, out) is True
    assert cv2.imread(out) is not None
    assert render_bestiary_chart({}, out) is False     # nothing to plot
