"""Factual entity roster for the bestiary — Doom/Freedoom actor classes.

ViZDoom reports each visible actor's real class name (`object_name`) via the objects
buffer, so monster identification is a FACT from the engine, not a guess from a vision
model. This module just classifies those names (monster / projectile / ignore) and maps a
projectile back to its caster, so the world model can be built from telemetry alone.
"""

# The Doom monster roster (Freedoom reuses the same actor classes, only the sprites
# differ). Anything here counts as an enemy in the bestiary.
MONSTERS = {
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre",
    "Cacodemon", "LostSoul", "BaronOfHell", "HellKnight", "Revenant", "Fatso",
    "Arachnotron", "PainElemental", "SpiderMastermind", "Cyberdemon", "WolfensteinSS",
}

# Projectiles → the monster that fires them. A monster whose projectile we see is RANGED;
# this is how we tell "throws fireballs from afar" from "charges you" without hardcoding.
PROJECTILE_CASTER = {
    "DoomImpBall": "DoomImp",
    "CacodemonBall": "Cacodemon",
    "BaronBall": "BaronOfHell",            # also HellKnight in vanilla
    "RevenantTracer": "Revenant",
    "FatShot": "Fatso",
    "ArachnotronPlasma": "Arachnotron",
}


# Hitscan shooters fire instant bullets (no projectile ACTOR to observe), so they'd look
# "melee" to projectile-detection — but their class IS ranged. This set restores that fact.
HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "SpiderMastermind", "WolfensteinSS"}


def is_monster(name: str) -> bool:
    return name in MONSTERS


def is_projectile(name: str) -> bool:
    return name in PROJECTILE_CASTER


def visible_enemies(labels, screen_width: float = 84.0) -> dict:
    """Summarise the enemies the agent can SEE from the ViZDoom labels buffer.

    `labels` is `state.labels` — each has `.object_name` and a screen bounding box
    (`.x`, `.width`). Ground-truth on-screen detection (unlike the map-wide objects list,
    this is only what's actually in view). Returns:
        count            — how many monsters are visible
        nearest_centered — min |bbox-center - screen-center| / (screen_width/2) in [0,1],
                           0 = an enemy is dead-centred (in the crosshair), 1 = at the edge.
                           None if no enemy is visible.

    Pure + side-effect free so it unit-tests without ViZDoom (pass simple objects/dicts).
    """
    half = screen_width / 2.0 or 1.0
    count = 0
    best_off = None
    for lab in labels or []:
        name = getattr(lab, "object_name", None)
        if name is None and isinstance(lab, dict):
            name = lab.get("object_name")
        if not name or name not in MONSTERS:
            continue
        count += 1
        x = getattr(lab, "x", None)
        w = getattr(lab, "width", None)
        if x is None and isinstance(lab, dict):
            x, w = lab.get("x"), lab.get("width")
        if x is None:
            continue
        center = float(x) + (float(w or 0) / 2.0)
        off = abs(center - half) / half          # 0 centred .. 1 at the edge
        best_off = off if best_off is None else min(best_off, off)
    return {"count": count, "nearest_centered": best_off}


# Names that aren't "objectives" to discover: the player itself and walls/decor we don't
# want to pay for. Monsters are handled by the kill/bestiary rewards, so they're excluded too.
_NON_DISCOVERY = {"DoomPlayer", "Player", "BulletPuff", "Blood"}


def visible_object_names(labels) -> set:
    """The set of notable object names currently in view (keys, switches, weapons, items,
    new monster types) — for the discovery reward. Excludes the player and pure decor.
    Pure + side-effect free so it unit-tests without ViZDoom."""
    names = set()
    for lab in labels or []:
        name = getattr(lab, "object_name", None)
        if name is None and isinstance(lab, dict):
            name = lab.get("object_name")
        if name and name not in _NON_DISCOVERY:
            names.add(name)
    return names
