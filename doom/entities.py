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
