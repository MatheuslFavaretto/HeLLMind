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


# Object categories for the on-screen detector overlay (squares around everything the agent
# sees), colour-coded by what they ARE. Matched by substring so Freedoom variants still hit.
_CATEGORY_KEYWORDS = [
    ("weapon", ("Shotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "BFG",
                "Chainsaw", "SuperShotgun")),
    ("health", ("Medikit", "Stimpack", "HealthBonus", "Soulsphere", "Megasphere", "Berserk")),
    ("armor",  ("Armor", "GreenArmor", "BlueArmor", "MegaArmor")),
    ("ammo",   ("Clip", "Shell", "RocketAmmo", "RocketBox", "Cell", "Ammo", "Cartridge")),
    ("key",    ("Card", "Skull", "Key")),
    ("powerup",("Invulnerability", "Invisibility", "RadSuit", "Allmap", "Infrared",
                "Backpack")),
]


# Category → uint8 code painted into the SEMANTIC obs channel (the network's "what is where").
# Spread across 0..255 so the CNN can separate them; 0 = empty. Doors get their own code.
SEMANTIC_CODES = {
    "enemy": 255, "weapon": 210, "health": 180, "armor": 150, "ammo": 120,
    "key": 95, "powerup": 70, "item": 45, "door": 30, "projectile": 0, "self": 0,
}


def semantic_code(category: str) -> int:
    """uint8 value for a category in the semantic channel (0 if it shouldn't be painted)."""
    return SEMANTIC_CODES.get(category, 0)


def screen_x_of(px: float, py: float, angle_deg: float, tx: float, ty: float,
                fov_deg: float = 90.0):
    """Project a world point (tx,ty) into normalised screen-x [0,1] for an agent at (px,py)
    facing angle_deg (Doom convention: 0=east, CCW). Returns None if the point is behind the
    agent or outside the horizontal FOV. Used to paint DOORS (which aren't actors) into the
    semantic channel. Left of view → 0.0, right → 1.0."""
    import math
    bearing = math.degrees(math.atan2(ty - py, tx - px))
    rel = (bearing - angle_deg + 180.0) % 360.0 - 180.0   # [-180,180], +=CCW=left
    if abs(rel) > fov_deg / 2.0:
        return None
    return 0.5 - rel / fov_deg                            # CCW(left) → smaller x


def classify_object(name: str) -> str:
    """Category of a visible object for the detector overlay: enemy / weapon / health /
    armor / ammo / key / powerup / self / item. Drives the box colour + label."""
    if not name:
        return "item"
    if name in MONSTERS:
        return "enemy"
    if name == "DoomPlayer":
        return "self"
    # Narrow projectile check FIRST among in-flight shots (but BEFORE keyword match so a
    # fireball isn't miscategorised) — must not catch weapons like "Shotgun" (has "Shot").
    if name in PROJECTILE_CASTER or name.endswith("Ball") or "Tracer" in name:
        return "projectile"
    # Pickups by keyword. Weapons checked here, so "Shotgun" → weapon (not projectile).
    for cat, keys in _CATEGORY_KEYWORDS:
        if any(k in name for k in keys):
            return cat
    return "item"


def visible_objects(labels, screen_w: float, screen_h: float):
    """Every visible labelled object as a dict with NORMALISED bbox [0,1] + category, so the
    overlay can scale it to any render size. Skips the agent's own body (DoomPlayer self)."""
    out = []
    for lab in (labels or []):
        name = getattr(lab, "object_name", None) or (
            lab.get("object_name") if isinstance(lab, dict) else None)
        if not name:
            continue
        cat = classify_object(name)
        if cat == "self":
            continue
        gx = getattr(lab, "x", None);  gx = lab.get("x", 0) if gx is None else gx
        gy = getattr(lab, "y", None);  gy = lab.get("y", 0) if gy is None else gy
        gw = getattr(lab, "width", None);  gw = lab.get("width", 0) if gw is None else gw
        gh = getattr(lab, "height", None); gh = lab.get("height", 0) if gh is None else gh
        out.append({
            "name": name, "category": cat,
            "x": float(gx) / screen_w, "y": float(gy) / screen_h,
            "w": float(gw) / screen_w, "h": float(gh) / screen_h,
        })
    return out


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
