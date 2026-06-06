"""Tests for doom.entities.visible_enemies — ground-truth on-screen enemy detection."""
from doom.entities import MONSTERS, classify_object, visible_enemies, visible_objects


class _Label:
    """Mimics a ViZDoom Label object (object_name + screen bbox)."""
    def __init__(self, object_name, x=0.0, width=0.0, y=0.0, height=0.0):
        self.object_name = object_name
        self.x = x
        self.width = width
        self.y = y
        self.height = height


def test_classify_object_categories():
    # The detector overlay colours boxes by category — these must be right.
    assert classify_object("DoomImp") == "enemy"
    assert classify_object("ShotgunGuy") == "enemy"        # a monster, not a weapon
    assert classify_object("Shotgun") == "weapon"          # NOT projectile (has "Shot")
    assert classify_object("Chaingun") == "weapon"
    assert classify_object("Medikit") == "health"
    assert classify_object("Clip") == "ammo"
    assert classify_object("RedCard") == "key"
    assert classify_object("GreenArmor") == "armor"
    assert classify_object("DoomImpBall") == "projectile"
    assert classify_object("DoomPlayer") == "self"
    assert classify_object("") == "item"


def test_visible_objects_normalises_bbox_and_skips_self():
    labels = [
        _Label("DoomImp", x=80, width=40, y=60, height=80),
        _Label("Shotgun", x=160, width=20, y=100, height=20),
        _Label("DoomPlayer", x=0, width=10, y=0, height=10),   # self → skipped
    ]
    objs = visible_objects(labels, screen_w=320.0, screen_h=200.0)
    assert len(objs) == 2                                   # self dropped
    imp = next(o for o in objs if o["name"] == "DoomImp")
    assert imp["category"] == "enemy"
    assert imp["x"] == 80 / 320 and imp["w"] == 40 / 320    # normalised [0,1]
    assert {o["category"] for o in objs} == {"enemy", "weapon"}


def _a_monster():
    return next(iter(MONSTERS))


def test_no_labels_is_empty():
    out = visible_enemies(None)
    assert out["count"] == 0
    assert out["nearest_centered"] is None


def test_counts_only_monsters():
    labels = [_Label(_a_monster(), 40, 4), _Label("Medikit", 10, 4),
              _Label("Shotgun", 20, 4)]
    out = visible_enemies(labels, screen_width=84.0)
    assert out["count"] == 1  # only the monster counts


def test_two_monsters_counted():
    m = _a_monster()
    out = visible_enemies([_Label(m, 10, 4), _Label(m, 70, 4)], screen_width=84.0)
    assert out["count"] == 2


def test_centred_enemy_offset_near_zero():
    m = _a_monster()
    # bbox centred on the screen middle (84/2 = 42): x=40,width=4 -> center=42
    out = visible_enemies([_Label(m, 40, 4)], screen_width=84.0)
    assert out["nearest_centered"] == 0.0


def test_edge_enemy_offset_near_one():
    m = _a_monster()
    out = visible_enemies([_Label(m, 0, 0)], screen_width=84.0)  # center=0 -> far left
    assert out["nearest_centered"] == 1.0


def test_nearest_is_the_most_centred():
    m = _a_monster()
    out = visible_enemies([_Label(m, 0, 0), _Label(m, 41, 2)], screen_width=84.0)
    assert out["nearest_centered"] < 0.1  # the centred one wins


def test_accepts_dict_labels():
    m = _a_monster()
    out = visible_enemies([{"object_name": m, "x": 40, "width": 4}], screen_width=84.0)
    assert out["count"] == 1 and out["nearest_centered"] == 0.0


# --------------------------- discovery (visible_object_names) ---------------------------
def test_visible_object_names_collects_items():
    from doom.entities import visible_object_names
    labs = [{"object_name": "RedCard"}, {"object_name": "Shotgun"},
            {"object_name": "DoomImp"}]
    assert visible_object_names(labs) == {"RedCard", "Shotgun", "DoomImp"}


def test_visible_object_names_excludes_player_and_decor():
    from doom.entities import visible_object_names
    labs = [{"object_name": "DoomPlayer"}, {"object_name": "BulletPuff"},
            {"object_name": "Blood"}, {"object_name": "BlueSkull"}]
    assert visible_object_names(labs) == {"BlueSkull"}


def test_visible_object_names_empty():
    from doom.entities import visible_object_names
    assert visible_object_names(None) == set()
    assert visible_object_names([]) == set()


def test_steer_toward_turns_to_target():
    from doom.campaign import steer_toward
    N, FWD, TL, TR = 10, 0, 2, 3
    # facing east (0deg); target north -> turn left (ccw); target south -> turn right
    assert steer_toward([0]*N, 0, 0, 0, 0, 100, FWD, TL, TR)[TL] == 1
    assert steer_toward([0]*N, 0, 0, 0, 0, -100, FWD, TL, TR)[TR] == 1
    # aligned east -> go forward, no turns
    al = steer_toward([0]*N, 0, 0, 0, 100, 0, FWD, TL, TR)
    assert al[FWD] == 1 and al[TL] == 0 and al[TR] == 0


def test_map_doors_finds_real_doors():
    import os
    from doom.wad_doors import map_doors
    from doom.campaign import default_wad
    wad = default_wad()
    if not os.path.exists(wad):
        import pytest
        pytest.skip("freedoom2.wad not bundled")
    doors = map_doors(wad, "MAP01")
    assert 5 <= len(doors) <= 40            # the real handful, not the 103 geometry false-positives
    assert map_doors(wad, "NOPE") == ()     # missing map → graceful empty


def test_vision_steer_goes_to_open_space():
    from doom.campaign import vision_steer
    N, FWD, TL, TR = 10, 0, 2, 3
    fwd = vision_steer([0]*N, 0.2, 0.9, 0.2, FWD, TL, TR)   # centre open
    assert fwd[FWD] == 1 and fwd[TL] == 0 and fwd[TR] == 0
    left = vision_steer([0]*N, 0.9, 0.1, 0.1, FWD, TL, TR)  # left open, centre walled
    assert left[TL] == 1 and left[TR] == 0
    right = vision_steer([0]*N, 0.1, 0.1, 0.9, FWD, TL, TR)
    assert right[TR] == 1 and right[TL] == 0


def test_semantic_code_distinct_per_category():
    from doom.entities import semantic_code
    codes = {c: semantic_code(c) for c in
             ("enemy", "weapon", "health", "ammo", "key", "powerup", "item", "door")}
    assert codes["enemy"] == 255 and codes["door"] == 30
    assert len(set(codes.values())) == len(codes)        # every category is distinguishable
    assert semantic_code("projectile") == 0 and semantic_code("self") == 0  # not painted


def test_screen_x_projection():
    from doom.entities import screen_x_of
    # agent at origin facing east (0 deg), FOV 90 -> half-FOV 45
    assert abs(screen_x_of(0, 0, 0, 100, 0) - 0.5) < 1e-6      # dead ahead -> centre
    assert screen_x_of(0, 0, 0, 0, 100) is None               # 90 deg left -> outside FOV
    assert screen_x_of(0, 0, 0, -100, 0) is None              # behind
    left = screen_x_of(0, 0, 0, 100, 100)                     # 45 deg left (CCW) -> left edge
    assert left is not None and left < 0.1
