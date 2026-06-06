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
