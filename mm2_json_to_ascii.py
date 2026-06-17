#!/usr/bin/env python3
import json, os, argparse
from pathlib import Path

CAT_TERRAIN  = "terrain"
CAT_ENEMY    = "enemy"
CAT_ITEM     = "item"
CAT_PLATFORM = "platform"
CAT_DOOR     = "door"
CAT_HAZARD   = "hazard"
CAT_DECO     = "deco"
CAT_OTHER    = "other"

OBJ_META = {
    # terrain
    "Ground":              ("#", "#8B6914", CAT_TERRAIN),
    "Block":               ("B", "#C8A050", CAT_TERRAIN),
    "Hard Block":          ("H", "#888888", CAT_TERRAIN),
    "? Block":             ("?", "#F0C030", CAT_TERRAIN),
    "Hidden Block":        ("h", "#CCCCCC", CAT_TERRAIN),
    "Note Block":          ("N", "#E8A020", CAT_TERRAIN),
    "Donut Block":         ("d", "#F09050", CAT_TERRAIN),
    "Ice Block":           ("I", "#A0D8EF", CAT_TERRAIN),
    "P Block":             ("p", "#CC44CC", CAT_TERRAIN),
    "ON/OFF Block":        ("O", "#FF6600", CAT_TERRAIN),
    "Dotted-Line Block":   (".", "#AAAAAA", CAT_TERRAIN),
    "Blinking Block":      ("¤", "#FFAA00", CAT_TERRAIN),
    "Spike Block":         ("Ç", "#AA0000", CAT_TERRAIN),
    "Crate":               ("£", "#B87333", CAT_TERRAIN),
    "Stone":               ("H", "#888888", CAT_TERRAIN),  # decoder's actual name for Hard Block (id 6); "Stone Block" (id 75) never appears in practice
    "Goal Ground":         ("_", "#00AA00", CAT_TERRAIN),
    "Starting Brick":      ("{", "#C8A050", CAT_TERRAIN),
    "Castle Bridge":       ("·", "#885522", CAT_TERRAIN),
    "Tree":                ("¯", "#228B22", CAT_TERRAIN),
    "Slight Slope":        ("/", "#AA8833", CAT_TERRAIN),
    "Steep Slope":         ("\\","#CC9933", CAT_TERRAIN),
    # doors / warps
    "Pipe":                ("|", "#00BB00", CAT_DOOR),
    "Door":                ("D", "#4466FF", CAT_DOOR),
    "Warp Box":            ("§", "#6644FF", CAT_DOOR),
    "Key":                 ("±", "#FFD700", CAT_DOOR),
    "Checkpoint Flag":     ("f", "#00DDAA", CAT_DOOR),
    "Goal":                ("G", "#00FF44", CAT_DOOR),
    "Clear Pipe":          ("¢", "#44FFCC", CAT_DOOR),
    # enemies
    "Goomba":              ("g", "#CC6600", CAT_ENEMY),
    "Koopa":               ("K", "#44AA00", CAT_ENEMY),
    "Piranha Plant":       ("P", "#DD2200", CAT_ENEMY),
    "Piranha Flower":      ("P", "#DD2200", CAT_ENEMY),  # decoder's actual name for this object
    "Hammer Bro":          ("m", "#2244AA", CAT_ENEMY),
    "Thwomp":              ("t", "#6655AA", CAT_ENEMY),
    "Bob-omb":             ("o", "#444444", CAT_ENEMY),
    "Spiny":               ("s", "#CC2222", CAT_ENEMY),
    "Buzzy Beetle":        ("b", "#334488", CAT_ENEMY),
    "Lakitu":              ("L", "#DDAA00", CAT_ENEMY),
    "Lakitu's Cloud":      ("Â", "#CCCCAA", CAT_ENEMY),
    "Banzai Bill":         ("Z", "#333333", CAT_ENEMY),
    "Bullet Bill Blaster": ("V", "#333333", CAT_ENEMY),
    "Magikoopa":           ("y", "#8844CC", CAT_ENEMY),
    "Spike Top":           ("Å", "#AA3322", CAT_ENEMY),
    "Boo":                 ("u", "#DDDDDD", CAT_ENEMY),
    "Bowser":              ("X", "#BB3300", CAT_ENEMY),
    "Bowser Jr.":          ("x", "#CC5511", CAT_ENEMY),
    "Bowser Jr":           ("x", "#CC5511", CAT_ENEMY),  # decoder's actual name (no period)
    "Chain Chomp":         ("@", "#333333", CAT_ENEMY),
    "Cheep Cheep":         ("~", "#FF4488", CAT_ENEMY),
    "Blooper":             ("q", "#DDDDDD", CAT_ENEMY),
    "Wiggler":             ("w", "#AADD00", CAT_ENEMY),
    "Pokey":               ("Y", "#CCAA22", CAT_ENEMY),
    "Piranha Creeper":     ("¸", "#AA2200", CAT_ENEMY),
    "Porcupuffer":         ("µ", "#8866AA", CAT_ENEMY),
    "Fish Bone":           ("¿", "#AAAAAA", CAT_ENEMY),
    "Lava Bubble":         ("&", "#FF4400", CAT_ENEMY),
    "Rocky Wrench":        ("r", "#888844", CAT_ENEMY),
    "Muncher":             (",", "#00AA22", CAT_ENEMY),
    "Ant Trooper":         ("´", "#AA3300", CAT_ENEMY),
    "Monty Mole":          ("n", "#885522", CAT_ENEMY),
    "Mechakoopa":          ("R", "#666666", CAT_ENEMY),
    "Boom Boom":           ("!", "#BB4400", CAT_ENEMY),
    "Dry Bones":           ("9", "#BBBBAA", CAT_ENEMY),
    "Skipsqueak":          ("³", "#FFAA88", CAT_ENEMY),
    "Stingby":             ("É", "#DDCC00", CAT_ENEMY),
    "Angry Sun":           ("A", "#FF8800", CAT_ENEMY),
    "Charvaargh":          ("Ä", "#FF3300", CAT_ENEMY),
    "Bully":               ("Æ", "#883300", CAT_ENEMY),
    "Lemmy":               ("1", "#FF88CC", CAT_ENEMY),
    "Morton":              ("2", "#888888", CAT_ENEMY),
    "Larry":               ("3", "#44AA44", CAT_ENEMY),
    "Wendy":               ("4", "#FF44AA", CAT_ENEMY),
    "Iggy":                ("5", "#44AAFF", CAT_ENEMY),
    "Roy":                 ("6", "#AA44FF", CAT_ENEMY),
    "Ludwig":              ("7", "#4444CC", CAT_ENEMY),
    # items
    "Coin":                ("c", "#FFD700", CAT_ITEM),
    "Red Coin":            ("$", "#FF2200", CAT_ITEM),
    "1-Up Mushroom":       ("U", "#00CC00", CAT_ITEM),
    "1UP":                 ("U", "#00CC00", CAT_ITEM),  # decoder's actual name for this object
    "Fire Flower":         ("i", "#FF5500", CAT_ITEM),
    "Super Star":          ("*", "#FFFF00", CAT_ITEM),
    "Super Mushroom":      ("M", "#EE2222", CAT_ITEM),
    "Big Mushroom":        ("E", "#CC1111", CAT_ITEM),
    "SMB2 Mushroom":       ("Q", "#884488", CAT_ITEM),
    # Style Power-up A (id 44) gamestyle variants — see resolve_obj_name()
    "Super Leaf":          ("E", "#CC1111", CAT_ITEM),
    "Cape Feather":        ("E", "#CC1111", CAT_ITEM),
    "Propeller Mushroom":  ("E", "#CC1111", CAT_ITEM),
    # Style Power-up B (id 81) gamestyle variants — see resolve_obj_name()
    "Link":                ("Q", "#884488", CAT_ITEM),
    "Frog Suit":           ("Q", "#884488", CAT_ITEM),
    "Power Balloon":       ("Q", "#884488", CAT_ITEM),
    "Super Acorn":         ("Q", "#884488", CAT_ITEM),
    "Super Hammer":        ("¬", "#996622", CAT_ITEM),
    "Big Coin":            ("£", "#FFAA00", CAT_ITEM),
    "P Switch":            ("S", "#4488FF", CAT_ITEM),
    "POW Block":           ("W", "#3366FF", CAT_ITEM),
    "POW":                 ("W", "#3366FF", CAT_ITEM),  # decoder's actual name (no "Block")
    "Spring":              ("J", "#DDDD00", CAT_ITEM),
    # Style Ride (id 45) gamestyle variants — see resolve_obj_name()
    "Goomba's Shoe":       ("z", "#CC6600", CAT_ITEM),
    "Yoshi's Egg":         ("z", "#CC6600", CAT_ITEM),
    "Cannon Box":          ("È", "#666666", CAT_ITEM),
    "Propeller Box":       ("}", "#8888FF", CAT_ITEM),
    "Goomba Mask":         ("Ê", "#CC6600", CAT_ITEM),
    "Bullet Bill Mask":    ("°", "#333333", CAT_ITEM),
    "Red POW Box":         ("²", "#FF3333", CAT_ITEM),
    # platforms
    "Lift":                ("-", "#DDAA55", CAT_PLATFORM),
    "Mushroom Platform":   ("T", "#FF6688", CAT_PLATFORM),
    "Semisolid Platform":  ("k", "#AAAAFF", CAT_PLATFORM),
    "Bridge":              ("=", "#AA8833", CAT_PLATFORM),
    "Lava Lift":           ("F", "#FF4400", CAT_PLATFORM),
    "Snake Block":         ("¹", "#44CC44", CAT_PLATFORM),
    "Track Block":         ("º", "#AA6622", CAT_PLATFORM),
    "Conveyor Belt":       ("»", "#888888", CAT_PLATFORM),
    "Fast Conveyor Belt":  ("¼", "#555555", CAT_PLATFORM),
    "Sprint Platform":     ("½", "#FF8800", CAT_PLATFORM),
    "Seesaw":              ("¾", "#AA8844", CAT_PLATFORM),
    "Swinging Claw":       ("j", "#AAAAAA", CAT_PLATFORM),
    "ON/OFF Trampoline":   ("À", "#FF6600", CAT_PLATFORM),
    "Mushroom Trampoline": ("Á", "#FF4488", CAT_PLATFORM),
    "Jumping Machine":     ("¦", "#8844FF", CAT_PLATFORM),
    "Half-Collision Platform": ("a", "#CCCCAA", CAT_PLATFORM),
    "Donut":               ("Ã", "#F09050", CAT_PLATFORM),
    # hazards
    "Fire Bar":            ("e", "#FF4400", CAT_HAZARD),
    "Saw":                 ("%", "#AAAAAA", CAT_HAZARD),
    "Burner":              ("l", "#FF6600", CAT_HAZARD),
    "Spikes":              ("^", "#888888", CAT_HAZARD),
    "Spike Ball":          ("v", "#884444", CAT_HAZARD),
    "Skewer":              ("0", "#666666", CAT_HAZARD),
    "Twister":             ("8", "#AADDFF", CAT_HAZARD),
    "Icicle":              ("Ë", "#AADDFF", CAT_HAZARD),
    # deco
    "Cloud":               ("<", "#CCCCFF", CAT_TERRAIN),
    "Vine":                ("[", "#00BB00", CAT_DECO),
    "Water Marker":        ("Î", "#0055FF", CAT_DECO),
    "Arrow":               ("Ï", "#FFFF00", CAT_DECO),
    "One-Way Wall":        ("]", "#FFFF88", CAT_DECO),
    "One-Way":             ("]", "#FFFF88", CAT_DECO),  # decoder's actual name (no "Wall")
    "Reel Camera":         ("Ñ", "#AAAAAA", CAT_DECO),
    "Sound Effect":        ("Ò", "#FFAAFF", CAT_DECO),
    # other
    "Player":              ("Ó", "#0000FF", CAT_OTHER),
    "Clown Car":           (";", "#FF4488", CAT_OTHER),
    "Koopa Clown Car":     ("Õ", "#44AA00", CAT_OTHER),
    "Track":               ("Ö", "#AAAAAA", CAT_OTHER),
    "Starting Arrow":      ("×", "#FFFF00", CAT_OTHER),
    "Cannon":              (")", "#444444", CAT_OTHER),
    "! Block":             ("Ù", "#FFAA00", CAT_OTHER),
    "_unknown":            ("?", "#FF00FF", CAT_OTHER),
}

GROUND_COLOR = "#8B6914"
GROUND_CHAR  = "#"

# ASCII export replacements: MM2 objects that convert to a different SWE object.
# Mirrors the approximations in json_to_swe.py OBJ_SWE_IDS.
ASCII_REPLACEMENTS = {
    "Spike Top":           "s",  # → Spiny
    "Fish Bone":           "~",  # → Cheep Cheep
    "Lakitu's Cloud":      ";",  # → Clown Car
    "Jumping Machine":     "J",  # → Spring
    "Mushroom Trampoline": "J",  # → Spring
    "ON/OFF Trampoline":   "J",  # → Spring
}

CAT_COLORS = {
    CAT_TERRAIN:  "#C8A050",
    CAT_ENEMY:    "#CC4444",
    CAT_ITEM:     "#FFD700",
    CAT_PLATFORM: "#5599FF",
    CAT_DOOR:     "#44AAFF",
    CAT_HAZARD:   "#FF6600",
    CAT_DECO:     "#88BB88",
    CAT_OTHER:    "#AAAAAA",
}

# Style Power-ups: objects.id 44 ("Big Mushroom") and 81 ("SMB2 Mushroom")
# are decoded with fixed SMB1 names, but the power-up actually granted (and
# its sprite) depends on the level's gamestyle_raw. See
# mm2_json_field_dictionary.txt §5 for the full mapping.
STYLE_POWERUP_NAMES = {
    "Big Mushroom": {     # Slot A (id 44, since v1.0.0)
        12621: "Big Mushroom",       # SMB1   -> Mega Mario
        13133: "Super Leaf",         # SMB3   -> Raccoon Mario
        22349: "Cape Feather",        # SMW    -> Cape Mario
        21847: "Propeller Mushroom",  # NSMBU  -> Propeller Mario
    },
    "SMB2 Mushroom": {    # Slot B (id 81, since v3.0.0)
        12621: "Link",                 # SMB1   -> Link Mario (Master Sword)
        13133: "Frog Suit",           # SMB3   -> Frog Mario
        22349: "Power Balloon",        # SMW    -> Balloon Mario
        21847: "Super Acorn",          # NSMBU  -> Flying Squirrel Mario
    },
}

# Style Ride: objects.id 45 (decoded by the level parser as "Shoe Goomba")
# is a fixed SMB1/SMB3 name, but in SMW/NSMBU the same slot is a Yoshi's Egg
# that hatches into a rideable Yoshi. See mm2_json_field_dictionary.txt §6.
STYLE_RIDE_NAMES = {
    "Shoe Goomba": {    # Slot (id 45) — decoder's actual name for this object
        12621: "Goomba's Shoe",  # SMB1
        13133: "Goomba's Shoe",  # SMB3
        22349: "Yoshi's Egg",    # SMW
        21847: "Yoshi's Egg",    # NSMBU
    },
}


def resolve_obj_name(obj_name: str, gamestyle_raw: int) -> str:
    """Map a decoded object name to its gamestyle-correct name for
    Style Power-up slots (id 44 / 81) and the Style Ride slot (id 45);
    passes through unchanged otherwise."""
    for table in (STYLE_POWERUP_NAMES, STYLE_RIDE_NAMES):
        variants = table.get(obj_name)
        if variants:
            return variants.get(gamestyle_raw, obj_name)
    return obj_name


def get_meta(name: str):
    return OBJ_META.get(name, OBJ_META["_unknown"])


# ---------------------------------------------------------------------------
# Pipe direction helpers (flag % 0x80: 0x00=R, 0x20=L, 0x40=U, 0x60=D)
# ---------------------------------------------------------------------------
def _pipe_direction(flag: int) -> str:
    d = flag % 0x80
    if d == 0x00: return 'R'
    if d == 0x20: return 'L'
    if d == 0x40: return 'U'
    return 'D'

_PIPE_DIR_CHAR = {'R': '→', 'L': '←', 'U': '↑', 'D': '↓'}


# ---------------------------------------------------------------------------
# Tile size helper — uses w/h from JSON directly (already tile counts)
# ---------------------------------------------------------------------------
def obj_tile_size(obj: dict):
    """Return (w_tiles, h_tiles). The JSON w/h fields are direct tile counts.

    Pipes use h as the pipe length (C++ objH) regardless of direction;
    the cross-section is always 2 tiles wide/tall.
    """
    if obj.get("name") == "Pipe":
        direction = _pipe_direction(obj.get("flag", 0))
        length = max(1, obj.get("h", 1))
        if direction in ('U', 'D'):
            return 2, length
        else:
            return length, 2
    w = max(1, obj.get("w", 1))
    h = max(1, obj.get("h", 1))
    return w, h


# Objects whose x coordinate is the left-tile center (x = col*160 + 80).
# The C++ drawer uses the per-tile formula  j - 0.5 + x/160  for these,
# so  col = x // 160  is already correct — no w//2 correction.
# Everything else (Thwomp, Skewer, Lift, Saw, Arrow, Donut, …) uses the
# center-of-span formula  -w/2 + x/160  →  col = x//160 - w//2.
_LEFT_ANCHOR = frozenset({
    "Pipe",
    "Bridge",
    "Conveyor Belt",
    "Fast Conveyor Belt",
    "Mushroom Platform",
    "Semisolid Platform",
    "Slight Slope",
    "Steep Slope",
    "Half-Collision Platform",
    # synthetic objects injected by _normalize_level use tile coords directly
    "Ground",
    "Starting Brick",
    "Goal",
})


def obj_anchor(obj: dict):
    """Return (col, row) — bottom-left tile of the object.

    Left-anchor objects store x as the left-tile center (x = col*160 + 80)
    and are drawn per-tile with  j - 0.5 + x/160  in the C++ renderer.
    All other objects store x as the center of their full bounding span and
    are drawn with  -w/2 + x/160  — equivalent to  x//160 - w//2  for both
    even-width (x%160==0) and odd-width (x%160==80) cases.
    y is always the bottom-tile center for all JSON objects, so
    row = y // 160 is always correct.

    Pipes require direction-specific anchor adjustment derived from the C++
    rendering offsets for each direction case.
    """
    if obj.get("name") == "Pipe":
        direction = _pipe_direction(obj.get("flag", 0))
        base_col = obj["x"] // 160
        base_row = obj["y"] // 160
        w, h = obj_tile_size(obj)
        if direction == 'U':
            # columns [col, col+1], rows [base_row, base_row+h-1]
            return base_col, base_row
        elif direction == 'D':
            # x offset -1 tile; pipe extends downward (decreasing row)
            return base_col - 1, base_row - h + 1
        elif direction == 'R':
            # columns [col, col+w-1], rows [base_row, base_row]
            return base_col, base_row - 1
        else:  # L
            # pipe extends left; y stays the same
            return base_col - w + 1, base_row

    w, h = obj_tile_size(obj)
    x = obj["x"]
    if obj.get("name", "") in _LEFT_ANCHOR:
        col = x // 160
    else:
        col = x // 160 - w // 2
    row = obj["y"] // 160
    return col, row


# ---------------------------------------------------------------------------
# Main viewer window
# ---------------------------------------------------------------------------
_SLOPE_NAMES = frozenset({"Slight Slope", "Steep Slope"})


# ---------------------------------------------------------------------------


def normalize_level(lvl):
    if lvl.get("_normalized"):
        return
    lvl["_normalized"] = True

    objects = lvl.get("objects", [])

    for g in lvl.get("ground", []):
        objects.append({
            "name":"Ground","x":g["x"]*160,"y":g["y"]*160,"w":1,"h":1
        })

    if "is_overworld" in lvl:
        is_overworld = bool(lvl["is_overworld"])
    else:
        src_name = os.path.basename(lvl.get("_source_file","")).lower()
        is_overworld = "_subworld" not in src_name

    if not is_overworld:
        lvl["objects"] = objects
        return

    start_y = lvl.get("start_y",0)

    for col in range(0,7):
        for row in range(0,start_y):
            objects.append({
                "name":"Ground",
                "x":col*160,
                "y":row*160,
                "w":1,
                "h":1
            })

    goal_x = lvl.get("goal_x",0)
    goal_y = lvl.get("goal_y",0)
    goal_col = goal_x // 10

    is_castle = (lvl.get("theme_raw",-1)==2 or lvl.get("theme","")=="Castle")
    is_3dw = (lvl.get("gamestyle","")=="SM3DW" or lvl.get("gamestyle_raw",0)==22323)

    if is_castle and not is_3dw:
        objects.append({
            "name":"Goal","x":goal_col*160,"y":goal_y*160,"w":2,"h":4
        })
    else:
        objects.append({
            "name":"Goal","x":goal_col*160,"y":goal_y*160,"w":1,"h":11
        })

    for col in range(goal_col, goal_col+13):
        for row in range(0, goal_y):
            objects.append({
                "name":"Ground",
                "x":col*160,
                "y":row*160,
                "w":1,
                "h":1
            })

    lvl["objects"] = objects


def grid_bounds(lvl):
    top_b = lvl.get("top_boundary",0)
    right_b = lvl.get("right_boundary",0)
    if top_b > 0 and right_b > 0:
        max_tx = right_b // 16
        max_ty = top_b // 16
    else:
        max_tx,max_ty = 40,20
        for o in lvl.get("objects",[]):
            col,row = obj_anchor(o)
            w,h = obj_tile_size(o)
            max_tx = max(max_tx, col+w+1)
            max_ty = max(max_ty, row+h+1)
    return min(max_tx,240), min(max_ty,28)


def build_ascii_grid(level):
    objects = level.get("objects", [])
    max_tx,max_ty = grid_bounds(level)

    grid = [[" "] * max_tx for _ in range(max_ty)]

    def set_cell(col,row_game,ch):
        if 0 <= col < max_tx and 0 <= row_game < max_ty:
            grid[max_ty - 1 - row_game][col] = ch

    BG_TYPES = {"Semisolid Platform","Mushroom Platform"}

    for pass_n in range(2):
        for obj in objects:
            obj_name = obj.get("name","_unknown")
            is_bg = obj_name in BG_TYPES
            if pass_n == 0 and not is_bg:
                continue
            if pass_n == 1 and is_bg:
                continue

            char,_,_ = get_meta(resolve_obj_name(obj_name, level.get("gamestyle_raw", 0)))

            if obj_name in ASCII_REPLACEMENTS:
                char = ASCII_REPLACEMENTS[obj_name]

            if obj_name in _SLOPE_NAMES:
                # Slopes have no flat-ASCII diagonal equivalent, so fill their
                # footprint as a solid ascending/descending staircase of
                # ground -- mirrors slope_fill_cells() in json_to_swe.py.
                col, row = obj_anchor(obj)
                w, h = obj_tile_size(obj)
                step = 2 if obj.get("id") == 87 else 1
                descending = (obj.get("flag", 0) & 0x100000) != 0
                for x in range(w):
                    run = (w - x) if descending else (x + 1)
                    height = min((run + step - 1) // step, h)
                    for y in range(height):
                        set_cell(col + x, row + y, GROUND_CHAR)
            elif obj_name == "Mushroom Platform":
                col,row = obj_anchor(obj)
                w,h = obj_tile_size(obj)
                sc = col + w // 2
                # stem: centered column, all rows below cap
                for dy in range(h - 1):
                    set_cell(sc, row + dy, char)
                # cap: full width at top row
                for dx in range(w):
                    set_cell(col + dx, row + h - 1, char)
            elif obj_name == "Bridge":
                col,row = obj_anchor(obj)
                w,_ = obj_tile_size(obj)
                # Only the bottom (walkable) row is kept; the rope/chain
                # row above it is decorative and gets dropped.
                for dx in range(w):
                    set_cell(col+dx,row,char)
            elif obj_name == "Big Coin":
                # Big Coin is being phased out; represent it as a 2x2 cluster
                # of regular coins instead of its own glyph.
                col,row = obj_anchor(obj)
                coin_char,_,_ = get_meta("Coin")
                for dx in range(2):
                    for dy in range(2):
                        set_cell(col+dx,row+dy,coin_char)
            else:
                col,row = obj_anchor(obj)
                w,h = obj_tile_size(obj)

                for dx in range(w):
                    for dy in range(h):
                        set_cell(col+dx,row+dy,char)

    return ["".join(r).rstrip() for r in grid]


def convert_file(infile, outdir):
    data = json.loads(Path(infile).read_text(encoding="utf-8"))
    levels = data if isinstance(data, list) else [data]

    for lvl in levels:
        lvl.setdefault("_source_file", str(infile))
        normalize_level(lvl)

    for idx,lvl in enumerate(levels, start=1):
        stem = Path(infile).stem
        suffix = f"_{idx}" if len(levels) > 1 else ""
        outfile = Path(outdir) / f"{stem}{suffix}.txt"
        outfile.write_text("\n".join(build_ascii_grid(lvl)) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_folder")
    ap.add_argument("output_folder")
    args = ap.parse_args()

    outdir = Path(args.output_folder)
    outdir.mkdir(parents=True, exist_ok=True)

    for jf in sorted(Path(args.input_folder).glob("*.json")):
        try:
            convert_file(jf, outdir)
            print(f"Converted {jf.name}")
        except Exception as e:
            print(f"Failed {jf.name}: {e}")

if __name__ == "__main__":
    main()
