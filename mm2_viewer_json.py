"""
MM2 Level Viewer (simplified)
==============================
Reads .json level files exported in the TheGreatRambler/mm2_level format.

Usage
-----
    python mm2_viewer.py                  # open GUI, use Load JSON button
    python mm2_viewer.py my_level.json    # auto-load on startup

Coordinate system
-----------------
    Objects: x/y in sub-pixels where 160 sub-pixels = 1 tile.
    Tile col = x // 160,  tile row = y // 160
    Display: Y is flipped so row 0 appears at the BOTTOM of the canvas.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, sys, os, math, re

try:
    from PIL import Image, ImageTk
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ---------------------------------------------------------------------------
# Object metadata: name → (char, color, category)
# ---------------------------------------------------------------------------
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
    "Spikes":              ("^", "#888888", CAT_HAZARD),
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
    "Big Coin":            ("£", "#FFAA00", CAT_ITEM),
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
    "Spike Block":         ("Ç", "#AA0000", CAT_TERRAIN),
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

# Cell that toost's render shows as occupied but no object's heuristic
# bbox claims — keeps the ASCII silhouette pixel-faithful to toost.
UNKNOWN_CHAR = "▒"

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

# ---------------------------------------------------------------------------
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
    "Track",
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
# Slope tile iterator
# ---------------------------------------------------------------------------
_SLOPE_NAMES = frozenset({"Slight Slope", "Steep Slope"})

# Burner (id 54) flame direction, keyed by flag % 0x100 — derived from
# LevelDrawer::DrawFire's per-case sprite placement (OBJ_54A1/A3/A5/A7 and
# their 0x44/0x4C/0x54/0x5C "moving" variants share the same offsets). The
# flame is a 3-tile jet extending from the burner's edge in this direction.
_BURNER_FLAME_DIR = {
    0x40: 'U', 0x44: 'U',
    0x48: 'R', 0x4C: 'R',
    0x50: 'D', 0x54: 'D',
    0x58: 'L', 0x5C: 'L',
}

def slope_tiles(obj: dict):
    """
    Generate all solid terrain cells occupied by a Mario Maker slope.

    ID 87 = Slight Slope (rise 1, run 2)
    ID 88 = Steep Slope  (rise 1, run 1)

    Direction bit:
        flag & 0x100000 == 0  -> left slope
        flag & 0x100000 != 0  -> right slope

    Returns:
        (col, row)
    """

    base_col, base_row = obj_anchor(obj)
    w, h = obj_tile_size(obj)

    if w <= 0 or h <= 0:
        return

    obj_id = obj["id"]

    if obj_id == 87:
        step = 2      # gentle slope
    elif obj_id == 88:
        step = 1      # steep slope
    else:
        return

    right_slope = (obj.get("flag", 0) & 0x100000) != 0

    for row in range(h):

        if right_slope:
            x_start = row * step
            x_end = min(w, (row + 1) * step)

            # fill behind the slope edge
            fill_start = x_end
            fill_end = w

        else:
            x_start = max(0, w - (row + 1) * step)
            x_end = w - row * step

            # fill behind the slope edge
            fill_start = 0
            fill_end = x_start

        x_start = max(0, x_start)
        x_end = min(w, x_end)

        # slope cells
        for x in range(x_start, x_end):
            yield (
                base_col + x,
                base_row + (h - row - 1)
            )

    
        
    


# ---------------------------------------------------------------------------
class MM2Viewer(tk.Tk):
    TILE_PX  = 160
    MAX_COLS = 240
    MAX_ROWS = 28

    def __init__(self):
        super().__init__()
        self.title("MM2 Level Viewer")
        self.resizable(True, True)

        self.levels      = []
        self.current_idx = 0
        self.tile_size   = 16

        self.show_objects = tk.BooleanVar(value=True)
        self.show_grid    = tk.BooleanVar(value=True)
        self.show_labels  = tk.BooleanVar(value=True)
        self.ascii_mode   = tk.BooleanVar(value=False)
        self.toost_mode   = tk.BooleanVar(value=True)
        self._cat_vars    = {}
        self._tooltip_win = None

        # Toost drawing_instructions replay state
        self._id_to_sprite   = {}   # {combined_id: (x, y, w, h)} from LevelData.hpp ObjectLocation
        self._spritesheet    = None # PIL Image of img/spritesheet.png
        self._tile_sheet_cache = {} # {basename: PIL Image | None}
        self._toost_cache    = {}   # {level_idx: PIL Image (native res) | None}
        self._occ_cache      = {}   # {level_idx: [[bool]] pixel-occupancy grid | None}
        self._photo_ref      = None # keep PhotoImage alive

        self._build_ui()
        self._load_level_data()

    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        tb = tk.Frame(self, bd=1, relief=tk.RAISED)
        tb.pack(fill=tk.X, side=tk.TOP, padx=2, pady=2)

        tk.Button(tb, text="Load JSON", command=self._load_json).pack(side=tk.LEFT, padx=4)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Checkbutton(tb, text="Objects", variable=self.show_objects, command=self._redraw).pack(side=tk.LEFT)
        tk.Checkbutton(tb, text="Grid",    variable=self.show_grid,    command=self._redraw).pack(side=tk.LEFT)
        tk.Checkbutton(tb, text="Labels",  variable=self.show_labels,  command=self._redraw).pack(side=tk.LEFT)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Label(tb, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_var = tk.IntVar(value=16)
        tk.Scale(tb, from_=6, to=40, orient=tk.HORIZONTAL, variable=self.zoom_var,
                 command=lambda _: self._on_zoom(), showvalue=True, length=120).pack(side=tk.LEFT)

        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Checkbutton(tb, text="Toost Render", variable=self.toost_mode,
                       command=self._redraw).pack(side=tk.LEFT, padx=4)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Checkbutton(tb, text="ASCII mode", variable=self.ascii_mode,
                       command=self._redraw).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Export ASCII", command=self._export_ascii).pack(side=tk.LEFT, padx=2)

        # category filter bar
        fb = tk.Frame(self)
        fb.pack(fill=tk.X, padx=2)
        tk.Label(fb, text="Categories:").pack(side=tk.LEFT)
        for cat, col in CAT_COLORS.items():
            v = tk.BooleanVar(value=True)
            self._cat_vars[cat] = v
            tk.Checkbutton(fb, text=cat, variable=v,
                           fg=col, activeforeground=col,
                           command=self._redraw).pack(side=tk.LEFT, padx=2)

        # canvas + scrollbars
        cf = tk.Frame(self)
        cf.pack(fill=tk.BOTH, expand=True)
        hbar = tk.Scrollbar(cf, orient=tk.HORIZONTAL)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar = tk.Scrollbar(cf, orient=tk.VERTICAL)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas = tk.Canvas(cf, bg="#5C94FC",
                                xscrollcommand=hbar.set,
                                yscrollcommand=vbar.set,
                                cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        hbar.config(command=self.canvas.xview)
        vbar.config(command=self.canvas.yview)

        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>",     self._drag_move)
        self.canvas.bind("<Motion>",        self._on_hover)
        self.canvas.bind("<Leave>",         lambda _: self._hide_tip())

        # nav bar
        nav = tk.Frame(self)
        nav.pack(fill=tk.X, padx=4, pady=2)
        tk.Button(nav, text="<< Prev", command=self._prev).pack(side=tk.LEFT)
        tk.Button(nav, text="Next >>", command=self._next).pack(side=tk.LEFT, padx=4)
        tk.Label(nav, text="Jump:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(nav, width=6)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self._jump)
        self.info_lbl = tk.Label(nav, text="No level loaded", anchor=tk.W)
        self.info_lbl.pack(side=tk.LEFT, padx=12)

        # legend
        leg = tk.Frame(self, bd=1, relief=tk.SUNKEN)
        leg.pack(fill=tk.X, side=tk.BOTTOM, padx=2, pady=2)
        tk.Label(leg, text="Legend:").pack(side=tk.LEFT)
        for cat, col in CAT_COLORS.items():
            tk.Label(leg, text=f" {cat} ", bg=col, fg="white", padx=3).pack(side=tk.LEFT, padx=2)

        self.bind("<Right>", lambda _: self._next())
        self.bind("<Left>",  lambda _: self._prev())

    # --------------------------------------------------------------- loading --

    def _normalize_level(self, lvl):
        """Injects ground, start, and goal directly into the objects list."""
        if lvl.get("_normalized"): return
        lvl["_normalized"] = True

        objects = lvl.get("objects", [])

        # 1. Explicit terrain from the ground array (always present, overworld or not)
        for g in lvl.get("ground", []):
            objects.append({
                "name": "Ground",
                "x":    g["x"] * 160,
                "y":    g["y"] * 160,
                "w":    1,
                "h":    1,
            })

        # 1b. Track rails. LevelDrawer::DrawTrack draws a 3x3-tile graphic
        # centered on (x, y) for type < 8, or a 5x5 graphic offset to
        # cols [x-1, x+3] / rows [y-1, y+3] for type >= 8 ("Y" junctions).
        # Connector tiles linking adjacent track pieces always fall inside
        # that box, so the box is enough for occ-gating to pick out the
        # real rail shape.
        for t in lvl.get("track", []):
            tx, ty = t.get("x", 0), t.get("y", 0)
            size = 3 if t.get("type", 0) < 8 else 5
            objects.append({
                "name": "Track",
                "x":    (tx - 1) * 160,
                "y":    (ty - 1) * 160,
                "w":    size,
                "h":    size,
            })

        # Determine overworld vs subworld (C++ NowIO check).
        # Prefer the JSON boolean; fall back to the source filename injected by the loader.
        if "is_overworld" in lvl:
            is_overworld = bool(lvl["is_overworld"])
        else:
            src_name = os.path.basename(lvl.get("_source_file", "")).lower()
            is_overworld = "_subworld" not in src_name

        # Subworlds have no fixed start/goal structure.
        if not is_overworld:
            lvl["objects"] = objects
            return

        start_y = lvl.get("start_y", 0)

        for col in range(0, 7):
            for row in range(0, start_y):
                objects.append({
                    "name": "Ground",
                    "x":    col * 160,
                    "y":    row * 160,
                    "w":    1,
                    "h":    1,
                })

        # 3. Goal — castle (axe + bridge) for all styles except SM3DW; flagpole otherwise.
        # C++ DrawGrd uses theme==2 (castle) for the axe, but SM3DW has no castle variant.
        goal_x = lvl.get("goal_x", 0)
        goal_y = lvl.get("goal_y", 0)
        goal_col = goal_x // 10

        is_castle = (lvl.get("theme_raw", -1) == 2 or lvl.get("theme", "") == "Castle")
        is_3dw    = (lvl.get("gamestyle", "") == "SM3DW" or lvl.get("gamestyle_raw", 0) == 22323)

        if is_castle and not is_3dw:
            # Axe: 2 wide × 4 tall
            objects.append({
                "name": "Goal",
                "x": goal_col * 160,
                "y": goal_y * 160,
                "w": 2,
                "h": 4,
            })
        else:
            # Flagpole: 1 wide × 11 tall
            objects.append({
                "name": "Goal",
                "x": goal_col * 160,
                "y": goal_y * 160,
                "w": 1,
                "h": 11,
            })

        # Ground columns extending rightward from the goal
        for col in range(goal_col, goal_col + 13):
            for row in range(0, goal_y):
                objects.append({
                    "name": "Ground",
                    "x":    col * 160,
                    "y":    row * 160,
                    "w":    1,
                    "h":    1,
                })

        lvl["objects"] = objects


    def _load_json(self):
        path = filedialog.askopenfilename(
            title="Select MM2 level JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
                
            # Normalize the level data before rendering
            for lvl in data:
                lvl.setdefault("_source_file", path)
                self._normalize_level(lvl)
                
            self.levels = data
            self.current_idx = 0
            self._toost_cache.clear()
            self._occ_cache.clear()
            self._redraw()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    # ------------------------------------------------------------ navigation --
    def _prev(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self._redraw()

    def _next(self):
        if self.current_idx < len(self.levels) - 1:
            self.current_idx += 1
            self._redraw()

    def _jump(self, _=None):
        try:
            idx = int(self.jump_entry.get()) - 1
            if 0 <= idx < len(self.levels):
                self.current_idx = idx
                self._redraw()
        except ValueError:
            pass

    def _on_zoom(self):
        self.tile_size = self.zoom_var.get()
        self._redraw()

    def _active_cats(self):
        return {cat for cat, v in self._cat_vars.items() if v.get()}

    # --------------------------------------------------------------- drawing --

    def _load_level_data(self):
        """Parse LevelData.hpp's ObjectLocation map and load img/spritesheet.png.

        ObjectLocation entries look like ``{ OBJ_27 | SMB1, { 1760, 224, 16, 176 } }``.
        Every enum constant in the file is a plain ``NAME = NUMBER`` line, so the
        whole table can be flattened and the map keys (NAME or NAME | NAME) can be
        evaluated with a simple bitwise OR. The resulting {int: (x, y, w, h)} table
        is exactly what's needed to resolve the numeric `path` ids used by
        is_tile=false drawing_instructions (see _render_toost_image).
        """
        self._id_to_sprite = {}
        self._spritesheet  = None
        if not _PIL_OK:
            return

        script_dir = os.path.dirname(os.path.abspath(__file__))
        hpp_path = next((p for p in (
            os.path.join(script_dir, "LevelData.hpp"),
            os.path.join(script_dir, "toost_stuff", "LevelData.hpp"),
            os.path.join(script_dir, "toost_stuff", "src", "LevelData.hpp"),
            os.path.join(script_dir, "..", "toost", "src", "LevelData.hpp"),
        ) if os.path.exists(p)), None)
        sheet_path = next((p for p in (
            os.path.join(script_dir, "img", "spritesheet.png"),
            os.path.join(script_dir, "toost_stuff", "img", "spritesheet.png"),
        ) if os.path.exists(p)), None)
        if not hpp_path or not sheet_path:
            return

        with open(hpp_path, "r", encoding="utf-8") as fh:
            hpp = fh.read()

        # Flatten every `NAME = NUMBER,` enum constant (Object, Gamestyle, etc.)
        const_map = {}
        for m in re.finditer(r'^\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*,?\s*$', hpp, re.M):
            const_map[m.group(1)] = int(m.group(2))

        # ObjectLocation entries: { NAME [| NAME], { x, y, w, h } }
        entry_pat = re.compile(
            r'\{\s*([A-Za-z_]\w*)(?:\s*\|\s*([A-Za-z_]\w*))?\s*,'
            r'\s*\{\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\}\s*\}'
        )
        id_to_sprite = {}
        for a, b, x, y, w, h in entry_pat.findall(hpp):
            if a not in const_map:
                continue
            key = const_map[a]
            if b:
                if b not in const_map:
                    continue
                key |= const_map[b]
            id_to_sprite[key] = (int(x), int(y), int(w), int(h))

        self._id_to_sprite = id_to_sprite
        self._spritesheet  = Image.open(sheet_path).convert("RGBA")

    def _get_tile_sheet(self, basename: str):
        """Return a cached RGBA PIL Image for img/tile/<basename>, or None."""
        if basename in self._tile_sheet_cache:
            return self._tile_sheet_cache[basename]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sheet = None
        for d in (os.path.join(script_dir, "img", "tile"),
                  os.path.join(script_dir, "toost_stuff", "img", "tile")):
            p = os.path.join(d, basename)
            if os.path.exists(p):
                sheet = Image.open(p).convert("RGBA")
                break
        self._tile_sheet_cache[basename] = sheet
        return sheet

    def _render_toost_image(self, lvl):
        """Replay drawing_instructions to reproduce toost's own PNG render.

        Each instruction is either a tile-sheet blit (is_tile=true: crop
        tile_x/tile_y/tile_w/tile_h * TileW from img/tile/<sheet>.png) or a
        spritesheet blit (is_tile=false: path is the combined OBJ|Gamestyle
        id, looked up in self._id_to_sprite -> img/spritesheet.png). Both are
        scaled to target_width/target_height and pasted at x,y — the exact
        draw script toost used, so this should match the reference PNG
        pixel-for-pixel (at native 16px/tile resolution).
        """
        dis = lvl.get("drawing_instructions")
        if not dis or not _PIL_OK:
            return None

        max_tx, max_ty = self._grid_bounds(lvl)
        W, H = max_tx * 16, max_ty * 16
        if W <= 0 or H <= 0:
            return None

        # toost's cairo surface starts fully transparent; the sky-blue look
        # comes from the canvas background (#5C94FC) showing through.
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        for instr in dis:
            tw = instr.get("target_width", 0)
            th = instr.get("target_height", 0)
            if tw <= 0 or th <= 0:
                continue
            x = instr.get("x", 0)
            y = instr.get("y", 0)

            if instr.get("is_tile"):
                sheet = self._get_tile_sheet(os.path.basename(instr.get("path", "")))
                if sheet is None:
                    continue
                tile_px = sheet.width // 16
                sx = instr.get("tile_x", 0) * tile_px
                sy = instr.get("tile_y", 0) * tile_px
                sw = max(1, instr.get("tile_w", 1)) * tile_px
                sh = max(1, instr.get("tile_h", 1)) * tile_px
                crop = sheet.crop((sx, sy, sx + sw, sy + sh))
            else:
                if self._spritesheet is None:
                    continue
                try:
                    sprite_id = int(instr.get("path", ""))
                except ValueError:
                    continue
                coords = self._id_to_sprite.get(sprite_id)
                if coords is None:
                    continue
                sx, sy, sw, sh = coords
                crop = self._spritesheet.crop((sx, sy, sx + sw, sy + sh))

            if crop.size != (tw, th):
                crop = crop.resize((tw, th), Image.NEAREST)

            opacity = instr.get("opacity", 1.0)
            if opacity < 1.0:
                crop = crop.copy()
                alpha = crop.getchannel("A").point(lambda a, o=opacity: int(a * o))
                crop.putalpha(alpha)

            angle = instr.get("angle", 0.0)
            if angle:
                crop = crop.rotate(-math.degrees(angle), resample=Image.NEAREST, expand=True)
                x -= (crop.width - tw) // 2
                y -= (crop.height - th) // 2

            # Proper "over" compositing (PIL's paste-with-mask mishandles the
            # alpha channel itself, which matters once the canvas starts
            # transparent and instructions carry partial opacity).
            region = img.crop((x, y, x + crop.width, y + crop.height))
            img.paste(Image.alpha_composite(region, crop), (x, y))

        return img

    def _get_toost_image(self, idx, lvl):
        """Native-resolution (16px/tile) toost render, cached per level."""
        if idx not in self._toost_cache:
            self._toost_cache[idx] = self._render_toost_image(lvl)
        return self._toost_cache[idx]

    def _get_pixel_occupancy(self, idx, lvl, max_tx, max_ty):
        """Per-tile bool grid (row 0 = bottom) — True where toost's render
        has any non-transparent pixel. This is the ground truth for "is
        anything actually here", independent of obj_anchor/obj_tile_size."""
        if idx not in self._occ_cache:
            img = self._get_toost_image(idx, lvl)
            if img is None:
                occ = None
            else:
                alpha = img.getchannel("A")
                occ = [[False] * max_tx for _ in range(max_ty)]
                for row in range(max_ty):
                    py = (max_ty - 1 - row) * 16
                    for col in range(max_tx):
                        px = col * 16
                        cell = alpha.crop((px, py, px + 16, py + 16))
                        if cell.getbbox() is not None:
                            occ[row][col] = True
            self._occ_cache[idx] = occ
        return self._occ_cache[idx]

    def _draw_object_overlay(self, lvl, max_tx, max_ty, ts):
        """Outline each object's (col,row,w,h) bounding box (from obj_anchor /
        obj_tile_size) on top of the Toost render, for fact-checking those
        heuristics against toost's own ground-truth tile placement."""
        active = self._active_cats()
        show_lbl = self.show_labels.get() and ts >= 14
        font_sz = ("Courier", max(ts // 2, 7), "bold")
        for obj in lvl.get("objects", []):
            name = resolve_obj_name(obj.get("name", "_unknown"), lvl.get("gamestyle_raw", 0))
            char, color, cat = get_meta(name)
            if cat not in active:
                continue
            col, row = obj_anchor(obj)
            w, h = obj_tile_size(obj)
            if col + w <= 0 or row + h <= 0 or col >= max_tx or row >= max_ty:
                continue
            px0 = col * ts
            px1 = (col + w) * ts
            py0 = (max_ty - (row + h)) * ts
            py1 = (max_ty - row) * ts
            self.canvas.create_rectangle(px0, py0, px1, py1, outline=color, width=2)
            if show_lbl:
                self.canvas.create_text((px0 + px1) // 2, (py0 + py1) // 2,
                                        text=char, fill=color, font=font_sz)

    def _grid_bounds(self, lvl):
        """Return (max_tx, max_ty) matching the C++ H = BorT/16, W = BorR/16."""
        top_b   = lvl.get("top_boundary", 0)
        right_b = lvl.get("right_boundary", 0)
        if top_b > 0 and right_b > 0:
            max_tx = right_b // 16
            max_ty = top_b // 16
        else:
            # fallback for files without boundary fields
            objects = lvl.get("objects", [])
            max_tx, max_ty = 40, 20
            for o in objects:
                col, row = obj_anchor(o)
                w, h = obj_tile_size(o)
                max_tx = max(max_tx, col + w + 1)
                max_ty = max(max_ty, row + h + 1)
        return min(max_tx, self.MAX_COLS), min(max_ty, self.MAX_ROWS)

    def _redraw(self):
        self.canvas.delete("all")
        if not self.levels:
            self.info_lbl.config(text="No level loaded")
            return

        if self.ascii_mode.get():
            self._render_ascii()
            return

        lvl     = self.levels[self.current_idx]
        objects = lvl.get("objects", [])
        name    = lvl.get("name", f"Level {self.current_idx + 1}")
        ts      = self.tile_size
        active  = self._active_cats()

        max_tx, max_ty = self._grid_bounds(lvl)

        W = max_tx * ts
        H = max_ty * ts
        self.canvas.config(scrollregion=(0, 0, W, H))

        # ---- Toost Render mode (replays drawing_instructions) ----
        if self.toost_mode.get() and _PIL_OK:
            native = self._get_toost_image(self.current_idx, lvl)
            if native is not None:
                disp = native if ts == 16 else native.resize((W, H), Image.NEAREST)
                photo = ImageTk.PhotoImage(disp)
                self.canvas.create_image(0, 0, image=photo, anchor="nw")
                self._photo_ref = photo          # prevent garbage collection
                if self.show_grid.get():
                    gc = "#555555" if ts > 10 else "#333333"
                    for col in range(max_tx + 1):
                        self.canvas.create_line(col*ts, 0, col*ts, H, fill=gc)
                    for row in range(max_ty + 1):
                        self.canvas.create_line(0, row*ts, W, row*ts, fill=gc)
                if self.show_objects.get():
                    self._draw_object_overlay(lvl, max_tx, max_ty, ts)
                self.info_lbl.config(
                    text=f"[{self.current_idx+1}/{len(self.levels)}]  {name}  "
                         f"[Toost Render | style={lvl.get('gamestyle','?')}]  "
                         f"grid {max_tx}×{max_ty}")
                return
            # No drawing_instructions in this JSON — fall back to rectangle view below.

        self.canvas.create_rectangle(0, 0, W, H, fill="#5C94FC", outline="")

        # grid lines
        if self.show_grid.get():
            grid_color = "#888888" if ts > 10 else "#666666"
            for col in range(max_tx + 1):
                self.canvas.create_line(col * ts, 0, col * ts, H, fill=grid_color)
            for row in range(max_ty + 1):
                self.canvas.create_line(0, row * ts, W, row * ts, fill=grid_color)

        show_lbl = self.show_labels.get() and ts >= 14
        pad = max(1, ts // 8)

        # draw objects — semisolids first (background layer), then foreground
        if self.show_objects.get():
            BG_TYPES = {"Semisolid Platform", "Mushroom Platform"}
            for pass_n in range(2):
                for obj in objects:
                    obj_name = obj.get("name", "_unknown")
                    is_bg = obj_name in BG_TYPES
                    if pass_n == 0 and not is_bg: continue
                    if pass_n == 1 and is_bg:     continue
                    char, color, cat = get_meta(resolve_obj_name(obj_name, lvl.get("gamestyle_raw", 0)))
                    if obj_name == "Pipe":
                        char = _PIPE_DIR_CHAR.get(_pipe_direction(obj.get("flag", 0)), char)
                    if obj_name in _SLOPE_NAMES:
                        char = "/" if (obj.get("flag", 0) >> 20) & 1 else "\\"
                    if cat not in active:
                        continue
                    col, row = obj_anchor(obj)
                    w, h = obj_tile_size(obj)
                    outline_col = "#888888" if is_bg else "#000000"
                    font_sz = ("Courier", max(ts // 2, 7), "bold")
                    if obj_name in _SLOPE_NAMES:
                        for tc, tr in slope_tiles(obj):
                            if tc < 0 or tc >= max_tx or tr < 0 or tr >= max_ty:
                                continue
                            spx0 = tc * ts + pad
                            spx1 = (tc + 1) * ts - pad
                            spy0 = (max_ty - 1 - tr) * ts + pad
                            spy1 = (max_ty - 1 - tr) * ts + ts - pad
                            self.canvas.create_rectangle(spx0, spy0, spx1, spy1,
                                                         fill=color, outline=outline_col)
                            if show_lbl:
                                self.canvas.create_text((spx0 + spx1) // 2, (spy0 + spy1) // 2,
                                                        text=char, fill="white", font=font_sz)
                            # ---- Fill supporting ground blocks ----

                            right_slope = (obj.get("flag", 0) & 0x100000) != 0
                            step = 2 if obj["id"] == 87 else 1

                            # Gentle slopes: only place support once per pair of slope tiles
                            base_col, _ = obj_anchor(obj)
                            if step == 2 or ((tc - base_col) % step == 0):
                                
                                fill_tc = tc + 1 if right_slope else tc - 1

                                if 0 <= fill_tc < max_tx:

                                    fpx0 = fill_tc * ts + pad
                                    fpx1 = (fill_tc + 1) * ts - pad

                                    self.canvas.create_rectangle(
                                        fpx0, spy0, fpx1, spy1,
                                        fill=GROUND_COLOR,
                                        outline=outline_col
                                    )

                                    if show_lbl:
                                        self.canvas.create_text(
                                            (fpx0 + fpx1) // 2,
                                            (spy0 + spy1) // 2,
                                            text="#",
                                            fill="white",
                                            font=font_sz
                                        )
                    else:
                        if col >= max_tx or row >= max_ty:
                            continue
                        if obj_name == "Mushroom Platform":
                            sc = col + w // 2
                            cr = row + h - 1
                            # stem: 1 tile wide, from bottom up to (but not including) cap
                            spx0 = sc * ts + pad
                            spx1 = (sc + 1) * ts - pad
                            spy0 = (max_ty - 1 - cr) * ts + ts - pad  # bottom of stem = just below cap
                            spy1 = (max_ty - 1 - row) * ts + ts - pad  # bottom of bounding box
                            if spy0 < spy1:  # only draw stem if h > 1
                                self.canvas.create_rectangle(spx0, spy0, spx1, spy1,
                                                             fill=color, outline=outline_col)
                                if show_lbl:
                                    self.canvas.create_text((spx0 + spx1) // 2, (spy0 + spy1) // 2,
                                                            text=char, fill="white", font=font_sz)
                            # cap: full width at top row
                            cpx0 = col * ts + pad
                            cpx1 = (col + w) * ts - pad
                            cpy0 = (max_ty - 1 - cr) * ts + pad
                            cpy1 = (max_ty - 1 - cr) * ts + ts - pad
                            self.canvas.create_rectangle(cpx0, cpy0, cpx1, cpy1,
                                                         fill=color, outline=outline_col)
                            if show_lbl:
                                self.canvas.create_text((cpx0 + cpx1) // 2, (cpy0 + cpy1) // 2,
                                                        text=char, fill="white", font=font_sz)
                        else:
                            px0 = col * ts + pad
                            py1 = (max_ty - 1 - row) * ts + ts - pad
                            px1 = (col + w) * ts - pad
                            py0 = (max_ty - 1 - (row + h - 1)) * ts + pad
                            self.canvas.create_rectangle(px0, py0, px1, py1,
                                                         fill=color, outline=outline_col)
                            if show_lbl:
                                self.canvas.create_text((px0 + px1) // 2, (py0 + py1) // 2,
                                                        text=char, fill="white", font=font_sz)

        note = ""
        if self.toost_mode.get():
            note = "  |  [Toost Render unavailable: no drawing_instructions]"
        self.info_lbl.config(
            text=f"[{self.current_idx + 1}/{len(self.levels)}]  {name}  |  "
                 f"style={lvl.get('gamestyle', '?')}  theme={lvl.get('theme', '?')}  |  "
                 f"{len(objects)} objects  |  grid {max_tx}×{max_ty}{note}")

    # ----------------------------------------------------------- ASCII mode --
    def _build_ascii_grid(self):
        lvl     = self.levels[self.current_idx]
        objects = lvl.get("objects", [])
        max_tx, max_ty = self._grid_bounds(lvl)

        grid = [[" "] * max_tx for _ in range(max_ty)]

        # Ground truth from toost's own render (None if drawing_instructions
        # is unavailable, in which case we fall back to pure heuristics).
        occ = self._get_pixel_occupancy(self.current_idx, lvl, max_tx, max_ty) \
            if _PIL_OK else None

        def set_cell(col, row_game, ch, only_if_blank=False, force=False):
            if 0 <= col < max_tx and 0 <= row_game < max_ty:
                if not force and occ is not None and not occ[row_game][col]:
                    return  # toost shows nothing here — skip the heuristic ghost
                r = max_ty - 1 - row_game
                if only_if_blank and grid[r][col] != " ":
                    return  # a real object already claims this cell
                grid[r][col] = ch

        BG_TYPES = {"Semisolid Platform", "Mushroom Platform"}
        # Slopes and track rails sit "between" background platforms and real
        # objects: their rectangular bbox commonly overlaps a Block, enemy,
        # etc. that shares one of its tiles (perfectly normal in Mario
        # Maker), and that object should stay visible. Draw order decides
        # the winner for a shared cell, so background platforms go first,
        # slopes/tracks next, and everything else last.
        FILL_TYPES = _SLOPE_NAMES | {"Track"}

        def _pass_for(obj_name):
            if obj_name in BG_TYPES:
                return 0
            if obj_name in FILL_TYPES:
                return 1
            return 2

        for pass_n in range(3):
            for obj in objects:
                obj_name = obj.get("name", "_unknown")
                if _pass_for(obj_name) != pass_n:
                    continue
                char, _, _ = get_meta(resolve_obj_name(obj_name, lvl.get("gamestyle_raw", 0)))
                if obj_name in ASCII_REPLACEMENTS:
                    char = ASCII_REPLACEMENTS[obj_name]
                col, row = obj_anchor(obj)
                w, h = obj_tile_size(obj)
                if obj_name == "Bridge":
                    # Only the bottom (walkable) row is kept; the rope/chain
                    # row above it is decorative and gets dropped.
                    for dx in range(w):
                        set_cell(col + dx, row, char)
                    continue
                if obj_name in _SLOPE_NAMES:
                    # Slopes have no flat-ASCII diagonal equivalent, so fill
                    # their footprint as a solid ascending/descending
                    # staircase of ground -- mirrors slope_fill_cells() in
                    # json_to_swe.py and the same fix in mm2_json_to_ascii.py.
                    step = 2 if obj.get("id") == 87 else 1
                    descending = (obj.get("flag", 0) & 0x100000) != 0
                    for x in range(w):
                        run = (w - x) if descending else (x + 1)
                        height = min((run + step - 1) // step, h)
                        for y in range(height):
                            set_cell(col + x, row + y, GROUND_CHAR)
                    continue
                if obj_name == "Big Coin":
                    # Big Coin is being phased out; represent it as a 2x2
                    # cluster of regular coins instead of its own glyph.
                    # force=True since toost's actual sprite only occupies a
                    # single tile, not the 2x2 footprint we want here.
                    coin_char, _, _ = get_meta("Coin")
                    for dx in range(2):
                        for dy in range(2):
                            set_cell(col + dx, row + dy, coin_char, force=True)
                    continue
                # Claim the object's full bounding box; occ (toost's actual
                # rendered pixels) carves out the real silhouette — slope
                # diagonals, mushroom-platform caps/stems, etc. fall out for
                # free instead of needing per-shape fill formulas.
                for dx in range(w):
                    for dy in range(h):
                        set_cell(col + dx, row + dy, char)

                if obj_name == "Burner":
                    # The flame jet isn't part of the burner's own bbox, so
                    # it's claimed separately based on facing direction.
                    direction = _BURNER_FLAME_DIR.get(obj.get("flag", 0) % 0x100)
                    if direction == 'U':
                        flame = [(col + dx, row + h + dy) for dx in range(w) for dy in range(3 * h)]
                    elif direction == 'D':
                        flame = [(col + dx, row - 1 - dy) for dx in range(w) for dy in range(3 * h)]
                    elif direction == 'R':
                        flame = [(col + w + dx, row + dy) for dx in range(3 * w) for dy in range(h)]
                    elif direction == 'L':
                        flame = [(col - 1 - dx, row + dy) for dx in range(3 * w) for dy in range(h)]
                    else:
                        flame = []
                    for fx, fy in flame:
                        set_cell(fx, fy, char, only_if_blank=True)

        # Anything toost actually drew that no object claimed — keeps the
        # silhouette pixel-faithful even where the heuristics fall short.
        if occ is not None:
            for row_game in range(max_ty):
                for col in range(max_tx):
                    if occ[row_game][col] and grid[max_ty - 1 - row_game][col] == " ":
                        grid[max_ty - 1 - row_game][col] = UNKNOWN_CHAR
        return grid, max_tx, max_ty

    def _render_ascii(self):
        lvl  = self.levels[self.current_idx]
        name = lvl.get("name", f"Level {self.current_idx + 1}")
        grid, max_tx, max_ty = self._build_ascii_grid()
        ts   = self.tile_size
        font = ("Courier", max(ts - 2, 7), "bold")
        W, H = max_tx * ts, max_ty * ts
        self.canvas.config(scrollregion=(0, 0, W, H))
        self.canvas.create_rectangle(0, 0, W, H, fill="#111111", outline="")
        for row_canvas, row_chars in enumerate(grid):
            for col, ch in enumerate(row_chars):
                x0, y0 = col * ts, row_canvas * ts
                if ch == GROUND_CHAR:     bg = "#8B6914"
                elif ch in ("/", "\\"):  bg = "#AA8833"
                elif ch == UNKNOWN_CHAR: bg = "#553355"
                elif ch == " ":           bg = "#111111"
                else:                     bg = "#222222"
                self.canvas.create_rectangle(x0, y0, x0 + ts, y0 + ts,
                                             fill=bg, outline="")
                if ch != " ":
                    self.canvas.create_text(x0 + ts // 2, y0 + ts // 2,
                                            text=ch, fill="#EEEEEE", font=font)
        self.info_lbl.config(
            text=f"[{self.current_idx + 1}/{len(self.levels)}]  {name}  [ASCII]  "
                 f"grid {max_tx}×{max_ty}")

    def _export_ascii(self):
        if not self.levels:
            return
        grid, _, _ = self._build_ascii_grid()
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt"), ("All", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            for row in grid:
                f.write("".join(row).rstrip() + "\n")

    # ------------------------------------------------------------ tooltip --
    def _on_hover(self, event):
        if not self.levels:
            return
        lvl = self.levels[self.current_idx]
        ts  = self.tile_size
        max_tx, max_ty = self._grid_bounds(lvl)

        # account for canvas scroll offset
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        col      = int(cx // ts)
        row_game = max_ty - 1 - int(cy // ts)

        hits = []
        for obj in lvl.get("objects", []):
            oc, or_ = obj_anchor(obj)
            ow, oh  = obj_tile_size(obj)
            if oc <= col < oc + ow and or_ <= row_game < or_ + oh:
                hits.append(f"{obj.get('name', '?')}  size={ow}×{oh}  @({oc},{or_})")

        # Cross-check against toost's own render (ground truth for "is
        # anything actually drawn here") so the tooltip flags heuristic
        # mismatches instead of silently trusting obj_anchor/obj_tile_size.
        occ = self._get_pixel_occupancy(self.current_idx, lvl, max_tx, max_ty) \
            if _PIL_OK else None
        in_bounds = 0 <= col < max_tx and 0 <= row_game < max_ty
        occupied = occ is not None and in_bounds and occ[row_game][col]

        if hits:
            tip = "\n".join(hits)
            if occ is not None and not occupied:
                tip += "\n[no toost pixels here]"
        elif occupied:
            tip = f"tile ({col}, {row_game})  [occupied, unidentified]"
        else:
            tip = f"tile ({col}, {row_game})"
        self._show_tip(event.x_root, event.y_root, tip)

    def _show_tip(self, rx, ry, text):
        self._hide_tip()
        self._tooltip_win = tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{rx + 14}+{ry - 10}")
        tk.Label(tw, text=text, justify=tk.LEFT,
                 background="#FFFFCC", relief=tk.SOLID, borderwidth=1,
                 font=("Courier", 9)).pack()

    def _hide_tip(self):
        if self._tooltip_win:
            self._tooltip_win.destroy()
            self._tooltip_win = None

    def _drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _drag_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = MM2Viewer()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        try:
            with open(sys.argv[1], encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
                
            # Normalize the CLI-loaded data
            for lvl in data:
                lvl.setdefault("_source_file", sys.argv[1])
                app._normalize_level(lvl)
                
            app.levels = data
            app.current_idx = 0
            app.after(100, app._redraw)
        except Exception as e:
            print(f"Could not load {sys.argv[1]}: {e}")

    app.protocol("WM_DELETE_WINDOW", lambda: (app.destroy(), sys.exit(0)))
    app.mainloop()