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
    from PIL import Image, ImageDraw, ImageTk
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
    "Blinking Block":      ("*", "#FFAA00", CAT_TERRAIN),
    "Spike Block":         ("^", "#AA0000", CAT_TERRAIN),
    "Crate":               ("C", "#B87333", CAT_TERRAIN),
    "Stone":               ("S", "#999999", CAT_TERRAIN),
    "Goal Ground":         ("_", "#00AA00", CAT_TERRAIN),
    "Starting Brick":      ("{", "#C8A050", CAT_TERRAIN),
    "Castle Bridge":       ("=", "#885522", CAT_TERRAIN),
    "Tree":                ("T", "#228B22", CAT_TERRAIN),
    "Slight Slope":        ("/", "#AA8833", CAT_TERRAIN),
    "Steep Slope":         ("\\","#CC9933", CAT_TERRAIN),
    # doors / warps
    "Pipe":                ("|", "#00BB00", CAT_DOOR),
    "Door":                ("D", "#4466FF", CAT_DOOR),
    "Warp Box":            ("W", "#6644FF", CAT_DOOR),
    "Key":                 ("k", "#FFD700", CAT_DOOR),
    "Checkpoint Flag":     ("f", "#00DDAA", CAT_DOOR),
    "Goal":                ("G", "#00FF44", CAT_DOOR),
    "Clear Pipe":          ("c", "#44FFCC", CAT_DOOR),
    # enemies
    "Goomba":              ("g", "#CC6600", CAT_ENEMY),
    "Koopa":               ("K", "#44AA00", CAT_ENEMY),
    "Piranha Plant":       ("P", "#DD2200", CAT_ENEMY),
    "Hammer Bro":          ("m", "#2244AA", CAT_ENEMY),
    "Thwomp":              ("t", "#6655AA", CAT_ENEMY),
    "Bob-omb":             ("o", "#444444", CAT_ENEMY),
    "Spiny":               ("s", "#CC2222", CAT_ENEMY),
    "Buzzy Beetle":        ("b", "#334488", CAT_ENEMY),
    "Lakitu":              ("L", "#DDAA00", CAT_ENEMY),
    "Lakitu's Cloud":      ("l", "#CCCCAA", CAT_ENEMY),
    "Banzai Bill":         ("Z", "#333333", CAT_ENEMY),
    "Bullet Bill Blaster": ("V", "#333333", CAT_ENEMY),
    "Magikoopa":           ("y", "#8844CC", CAT_ENEMY),
    "Spike Top":           ("<", "#AA3322", CAT_ENEMY),
    "Boo":                 ("u", "#DDDDDD", CAT_ENEMY),
    "Bowser":              ("X", "#BB3300", CAT_ENEMY),
    "Bowser Jr.":          ("x", "#CC5511", CAT_ENEMY),
    "Chain Chomp":         ("@", "#333333", CAT_ENEMY),
    "Cheep Cheep":         ("~", "#FF4488", CAT_ENEMY),
    "Blooper":             ("q", "#DDDDDD", CAT_ENEMY),
    "Wiggler":             ("w", "#AADD00", CAT_ENEMY),
    "Pokey":               ("Y", "#CCAA22", CAT_ENEMY),
    "Piranha Creeper":     ("e", "#AA2200", CAT_ENEMY),
    "Porcupuffer":         ("F", "#8866AA", CAT_ENEMY),
    "Fish Bone":           ("%", "#AAAAAA", CAT_ENEMY),
    "Lava Bubble":         ("&", "#FF4400", CAT_ENEMY),
    "Rocky Wrench":        ("r", "#888844", CAT_ENEMY),
    "Muncher":             (",", "#00AA22", CAT_ENEMY),
    "Ant Trooper":         ("a", "#AA3300", CAT_ENEMY),
    "Monty Mole":          ("n", "#885522", CAT_ENEMY),
    "Mechakoopa":          ("R", "#666666", CAT_ENEMY),
    "Boom Boom":           ("!", "#BB4400", CAT_ENEMY),
    "Dry Bones":           ("9", "#BBBBAA", CAT_ENEMY),
    "Skipsqueak":          ("j", "#FFAA88", CAT_ENEMY),
    "Stingby":             (";", "#DDCC00", CAT_ENEMY),
    "Angry Sun":           ("A", "#FF8800", CAT_ENEMY),
    "Charvaargh":          ("v", "#FF3300", CAT_ENEMY),
    "Bully":               ("[", "#883300", CAT_ENEMY),
    "Lemmy":               ("1", "#FF88CC", CAT_ENEMY),
    "Morton":              ("2", "#888888", CAT_ENEMY),
    "Larry":               ("3", "#44AA44", CAT_ENEMY),
    "Wendy":               ("4", "#FF44AA", CAT_ENEMY),
    "Iggy":                ("5", "#44AAFF", CAT_ENEMY),
    "Roy":                 ("6", "#AA44FF", CAT_ENEMY),
    "Ludwig":              ("7", "#4444CC", CAT_ENEMY),
    # items
    "Coin":                ("¢", "#FFD700", CAT_ITEM),
    "Red Coin":            ("$", "#FF2200", CAT_ITEM),
    "Large Coin":          ("£", "#FFAA00", CAT_ITEM),
    "1-Up Mushroom":       ("U", "#00CC00", CAT_ITEM),
    "Fire Flower":         ("i", "#FF5500", CAT_ITEM),
    "Super Star":          ("*", "#FFFF00", CAT_ITEM),
    "Super Mushroom":      ("M", "#EE2222", CAT_ITEM),
    "Big Mushroom":        ("¶", "#CC1111", CAT_ITEM),
    "SMB2 Mushroom":       ("§", "#884488", CAT_ITEM),
    "Super Hammer":        ("¬", "#996622", CAT_ITEM),
    "P Switch":            ("¦", "#4488FF", CAT_ITEM),
    "POW Block":           ("¯", "#3366FF", CAT_ITEM),
    "Spring":              ("±", "#DDDD00", CAT_ITEM),
    "Goomba's Shoe":       ("µ", "#CC6600", CAT_ITEM),
    "Cannon Box":          ("]", "#666666", CAT_ITEM),
    "Propeller Box":       ("}", "#8888FF", CAT_ITEM),
    "Goomba Mask":         (")", "#CC6600", CAT_ITEM),
    "Bullet Bill Mask":    ("°", "#333333", CAT_ITEM),
    "Red POW Box":         ("²", "#FF3333", CAT_ITEM),
    # platforms
    "Lift":                ("-", "#DDAA55", CAT_PLATFORM),
    "Mushroom Platform":   ("³", "#FF6688", CAT_PLATFORM),
    "Semisolid Platform":  ("´", "#AAAAFF", CAT_PLATFORM),
    "Bridge":              ("·", "#AA8833", CAT_PLATFORM),
    "Lava Lift":           ("¸", "#FF4400", CAT_PLATFORM),
    "Snake Block":         ("¹", "#44CC44", CAT_PLATFORM),
    "Track Block":         ("º", "#AA6622", CAT_PLATFORM),
    "Conveyor Belt":       ("»", "#888888", CAT_PLATFORM),
    "Fast Conveyor Belt":  ("¼", "#555555", CAT_PLATFORM),
    "Sprint Platform":     ("½", "#FF8800", CAT_PLATFORM),
    "Seesaw":              ("¾", "#AA8844", CAT_PLATFORM),
    "Swinging Claw":       ("¿", "#AAAAAA", CAT_PLATFORM),
    "ON/OFF Trampoline":   ("À", "#FF6600", CAT_PLATFORM),
    "Mushroom Trampoline": ("Á", "#FF4488", CAT_PLATFORM),
    "Jumping Machine":     ("J", "#8844FF", CAT_PLATFORM),
    "Half-Collision Platform": ("Â", "#CCCCAA", CAT_PLATFORM),
    "Donut":               ("Ã", "#F09050", CAT_PLATFORM),
    # hazards
    "Fire Bar":            ("Ä", "#FF4400", CAT_HAZARD),
    "Saw":                 ("Å", "#AAAAAA", CAT_HAZARD),
    "Burner":              ("Æ", "#FF6600", CAT_HAZARD),
    "Spikes":              ("Ç", "#888888", CAT_HAZARD),
    "Spike Ball":          ("È", "#884444", CAT_HAZARD),
    "Skewer":              ("É", "#666666", CAT_HAZARD),
    "Twister":             ("Ê", "#AADDFF", CAT_HAZARD),
    "Icicle":              ("Ë", "#AADDFF", CAT_HAZARD),
    # deco
    "Cloud":               ("Ì", "#CCCCFF", CAT_DECO),
    "Vine":                ("Í", "#00BB00", CAT_DECO),
    "Water Marker":        ("Î", "#0055FF", CAT_DECO),
    "Arrow":               ("Ï", "#FFFF00", CAT_DECO),
    "One-Way Wall":        ("Ð", "#FFFF88", CAT_DECO),
    "Reel Camera":         ("Ñ", "#AAAAAA", CAT_DECO),
    "Sound Effect":        ("Ò", "#FFAAFF", CAT_DECO),
    # other
    "Player":              ("Ó", "#0000FF", CAT_OTHER),
    "Clown Car":           ("Ô", "#FF4488", CAT_OTHER),
    "Koopa Clown Car":     ("Õ", "#44AA00", CAT_OTHER),
    "Track":               ("Ö", "#AAAAAA", CAT_OTHER),
    "Starting Arrow":      ("×", "#FFFF00", CAT_OTHER),
    "Cannon":              ("Ø", "#444444", CAT_OTHER),
    "! Block":             ("Ù", "#FFAA00", CAT_OTHER),
    "_unknown":            ("?", "#FF00FF", CAT_OTHER),
}

GROUND_COLOR = "#8B6914"
GROUND_CHAR  = "#"

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
# Sprite support
# ---------------------------------------------------------------------------
# Maps display names (keys of OBJ_META) to the game's internal object ID.
# These IDs correspond to OBJ_N in LevelData.hpp where N * 32768 = enum value.
NAME_TO_GAME_ID = {
    "Ground": 7,    "Block": 4,         "Hard Block": 6,     "? Block": 5,
    "Hidden Block": 29, "Note Block": 23, "Donut Block": 21,  "Ice Block": 63,
    "P Block": 79,  "ON/OFF Block": 99, "Dotted-Line Block": 100,
    "Blinking Block": 108, "Spike Block": 110, "Crate": 112,  "Stone": 75,
    "Goal Ground": 26, "Starting Brick": 37, "Castle Bridge": 49,
    "Tree": 106,    "Slight Slope": 87, "Steep Slope": 88,
    "Pipe": 9,      "Door": 55,         "Warp Box": 97,       "Key": 95,
    "Checkpoint Flag": 90, "Goal": 27,  "Clear Pipe": 93,
    "Goomba": 0,    "Koopa": 1,         "Piranha Plant": 2,   "Hammer Bro": 3,
    "Thwomp": 12,   "Bob-omb": 15,      "Spiny": 25,          "Buzzy Beetle": 28,
    "Lakitu": 30,   "Lakitu's Cloud": 31, "Banzai Bill": 32,
    "Bullet Bill Blaster": 13, "Magikoopa": 39, "Spike Top": 40,
    "Boo": 41,      "Bowser": 62,       "Bowser Jr.": 98,     "Chain Chomp": 61,
    "Cheep Cheep": 56, "Blooper": 48,   "Wiggler": 52,        "Pokey": 78,
    "Piranha Creeper": 107, "Porcupuffer": 114, "Fish Bone": 103,
    "Lava Bubble": 60, "Rocky Wrench": 58, "Muncher": 57,
    "Ant Trooper": 96, "Monty Mole": 102, "Mechakoopa": 111,  "Boom Boom": 77,
    "Dry Bones": 46, "Skipsqueak": 51, "Stingby": 65,         "Angry Sun": 104,
    "Charvaargh": 86, "Bully": 117,
    "Lemmy": 120,   "Morton": 121,      "Larry": 122,         "Wendy": 123,
    "Iggy": 124,    "Roy": 125,         "Ludwig": 126,
    "Coin": 8,      "Red Coin": 92,     "Large Coin": 70,     "1-Up Mushroom": 33,
    "Fire Flower": 34, "Super Star": 35, "Super Mushroom": 20, "Big Mushroom": 44,
    "SMB2 Mushroom": 81, "Super Hammer": 116, "P Switch": 18, "POW Block": 19,
    "Spring": 10,   "Goomba's Shoe": 45, "Cannon Box": 127,  "Propeller Box": 128,
    "Goomba Mask": 129, "Bullet Bill Mask": 130, "Red POW Box": 131,
    "Lift": 11,     "Mushroom Platform": 14, "Semisolid Platform": 16,
    "Bridge": 17,   "Lava Lift": 36,    "Snake Block": 84,    "Track Block": 85,
    "Conveyor Belt": 94, "Fast Conveyor Belt": 53, "Sprint Platform": 80,
    "Seesaw": 91,   "Swinging Claw": 105, "ON/OFF Trampoline": 132,
    "Mushroom Trampoline": 113, "Jumping Machine": 50,
    "Half-Collision Platform": 71, "Donut": 82,
    "Fire Bar": 24, "Saw": 68,          "Burner": 54,         "Spikes": 43,
    "Spike Ball": 74, "Skewer": 83,     "Twister": 76,        "Icicle": 118,
    "Cloud": 22,    "Vine": 64,         "Water Marker": 101,  "Arrow": 66,
    "One-Way Wall": 67, "Reel Camera": 89, "Sound Effect": 109,
    "Player": 69,   "Clown Car": 42,    "Koopa Clown Car": 72, "Track": 59,
    "Starting Arrow": 38, "Cannon": 47, "! Block": 119,
}

# Normalise various gamestyle representations → LevelData.hpp enum name
_GS_NORM = {
    "smb1": "SMB1", "smb 1": "SMB1", "smb_1": "SMB1",
    "smb3": "SMB3", "smb 3": "SMB3", "smb_3": "SMB3",
    "smw":  "SMW",
    "nsmbu":"NSMBU",
    "sm3dw":"SM3DW", "sm3d world": "SM3DW",
}
_GS_INT = {12621: "SMB1", 13133: "SMB3", 22349: "SMW", 21847: "NSMBU", 22323: "SM3DW"}


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
# Slope tile iterator
# ---------------------------------------------------------------------------
_SLOPE_NAMES = frozenset({"Slight Slope", "Steep Slope"})

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
        self.sprite_mode  = tk.BooleanVar(value=False)
        self._cat_vars    = {}
        self._tooltip_win = None

        # Sprite rendering state
        self._sprite_map  = None   # {(obj_enum_name, gs_name): (x, y, w, h)}
        self._sheet       = None   # PIL Image of spritesheet
        self._spr_cache   = {}     # {(name, gs, w_px, h_px): PIL Image | None}
        self._photo_ref   = None   # keep PhotoImage alive

        self._build_ui()
        self._load_sprites()

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
        tk.Checkbutton(tb, text="Sprites", variable=self.sprite_mode,
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

        # 2. Starting structure — C++ draws at col 1, 3 wide, 3 tall at start_y.
        # Left-anchor convention: x = col*160 + 80.
        start_y = lvl.get("start_y", 0)
        objects.append({
            "name": "Starting Brick",
            "x": 1 * 160 + 80,
            "y": start_y * 160,
            "w": 3,
            "h": 3,
        })

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
            # Castle bridge: 14 tiles extending left up to the axe
            for i in range(14):
                objects.append({
                    "name": "Castle Bridge",
                    "x": (goal_col - 14 + i) * 160,
                    "y": (goal_y * 160) - 1,
                    "w": 1,
                    "h": 1,
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
        self._spr_cache.clear()
        self._redraw()

    def _active_cats(self):
        return {cat for cat, v in self._cat_vars.items() if v.get()}

    # --------------------------------------------------------------- drawing --

    def _load_sprites(self):
        if not _PIL_OK:
            return
        script_dir = os.path.dirname(os.path.abspath(__file__))
        hpp_path   = os.path.join(script_dir, "LevelData.hpp")
        sheet_path = os.path.join(script_dir, "img", "spritesheet.png")
        if not (os.path.exists(hpp_path) and os.path.exists(sheet_path)):
            return
        with open(hpp_path, "r") as fh:
            hpp = fh.read()
        # Parse ObjectLocation: { OBJ_xxx | GS_yyy, { x, y, w, h } }
        pat = re.compile(
            r'\{\s*(\w+)\s*\|\s*(\w+)\s*,\s*\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}\s*\}'
        )
        smap = {}
        for m in pat.finditer(hpp):
            smap[(m.group(1), m.group(2))] = (
                int(m.group(3)), int(m.group(4)),
                int(m.group(5)), int(m.group(6)),
            )
        self._sprite_map = smap
        self._sheet = Image.open(sheet_path).convert("RGBA")

    def _resolve_gs(self, raw) -> str:
        if isinstance(raw, int):
            return _GS_INT.get(raw, "NSMBU")
        s = str(raw).strip().lower()
        return _GS_NORM.get(s, _GS_NORM.get(s.replace(" ", ""), "NSMBU"))

    def _get_sprite(self, obj_name: str, gs: str, w_px: int, h_px: int):
        """Return a scaled RGBA PIL Image for obj_name in gamestyle gs, or None."""
        if not self._sprite_map or not self._sheet:
            return None
        game_id = NAME_TO_GAME_ID.get(obj_name)
        if game_id is None:
            return None
        key = (obj_name, gs, w_px, h_px)
        if key in self._spr_cache:
            return self._spr_cache[key]
        # Try base variant then A variant
        coords = None
        for enum_name in (f"OBJ_{game_id}", f"OBJ_{game_id}A"):
            coords = self._sprite_map.get((enum_name, gs))
            if coords:
                break
        if not coords:
            self._spr_cache[key] = None
            return None
        sx, sy, sw, sh = coords
        crop = self._sheet.crop((sx, sy, sx + sw, sy + sh))
        scaled = crop.resize((w_px, h_px), Image.NEAREST)
        self._spr_cache[key] = scaled
        return scaled

    def _render_sprite_image(self, lvl) -> "Image.Image":
        """Compose the full level into a PIL RGBA image using sprite data."""
        ts       = self.tile_size
        max_tx, max_ty = self._grid_bounds(lvl)
        W, H     = max_tx * ts, max_ty * ts
        gs       = self._resolve_gs(lvl.get("gamestyle", ""))
        active   = self._active_cats()
        objects  = lvl.get("objects", [])

        img  = Image.new("RGBA", (W, H), (92, 148, 252, 255))   # sky blue
        draw = ImageDraw.Draw(img)

        BG_TYPES = {"Semisolid Platform", "Mushroom Platform"}

        for pass_n in range(2):
            for obj in objects:
                name   = obj.get("name", "_unknown")
                is_bg  = name in BG_TYPES
                if pass_n == 0 and not is_bg: continue
                if pass_n == 1 and is_bg:     continue

                _, color, cat = get_meta(name)
                if cat not in active:
                    continue

                col, row = obj_anchor(obj)
                w,   h   = obj_tile_size(obj)

                # Slopes: draw one sprite tile per slope cell
                if name in _SLOPE_NAMES:
                    for tc, tr in slope_tiles(obj):
                        if tc < 0 or tc >= max_tx or tr < 0 or tr >= max_ty:
                            continue
                        px0 = tc * ts
                        py0 = (max_ty - tr - 1) * ts
                        spr = self._get_sprite(name, gs, ts, ts)
                        if spr:
                            img.paste(spr, (px0, py0), spr)
                        else:
                            r_, g_, b_ = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                            draw.rectangle([px0, py0, px0+ts-1, py0+ts-1], fill=(r_, g_, b_))
                    continue

                # Compute pixel rect (y-flipped)
                px0 = col * ts
                py0 = (max_ty - row - h) * ts
                px1 = px0 + w * ts
                py1 = py0 + h * ts

                # Skip fully off-canvas objects
                if px1 <= 0 or py1 <= 0 or px0 >= W or py0 >= H:
                    continue

                # Clamp to canvas
                cx0, cy0 = max(0, px0), max(0, py0)
                cx1, cy1 = min(W, px1), min(H, py1)
                w_px, h_px = cx1 - cx0, cy1 - cy0

                spr = self._get_sprite(name, gs, w_px, h_px)
                if spr:
                    img.paste(spr, (cx0, cy0), spr)
                else:
                    r_, g_, b_ = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                    draw.rectangle([cx0, cy0, cx1-1, cy1-1], fill=(r_, g_, b_, 220))

        return img

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

        # ---- Sprite mode ----
        if self.sprite_mode.get() and _PIL_OK and self._sprite_map:
            pil_img = self._render_sprite_image(lvl)
            photo   = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, image=photo, anchor="nw")
            self._photo_ref = photo          # prevent garbage collection
            if self.show_grid.get():
                gc = "#555555" if ts > 10 else "#333333"
                for col in range(max_tx + 1):
                    self.canvas.create_line(col*ts, 0, col*ts, H, fill=gc)
                for row in range(max_ty + 1):
                    self.canvas.create_line(0, row*ts, W, row*ts, fill=gc)
            self.info_lbl.config(
                text=f"[{self.current_idx+1}/{len(self.levels)}]  {name}  "
                     f"[Sprites | style={lvl.get('gamestyle','?')}]  "
                     f"grid {max_tx}×{max_ty}")
            return

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
                    char, color, cat = get_meta(obj_name)
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

        self.info_lbl.config(
            text=f"[{self.current_idx + 1}/{len(self.levels)}]  {name}  |  "
                 f"style={lvl.get('gamestyle', '?')}  theme={lvl.get('theme', '?')}  |  "
                 f"{len(objects)} objects  |  grid {max_tx}×{max_ty}")

    # ----------------------------------------------------------- ASCII mode --
    def _build_ascii_grid(self):
        lvl     = self.levels[self.current_idx]
        objects = lvl.get("objects", [])
        max_tx, max_ty = self._grid_bounds(lvl)

        grid = [[" "] * max_tx for _ in range(max_ty)]

        def set_cell(col, row_game, ch):
            if 0 <= col < max_tx and 0 <= row_game < max_ty:
                grid[max_ty - 1 - row_game][col] = ch

        BG_TYPES = {"Semisolid Platform", "Mushroom Platform"}
        for pass_n in range(2):
            for obj in objects:
                obj_name = obj.get("name", "_unknown")
                is_bg = obj_name in BG_TYPES
                if pass_n == 0 and not is_bg: continue
                if pass_n == 1 and is_bg:     continue
                char, _, _ = get_meta(obj_name)
                if obj_name == "Pipe":
                    char = _PIPE_DIR_CHAR.get(_pipe_direction(obj.get("flag", 0)), char)
                col, row = obj_anchor(obj)
                w, h = obj_tile_size(obj)
                if obj_name in _SLOPE_NAMES:
                    right_slope = (obj.get("flag", 0) & 0x100000) != 0
                    slope_char = "/" if right_slope else "\\"
                    face_cells = list(slope_tiles(obj))
                    for tc, tr in face_cells:
                        if right_slope:
                            for fill_x in range(tc + 1, col + w):
                                set_cell(fill_x, tr, GROUND_CHAR)
                        else:
                            for fill_x in range(col, tc):
                                set_cell(fill_x, tr, GROUND_CHAR)
                    for tc, tr in face_cells:
                        set_cell(tc, tr, slope_char)

                elif obj_name == "Mushroom Platform":
                    sc = col + w // 2
                    # stem: centered column, all rows below cap
                    for dy in range(h - 1):
                        set_cell(sc, row + dy, char)
                    # cap: full width at top row
                    for dx in range(w):
                        set_cell(col + dx, row + h - 1, char)
                else:
                    for dx in range(w):
                        for dy in range(h):
                            set_cell(col + dx, row + dy, char)
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
                if ch == GROUND_CHAR:    bg = "#8B6914"
                elif ch in ("/", "\\"): bg = "#AA8833"
                elif ch == " ":          bg = "#111111"
                else:                    bg = "#222222"
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
        _, max_ty = self._grid_bounds(lvl)

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

        tip = "\n".join(hits) if hits else f"tile ({col}, {row_game})"
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