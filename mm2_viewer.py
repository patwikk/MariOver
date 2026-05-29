"""
MM2 Level Viewer
================
Visual browser for Super Mario Maker 2 level data parsed via level.py.
 
Usage
-----
    python mm2_viewer.py                  # open GUI, use buttons to load
    python mm2_viewer.py my_level.json    # auto-load a JSON export
 
Coordinate systems
------------------
  Objects : pixel coords (multiples of 160).  Tile col/row = coord // 160
  Ground  : already raw tile indices (small ints). Y=0 is the bottom row.
  Display : Y is flipped so Y=0 appears at the BOTTOM of the canvas.
"""
 
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json, sys, os, zlib
from io import BytesIO
from level import Level
import zlib
import math
import numpy as np
import multiprocessing
import queue as _queue

# Append the Isaac Karth WFC framework dependency path
sys.path.append(os.path.join(os.path.dirname(__file__), "wfc_2019f"))

 
# ---------------------------------------------------------------------------
# ObjId integer → name  (matches level.py ObjId enum exactly)
# ---------------------------------------------------------------------------
OBJID_INT_TO_STR = {
    0:"goomba", 1:"koopa", 2:"piranha_flower", 3:"hammer_bro",
    4:"block", 5:"question_block", 6:"hard_block", 7:"ground",
    8:"coin", 9:"pipe", 10:"spring", 11:"lift", 12:"thwomp",
    13:"bullet_bill_blaster", 14:"mushroom_platform", 15:"bob_omb",
    16:"semisolid_platform", 17:"bridge", 18:"p_switch", 19:"pow",
    20:"super_mushroom", 21:"donut_block", 22:"cloud", 23:"note_block",
    24:"fire_bar", 25:"spiny", 26:"goal_ground", 27:"goal",
    28:"buzzy_beetle", 29:"hidden_block", 30:"lakitu", 31:"lakitu_cloud",
    32:"banzai_bill", 33:"one_up", 34:"fire_flower", 35:"super_star",
    36:"lava_lift", 37:"starting_brick", 38:"starting_arrow",
    39:"magikoopa", 40:"spike_top", 41:"boo", 42:"clown_car",
    43:"spikes", 44:"big_mushroom", 45:"shoe_goomba", 46:"dry_bones",
    47:"cannon", 48:"blooper", 49:"castle_bridge", 50:"jumping_machine",
    51:"skipsqueak", 52:"wiggler", 53:"fast_conveyor_belt", 54:"burner",
    55:"door", 56:"cheep_cheep", 57:"muncher", 58:"rocky_wrench",
    59:"track", 60:"lava_bubble", 61:"chain_chomp", 62:"bowser",
    63:"ice_block", 64:"vine", 65:"stingby", 66:"arrow",
    67:"one_way", 68:"saw", 69:"player", 70:"big_coin",
    71:"half_collision_platform", 72:"koopa_car", 73:"cinobio",
    74:"spike_ball", 75:"stone", 76:"twister", 77:"boom_boom",
    78:"pokey", 79:"p_block", 80:"sprint_platform", 81:"smb2_mushroom",
    82:"donut", 83:"skewer", 84:"snake_block", 85:"track_block",
    86:"charvaargh", 87:"slight_slope", 88:"steep_slope",
    89:"reel_camera", 90:"checkpoint_flag", 91:"seesaw",
    92:"red_coin", 93:"clear_pipe", 94:"conveyor_belt", 95:"key",
    96:"ant_trooper", 97:"warp_box", 98:"bowser_jr", 99:"on_off_block",
    100:"dotted_line_block", 101:"water_marker", 102:"monty_mole",
    103:"fish_bone", 104:"angry_sun", 105:"swinging_claw", 106:"tree",
    107:"piranha_creeper", 108:"blinking_block", 109:"sound_effect",
    110:"spike_block", 111:"mechakoopa", 112:"crate",
    113:"mushroom_trampoline", 114:"porkupuffer", 115:"cinobic",
    116:"super_hammer", 117:"bully", 118:"icicle",
    119:"exclamation_block", 120:"lemmy", 121:"morton", 122:"larry",
    123:"wendy", 124:"iggy", 125:"roy", 126:"ludwig",
    127:"cannon_box", 128:"propeller_box", 129:"goomba_mask",
    130:"bullet_bill_mask", 131:"red_pow_box", 132:"on_off_trampoline",
}
 
def obj_id_to_str(obj_id) -> str:
    """Handle enum objects, 'ObjId.door' strings, or raw integers."""
    if isinstance(obj_id, int):
        return OBJID_INT_TO_STR.get(obj_id, "_unknown")
    s = str(obj_id)
    if "." in s:
        # "ObjId.door"  or  "<ObjId.door: 55>"
        return s.split(".")[-1].split(":")[0].strip().rstrip(">").strip()
    try:
        return OBJID_INT_TO_STR.get(int(s), "_unknown")
    except ValueError:
        return s if s else "_unknown"
 
# ---------------------------------------------------------------------------
# Category → (char, bg_color)
# ---------------------------------------------------------------------------
CAT_TERRAIN  = "terrain"
CAT_ENEMY    = "enemy"
CAT_ITEM     = "item"
CAT_PLATFORM = "platform"
CAT_DOOR     = "door"
CAT_HAZARD   = "hazard"
CAT_DECO     = "deco"
CAT_OTHER    = "other"
 
# name → (char, color, category)
OBJ_META = {
    # terrain
    "ground":              ("#", "#8B6914", CAT_TERRAIN),
    "block":               ("B", "#C8A050", CAT_TERRAIN),
    "hard_block":          ("H", "#888888", CAT_TERRAIN),
    "question_block":      ("?", "#F0C030", CAT_TERRAIN),
    "hidden_block":        ("h", "#CCCCCC", CAT_TERRAIN),
    "note_block":          ("N", "#E8A020", CAT_TERRAIN),
    "donut_block":         ("d", "#F09050", CAT_TERRAIN),
    "ice_block":           ("I", "#A0D8EF", CAT_TERRAIN),
    "p_block":             ("p", "#CC44CC", CAT_TERRAIN),
    "on_off_block":        ("O", "#FF6600", CAT_TERRAIN),
    "dotted_line_block":   (".", "#AAAAAA", CAT_TERRAIN),
    "blinking_block":      ("*", "#FFAA00", CAT_TERRAIN),
    "spike_block":         ("^", "#AA0000", CAT_TERRAIN),
    "crate":               ("C", "#B87333", CAT_TERRAIN),
    "stone":               ("S", "#999999", CAT_TERRAIN),
    "goal_ground":         ("_", "#00AA00", CAT_TERRAIN),
    "starting_brick":      ("{", "#C8A050", CAT_TERRAIN),
    "castle_bridge":       ("=", "#885522", CAT_TERRAIN),
    "tree":                ("T", "#228B22", CAT_TERRAIN),
    "slight_slope":        ("/", "#AA8833", CAT_TERRAIN),
    "steep_slope":         ("\\","#CC9933", CAT_TERRAIN),
    # doors / warps
    "pipe":                ("|", "#00BB00", CAT_DOOR),
    "door":                ("D", "#4466FF", CAT_DOOR),
    "warp_box":            ("W", "#6644FF", CAT_DOOR),
    "key":                 ("k", "#FFD700", CAT_DOOR),
    "checkpoint_flag":     ("f", "#00DDAA", CAT_DOOR),
    "goal":                ("G", "#00FF44", CAT_DOOR),
    "clear_pipe":          ("c", "#44FFCC", CAT_DOOR),
    # enemies
    "goomba":              ("g", "#CC6600", CAT_ENEMY),
    "koopa":               ("K", "#44AA00", CAT_ENEMY),
    "piranha_flower":      ("P", "#DD2200", CAT_ENEMY),
    "hammer_bro":          ("m", "#2244AA", CAT_ENEMY), 
    "thwomp":              ("t", "#6655AA", CAT_ENEMY),
    "bob_omb":             ("o", "#444444", CAT_ENEMY),
    "spiny":               ("s", "#CC2222", CAT_ENEMY),
    "buzzy_beetle":        ("b", "#334488", CAT_ENEMY),
    "lakitu":              ("L", "#DDAA00", CAT_ENEMY),
    "lakitu_cloud":        ("l", "#CCCCAA", CAT_ENEMY),
    "banzai_bill":         ("Z", "#333333", CAT_ENEMY),
    "bullet_bill_blaster": ("V", "#333333", CAT_ENEMY),
    "magikoopa":           ("y", "#8844CC", CAT_ENEMY), 
    "spike_top":           ("<", "#AA3322", CAT_ENEMY), 
    "boo":                 ("u", "#DDDDDD", CAT_ENEMY),
    "bowser":              ("X", "#BB3300", CAT_ENEMY),
    "bowser_jr":           ("x", "#CC5511", CAT_ENEMY),
    "chain_chomp":         ("@", "#333333", CAT_ENEMY),
    "cheep_cheep":         ("~", "#FF4488", CAT_ENEMY),
    "blooper":             ("q", "#DDDDDD", CAT_ENEMY),
    "wiggler":             ("w", "#AADD00", CAT_ENEMY),
    "pokey":               ("Y", "#CCAA22", CAT_ENEMY), 
    "piranha_creeper":     ("e", "#AA2200", CAT_ENEMY),
    "porkupuffer":         ("F", "#8866AA", CAT_ENEMY),
    "fish_bone":           ("%", "#AAAAAA", CAT_ENEMY),
    "lava_bubble":         ("&", "#FF4400", CAT_ENEMY),
    "rocky_wrench":        ("r", "#888844", CAT_ENEMY),
    "muncher":             (",", "#00AA22", CAT_ENEMY),
    "ant_trooper":         ("a", "#AA3300", CAT_ENEMY),
    "monty_mole":          ("n", "#885522", CAT_ENEMY),
    "mechakoopa":          ("R", "#666666", CAT_ENEMY),
    "boom_boom":           ("!", "#BB4400", CAT_ENEMY),
    "dry_bones":           ("9", "#BBBBAA", CAT_ENEMY),
    "skipsqueak":          ("j", "#FFAA88", CAT_ENEMY),
    "cinobio":             ("+", "#DD4444", CAT_ENEMY),
    "cinobic":             ("¡", "#CC3333", CAT_ENEMY),
    "stingby":             (";", "#DDCC00", CAT_ENEMY),
    "angry_sun":           ("A", "#FF8800", CAT_ENEMY),
    "charvaargh":          ("v", "#FF3300", CAT_ENEMY),
    "bully":               ("[", "#883300", CAT_ENEMY),
    "lemmy":               ("1", "#FF88CC", CAT_ENEMY),
    "morton":              ("2", "#888888", CAT_ENEMY),
    "larry":               ("3", "#44AA44", CAT_ENEMY),
    "wendy":               ("4", "#FF44AA", CAT_ENEMY),
    "iggy":                ("5", "#44AAFF", CAT_ENEMY),
    "roy":                 ("6", "#AA44FF", CAT_ENEMY),
    "ludwig":              ("7", "#4444CC", CAT_ENEMY),
    # items
    "coin":                ("¢", "#FFD700", CAT_ITEM), 
    "red_coin":            ("$", "#FF2200", CAT_ITEM),
    "big_coin":            ("£", "#FFAA00", CAT_ITEM), 
    "one_up":              ("U", "#00CC00", CAT_ITEM), 
    "fire_flower":         ("i", "#FF5500", CAT_ITEM), 
    "super_star":          ("¤", "#FFFF00", CAT_ITEM), 
    "super_mushroom":      ("M", "#EE2222", CAT_ITEM), 
    "big_mushroom":        ("¶", "#CC1111", CAT_ITEM), 
    "smb2_mushroom":       ("§", "#884488", CAT_ITEM), 
    "super_hammer":        ("¬", "#996622", CAT_ITEM), 
    "p_switch":            ("¦", "#4488FF", CAT_ITEM), 
    "pow":                 ("¯", "#3366FF", CAT_ITEM), 
    "spring":              ("±", "#DDDD00", CAT_ITEM), 
    "shoe_goomba":         ("µ", "#CC6600", CAT_ITEM), 
    "cannon_box":          ("]", "#666666", CAT_ITEM),
    "propeller_box":       ("}", "#8888FF", CAT_ITEM), 
    "goomba_mask":         (")", "#CC6600", CAT_ITEM), 
    "bullet_bill_mask":    ("°", "#333333", CAT_ITEM),
    "red_pow_box":         ("²", "#FF3333", CAT_ITEM),
    # platforms
    "lift":                ("-", "#DDAA55", CAT_PLATFORM),
    "mushroom_platform":   ("³", "#FF6688", CAT_PLATFORM), 
    "semisolid_platform":  ("´", "#AAAAFF", CAT_PLATFORM),
    "bridge":              ("·", "#AA8833", CAT_PLATFORM), 
    "lava_lift":           ("¸", "#FF4400", CAT_PLATFORM), 
    "snake_block":         ("¹", "#44CC44", CAT_PLATFORM), 
    "track_block":         ("º", "#AA6622", CAT_PLATFORM), 
    "conveyor_belt":       ("»", "#888888", CAT_PLATFORM),
    "fast_conveyor_belt":  ("¼", "#555555", CAT_PLATFORM), 
    "sprint_platform":     ("½", "#FF8800", CAT_PLATFORM), 
    "seesaw":              ("¾", "#AA8844", CAT_PLATFORM),
    "swinging_claw":       ("¿", "#AAAAAA", CAT_PLATFORM), 
    "on_off_trampoline":   ("À", "#FF6600", CAT_PLATFORM), 
    "mushroom_trampoline": ("Á", "#FF4488", CAT_PLATFORM), 
    "jumping_machine":     ("J", "#8844FF", CAT_PLATFORM),
    "half_collision_platform": ("Â", "#CCCCAA", CAT_PLATFORM), 
    "donut":               ("Ã", "#F09050", CAT_PLATFORM), 
    # hazards
    "fire_bar":            ("Ä", "#FF4400", CAT_HAZARD), 
    "saw":                 ("Å", "#AAAAAA", CAT_HAZARD), 
    "burner":              ("Æ", "#FF6600", CAT_HAZARD), 
    "spikes":              ("Ç", "#888888", CAT_HAZARD), 
    "spike_ball":          ("È", "#884444", CAT_HAZARD), 
    "skewer":              ("É", "#666666", CAT_HAZARD), 
    "twister":             ("Ê", "#AADDFF", CAT_HAZARD), 
    "icicle":              ("Ë", "#AADDFF", CAT_HAZARD), 
    # deco
    "cloud":               ("Ì", "#CCCCFF", CAT_DECO), 
    "vine":                ("Í", "#00BB00", CAT_DECO), 
    "water_marker":        ("Î", "#0055FF", CAT_DECO), 
    "arrow":               ("Ï", "#FFFF00", CAT_DECO), 
    "one_way":             ("Ð", "#FFFF88", CAT_DECO), 
    "reel_camera":         ("Ñ", "#AAAAAA", CAT_DECO), 
    "sound_effect":        ("Ò", "#FFAAFF", CAT_DECO), 
    # other
    "player":              ("Ó", "#0000FF", CAT_OTHER), 
    "clown_car":           ("Ô", "#FF4488", CAT_OTHER), 
    "koopa_car":           ("Õ", "#44AA00", CAT_OTHER), 
    "track":               ("Ö", "#AAAAAA", CAT_OTHER), 
    "starting_arrow":      ("×", "#FFFF00", CAT_OTHER),
    "cannon":              ("Ø", "#444444", CAT_OTHER), 
    "exclamation_block":   ("Ù", "#FFAA00", CAT_OTHER), 
    "_ground_tile":        ("#", "#8B6914", CAT_TERRAIN),
    "_unknown":            ("?", "#FF00FF", CAT_OTHER),
}

ASCII_MAP = {k: v[0] for k, v in OBJ_META.items() if k != "_unknown"}
ASCII_MAP["_unknown"] = "?"

GROUND_COLOR = "#8B6914"
GROUND_CHAR  = "#"
 
 
# ---------------------------------------------------------------------------
# ASCII map — obj name → single character
# ---------------------------------------------------------------------------
ASCII_MAP = {
    "ground":"#","_ground_tile":"#","block":"B","hard_block":"H",
    "question_block":"?","hidden_block":"h","note_block":"N",
    "donut_block":"d","ice_block":"I","p_block":"p","on_off_block":"O",
    "dotted_line_block":".","blinking_block":"*","spike_block":"^",
    "crate":"C","stone":"S","goal_ground":"_","starting_brick":"{",
    "castle_bridge":"=","tree":"T","slight_slope":"/","steep_slope":"\\",
    "pipe":"|","door":"D","warp_box":"W","key":"k",
    "checkpoint_flag":"f","goal":"G","clear_pipe":"c",
    "goomba":"g","koopa":"K","piranha_flower":"P","hammer_bro":"m",
    "thwomp":"t","bob_omb":"o","spiny":"s","buzzy_beetle":"b",
    "lakitu":"L","lakitu_cloud":"l","banzai_bill":"Z",
    "bullet_bill_blaster":"V","magikoopa":"y","spike_top":"<",
    "boo":"u","bowser":"X","bowser_jr":"x","chain_chomp":"@",
    "cheep_cheep":"~","blooper":"q","wiggler":"w","pokey":"Y",
    "piranha_creeper":"e","porkupuffer":"F","fish_bone":"%",
    "lava_bubble":"&","rocky_wrench":"r","muncher":",",
    "ant_trooper":"a","monty_mole":"n","mechakoopa":"R",
    "boom_boom":"!","dry_bones":"9","skipsqueak":"j",
    "cinobio":"+","cinobic":"¡","stingby":";","angry_sun":"A",
    "charvaargh":"v","bully":"[","lemmy":"1","morton":"2",
    "larry":"3","wendy":"4","iggy":"5","roy":"6","ludwig":"7",
    "coin":"¢","red_coin":"$","big_coin":"£","one_up":"U",
    "fire_flower":"i","super_star":"¤","super_mushroom":"M",
    "big_mushroom":"¶","smb2_mushroom":"§","super_hammer":"¬",
    "p_switch":"¦","pow":"¯","spring":"±","shoe_goomba":"µ",
    "cannon_box":"]","propeller_box":"}","goomba_mask":")",
    "bullet_bill_mask":"°","red_pow_box":"²",
    "lift":"-","mushroom_platform":"³","semisolid_platform":"´",
    "bridge":"·","lava_lift":"¸","snake_block":"¹","track_block":"º",
    "conveyor_belt":"»","fast_conveyor_belt":"¼","sprint_platform":"½",
    "seesaw":"¾","swinging_claw":"¿","on_off_trampoline":"À",
    "mushroom_trampoline":"Á","jumping_machine":"J",
    "half_collision_platform":"Â","donut":"Ã",
    "fire_bar":"Ä","saw":"Å","burner":"Æ","spikes":"Ç",
    "spike_ball":"È","skewer":"É","twister":"Ê","icicle":"Ë",
    "cloud":"Ì","vine":"Í","water_marker":"Î","arrow":"Ï",
    "one_way":"Ð","reel_camera":"Ñ","sound_effect":"Ò",
    "player":"Ó","clown_car":"Ô","koopa_car":"Õ","track":"Ö",
    "starting_arrow":"×","cannon":"Ø","exclamation_block":"Ù",
    "_unknown":"?",
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
 
def get_meta(name: str):
    return OBJ_META.get(name, OBJ_META["_unknown"])
 
# ---------------------------------------------------------------------------
# Convert a parsed kaitai Level object → plain dict
# ---------------------------------------------------------------------------
def level_to_dict(level, name: str = "") -> dict:
    def parse_map(world):
        objs = []
        for i in range(world.object_count):
            o = world.objects[i]
            objs.append({"x": int(o.x), "y": int(o.y), "id": str(o.id)})
        gnd = []
        for i in range(world.ground_count):
            g = world.ground[i]
            gnd.append({"x": int(g.x), "y": int(g.y),
                        "tile_id": int(g.id), "background_id": int(g.background_id)})
        return objs, gnd
 
    ow_objs, ow_gnd = parse_map(level.overworld)
    sw_objs, sw_gnd = parse_map(level.subworld)
    level_name = name or getattr(level, "name", "Unknown")
    if isinstance(level_name, (bytes, bytearray)):
        level_name = level_name.decode("utf-16-le", errors="replace")
    level_name = level_name.strip("\x00").strip()
 
    start_y = int(getattr(level, "start_y", 0))
    goal_x  = int(getattr(level, "goal_x",  0))
    goal_y  = int(getattr(level, "goal_y",  0))
 
    # boundary_right is the actual right edge of this level in pixels
    boundary_right = int(level.overworld.boundary_right)
 
    gamestyle = str(getattr(level, "gamestyle", "")).lower()
    theme     = str(level.overworld.theme).lower()
    if "." in gamestyle:
        gamestyle = gamestyle.split(".")[-1]
    if "." in theme:
        theme = theme.split(".")[-1]
 
    # --- DEBUG ---
    print(f"\n=== LEVEL: {level_name} ===")
    print(f"  gamestyle={gamestyle}  theme={theme}")
    print(f"  header: start_y={start_y}  goal_x={goal_x}  goal_y={goal_y}  (goal_x//160={goal_x//160 if goal_x else 0})  (goal_x//10={goal_x//10 if goal_x else 0})")
    print(f"  all objects ({len(ow_objs)}):")
    for o in ow_objs:
        print(f"    id={o['id']}  x={o['x']}  y={o['y']}  tile_col={o['x']//160}  tile_row={o['y']//160}")
    # --- END DEBUG ---
 
    return {
        "name":           level_name,
        "gamestyle":      gamestyle,
        "theme":          theme,
        "start_y":        start_y,
        "goal_x_raw":     goal_x,
        "goal_y_raw":     goal_y,
        "boundary_right": boundary_right,
        "objects":        ow_objs,
        "ground":         ow_gnd,
        "subworld_objects": sw_objs,
        "subworld_ground":  sw_gnd,
    }
 
def export_level_json(level, path: str, name: str = ""):
    with open(path, "w") as f:
        json.dump(level_to_dict(level, name), f, indent=2)
    print(f"Exported to {path}")
 
# ---------------------------------------------------------------------------
# Main viewer
# ---------------------------------------------------------------------------
class MM2Viewer(tk.Tk):
    TILE_PX   = 160   # object pixel coords → divide by this to get tile index
    MAX_COLS  = 240
    MAX_ROWS  = 28
 
    def __init__(self):
        super().__init__()
        self.title("MM2 Level Viewer")
        self.resizable(True, True)
 
        self.levels      = []
        self.current_idx = 0
        self.tile_size   = 16
 
        self.show_ground  = tk.BooleanVar(value=True)
        self.show_objects = tk.BooleanVar(value=True)
        self.show_grid    = tk.BooleanVar(value=True)
        self.show_labels  = tk.BooleanVar(value=True)
        self.ascii_mode   = tk.BooleanVar(value=False)
        self._cat_vars    = {}   # cat → BooleanVar
        self._tooltip_win = None
 
        self._build_ui()
 
    # ------------------------------------------------------------------ UI --
    def _build_ui(self):
        # toolbar
        tb = tk.Frame(self, bd=1, relief=tk.RAISED)
        tb.pack(fill=tk.X, side=tk.TOP, padx=2, pady=2)
 
        tk.Button(tb, text="Load JSON",            command=self._load_json).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Load ASCII",           command=self._load_ascii).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Load from dataset",    command=self._load_dataset).pack(side=tk.LEFT, padx=4)
 
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Checkbutton(tb, text="Ground",  variable=self.show_ground,  command=self._redraw).pack(side=tk.LEFT)
        tk.Checkbutton(tb, text="Objects", variable=self.show_objects, command=self._redraw).pack(side=tk.LEFT)
        tk.Checkbutton(tb, text="Grid",    variable=self.show_grid,    command=self._redraw).pack(side=tk.LEFT)
        tk.Checkbutton(tb, text="Labels",  variable=self.show_labels,  command=self._redraw).pack(side=tk.LEFT)
 
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Label(tb, text="Zoom:").pack(side=tk.LEFT)
        self.zoom_var = tk.IntVar(value=16)
        tk.Scale(tb, from_=6, to=40, orient=tk.HORIZONTAL, variable=self.zoom_var,
                 command=lambda _: self._on_zoom(), showvalue=True, length=120).pack(side=tk.LEFT)
 
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        tk.Checkbutton(tb, text="ASCII mode", variable=self.ascii_mode,
                       command=self._redraw).pack(side=tk.LEFT, padx=4)
        tk.Button(tb, text="Export ASCII", command=self._export_ascii).pack(side=tk.LEFT, padx=2)
        tk.Button(tb, text="Capture PNG",  command=self._capture_png).pack(side=tk.LEFT, padx=2)
 
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
 
        self.canvas.bind("<ButtonPress-1>",  self._drag_start)
        self.canvas.bind("<B1-Motion>",      self._drag_move)
        self.canvas.bind("<Motion>",         self._on_hover)
        self.canvas.bind("<Leave>",          lambda _: self._hide_tip())
 
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
    def _load_json(self):
        path = filedialog.askopenfilename(
            title="Select MM2 level JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            self.levels = data
            self.current_idx = 0
            self._redraw()
        except Exception as e:
            messagebox.showerror("Load error", str(e))
    
    def _load_ascii(self):
        paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not paths:
            return
            
        loaded = []
        errors = []

        # Build reverse lookup character table dynamically from our unique mapping dictionary
        REVERSE_ASCII_MAP = {}
        for asset_name, meta in OBJ_META.items():
            char_symbol = meta[0]
            if asset_name not in ["_unknown", "_ground_tile"]:
                REVERSE_ASCII_MAP[char_symbol] = asset_name

        for path in paths:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    raw_lines = f.read().splitlines()

                # Trim empty outer line bounds while maintaining internal sky gaps
                while raw_lines and not raw_lines[0].strip():
                    raw_lines.pop(0)
                while raw_lines and not raw_lines[-1].strip():
                    raw_lines.pop()

                if not raw_lines:
                    errors.append(f"{os.path.basename(path)}: file is empty")
                    continue

                # Normalise row boundaries using your original dotted block rule
                max_w = max(len(r) for r in raw_lines)
                rows  = [r.ljust(max_w, ".") for r in raw_lines]

                name = os.path.splitext(os.path.basename(path))[0]
                print(f"[Load ASCII] '{name}'  {max_w}w × {len(rows)}h")

                grid_h = len(rows)
                spawn_col_found = None
                spawn_row_found = None
                goal_col_found  = None
                goal_row_found  = None
                is_castle        = False

                # RESTORED CRITICAL LOGIC: Locate level markers from text representation
                for ri, row_str in enumerate(rows):
                    for ci, ch in enumerate(row_str):
                        if ch == "S":
                            spawn_col_found = ci
                            spawn_row_found = ri
                        elif ch in ("G", "X"):
                            goal_col_found = ci
                            goal_row_found = ri
                            is_castle = (ch == "X")

                # Map text indices back to the vertical coordinate system
                if spawn_row_found is not None:
                    start_y = (grid_h - 1) - spawn_row_found
                else:
                    start_y = 1

                if goal_row_found is not None:
                    goal_y_raw = (grid_h - 1) - goal_row_found
                else:
                    goal_y_raw = 0

                if goal_col_found is not None:
                    goal_x_raw = goal_col_found * 10
                    boundary_right = (goal_col_found + 10) * 16
                else:
                    goal_x_raw     = 0
                    boundary_right = max_w * 16

                print(f"[Load ASCII]   spawn col={spawn_col_found} canvas_row={spawn_row_found} → start_y={start_y}")
                print(f"[Load ASCII]   goal  col={goal_col_found}  canvas_row={goal_row_found}  → goal_y_raw={goal_y_raw}  goal_x_raw={goal_x_raw}  castle={is_castle}")

                # Construct baseline structure payload definitions matching level_to_dict format
                base = {
                    "name":             name,
                    "gamestyle":        "smb1",
                    "theme":            "castle" if is_castle else "overworld",
                    "start_y":          start_y,
                    "goal_x_raw":       goal_x_raw,
                    "goal_y_raw":       goal_y_raw,
                    "boundary_right":   boundary_right,
                    "objects":          [],
                    "ground":           [],
                    "subworld_objects": [],
                    "subworld_ground":  [],
                    "_wfc_min_y":       0,
                }

                # Dynamically compile structural elements into objects and ground lists 
                for r_idx, line in enumerate(rows):
                    tile_y = (grid_h - 1) - r_idx
                    for c_idx, char in enumerate(line):
                        # Treat special placement anchors as structural air now that metadata is extracted
                        if char in ("S", "G", "X", " "):
                            continue
                        
                        asset_name = REVERSE_ASCII_MAP.get(char, None)
                        if not asset_name:
                            continue
                        
                        if asset_name == "ground":
                            base["ground"].append({
                                "x": c_idx, 
                                "y": tile_y, 
                                "tile_id": 7, 
                                "background_id": 0
                            })
                        else:
                            pixel_x = c_idx * self.TILE_PX
                            pixel_y = tile_y * self.TILE_PX
                            
                            matched_id = 0
                            for op_id, op_str in OBJID_INT_TO_STR.items():
                                if op_str == asset_name:
                                    matched_id = op_id
                                    break
                            base["objects"].append({
                                "x": pixel_x, 
                                "y": pixel_y, 
                                "id": str(matched_id)
                            })

                base["name"] = name
                loaded.append(base)
                print(f"[Load ASCII] Decoded: {len(base['ground'])} ground, {len(base['objects'])} objects")

            except Exception as exc:
                errors.append(f"{os.path.basename(path)}: {exc}")

        if errors:
            messagebox.showwarning("Load ASCII — some files failed", "\n".join(errors))

        if loaded:
            self.levels = loaded
            self.current_idx = 0
            self._redraw()
 
    def _load_dataset(self):
        dlg = _DatasetDialog(self)
        self.wait_window(dlg)
        if dlg.result:
            self.levels = dlg.result
            self.current_idx = 0
            self._redraw()
 
    def load_level_from_parsed(self, level, name=""):
        self.levels = [level_to_dict(level, name)]
        self.current_idx = 0
        self._redraw()
 
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
 
    # --------------------------------------------------------------- drawing --
    def _active_cats(self):
        return {cat for cat, v in self._cat_vars.items() if v.get()}
 
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
        ground  = lvl.get("ground",  [])
        name    = lvl.get("name", f"Level {self.current_idx + 1}")
        ts      = self.tile_size
        active  = self._active_cats()
 
        # compute grid size from actual data
        max_tx = 40
        max_ty = 20
        for o in objects:
            max_tx = max(max_tx, int(math.ceil(o["x"] / self.TILE_PX)) + 2)
            max_ty = max(max_ty, o["y"] // self.TILE_PX + 2)
        for g in ground:
            max_tx = max(max_tx, g["x"] + 2)
            max_ty = max(max_ty, g["y"] + 2)
        br = lvl.get("boundary_right", 0)
        boundary_cols = (br // 16) if br > 0 else 0
        max_tx = min(max_tx, self.MAX_COLS) - 1
        max_ty = min(max_ty, self.MAX_ROWS)
        print(f"DEBUG max_tx={max_tx}  boundary_right={br}  boundary_cols={boundary_cols}")
 
        # compute theme/gamestyle early — needed for goal X and canvas width
        theme     = lvl.get("theme",     "overworld")
        gamestyle = lvl.get("gamestyle", "smb1")
        is_castle_axe = (theme == "castle" and gamestyle != "sm3dw")
        print(f"DEBUG _redraw: theme={repr(theme)}  gamestyle={repr(gamestyle)}  is_castle_axe={is_castle_axe}")
 
        # goal_x_raw is in units of 1/10 tile — divide by 10 to get tile col.
        goal_x_raw = int(lvl.get("goal_x_raw", 0))
        goal_x_tile = math.ceil(goal_x_raw // 10) if goal_x_raw > 0 else 0
 
        if is_castle_axe:
            goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 10
            max_tx = max(max_tx, goal_base_col + 2)
        else:
            goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 9
            max_tx = goal_base_col + 10
            max_ty -= 1
 
        W = max_tx * ts
        H = max_ty * ts
        self.canvas.config(scrollregion=(0, 0, W, H))
 
        # sky background
        self.canvas.create_rectangle(0, 0, W, H, fill="#5C94FC", outline="")
 
        # grid lines
        if self.show_grid.get():
            grid_color = "#888888" if ts > 10 else "#666666"
            for col in range(max_tx + 1):
                self.canvas.create_line(col * ts, 0, col * ts, H, fill=grid_color)
            for row in range(max_ty + 1):
                self.canvas.create_line(0, row * ts, W, row * ts, fill=grid_color)
 
        show_lbl = self.show_labels.get() and ts >= 14
 
        # ground tiles  (Y=0 game → bottom row on canvas)
        if self.show_ground.get() and CAT_TERRAIN in active:
            for g in ground:
                col = g["x"]
                row_game = g["y"]
                if col >= max_tx or row_game >= max_ty:
                    continue
                row_canvas = max_ty - 1 - row_game
                x0, y0 = col * ts, row_canvas * ts
                self.canvas.create_rectangle(x0, y0, x0 + ts, y0 + ts,
                                             fill=GROUND_COLOR, outline="#5A3E00")
                if show_lbl:
                    self.canvas.create_text(x0 + ts // 2, y0 + ts // 2,
                                            text=GROUND_CHAR, fill="#EDD090",
                                            font=("Courier", max(ts // 2, 7), "bold"))
 
        # objects
        if self.show_objects.get():
            for obj in objects:
                name_str = obj_id_to_str(obj["id"])
                char, color, cat = get_meta(name_str)
                if cat not in active:
                    continue
                col      = int(math.ceil(obj["x"] // self.TILE_PX))
                row_game = obj["y"] // self.TILE_PX
                if col >= max_tx or row_game >= max_ty:
                    continue
                row_canvas = max_ty - 1 - row_game
                x0, y0 = col * ts, row_canvas * ts
                pad = max(1, ts // 8)
                self.canvas.create_rectangle(x0 + pad, y0 + pad,
                                             x0 + ts - pad, y0 + ts - pad,
                                             fill=color, outline="#000000")
                if show_lbl:
                    self.canvas.create_text(x0 + ts // 2, y0 + ts // 2,
                                            text=char, fill="white",
                                            font=("Courier", max(ts // 2, 7), "bold"))
 
        # ----------------------------------------------------------------
        # START GROUND  —  7 tiles wide, fills from row 0 up to start_y
        # (the game enforces this; no objects can be placed in this zone)
        # ----------------------------------------------------------------
        START_W    = 7
        start_ygame = lvl.get("start_y", 1)
        ground_color = GROUND_COLOR   # same brown used for regular ground
 
        for sc_col in range(START_W):
            if sc_col >= max_tx:
                continue
            # fill every row from 0 up to start_y-1 (start_y is where Mario stands)
            for row in range(0, start_ygame):
                if row >= max_ty:
                    continue
                row_canvas = max_ty - 1 - row
                x0 = sc_col * ts
                y0 = row_canvas * ts
                self.canvas.create_rectangle(x0, y0, x0 + ts, y0 + ts,
                                             fill=ground_color, outline="#5A3E00")
                if show_lbl and row == start_ygame - 1:
                    # top surface label
                    self.canvas.create_text(x0 + ts // 2, y0 + ts // 2,
                                            text="#", fill="#EDD090",
                                            font=("Courier", max(ts // 2, 7), "bold"))
 
        # spawn marker: green S on column 3 (centre of 7-wide zone), at start_y+1
        spawn_label_row = start_ygame
        if 3 < max_tx and spawn_label_row < max_ty:
            sx = 3 * ts
            sy = (max_ty - 1 - spawn_label_row) * ts
            self.canvas.create_rectangle(sx, sy, sx + ts, sy + ts,
                                         fill="#00CC00", outline="#006600")
            self.canvas.create_text(sx + ts // 2, sy + ts // 2,
                                    text="S", fill="white",
                                    font=("Courier", max(ts // 2, 7), "bold"))
 
        # ----------------------------------------------------------------
        # GOAL — X is always 9 tiles from the right edge of the level.
        # Y comes from the header goal_y field.
        # ----------------------------------------------------------------
        GOAL_W = 11 if not is_castle_axe else 10
 
        goal_base_ygame = int(lvl.get("goal_y_raw", 0))
 
        # ---- goal ground: 10 tiles wide, filled from row 0 to goal_base_ygame ----
        for gc_col in range(GOAL_W):
            col_abs = goal_base_col + gc_col
            if col_abs < 0 or col_abs >= max_tx:
                continue
            for row in range(0, goal_base_ygame):
                if row >= max_ty:
                    continue
                row_canvas = max_ty - 1 - row
                x0 = col_abs * ts
                y0 = row_canvas * ts
                self.canvas.create_rectangle(x0, y0, x0 + ts, y0 + ts,
                                             fill=ground_color, outline="#5A3E00")
                if show_lbl and row == goal_base_ygame - 1:
                    self.canvas.create_text(x0 + ts // 2, y0 + ts // 2,
                                            text="#", fill="#EDD090",
                                            font=("Courier", max(ts // 2, 7), "bold"))
 
        top_row = goal_base_ygame  # row Mario stands on at the goal
 
        if is_castle_axe:
            BRIDGE_W = 14
            bridge_row_canvas = max_ty - 1 - (goal_base_ygame - 1)
            for b in range(BRIDGE_W):
                bc = goal_base_col - BRIDGE_W + b
                if bc < 0 or bc >= max_tx:
                    continue
                bx = bc * ts
                by = bridge_row_canvas * ts
                self.canvas.create_rectangle(bx, by, bx + ts, by + ts,
                                             fill="#8B4513", outline="#5A2E00")
                if show_lbl:
                    self.canvas.create_text(bx + ts // 2, by + ts // 2,
                                            text="=", fill="#DDAA88",
                                            font=("Courier", max(ts // 2, 7), "bold"))
            if goal_base_col < max_tx and top_row < max_ty:
                ax0 = goal_base_col * ts
                ay0 = (max_ty - 1 - top_row) * ts
                self.canvas.create_rectangle(ax0, ay0, ax0 + ts, ay0 + ts,
                                             fill="#DD0000", outline="#880000")
                self.canvas.create_text(ax0 + ts // 2, ay0 + ts // 2,
                                        text="X", fill="white",
                                        font=("Courier", max(ts // 2, 7), "bold"))
        else:
            flag_col = goal_base_col 
            if flag_col < max_tx and top_row < max_ty:
                fx0 = flag_col * ts
                fy0 = (max_ty - 1 - top_row) * ts
                self.canvas.create_rectangle(fx0, fy0, fx0 + ts, fy0 + ts,
                                             fill="#DD0000", outline="#880000")
                self.canvas.create_text(fx0 + ts // 2, fy0 + ts // 2,
                                        text="G", fill="white",
                                        font=("Courier", max(ts // 2, 7), "bold"))
 
        self.info_lbl.config(
            text=f"[{self.current_idx + 1}/{len(self.levels)}]  {name}  |  "
                 f"style={gamestyle}  theme={theme}  |  "
                 f"{len(objects)} objects  {len(ground)} ground tiles  |  "
                 f"S=(0,{start_ygame})  "
                 f"G=({goal_base_col},{goal_base_ygame})  |  "
                 f"grid {max_tx}x{max_ty}")
 
    # ----------------------------------------------------------- ASCII mode --
 
    def _build_ascii_grid(self):
        lvl     = self.levels[self.current_idx]
        objects = lvl.get("objects", [])
        ground  = lvl.get("ground",  [])
        theme     = lvl.get("theme",     "overworld")
        gamestyle = lvl.get("gamestyle", "smb1")
        is_castle_axe = (theme == "castle" and gamestyle != "sm3dw")
 
        max_tx = 40
        max_ty = 20
        for o in objects:
            max_tx = max(max_tx, o["x"] // 160 + 2)
            max_ty = max(max_ty, o["y"] // 160 + 2)
        for g in ground:
            max_tx = max(max_tx, g["x"] + 2)
            max_ty = max(max_ty, g["y"] + 2)
        br = lvl.get("boundary_right", 0)
        boundary_cols = (br // 16) if br > 0 else 0
        max_tx = min(max_tx, 240) - 1
        max_ty = min(max_ty, 28)
        goal_x_raw = int(lvl.get("goal_x_raw", 0))
        goal_x_tile = math.ceil(goal_x_raw // 10) if goal_x_raw > 0 else 0
        if is_castle_axe:
            goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 10
            max_tx = max(max_tx, goal_base_col + 2)
        else:
            goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 9
            max_tx = goal_base_col + 11
            max_ty -= 1
 
        start_ygame     = lvl.get("start_y", 1)
        goal_base_ygame = int(lvl.get("goal_y_raw", 0))
        GOAL_W          = 11 if not is_castle_axe else 10
 
        grid = [["-"] * max_tx for _ in range(max_ty)]
 
        def set_cell(col, row_game, ch):
            if 0 <= col < max_tx and 0 <= row_game < max_ty:
                grid[max_ty - 1 - row_game][col] = ch
 
        for g in ground:
            set_cell(g["x"], g["y"], "#")
        for obj in objects:
            ch = ASCII_MAP.get(obj_id_to_str(obj["id"]), "?")
            # Replace math.ceil with standard floor division to eliminate the +1 tile offset
            set_cell(obj["x"] // 160, obj["y"] // 160, ch)
        for col in range(7):
            for row in range(0, start_ygame):
                set_cell(col, row, "#")
        set_cell(3, start_ygame, "S")
        for gc in range(GOAL_W):
            for row in range(0, goal_base_ygame):
                set_cell(goal_base_col + gc, row, "#")
        if is_castle_axe:
            for b in range(14):
                set_cell(goal_base_col - 14 + b, goal_base_ygame - 1, "=")
            set_cell(goal_base_col, goal_base_ygame, "X")
        else:
            set_cell(goal_base_col, goal_base_ygame, "G")
 
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
                if ch == "#":   bg = "#8B6914"
                elif ch == "-": bg = "#111111"
                elif ch == "S": bg = "#00AA00"
                elif ch in ("G","X","F"): bg = "#CC0000"
                elif ch == "=": bg = "#7B3F10"
                else:           bg = "#222222"
                self.canvas.create_rectangle(x0, y0, x0+ts, y0+ts, fill=bg, outline="")
                if ts >= 8:
                    fg = "#EEEEEE" if ch != "-" else "#333333"
                    self.canvas.create_text(x0+ts//2, y0+ts//2, text=ch, fill=fg, font=font)
        self.info_lbl.config(text=f"[{self.current_idx+1}/{len(self.levels)}]  {name}  [ASCII]  grid {max_tx}x{max_ty}")
 
    def _export_ascii(self):
        if not self.levels:
            return
        lvl = self.levels[self.current_idx]
        
        default_filename = f"{lvl.get('name', 'level')}.txt"
        
        path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".txt", 
            filetypes=[("Text files", "*.txt")]
        )
        if not path:
            return
            
        try:
            # Create a character matrix initialized to default background air spaces
            grid_matrix = [[" " for _ in range(self.MAX_COLS)] for _ in range(self.MAX_ROWS)]
            
            # 1. Overlay ground elements
            for g in lvl.get("ground", []):
                col = g["x"]
                tile_y = g["y"]
                row = (self.MAX_ROWS - 1) - tile_y
                if 0 <= col < self.MAX_COLS and 0 <= row < self.MAX_ROWS:
                    grid_matrix[row][col] = "ground"

            # 2. Overlay object entries
            for o in lvl.get("objects", []):
                col = int(o["x"] // self.TILE_PX)
                tile_y = int(o["y"] // self.TILE_PX)
                row = (self.MAX_ROWS - 1) - tile_y
                if 0 <= col < self.MAX_COLS and 0 <= row < self.MAX_ROWS:
                    grid_matrix[row][col] = obj_id_to_str(o["id"])
            
            # 3. Fill start ground zone — cols 0-6, rows 0..(start_y-1) — same
            #    logic as _build_ascii_grid lines 949-951.  Without this the 7-tile
            #    wide platform Mario spawns on is missing from the exported file.
            start_y = lvl.get("start_y", 1)
            for sg_col in range(7):
                for sg_row in range(0, start_y):
                    tile_row = (self.MAX_ROWS - 1) - sg_row
                    if 0 <= sg_col < self.MAX_COLS and 0 <= tile_row < self.MAX_ROWS:
                        if grid_matrix[tile_row][sg_col] == " ":
                            grid_matrix[tile_row][sg_col] = "ground"

            # 4. Inject special structure overlay markers back into the grid layout
            # Parse start_y marker position
            spawn_row = (self.MAX_ROWS - 1) - start_y
            if 0 <= spawn_row < self.MAX_ROWS:
                grid_matrix[spawn_row][0] = "SPAWN_MARKER"

            # Parse goal endpoints position metrics
            goal_x_raw = lvl.get("goal_x_raw", 0)
            goal_y_raw = lvl.get("goal_y_raw", 0)
            goal_col = goal_x_raw // 10
            goal_row = (self.MAX_ROWS - 1) - goal_y_raw
            is_castle = (lvl.get("theme", "overworld") == "castle")

            if 0 <= goal_col < self.MAX_COLS and 0 <= goal_row < self.MAX_ROWS:
                grid_matrix[goal_row][goal_col] = "CASTLE_MARKER" if is_castle else "GOAL_MARKER"

            # 5. Generate final file layout lines
            rows = []
            for r in range(self.MAX_ROWS):
                row_chars = []
                for c in range(self.MAX_COLS):
                    cell = grid_matrix[r][c]
                    if cell == "SPAWN_MARKER":
                        char = "S"
                    elif cell == "GOAL_MARKER":
                        char = "G"
                    elif cell == "CASTLE_MARKER":
                        char = "X"
                    elif cell == " " or cell == "ground":
                        char = ASCII_MAP.get(cell, " ")
                    else:
                        char = ASCII_MAP.get(cell, "?")
                    row_chars.append(char)
                rows.append("".join(row_chars))
            
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(rows))
                
            messagebox.showinfo("Success", "ASCII file exported cleanly.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save layout: {str(e)}")
 
    def _capture_png(self):
        """Render the entire level to a PNG file using PIL.

        The image is drawn off-screen at the current zoom level, reproducing
        the same visual as the main canvas — sky background, ground tiles,
        objects with category colours, start zone, goal, grid lines — but
        covering the *full* level width rather than just the visible viewport.
        PIL is used directly so no window capture / screenshot hacks are needed.
        """
        if not self.levels:
            messagebox.showwarning("No level", "Load a level first.")
            return

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            messagebox.showerror(
                "PIL not found",
                "Pillow is required for PNG export.\n"
                "Install it with:  pip install Pillow")
            return

        lvl  = self.levels[self.current_idx]
        name = lvl.get("name", f"level_{self.current_idx+1}")

        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All", "*.*")],
            title="Capture level as PNG",
            initialfile=f"{name}.png")
        if not path:
            return

        # ---- reproduce the same geometry as _redraw ---------------------- #
        objects  = lvl.get("objects", [])
        ground   = lvl.get("ground",  [])
        ts       = self.tile_size

        theme     = lvl.get("theme",     "overworld")
        gamestyle = lvl.get("gamestyle", "smb1")
        is_castle_axe = (theme == "castle" and gamestyle != "sm3dw")

        max_tx, max_ty = 40, 20
        for o in objects:
            max_tx = max(max_tx, int(math.ceil(o["x"] / self.TILE_PX)) + 2)
            max_ty = max(max_ty, o["y"] // self.TILE_PX + 2)
        for g in ground:
            max_tx = max(max_tx, g["x"] + 2)
            max_ty = max(max_ty, g["y"] + 2)
        max_tx = min(max_tx, self.MAX_COLS) - 1
        max_ty = min(max_ty, self.MAX_ROWS)

        goal_x_raw  = int(lvl.get("goal_x_raw", 0))
        goal_x_tile = math.ceil(goal_x_raw // 10) if goal_x_raw > 0 else 0
        if is_castle_axe:
            goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 10
            max_tx = max(max_tx, goal_base_col + 2)
        else:
            goal_base_col = goal_x_tile if goal_x_tile > 0 else max_tx - 9
            max_tx = goal_base_col + 10
            max_ty -= 1

        W, H = max_tx * ts, max_ty * ts

                # ---- helpers ----------------------------------------------------- #
        def hex_to_rgb(h):
            h = h.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        SKY_RGB   = hex_to_rgb("#5C94FC")
        GND_RGB   = hex_to_rgb(GROUND_COLOR)
        GND_OUT   = hex_to_rgb("#5A3E00")
        GRID_RGB  = hex_to_rgb("#888888")

        # Try to load a small monospace font; fall back to PIL default
        try:
            font_size = max(ts // 2, 7)
            font = ImageFont.truetype("cour.ttf", font_size)
        except Exception:
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", max(ts // 2, 7))
            except Exception:
                font = ImageFont.load_default()

        img  = Image.new("RGB", (W, H), SKY_RGB)
        draw = ImageDraw.Draw(img)
        show_lbl = self.show_labels.get() and ts >= 14
        active   = self._active_cats()

        def draw_tile(col, row_canvas, fill_hex, outline_hex, char=None, char_hex="#FFFFFF"):
            x0, y0 = col * ts, row_canvas * ts
            x1, y1 = x0 + ts - 1, y0 + ts - 1
            fill_rgb   = hex_to_rgb(fill_hex)
            outline_rgb = hex_to_rgb(outline_hex)
            draw.rectangle([x0, y0, x1, y1], fill=fill_rgb, outline=outline_rgb)
            if show_lbl and char:
                draw.text((x0 + ts // 2, y0 + ts // 2), char,
                          fill=hex_to_rgb(char_hex), font=font, anchor="mm")

        # ---- grid lines -------------------------------------------------- #
        if self.show_grid.get():
            for col in range(max_tx + 1):
                draw.line([(col * ts, 0), (col * ts, H)], fill=GRID_RGB)
            for row in range(max_ty + 1):
                draw.line([(0, row * ts), (W, row * ts)], fill=GRID_RGB)

        # ---- ground tiles ------------------------------------------------ #
        if self.show_ground.get() and CAT_TERRAIN in active:
            for g in ground:
                col, row_game = g["x"], g["y"]
                if col >= max_tx or row_game >= max_ty:
                    continue
                draw_tile(col, max_ty - 1 - row_game,
                          GROUND_COLOR, "#5A3E00",
                          GROUND_CHAR if show_lbl else None, "#EDD090")

        # ---- objects ----------------------------------------------------- #
        if self.show_objects.get():
            pad = max(1, ts // 8)
            for obj in objects:
                name_str        = obj_id_to_str(obj["id"])
                char, color, cat = get_meta(name_str)
                if cat not in active:
                    continue
                col      = int(math.ceil(obj["x"] // self.TILE_PX))
                row_game = obj["y"] // self.TILE_PX
                if col >= max_tx or row_game >= max_ty:
                    continue
                rc = max_ty - 1 - row_game
                x0, y0 = col * ts + pad, rc * ts + pad
                x1, y1 = col * ts + ts - pad - 1, rc * ts + ts - pad - 1
                draw.rectangle([x0, y0, x1, y1],
                               fill=hex_to_rgb(color), outline=(0, 0, 0))
                if show_lbl:
                    draw.text((col * ts + ts // 2, rc * ts + ts // 2), char,
                              fill=(255, 255, 255), font=font, anchor="mm")

        # ---- start zone -------------------------------------------------- #
        START_W     = 7
        start_ygame = lvl.get("start_y", 1)
        for sc_col in range(START_W):
            if sc_col >= max_tx:
                continue
            for row in range(0, start_ygame):
                if row >= max_ty:
                    continue
                draw_tile(sc_col, max_ty - 1 - row,
                          GROUND_COLOR, "#5A3E00",
                          "#" if (show_lbl and row == start_ygame - 1) else None,
                          "#EDD090")
        if 3 < max_tx and start_ygame < max_ty:
            draw_tile(3, max_ty - 1 - start_ygame, "#00CC00", "#006600",
                      "S" if show_lbl else None)

        # ---- goal zone --------------------------------------------------- #
        goal_base_ygame = int(lvl.get("goal_y_raw", 0))
        GOAL_W = 11 if not is_castle_axe else 10
        for gc_col in range(GOAL_W):
            col_abs = goal_base_col + gc_col
            if col_abs < 0 or col_abs >= max_tx:
                continue
            for row in range(0, goal_base_ygame):
                if row >= max_ty:
                    continue
                draw_tile(col_abs, max_ty - 1 - row,
                          GROUND_COLOR, "#5A3E00",
                          "#" if (show_lbl and row == goal_base_ygame - 1) else None,
                          "#EDD090")
        top_row = goal_base_ygame
        if is_castle_axe:
            for b in range(14):
                bc = goal_base_col - 14 + b
                if 0 <= bc < max_tx:
                    draw_tile(bc, max_ty - 1 - (goal_base_ygame - 1),
                              "#8B4513", "#5A2E00",
                              "=" if show_lbl else None, "#DDAA88")
            if goal_base_col < max_tx and top_row < max_ty:
                draw_tile(goal_base_col, max_ty - 1 - top_row,
                          "#DD0000", "#880000", "X" if show_lbl else None)
        else:
            if goal_base_col < max_tx and top_row < max_ty:
                draw_tile(goal_base_col, max_ty - 1 - top_row,
                          "#DD0000", "#880000", "G" if show_lbl else None)

        # ---- save -------------------------------------------------------- #
        img.save(path)
        messagebox.showinfo("Captured", f"PNG saved to {path}\n({W}×{H} px)")
    
    # --------------------------------------------------------------- tooltip --
 
    def _on_hover(self, event):
        if not self.levels:
            return
        ts  = self.tile_size
        cx  = self.canvas.canvasx(event.x)
        cy  = self.canvas.canvasy(event.y)
        lvl = self.levels[self.current_idx]
 
        # recompute max_ty the same way _redraw does
        max_ty = 20
        for o in lvl.get("objects", []):
            max_ty = max(max_ty, o["y"] // self.TILE_PX + 2)
        for g in lvl.get("ground", []):
            max_ty = max(max_ty, g["y"] + 2)
        max_ty = min(max_ty, self.MAX_ROWS)
 
        col        = int(cx // ts)
        row_canvas = int(cy // ts)
        row_game   = max_ty - 1 - row_canvas
 
        hits = []
        start_ygame_tip = lvl.get("start_y", 1)
        # start ground zone (cols 0-6, rows 0 to start_y)
        if col < 7 and 0 <= row_game <= start_ygame_tip:
            hits.append(f"start ground  tile({col},{row_game})")
        # spawn marker row
        if col == 3 and row_game == start_ygame_tip + 1:
            hits.append(f"SPAWN  tile({col},{row_game})")
        # goal objects
        for o in lvl.get("objects", []):
            id_str = obj_id_to_str(o["id"])
            obj_col = int(math.ceil(o["x"] / self.TILE_PX))
            if obj_col == col and o["y"] // self.TILE_PX == row_game:
                prefix = "GOAL " if id_str in ("goal", "goal_ground") else "obj"
                hits.append(f"{prefix}: {id_str}  px({o['x']},{o['y']})")
        for g in lvl.get("ground", []):
            if g["x"] == col and g["y"] == row_game:
                hits.append(f"ground  tile={g.get('tile_id','?')}  bg={g.get('background_id','?')}  @({col},{row_game})")
 
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
# Dataset loader dialog
# ---------------------------------------------------------------------------
class _DatasetDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Load from HuggingFace dataset")
        self.result = None
        self.resizable(False, False)
 
        tk.Label(self, text="Keyword filter:").grid(row=0, column=0, sticky=tk.W, padx=8, pady=4)
        self.kw = tk.Entry(self, width=20)
        self.kw.insert(0, "kaizo")
        self.kw.grid(row=0, column=1, padx=4)
 
        tk.Label(self, text="Max levels:").grid(row=1, column=0, sticky=tk.W, padx=8)
        self.mx = tk.Entry(self, width=6)
        self.mx.insert(0, "5")
        self.mx.grid(row=1, column=1, sticky=tk.W, padx=4)
 
        self.status = tk.Label(self, text="", fg="gray")
        self.status.grid(row=2, column=0, columnspan=2, padx=8, pady=2)
 
        bf = tk.Frame(self)
        bf.grid(row=3, column=0, columnspan=2, pady=6)
        tk.Button(bf, text="Load",   command=self._do_load).pack(side=tk.LEFT, padx=4)
        tk.Button(bf, text="Cancel", command=self.destroy).pack(side=tk.LEFT)
        self.grab_set()
 
    def _do_load(self):
        keyword = self.kw.get().strip().lower()
        try:
            max_n = int(self.mx.get())
        except ValueError:
            max_n = 5
 
        self.status.config(text="Importing libraries...")
        self.update()
 
        try:
            from datasets import load_dataset
            from kaitaistruct import KaitaiStream
        except ImportError as e:
            messagebox.showerror("Missing library", str(e), parent=self)
            return
 
        # Load level.py from same folder as this script
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "level",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "level.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            Level = mod.Level
        except Exception as e:
            messagebox.showerror("level.py not found", str(e), parent=self)
            return
 
        self.status.config(text="Streaming dataset...")
        self.update()
 
        try:
            ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")
            if keyword:
                ds = ds.filter(lambda ex: keyword in ex["name"].lower())
 
            levels = []
            for ex in ds:
                if len(levels) >= max_n:
                    break
                self.status.config(text=f"Parsing {len(levels)+1}/{max_n}: {ex['name']}")
                self.update()
                try:
                    raw = zlib.decompress(ex["level_data"])
                    lv  = Level(KaitaiStream(BytesIO(raw)))
                    levels.append(level_to_dict(lv, ex["name"]))
                except Exception:
                    continue
 
            self.result = levels
            self.status.config(text=f"Done — {len(levels)} levels loaded.")
            self.update()
            self.after(800, self.destroy)
        except Exception as e:
            messagebox.showerror("Dataset error", str(e), parent=self)
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = MM2Viewer()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        input_path  = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None

        if input_path.lower().endswith(".txt"):
            # ----------------------------------------------------------------
            # ASCII .txt level -> convert to VGLC, optionally save, then view
            # ----------------------------------------------------------------
            try:
                import importlib.util
                _conv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm2view_to_vglc.py")
                _spec = importlib.util.spec_from_file_location("mm2view_to_vglc", _conv_path)
                _mod  = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)

                with open(input_path, encoding="utf-8") as f:
                    lines = f.readlines()

                vglc_rows = _mod.convert_level(lines)
                vglc_text = "\n".join(vglc_rows) + "\n"

                if output_path:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(vglc_text)
                    print(f"VGLC level saved to: {output_path}")
                else:
                    print(vglc_text, end="")

            except Exception as e:
                print(f"Could not convert {input_path}: {e}")

        else:
            # ----------------------------------------------------------------
            # JSON export -> load into viewer as before
            # ----------------------------------------------------------------
            try:
                with open(input_path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                app.levels = data
                app.current_idx = 0
                app.after(100, app._redraw)
            except Exception as e:
                print(f"Could not load {input_path}: {e}")

    app.protocol("WM_DELETE_WINDOW", lambda: (app.destroy(), sys.exit(0)))
    app.mainloop()