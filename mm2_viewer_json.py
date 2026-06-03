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
import json, sys, os, math

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

def get_meta(name: str):
    return OBJ_META.get(name, OBJ_META["_unknown"])


# ---------------------------------------------------------------------------
# Tile size helper — uses w/h from JSON directly (already tile counts)
# ---------------------------------------------------------------------------
def obj_tile_size(obj: dict):
    """Return (w_tiles, h_tiles). The JSON w/h fields are direct tile counts."""
    w = max(1, obj.get("w", 1))
    h = max(1, obj.get("h", 1))
    return w, h


def obj_anchor(obj: dict):
    """Return (col, row) — bottom-left tile of the object."""
    return obj["x"] // 160, obj["y"] // 160


# ---------------------------------------------------------------------------
# Main viewer window
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
        self._cat_vars    = {}
        self._tooltip_win = None

        self._build_ui()

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

        # 1. First, parse the explicit terrain blocks from the clean 'ground' array
        for g in lvl.get("ground", []):
            objects.append({
                "name": "Ground",
                "x":    g["x"] * 160,   # tile col → sub-pixels
                "y":    g["y"] * 160,   # tile row → sub-pixels (y=0 at bottom)
                "w":    1,
                "h":    1,
            })

        # 2. Add the Start block structure
        start_y = lvl.get("start_y", 0)
        objects.append({
            "name": "Starting Brick",
            "x": 0,
            "y": start_y * 160,
            "w": 3,
            "h": 1,
        })

        # Add 4 vertical columns (strips) of ground at the start (cols 0, 1, 2, 3)
        # filled from row 0 up to start_y
        for col in range(0, 7):
            for row in range(0, start_y):
                objects.append({
                    "name": "Ground",
                    "x":    col * 160,
                    "y":    row * 160,
                    "w":    1,
                    "h":    1,
                })

        # 3. Add the Goal flagpole structure
        goal_x = lvl.get("goal_x", 0)
        goal_y = lvl.get("goal_y", 0)
        goal_col = goal_x // 10
        
        objects.append({
            "name": "Goal",
            "x": goal_col * 160,
            "y": goal_y * 160,
            "w": 1,
            "h": 5,
        })

        # Add 9 vertical columns (strips) of ground extending rightward from the goal position
        # filled from row 0 up to goal_y
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
        self._redraw()

    def _active_cats(self):
        return {cat for cat, v in self._cat_vars.items() if v.get()}

    # --------------------------------------------------------------- drawing --


    def _grid_bounds(self, lvl):
        """Compute (max_tx, max_ty) from the level's object extents."""
        objects = lvl.get("objects", [])
        max_tx, max_ty = 40, 20
        for o in objects:
            col, row = obj_anchor(o)
            w, h = obj_tile_size(o)
            max_tx = max(max_tx, col + w + 1)
            max_ty = max(max_ty, row + h + 1)
        max_tx = min(max_tx, self.MAX_COLS)
        max_ty = min(max_ty, self.MAX_ROWS)
        return max_tx, max_ty

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
                    if cat not in active:
                        continue
                    col, row = obj_anchor(obj)
                    w, h = obj_tile_size(obj)
                    if col >= max_tx or row >= max_ty:
                        continue
                    px0 = col * ts + pad
                    py1 = (max_ty - 1 - row) * ts + ts - pad
                    px1 = (col + w) * ts - pad
                    py0 = (max_ty - 1 - (row + h - 1)) * ts + pad
                    outline_col = "#888888" if is_bg else "#000000"
                    self.canvas.create_rectangle(px0, py0, px1, py1,
                                                 fill=color, outline=outline_col)
                    if show_lbl:
                        self.canvas.create_text((px0 + px1) // 2, (py0 + py1) // 2,
                                                text=char, fill="white",
                                                font=("Courier", max(ts // 2, 7), "bold"))

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
                col, row = obj_anchor(obj)
                w, h = obj_tile_size(obj)
                for dc in range(w):
                    for dr in range(h):
                        set_cell(col + dc, row + dr, char)

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
                if ch == GROUND_CHAR: bg = "#8B6914"
                elif ch == " ":       bg = "#111111"
                else:                 bg = "#222222"
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
                app._normalize_level(lvl)
                
            app.levels = data
            app.current_idx = 0
            app.after(100, app._redraw)
        except Exception as e:
            print(f"Could not load {sys.argv[1]}: {e}")

    app.protocol("WM_DELETE_WINDOW", lambda: (app.destroy(), sys.exit(0)))
    app.mainloop()