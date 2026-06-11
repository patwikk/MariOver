"""
Convert an MM2 level JSON (the toost / level.py export consumed by
json_to_bcd.py) into a Super Mario Maker Worldwide Engine (.swe) save file.

Background
----------
A .swe file is just:

    base64( UTF-8 JSON )  +  40-char lowercase hex HMAC-SHA1 checksum

The checksum is HMAC-SHA1 of the *base64 text* (not the decoded bytes) keyed
with the literal string "2559F35097-2021" (the key shipped in EngineTribe's
SMMWESaveDecryptor; verified here against a real awesome.swe). SMMWE rejects
the file if the checksum doesn't match, so we recompute it exactly.

The decoded JSON has two worlds:

    {"S0": {...overworld...}, "SB1": {...subworld...}}

Each world is a dict of sections:

    S1  list with ONE level-metadata dict (gamestyle, theme, timer, goal, ...)
    S2  ground/terrain tiles      {xx, yy, i}
    S3  decorations               {xx, yy, i, ID, spr}      (left empty here)
    S4  objects / enemies         {xx, yy, ID, scl, dir, ...many flags}
    S5  S6  S7  S8                other categories           (left empty here)

An unused subworld is written as SB1 = {"S1": []}, matching the editor.

Coordinate systems
-------------------
MM2 JSON                                   SWE
--------                                   ---
ground x,y   tile units, y up from bottom  xx,yy in PIXELS (16 px / tile),
objects x,y  sub-pixels, 160 / tile,         y DOWN from top
             centered, y up from bottom
goal_x       tenths of a tile (goal_x//10)
goal_y       tiles from bottom
boundaries   pixels (16 / tile), top=432

The playfield is 27 tiles (432 px) tall, so a tile at row `r` (from the
bottom) maps to swe_yy = (27 - 1 - r) * 16. Object columns use the viewer's
formula  col = x//160 - w//2 ,  rows  row = y//160  (see mm2_viewer_json.py).

Lossy / best-effort
-------------------
This is a "best effort" conversion, not a perfect round trip:

  * Ground autotiling. SWE stores a resolved tile-graphic index `i` per cell.
    We recompute `i` from 4-neighbour occupancy using a table reverse-
    engineered from a sample level. SMMWE picks random decorative variants,
    so our indices are representative, not identical, but load fine.
  * Object IDs. MM2 uses integer ids (0-132); SWE uses string ids
    (obj_*_res). OBJ_ID_MAP covers the objects SMMWE actually has; anything
    SMMWE lacks (clear pipes, snake/track blocks, most koopalings, tree,
    crate, ...) is dropped with a warning.
  * Object flags. MM2 packs orientation/wings/parachute/etc. into `flag` /
    `cflag` bitfields whose per-object meaning isn't fully documented. We
    emit the SWE flag fields as 0 (default) and only set scl / a best-effort
    direction. Wings, parachutes, etc. are not carried over.
  * Decorations (S3) and the slope/track object families are not emitted.

Usage
-----
    python json_to_swe.py bcd_levels/json/3000009_overworld.json
    python json_to_swe.py bcd_levels/json/3000009_overworld.json -o awesome.swe
    python json_to_swe.py 3000009_overworld.json --user patwick --name "My Level"

If a matching *_subworld.json sits next to the overworld file it is converted
into SB1 automatically.
"""

import argparse
import base64
import hashlib
import hmac
import json
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWE_HMAC_KEY      = b"2559F35097-2021"   # EngineTribe SMMWESaveDecryptor key
FIELD_HEIGHT_TILES = 27                  # playfield height (432 px / 16)
PX                = 16                    # pixels per tile in SWE
SUBPX             = 160                   # sub-pixels per tile in MM2 object coords

# gamestyle_raw (MM2) -> SWE gamestyle int. SMMWE has SMB1/SMB3/SMW/NSMBU;
# SM3DW has no SMMWE equivalent, so it falls back to NSMBU.
GAMESTYLE_MAP = {
    12621: 0,   # SMB1
    13133: 1,   # SMB3
    22349: 2,   # SMW
    21847: 3,   # NSMBU
    22323: 3,   # SM3DW -> NSMBU (closest)
}

# theme_raw (MM2, 0-9) -> SWE gametheme string.
THEME_MAP = {
    0: "overworld",
    1: "underground",
    2: "castle",
    3: "airship",
    4: "underwater",
    5: "ghost",
    6: "snow",
    7: "desert",
    8: "sky",
    9: "forest",
}

# 4-neighbour occupancy mask (N=1, E=2, S=4, W=8) -> SWE ground tile index `i`.
# Reverse-engineered from a sample level's S2 array (the most common variant
# per neighbour configuration). Masks the sample never exercised (single
# isolated neighbour) use sensible caps. `i` is the autotile *slot*, which is
# theme-independent (the sprite sheet changes per theme, the slot doesn't).
GROUND_AUTOTILE = {
    0:  2,    # isolated block
    1:  21,   # N only          (bottom cap of a vertical strip)
    2:  9,    # E only          (left cap of a horizontal strip)
    3:  18,   # N+E             (bottom-left outer corner)
    4:  53,   # S only          (top cap / grass top)
    5:  45,   # N+S             (vertical middle)
    6:  9,    # E+S             (top-left corner)
    7:  61,   # N+E+S           (left edge)
    8:  3,    # W only          (right cap of a horizontal strip)
    9:  11,   # N+W             (bottom-right outer corner)
    10: 54,   # E+W             (horizontal middle)
    11: 43,   # N+E+W           (bottom edge)
    12: 10,   # S+W             (top-right corner)
    13: 62,   # N+S+W           (right edge)
    14: 53,   # E+S+W           (top edge / grass top)
    15: 43,   # all four        (interior fill)
}

# MM2 object id (0-132, see mm2_json_field_dictionary.txt) -> SWE obj_*_res.
# None  => SMMWE has no equivalent; the object is dropped (with a warning).
# Entries marked "approx" pick the closest available SMMWE object.
OBJ_ID_MAP = {
    0:   "obj_goomba_res",          # Goomba
    1:   "obj_koopa_res",           # Koopa Troopa
    2:   "obj_pplant_res",          # Piranha Plant
    3:   "obj_hammerbro_res",       # Hammer Bro
    4:   "obj_block_res",           # Brick Block
    5:   "obj_qblock_res",          # Question Block
    6:   "obj_rock_res",            # Hard Block (verified: MM2 "Stone" id=6)
    7:   "obj_rock_res",            # Ground-as-object (approx; terrain is S2)
    8:   "obj_coin_res",            # Coin
    9:   None,                      # Pipe -> emitted to S5 (see build_pipes)
    10:  "obj_spring_res",          # Spring / Trampoline
    11:  "obj_platform_res",        # Lift (approx)
    12:  "obj_thwomp_res",          # Thwomp
    13:  "obj_bullebill_base_res",  # Bullet Bill Blaster
    14:  "obj_mushroom_platform_res",  # Mushroom Platform
    15:  "obj_bobomb_res",          # Bob-omb
    16:  "obj_platform_res",        # Semisolid Platform (approx)
    17:  "obj_puente_res",          # Bridge
    18:  "obj_pswitch_res",         # P Switch
    19:  "obj_pow_res",             # POW Block
    20:  "obj_mushroom_res",        # Super Mushroom
    21:  "obj_donut_res",           # Donut Block
    22:  "obj_nube_res",            # Cloud
    23:  "obj_noteblock_res",       # Note Block
    24:  "obj_firebar_res",         # Fire Bar
    25:  "obj_spiny_res",           # Spiny
    26:  None,                      # Goal Ground (terrain; goal is in S1)
    27:  None,                      # Goal (flagpole; stored in S1 goal_x/y)
    28:  "obj_buzzybeetle_res",     # Buzzy Beetle
    29:  "obj_block_hidden_res",    # Hidden Block
    30:  "obj_lakitu_res",          # Lakitu
    31:  "obj_nube_res",            # Lakitu's Cloud (approx)
    32:  "obj_billbanzai_res",      # Banzai Bill
    33:  "obj_1up_res",             # 1-Up Mushroom
    34:  "obj_fireflower_res",      # Fire Flower
    35:  "obj_star_res",            # Super Star
    36:  "obj_lava_lift_res",       # Lava Lift
    37:  None,                      # Starting Brick (editor marker)
    38:  "obj_arrow_res",           # Starting Arrow
    39:  "obj_magikoopa_res",       # Magikoopa
    40:  "obj_spike_res",           # Spike Top (approx)
    41:  "obj_boo_res",             # Boo
    42:  "obj_clown_res",           # Clown Car
    43:  "obj_pinchos_res",         # Spike Trap
    44:  "obj_mushroom_res",        # Big Mushroom (approx; see scl)
    45:  None,                      # Goomba's Shoe (no SMMWE equiv)
    46:  "obj_drybones_res",        # Dry Bones
    47:  "obj_cannon_res",          # Cannon
    48:  "obj_blooper_res",         # Blooper
    49:  "obj_puente_res",          # Castle Bridge (approx)
    50:  "obj_spring_res",          # Jumping Machine (approx)
    51:  None,                      # Skipsqueak
    52:  None,                      # Wiggler
    53:  "obj_cinta_res",           # Fast Conveyor Belt
    54:  "obj_soplete_res",         # Burner
    55:  "obj_door_res",            # Door
    56:  "obj_cheepcheep_res",      # Cheep Cheep
    57:  "obj_muncher_res",         # Muncher
    58:  "obj_rocky_res",           # Rocky Wrench
    59:  "obj_rails_res",           # Track / rail
    60:  "obj_podoboo_res",         # Lava Bubble
    61:  "obj_chomp_res",           # Chain Chomp
    62:  "obj_bowser_res",          # Bowser
    63:  "obj_ice_res",             # Ice Block
    64:  "obj_vine_res",            # Vine
    65:  None,                      # Stingby
    66:  "obj_arrow_res",           # Arrow (decoration)
    67:  "obj_oneway_res",          # One-Way Wall
    68:  "obj_grinder_res",         # Saw
    69:  None,                      # Player spawn (stored in S1 start_y)
    70:  "obj_coin10_res",          # Big Coin (10-coin)
    71:  "obj_platform_res",        # Half-Collision Platform (approx)
    72:  "obj_clown_res",           # Koopa Car (approx)
    73:  None,                      # Cinobio
    74:  "obj_spike_ball_res",      # Spike Ball
    75:  "obj_rock_res",            # Stone Block
    76:  "obj_torbellino_res",      # Twister
    77:  "obj_boomboom_res",        # Boom Boom
    78:  "obj_pokey_res",           # Pokey
    79:  "obj_pblock_res",          # P Block
    80:  "obj_expandplatf_res",     # Sprint Platform (approx)
    81:  "obj_SMB2_mushroom_res",   # SMB2 Mushroom
    82:  "obj_donut_res",           # Donut Block Platform
    83:  "obj_skewer_res",          # Skewer
    84:  None,                      # Snake Block
    85:  None,                      # Track Block
    86:  "obj_floruga_res",         # Charvaargh (approx)
    87:  None,                      # Slight Slope (terrain)
    88:  None,                      # Steep Slope (terrain)
    89:  None,                      # Reel Camera (cutscene marker)
    90:  "obj_checkpoint_res",      # Checkpoint Flag
    91:  "obj_seesaw_res",          # Seesaw
    92:  "obj_pink_coin_res",       # Red Coin (approx -> pink coin)
    93:  None,                      # Clear Pipe
    94:  "obj_cinta_res",           # Conveyor Belt
    95:  "obj_key_res",             # Key
    96:  None,                      # Ant Trooper
    97:  None,                      # Warp Box
    98:  "obj_bowserjr_res",        # Bowser Jr.
    99:  "obj_onoffblock_res",      # ON/OFF Block
    100: None,                      # Dotted-Line Block
    101: None,                      # Water Marker (liquid is in S1 wl)
    102: "obj_monty_res",           # Monty Mole
    103: "obj_fishbone_res",        # Fish Bone
    104: "obj_angrysun_res",        # Angry Sun
    105: "obj_claw_res",            # Swinging Claw
    106: None,                      # Tree (decoration)
    107: None,                      # Piranha Creeper
    108: None,                      # Blinking Block
    109: None,                      # Sound Effect marker
    110: "obj_pinchos_res",         # Spike Block (approx)
    111: "obj_mechakoopa_res",      # Mechakoopa
    112: None,                      # Crate
    113: "obj_spring_res",          # Mushroom Trampoline (approx)
    114: None,                      # Porcupuffer
    115: None,                      # Cinobic
    116: None,                      # Super Hammer
    117: None,                      # Bully
    118: "obj_icicle_res",          # Icicle
    119: None,                      # ! Block
    120: "obj_ludwig_res",          # Lemmy   (approx -> only Ludwig in SMMWE)
    121: "obj_ludwig_res",          # Morton  (approx)
    122: "obj_ludwig_res",          # Larry   (approx)
    123: "obj_ludwig_res",          # Wendy   (approx)
    124: "obj_ludwig_res",          # Iggy    (approx)
    125: "obj_ludwig_res",          # Roy     (approx)
    126: "obj_ludwig_res",          # Ludwig
    127: None,                      # Cannon Box
    128: None,                      # Propeller Box
    129: "obj_cap_res",             # Goomba Mask (approx)
    130: "obj_cap_res",             # Bullet Bill Mask (approx)
    131: "obj_pow_res",             # Red POW Box (approx)
    132: "obj_spring_res",          # ON/OFF Trampoline (approx)
}

# MM2 pipe direction (flag % 0x80) -> orientation, used by build_pipes to pick
# which axis (xscl vs yscl) carries the pipe's length and to orient its
# footprint anchor.
MM2_PIPE_DIR = {0x00: "R", 0x20: "L", 0x40: "U", 0x60: "D"}

# SWE S5 pipe `dir` (0/1/2/3 = U/R/D/L, clockwise) and matching `rot`,
# verified against four hand-placed length-4 pipes (see build_pipes).
PIPE_DIR_MAP = {"U": 0, "R": 1, "D": 2, "L": 3}
PIPE_ROT = {0: 0, 1: -90, 2: 180, 3: -270}

# Constant/default fields for an S5 pipe entry, taken verbatim from a
# hand-placed "vertical, length 4" pipe saved by the SMMWE editor (see
# build_pipes). Per-pipe code overrides sz/sclx/rot/xscl/yscl/dir/xx/yy/
# t_x_pos/t_y_pos.
S5_PIPE_TEMPLATE = {
    "sz": 0, "t_dir": 0, "clr": 0, "sclx": 1, "t_rot": 0, "wrp": 0,
    "t_s_sclx": 1, "msk": 0, "xscl": 1, "t_yscl": 1, "rot": 0, "t_sz": 0,
    "t_clr": 0, "yscl": 1, "t_xscl": 1, "dir": 0,
}

# Some object types render one tile lower than the generic formula predicts
# (their SMMWE anchor differs from the generic top-left-cell convention).
# Value is a pixel delta subtracted from yy (positive = move up on screen,
# since SWE yy grows downward). Confirmed: Saw (68) and Big Coin (70) both
# need to move up by one tile.
OBJ_Y_OFFSET_PX = {
    68: PX,   # Saw -> obj_grinder_res
    70: PX,   # Big Coin (10-coin) -> obj_coin10_res
}

# Every key an S4 object dict carries (from a real .swe). All flags default to
# 0; we only fill ID / xx / yy / scl / dir.
S4_TEMPLATE = {
    "air": 0, "pinkcoin": 0, "fire": 0, "claw": 0, "key": 0, "rock": 0,
    "sprout": 0, "energy": 0, "wings": 0, "w_mode": 0, "rot": 0,
    "can_complement": 0, "parachute": 0, "progress": 0, "sierra": 0,
    "bumper": 0, "inup": 0, "ice": 0,
}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def ground_xx(tile_x):
    return tile_x * PX


def ground_yy(tile_y):
    """tile_y is rows-from-bottom; SWE yy is pixels-from-top."""
    return (FIELD_HEIGHT_TILES - 1 - tile_y) * PX


def marker_yy(tile_y):
    """Start/goal markers sit one row lower than the ground formula (their
    anchor differs from terrain tiles in SMMWE)."""
    return (FIELD_HEIGHT_TILES - tile_y) * PX


def object_cell(o):
    """Return (col, row_from_bottom) for an MM2 object, matching the viewer's
    formula:  col = x//160 - w//2 ,  row = y//160 ."""
    w = max(1, o.get("w", 1))
    col = o["x"] // SUBPX - w // 2
    row = o["y"] // SUBPX
    return col, row


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_ground(ground):
    """MM2 ground[] -> SWE S2, recomputing the autotile index from 4-neighbour
    occupancy."""
    occ = {(g["x"], g["y"]) for g in ground}
    out = []
    for g in ground:
        x, y = g["x"], g["y"]
        mask = (
            ((x, y + 1) in occ) << 0   # N: a higher row (y+1) is "above"
            | ((x + 1, y) in occ) << 1  # E
            | ((x, y - 1) in occ) << 2  # S: lower row (y-1) is "below"
            | ((x - 1, y) in occ) << 3  # W
        )
        out.append({"xx": ground_xx(x), "yy": ground_yy(y), "i": GROUND_AUTOTILE[mask]})
    return out


def build_pipes(objects):
    """MM2 pipes (id 9) -> SWE S5 entries.

    Reverse-engineered from four hand-placed length-4 pipes (one per
    direction) saved directly from the SMMWE editor -- both earlier S5
    endpoint-based schemes and an S4 "obj_tuberia_res" attempt were wrong.
    The four samples (sz, xscl/yscl, sclx, rot, dir, t_x_pos-xx, t_y_pos-yy):

        U: sz=2 yscl=2 xscl=1 sclx=1  rot=0    dir=0  t_x-xx=32 t_y-yy=32
        R: sz=2 yscl=1 xscl=2 sclx=1  rot=-90  dir=1  t_x-xx=32 t_y-yy=0
        D: sz=2 yscl=2 xscl=1 sclx=-1 rot=180  dir=2  t_x-xx=32 t_y-yy=0
        L: sz=2 yscl=1 xscl=2 sclx=-1 rot=-270 dir=3  t_x-xx=32 t_y-yy=0

    All four were length 4 (sz = length - 2 = 2). The pattern: `dir` is
    0/1/2/3 for U/R/D/L (clockwise), rot = -90*dir (dir=2 shown as +180),
    sclx flips to -1 for the "second half" (D/L), the length axis is yscl
    for vertical (U/D) and xscl for horizontal (R/L), t_x_pos is always
    xx+32, and t_y_pos is yy+32 only for U (else yy).

    (xx,yy) is the TOP-LEFT corner of the pipe's MM2 tile footprint, per
    obj_anchor()/obj_tile_size() in mm2_viewer_json.py (no `-w//2`
    correction; `length` always comes from `h` regardless of direction).
    This anchor convention is in-game verified for U. For R/D/L it is
    derived by applying the same bottom-left -> top-left transform that
    obj_anchor()/obj_tile_size() use for those directions (e.g. R's
    bottom-left (base_col, base_row-1) with a 2-row-tall footprint gives
    top-left (base_col, base_row), matching the formula below) -- but the
    resulting absolute placement against terrain is not yet in-game
    verified for R/D/L."""
    out = []
    for o in objects:
        if o.get("id") != 9:
            continue
        length = max(1, o.get("h", 1))
        base_col = o["x"] // SUBPX
        base_row = o["y"] // SUBPX
        direction = MM2_PIPE_DIR.get(o.get("flag", 0) % 0x80, "R")

        if direction == "U":
            col, row_top = base_col, base_row + length - 1
        elif direction == "D":
            col, row_top = base_col - 1, base_row
        elif direction == "R":
            col, row_top = base_col, base_row
        else:  # L
            col, row_top = base_col - length + 1, base_row + 1

        xx = col * PX
        yy = ground_yy(row_top)
        dir_idx = PIPE_DIR_MAP[direction]

        entry = dict(S5_PIPE_TEMPLATE)
        if dir_idx % 2 == 0:   # U, D
            entry["yscl"] = length / 2
        else:                  # R, L
            entry["xscl"] = length / 2
        entry.update({
            "sz": length - 2,
            "sclx": -1 if dir_idx >= 2 else 1,
            "rot": PIPE_ROT[dir_idx],
            "dir": dir_idx,
            "xx": xx, "yy": yy,
            "t_x_pos": xx + 2 * PX,
            "t_y_pos": yy + 2 * PX if direction == "U" else yy,
        })
        out.append(entry)
    return out


def build_objects(objects):
    """MM2 objects[] -> SWE S4. Returns (s4_list, dropped_counts).
    Pipes (id 9) are handled separately by build_pipes and skipped here."""
    out = []
    dropped = {}
    for o in objects:
        oid = o.get("id")
        if oid == 9:
            continue
        swe_id = OBJ_ID_MAP.get(oid)
        if swe_id is None:
            name = o.get("name", f"id={oid}")
            dropped[name] = dropped.get(name, 0) + 1
            continue
        col, row = object_cell(o)
        # Big variants (Big Mushroom etc.) -> scale 2, everything else 1.
        scl = 2 if oid == 44 else 1
        entry = dict(S4_TEMPLATE)
        entry.update({
            "ID": swe_id,
            "xx": col * PX,
            "yy": (FIELD_HEIGHT_TILES - 1 - row) * PX - OBJ_Y_OFFSET_PX.get(oid, 0),
            "scl": scl,
            "dir": 0,   # best-effort: MM2 flag bitfields aren't carried over
        })
        out.append(entry)
    return out, dropped


def build_metadata(j, *, user, name, desc, date_str, time_str):
    """MM2 level header -> SWE S1 metadata dict."""
    gamestyle_raw = j.get("gamestyle_raw", 0)
    theme_raw = j.get("theme_raw", 0)

    goal_x_tenths = j.get("goal_x", 0)            # tenths of a tile
    goal_col = goal_x_tenths // 10
    goal_y_tiles = j.get("goal_y", 0)             # tiles from bottom
    start_y_tiles = j.get("start_y", 0)           # tiles from bottom

    # Level width in pixels: prefer the stored boundary, else span the content.
    right = j.get("right_boundary", 0)
    if right > 0:
        size = right
    else:
        cols = [g["x"] for g in j.get("ground", [])]
        cols += [o["x"] // SUBPX for o in j.get("objects", [])]
        size = (max(cols) + 2) * PX if cols else FIELD_HEIGHT_TILES * PX

    # Liquid / water level.
    liquid_start = j.get("liquid_start_height", 0)
    if liquid_start:
        wl = (FIELD_HEIGHT_TILES - liquid_start) * PX
    else:
        wl = (FIELD_HEIGHT_TILES - 1) * PX
    wl_speed_map = {0: 0.0, 1: 0.2, 2: 0.4, 3: 0.6}
    wl_speed = wl_speed_map.get(j.get("liquid_speed_raw", 0), 0.0)

    return {
        "goal_y": marker_yy(goal_y_tiles),
        "nightmode": 1 if j.get("night_time") else 0,
        "label_1": 0,
        "autoscroll": 1 if j.get("autoscroll_type_raw", 0) else 0,
        "t_conditions": 0,
        "wl": wl,
        "mtrs": 0,
        "wl_speed": wl_speed,
        "c_conditions": 0,
        "conditions": 0,
        "t": 1,
        "start_y": marker_yy(start_y_tiles),
        "date": date_str,
        "goal_x": goal_col * PX,
        "size": size,
        "gamestyle": GAMESTYLE_MAP.get(gamestyle_raw, 3),
        "ds_s": 0,
        "desc": desc if desc is not None else (j.get("description", "") or ""),
        "label_2": -1,
        "user": user,
        "time": time_str,
        "gametheme": THEME_MAP.get(theme_raw, "overworld"),
        "wl_limit": wl,
        "o_conditions": 0,
        "timer": j.get("timer", 0),
    }


def build_world(j, *, user, name, desc, date_str, time_str):
    """Build a full SWE world dict (S0 or a populated SB1) from one map JSON."""
    objects = j.get("objects", [])
    s4, dropped = build_objects(objects)
    s5 = build_pipes(objects)

    world = {
        "S1": [build_metadata(j, user=user, name=name, desc=desc,
                              date_str=date_str, time_str=time_str)],
        "S2": build_ground(j.get("ground", [])),
        "S3": [],
        "S4": s4,
        "S5": s5,
        "S6": [],
        "S7": [],
        "S8": [],
    }
    return world, dropped


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_swe(level_dict):
    """level_dict -> .swe file bytes (base64 JSON + HMAC-SHA1 hex)."""
    payload = json.dumps(level_dict, separators=(",", ":"), ensure_ascii=False)
    b64 = base64.b64encode(payload.encode("utf-8"))
    checksum = hmac.new(SWE_HMAC_KEY, b64, hashlib.sha1).hexdigest()
    return b64 + checksum.encode("ascii")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_companion(path: Path):
    """Given an *_overworld.json / *_subworld.json, return
    (overworld_path, subworld_path, base_stem) with missing files as None."""
    stem = path.stem
    if stem.endswith("_overworld"):
        base = stem[: -len("_overworld")]
        ow, sub = path, path.with_name(base + "_subworld" + path.suffix)
    elif stem.endswith("_subworld"):
        base = stem[: -len("_subworld")]
        ow, sub = path.with_name(base + "_overworld" + path.suffix), path
    else:
        base = stem
        ow, sub = path, None
    ow = ow if ow and ow.exists() else None
    sub = sub if sub and sub.exists() else None
    return ow, sub, base


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert an MM2 level JSON into a Super Mario Maker "
                    "Worldwide Engine (.swe) save file."
    )
    p.add_argument("json_path", help="Path to a *_overworld.json (or plain) level JSON")
    p.add_argument("-o", "--output", help="Output .swe path (default: <stem>.swe)")
    p.add_argument("--user", default="MariOver", help="Author name stored in the level")
    p.add_argument("--name", default=None, help="Level name (default: JSON 'name')")
    p.add_argument("--desc", default=None, help="Description (default: JSON 'description')")
    p.add_argument("--height", type=int, default=FIELD_HEIGHT_TILES,
                   help=f"Playfield height in tiles (default {FIELD_HEIGHT_TILES})")
    return p.parse_args()


def main():
    global FIELD_HEIGHT_TILES
    args = parse_args()
    FIELD_HEIGHT_TILES = args.height

    in_path = Path(args.json_path)
    ow_path, sub_path, base = find_companion(in_path)
    if ow_path is None and sub_path is None:
        raise SystemExit(f"Could not find JSON data at {in_path}")

    overworld_json = load_json(ow_path) if ow_path else load_json(in_path)
    subworld_json = load_json(sub_path) if sub_path else None

    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    time_str = now.strftime("%H:%M")
    name = args.name if args.name is not None else overworld_json.get("name", base)

    s0, dropped = build_world(overworld_json, user=args.user, name=name,
                              desc=args.desc, date_str=date_str, time_str=time_str)

    if subworld_json is not None:
        sb1, sub_dropped = build_world(subworld_json, user=args.user, name=name,
                                       desc=args.desc, date_str=date_str, time_str=time_str)
        for k, v in sub_dropped.items():
            dropped[k] = dropped.get(k, 0) + v
    else:
        sb1 = {"S1": []}   # empty subworld, as the editor writes it

    level = {"S0": s0, "SB1": sb1}
    data = encode_swe(level)

    out_path = Path(args.output) if args.output else in_path.with_name(base + ".swe")
    out_path.write_bytes(data)

    print(f"Wrote {out_path} ({len(data)} bytes)")
    print(f"  ground tiles : {len(s0['S2'])}")
    print(f"  objects      : {len(s0['S4'])}")
    print(f"  pipes        : {len(s0['S5'])}")
    if dropped:
        total = sum(dropped.values())
        print(f"  dropped {total} object(s) with no SMMWE equivalent:")
        for name_, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"      {n:4d}  {name_}")


if __name__ == "__main__":
    main()
