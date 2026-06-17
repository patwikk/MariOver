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
    S5  pipes                     {xx, yy, dir, sz, xscl, yscl, ...}
    S6  "obj_cannon_res" entries  Cannon: {xx, yy, dir, rht, dwn, up, lft}
    S7  "stretchy sprite" objects Bullet Bill Blaster / Semisolid Platform:
                                   {xx, yy, dir, spr, wth, hht, dph, ...}
    S8                             (left empty here)

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
OBJ_LEFT_ANCHOR_IDS use  col = x//160  (no -w//2) instead. For
OBJ_H_ANCHOR_TOP_IDS, row is additionally shifted up by (h-1) tiles, since
their SWE sprite sits at the top of the object's h-tall MM2 footprint, not
its bottom row.

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
  * Bullet Bill Blaster (13), Semisolid Platform (16) and Cannon (47) are
    NOT S4 objects in SMMWE -- they're emitted to S6/S7 by build_cannons /
    build_platform_objects. Confusingly, SMMWE's internal names are swapped
    relative to MM2: the Cannon is "obj_cannon_res" (S6) and the Bullet Bill
    Blaster is "obj_bullebill_base_res" (S7), verified in-game against
    hand-placed reference saves. The Cannon's S6 entry has no size field, so
    its MM2 `h` is lost.
  * Object flags. MM2 packs orientation/wings/parachute/etc. into `flag` /
    `cflag` bitfields whose per-object meaning isn't fully documented. We
    emit the SWE flag fields as 0 (default) and only set scl / a best-effort
    direction. Wings, parachutes, etc. are not carried over.
  * Decorations (S3) and the track object families are not emitted. Slight/
    Steep Slopes (87/88) have no diagonal SWE equivalent, so their full
    w x h bounding box is filled in as solid S2 ground instead.

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
    # 11 Lift -> S7 "obj_platform_res", see build_platform_objects
    12:  "obj_thwomp_res",          # Thwomp
    # 13 Bullet Bill Blaster -> S7 "obj_bullebill_base_res", see build_platform_objects
    # 14 Mushroom Platform -> S7 "obj_mushroom_platform_res", see build_platform_objects
    14:  None,
    15:  "obj_bobomb_res",          # Bob-omb
    # 16 Semisolid Platform -> S7 "obj_semisolid_platform1", see build_platform_objects
    # 17 Bridge -> S7 "obj_puente_res", see build_platform_objects / PLATFORM_S7_IDS
    17:  None,
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
    31:  "obj_clown_res",           # Lakitu's Cloud -> Clown Car (nearest rideable vehicle)
    32:  "obj_billbanzai_res",      # Banzai Bill
    33:  "obj_1up_res",             # 1-Up Mushroom
    34:  "obj_fireflower_res",      # Fire Flower
    35:  "obj_star_res",            # Super Star
    36:  "obj_lava_lift_res",       # Lava Lift
    37:  None,                      # Starting Brick (editor marker)
    38:  "obj_arrow_res",           # Starting Arrow
    39:  "obj_magikoopa_res",       # Magikoopa
    40:  "obj_spiny_res",           # Spike Top -> Spiny (approx)
    41:  "obj_boo_res",             # Boo
    42:  None,                      # Clown Car -> obj_clown_res / obj_clown_fire_res, see ONOFF_SWE_IDS
    43:  "obj_pinchos_res",         # Spike Trap
    44:  None,                      # Style Power-up A -> see STYLE_POWERUP_A_MAP (gamestyle-dependent)
    45:  "obj_egg_res",              # Shoe Goomba / Yoshi's Egg -> obj_egg_res (green Yoshi); works for SMB1/SMB3/SMW/NSMBU
    46:  "obj_drybones_res",        # Dry Bones
    # 47 Cannon -> S6 "obj_cannon_res", see build_cannons
    48:  "obj_blooper_res",         # Blooper
    # 49 Castle Bridge (approx) -> S7 "obj_puente_res", see build_platform_objects / PLATFORM_S7_IDS
    49:  None,
    50:  "obj_spring_res",          # Jumping Machine (approx)
    51:  None,                      # Skipsqueak
    52:  None,                      # Wiggler
    53:  "obj_cinta_res",           # Fast Conveyor Belt
    54:  "obj_soplete_res",         # Burner
    55:  None,                      # Door -> S8 paired entry, see build_doors
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
    # 81 Style Power-up B -> see STYLE_POWERUP_B_MAP (gamestyle-dependent)
    82:  "obj_donut_res",           # Donut Block Platform
    83:  "obj_skewer_res",          # Skewer
    84:  None,                      # Snake Block
    85:  None,                      # Track Block
    86:  "obj_floruga_res",         # Charvaargh (approx)
    87:  None,                      # Slight Slope (terrain)
    88:  None,                      # Steep Slope (terrain)
    89:  None,                      # Reel Camera (cutscene marker)
    90:  "obj_checkpoint_res",      # Checkpoint Flag
    # 91 Seesaw -> S7 stretchy "obj_seesaw_res", sized to the seesaw's
    # footprint, see PLATFORM_S7_IDS / build_platform_objects
    91:  None,
    92:  "obj_pink_coin_res",       # Red Coin (approx -> pink coin)
    93:  None,                      # Clear Pipe
    94:  "obj_cinta_res",           # Conveyor Belt
    95:  "obj_key_res",             # Key
    96:  None,                      # Ant Trooper
    97:  None,                      # Warp Box
    98:  "obj_bowserjr_res",        # Bowser Jr.
    99:  None,                      # ON/OFF Block -> obj_onoffblock_res/_blue_res, see ONOFF_SWE_IDS
    100: None,                      # Dotted-Line Block -> obj_onoffplatform_res/_blue_res, see ONOFF_SWE_IDS
    101: None,                      # Water Marker (liquid is in S1 wl)
    102: "obj_monty_res",           # Monty Mole
    103: "obj_cheepcheep_res",      # Fish Bone -> Cheep Cheep (approx)
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

# MM2 items inside blocks: MM2 block `cid` value -> SMMWE S4 `sprout` value.
# In MM2 JSON, a block's `cid` field directly stores the MM2 object id of the
# item inside (e.g. cid=20 = Super Mushroom, cid=44 = Style Power-up A).
# The item is NOT a separate entry in objects[]; only the block is in the list.
# In SMMWE, the block's S4 `sprout` field encodes the item.
# sprout=0 -> empty block. sprout=-1 -> multi-coin brick (from GML).
# Verified sprout values from hand-made SMMWE test levels:
#   1 = Super Mushroom (cid=20),  2 = Fire Flower (cid=34)
#   -77 = Style Power-up A (cid=44),  -85 = Style Power-up B (cid=81)
# Blocks with cid not in this map are emitted as empty blocks.
BLOCK_SPROUT_MAP = {
    8:   -1,   # Coin -> multi-coin block (from GML scr_edit_to_play)
    20:   1,   # Super Mushroom (verified)
    34:   2,   # Fire Flower (verified)
    44: -77,   # Style Power-up A / Cape / Mega Mario (verified)
    81: -85,   # Style Power-up B / Link (verified)
    # 33: ?,  # 1-Up Mushroom -- sprout value not yet verified
    # 35: ?,  # Super Star    -- sprout value not yet verified
}

# MM2 block ids that can contain items via cid link: Brick (4), Question (5), Hidden (29)
_BLOCK_IDS = {4, 5, 29}

# MM2 pipe direction (flag % 0x80) -> orientation, used by build_pipes to pick
# which axis (xscl vs yscl) carries the pipe's length and to orient its
# footprint anchor.
MM2_PIPE_DIR = {0x00: "R", 0x20: "L", 0x40: "U", 0x60: "D"}

# SWE S5 pipe `dir` (0/1/2/3 = U/R/D/L, clockwise) and matching `rot`,
# verified against four hand-placed length-4 pipes (see build_pipes).
PIPE_DIR_MAP = {"U": 0, "R": 1, "D": 2, "L": 3}
PIPE_ROT = {0: 0, 1: -90, 2: 180, 3: -270}

# obj_skewer_res `dir`/`rot`. dir=0 (the generic S4 default) is not a valid
# case in scr_edit_to_play and crashes on entering play mode, so Skewer must
# always get one of 1-4. MM2 flag bit 23 (1<<23) distinguishes ceiling-mounted
# skewers from floor-mounted ones. Mapping reverse-engineered by hand-placing
# all 4 of this level's skewers correctly in the SMMWE editor and inspecting
# the resulting .swe: bit23 clear -> dir=1/rot=0, bit23 set -> dir=3/rot=180.
# left/right-mounted skewers (dir=2/4) aren't reverse-engineered yet.
SKEWER_CEILING_FLAG = 1 << 23
SKEWER_DIR = {0: 1, SKEWER_CEILING_FLAG: 3}
SKEWER_ROT = {1: 0, 3: 180}

# obj_thwomp_res requires dir=1 to aggro (chase) the player. dir=0 (the
# generic S4 default) produces an inert thwomp that never moves. Verified
# against "correct thwomps.swe": all three hand-placed thwomps carry dir=1.
# Flag-based direction (up/left/right variants) is not yet reverse-engineered;
# dir=1 (face-down, the standard orientation) is used as a constant.
THWOMP_DIR = 1

# obj_oneway_res requires a 1-4 `direct_t` value (1/2/3/4 = right/top/
# left/bottom in the source's switch statement); the SWE S4 JSON has no
# literal "direct_t" key -- the loader derives it from the generic "dir"
# field instead. dir=0, the generic S4 default used for every other object,
# is NOT a valid case in that switch (no default case), so a One-Way Wall
# left at dir=0 becomes permanently invisible on entering play mode -- it's
# hidden (visible=false) and never replaced. MM2 encodes the 4 facings in
# flag bits 22-23 ((flag>>22)&0x3), confirmed to take exactly 4 distinct
# values across the dataset; this maps them 1:1 onto direct_t (raw
# 0/1/2/3 -> dir 1/2/3/4). Confirmed end-to-end against 4 hand-placed
# reference saves, one per direction ("one way functional (right).swe",
# "one way dir up/left/down.swe"): dir 1/2/3/4 = right/up/left/down exactly
# as mapped. Each reference also carries a `rot` value of 0/810/540/270
# respectively (== 0/90/180/270 mod 360, i.e. rot = (dir-1)*90) -- the S4
# entry needs this set too, or the gate may render facing the editor's
# default orientation instead of the rotated one.
ONEWAY_DIR_MAP = {0: 1, 1: 2, 2: 3, 3: 4}

# obj_billbanzai_res needs `dir`/`rot`/`scl` set or the spawned obj_banzaibill
# never moves: scr_edit_to_play copies direct = other.s_scaley, rotacion =
# other.rotacion, direct_t = other.direct_t onto the play object, and the
# generic S4 defaults (dir=0, rot=0, scl=1) apparently correspond to no
# launch direction at all -- the generic-S4 path left every Banzai Bill
# stationary. Confirmed against 4 hand-placed reference saves ("banzai
# right/up/left/down.swe"): dir 1/2/3/4 = left/down/right/up, rot =
# (dir-1)*90 (0/90/180/270, observed as 360/630/540/450 i.e. mod-360
# equivalent -- editor rotation accumulates past 360 but only the
# congruence class matters), and scl = 1 for left/down (dir 1/2) or -1 for
# right/up (dir 3/4). MM2 encodes the 4 facings in flag bits 22-23, the
# same bit field/encoding as One-Way Wall (ONEWAY_DIR_MAP); raw 0/1/2/3 is
# assumed to map onto dir 1/2/3/4 by the same convention, though that
# specific correspondence (vs. e.g. an offset/reversed mapping) isn't
# independently confirmed since the 4 reference saves were hand-placed in
# SMMWE directly rather than converted from a known MM2 source flag.
BANZAI_DIR_MAP = {0: 1, 1: 2, 2: 3, 3: 4}
BANZAI_SCL = {1: 1, 2: 1, 3: -1, 4: -1}

# ON/OFF Block (99) / Dotted-Line Block (100) -> SWE S4 "ID", keyed by MM2
# flag bit 2 (flag & 4), a shared red/blue "color" flag for both ids.
# Verified exhaustively against "3000209_6 but right.swe": all 73 id-99/100
# objects in that level (4x id 99, 69x id 100) match this rule + the generic
# centered-anchor position formula exactly, including the surprising 1:1
# mapping of id 99 ("ON/OFF Block") to "obj_onoffblock_res" rather than the
# "obj_onoffplatform_*" pair (which is instead what id 100, "Dotted-Line
# Block", maps to). Value tuple is (flag&4 clear, flag&4 set).
# NOTE: "obj_onoffblock_blue_res" (id 99 with flag&4 set) has no example in
# the reference and is extrapolated from the verified obj_onoffplatform_res/
# _blue_res pair for id 100.
# MM2 flag bit 2 (0x4) selects between two SWE variants for several object
# types. For ON/OFF blocks it chooses red vs blue; for Clown Car it chooses
# regular vs fire. Tuple is (bit2-clear variant, bit2-set variant).
# id=99/100 verified against "3000209_6 but right.swe" (73 objects exact match).
# id=42 verified against "clown car regular.swe" / "clown car fire.swe".
ONOFF_COLOR_FLAG = 1 << 2
ONOFF_SWE_IDS = {
    42:  ("obj_clown_res",           "obj_clown_fire_res"),
    99:  ("obj_onoffblock_res",      "obj_onoffblock_blue_res"),
    100: ("obj_onoffplatform_blue_res", "obj_onoffplatform_res"),
}

# MM2 id=44 (Style Power-up A) maps to a different SMMWE object per gamestyle.
# SWE gamestyle int (see GAMESTYLE_MAP) -> swe_id string.
# gamestyle=2 (SMW): Cape Feather -> obj_cap_res (verified from "cape example .swe").
# Other gamestyles are unverified; SMB3 Super Leaf and NSMBU Propeller Mushroom
# are dropped rather than silently shown as the wrong object.
STYLE_POWERUP_A_MAP = {
    0: "obj_mushroom_res",       # SMB1: Big Mushroom -> Mega Mario (best-effort fallback)
    2: "obj_cap_res",            # SMW: Cape Feather -> Cape Mario  (verified)
}

# MM2 id=81 (Style Power-up B) maps to the same SMMWE slot across all gamestyles:
# SMB1 (Link), SMB3 (Frog Suit), SMW (Power Balloon), NSMBU (Super Acorn)
# all share obj_SMB2_mushroom_res; SMMWE shows the correct variant per gamestyle.
STYLE_POWERUP_B_MAP = {
    0: "obj_SMB2_mushroom_res",  # SMB1: Link (Master Sword)
    1: "obj_SMB2_mushroom_res",  # SMB3: Frog Suit
    2: "obj_SMB2_mushroom_res",  # SMW: Power Balloon
    3: "obj_SMB2_mushroom_res",  # NSMBU: Super Acorn
}

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

# MM2 object ids whose SWE counterpart anchors its sprite at the LEFT edge of
# the MM2 footprint (col = x // SUBPX), matching _LEFT_ANCHOR in
# mm2_viewer_json.py, instead of the generic centered formula
# col = x // SUBPX - w // 2.
OBJ_LEFT_ANCHOR_IDS = {14, 16, 17, 31, 42, 49, 67, 71, 72}  # Mushroom / Semisolid / Half-Collision Platform, Bridge / Castle Bridge, Lakitu's Cloud / Clown Car / Koopa Car, One-Way Wall

# MM2 object ids whose `h` (tile height, growing UP from the (x,y) anchor at
# the bottom row) is not represented by a stretched SWE sprite -- instead the
# single SWE sprite sits at the TOP of that h-tall span. Without this, these
# objects are placed on the object's bottom row, which for most level designs
# is the ground row and hides the sprite behind/under terrain ("doesn't show
# up at all"). yy is shifted up by (h-1) tiles to compensate. Verified exactly
# against toost's drawing_instructions for Bullet Bill Blaster (id 13: h=2/5/7
# all match (h-1)*PX precisely); applied to Banzai Bill / Half-Collision
# Platform as a best-effort extrapolation (not independently verified).
# Bullet Bill Blaster / Semisolid Platform / Cannon / Mushroom Platform
# (13/16/47/14) are no longer routed through this S4 path -- see
# build_cannons / build_platform_objects, which apply the same (h-1)*PX
# shift directly.
# Thwomp (12, h=2): without this, obj_thwomp_res lands on the same row as
# objects placed one row above it -- hiding them under the sprite. Verified
# against "3000209_6 but right.swe": one Thwomp was fully covering a
# side-by-side ON/OFF Block pair and one sat one tile too low above its pair.
OBJ_H_ANCHOR_TOP_IDS = {12, 32, 71}

# MM2 flag bit 14 (0x4000) marks enemies given a Super Mushroom (big form).
# Verified by comparing flag distributions across ~4000 levels: bit 14 is the
# one bit that appears alongside both goombas AND chain chomps in "big" amounts
# and is absent from normal baseline (0x6000040). Bit 12 (0x1000) is wings
# (paragoomba etc.), NOT big — an earlier hypothesis that it was big was wrong.
# SMMWE represents big form as separate "_b_res" objects rather than scaling.
# BIG_OBJ_ID_MAP maps the regular SWE ID to its big counterpart; BIG_Y_OFFSET
# is an extra upward yy shift (in px) to keep the bigger sprite grounded at
# the same tile row.
#
# Only goomba/chomp/koopa/hammerbro are confirmed (in-game tested) to have a
# working big variant in SMMWE. Every other enemy that can be made "big" in
# MM2 was tested directly in the SMMWE editor and either has no big-form
# option at all, or has one that doesn't actually work in the engine -- in
# both cases there is no usable "_b_res" resource to substitute. The earlier
# version of this map guessed "_b_res" names for all of these (Thwomp/Monty
# Mole/Bowser/Piranha Plant/etc.) by analogy; that guess was the cause of
# three independently-reported crashing levels, since instance_create on an
# unresolved asset name crashes GameMaker on entering play mode (the same
# failure mode as the missing-dph crashes elsewhere in this file). Enemies
# not in this map keep their normal (small) sprite when big-flagged, which
# is wrong cosmetically but cannot crash the engine.
BIG_ENEMY_FLAG = 1 << 14
BIG_OBJ_ID_MAP = {
    "obj_goomba_res":    "obj_goomba_b_res",     # verified
    "obj_chomp_res":     "obj_chomp_b_res",      # verified
    "obj_koopa_res":     "obj_koopa_b_res",      # verified
    "obj_hammerbro_res": "obj_hammerbro_b_res",  # verified
    # Confirmed NOT to have a working big variant in SMMWE (no option, or
    # broken in-engine) -- do not re-add without re-testing:
    # obj_pplant_res, obj_thwomp_res, obj_spiny_res, obj_magikoopa_res,
    # obj_blooper_res, obj_rocky_res, obj_bowser_res, obj_boomboom_res,
    # obj_bowserjr_res, obj_monty_res
}
BIG_Y_OFFSET = {
    "obj_goomba_res": PX,       # verified: big goomba 1 tile taller
    "obj_chomp_res":  2 * PX,   # verified: big chomp 2 tiles taller
}
_BIG_Y_DEFAULT = PX             # unverified: assume 1 tile taller

# Every key an S4 object dict carries (from a real .swe). All flags default to
# 0; we only fill ID / xx / yy / scl / dir.
#
# "dph" was missing here too (see S6_CANNON_TEMPLATE / PLATFORM_S7_IDS for the
# same crash on the S6/S7 side): without it, obj_creator_jugar_editar's
# scr_edit_to_play conversion hits "instance_create_depth argument 2 ...
# expecting a Number" for every S4 entry (e.g. obj_skewer_res/obj_claw_res) as
# soon as test-play / play mode starts. dph=0 is GameMaker's default draw
# depth, same value used for S6_CANNON_TEMPLATE.
S4_TEMPLATE = {
    "air": 0, "pinkcoin": 0, "fire": 0, "claw": 0, "key": 0, "rock": 0,
    "sprout": 0, "energy": 0, "wings": 0, "w_mode": 0, "rot": 0,
    "can_complement": 0, "parachute": 0, "progress": 0, "sierra": 0,
    "bumper": 0, "inup": 0, "ice": 0, "dph": 0,
}

# Every key an S6 Cannon ("obj_cannon_res") entry carries, from a hand-placed
# reference save. rht/dwn/up/lft/dir are direction-related flags whose
# per-direction meaning isn't reverse-engineered yet; the reference (default
# placement) had dwn=1 and everything else 0, used as a constant.
#
# "dph" was missing here too: like the Lift -> obj_platform_res crash (S4
# entry missing dph/wth/hht/spr), a level with a Cannon crashed on entering
# play mode with "instance_create_depth argument 2 ... expecting a Number",
# this time from obj_load_guardabot (the S6 loader) instead of
# obj_creator_jugar_editar (the S4 loader). dph=0 is GameMaker's default
# draw depth and matches obj_bullebill_base_res's dph (the other
# cannon/blaster-family S7 entry, see PLATFORM_S7_IDS).
S6_CANNON_TEMPLATE = {
    "rht": 0, "dwn": 1, "up": 0, "lft": 0, "dir": 0, "dph": 0,
}

# Every key an S7 "stretchy sprite" entry carries besides ID/xx/yy/dir/spr/
# wth/hht/dph, from a hand-placed reference save. All state flags default to 0
# (contents/effects like fire/ice/wings/parachute aren't carried over).
S7_TEMPLATE = {
    "air": 0, "pinkcoin": 0, "clr": 0, "fire": 0, "key": 0, "rock": 0,
    "energy": 0, "wings": 0, "parachute": 0, "ice": 0,
}

# MM2 object id -> S7 "ID" string for the platform-family stretchy sprites.
# See build_platform_objects for the wth/hht/dph formulas. SMMWE's naming
# swaps 13 <-> 47 relative to MM2: the Bullet Bill Blaster (13) is
# "obj_bullebill_base_res" and the Cannon (47) is "obj_cannon_res" (in S6,
# see build_cannons / S6_CANNON_TEMPLATE) -- verified in-game.
PLATFORM_S7_IDS = {
    16: "obj_semisolid_platform1",   # Semisolid Platform
    13: "obj_bullebill_base_res",    # Bullet Bill Blaster
    91: "obj_platform_res",          # Seesaw -> sized moving platform
    11: "obj_platform_res",          # Lift -> moving platform
    17: "obj_puente_res",            # Bridge
    49: "obj_puente_res",            # Castle Bridge (approx)
    14: "obj_mushroom_platform_res", # Mushroom Platform
}

# obj_puente_res sprite name. From a hand-placed reference save ("bridge
# example .swe", NSMBU/ghost) containing 5 bridges of width 3-7: every
# bridge used "spr_NSMBU_puente_underground" regardless of the level's
# theme, and dph=8/hht=3/dir=0 were constant across all 5. No theme/
# gamestyle variants are confirmed, so this single verified sprite is used
# for every bridge.
PUENTE_SPRITE = "spr_NSMBU_puente_underground"

# SWE gamestyle int (see GAMESTYLE_MAP) -> sprite-name prefix used by the S7
# "ssp1"/"bullebill_base" sprites. SMW (2) sprites have no prefix.
GAMESTYLE_SPR_PREFIX = {0: "SMB", 1: "SMB3", 2: "", 3: "NSMBU"}

# THEME_MAP theme names for which a `spr_<prefix>_ssp1_<theme>` sprite exists,
# per gamestyle prefix (reverse-engineered from data.win's string table).
# Themes not listed fall back to "overworld". Night variants are not used
# (best-effort: a level converted at night gets the day sprite).
SSP1_THEMES = {
    "SMB":   {"airship", "castle", "desert", "ghost", "overworld", "snow",
              "underground", "underwater"},
    "SMB3":  {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "sky", "snow", "underground", "underwater"},
    "":      {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "snow", "underground", "underwater"},
    "NSMBU": {"airship", "castle", "desert", "forest", "ghost", "overworld",
              "sky", "snow", "underground", "underwater"},
}


def ssp_sprite_name(gamestyle, theme):
    """Semisolid Platform sprite name for a given SWE gamestyle/theme."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    valid = SSP1_THEMES.get(prefix, SSP1_THEMES["NSMBU"])
    t = theme if theme in valid else "overworld"
    return f"spr_{prefix}_ssp1_{t}" if prefix else f"spr_ssp1_{t}"


def mp1_sprite_name(gamestyle, theme):
    """Mushroom Platform sprite name for a given SWE gamestyle/theme.
    Confirmed for NSMBU/ghost ("spr_NSMBU_mp1_ghost") via a hand-placed
    reference save; other gamestyles/themes follow the ssp1 naming pattern
    (best-effort, SSP1_THEMES reused since mp1 has no separate theme list)."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    valid = SSP1_THEMES.get(prefix, SSP1_THEMES["NSMBU"])
    t = theme if theme in valid else "overworld"
    return f"spr_{prefix}_mp1_{t}" if prefix else f"spr_mp1_{t}"


def bullebill_sprite_name(gamestyle):
    """Bullet Bill Blaster ("obj_bullebill_base_res") sprite name for a SWE
    gamestyle. This sprite has no theme variants (only SMB3/NSMBU have a
    dedicated sprite; SMB1/SMW share the unprefixed default)."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    if prefix in ("SMB3", "NSMBU"):
        return f"spr_{prefix}_bullebill_base"
    return "spr_bullebill_base"


def platform_sprite_name(gamestyle):
    """Moving platform ("obj_platform_res") sprite name for a SWE gamestyle.
    Confirmed for SMB1 (prefix "SMB") via a hand-placed reference save
    ("spr_SMB_platform"); other gamestyles follow the same
    spr_<prefix>_platform pattern, with SMW (no prefix) using "spr_platform"."""
    prefix = GAMESTYLE_SPR_PREFIX.get(gamestyle, "NSMBU")
    return f"spr_{prefix}_platform" if prefix else "spr_platform"


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
    formula:  col = x//160 - w//2 ,  row = y//160 -- except for
    OBJ_LEFT_ANCHOR_IDS, which use  col = x//160  (no -w//2), matching
    _LEFT_ANCHOR in mm2_viewer_json.py."""
    w = max(1, o.get("w", 1))
    if o.get("id") in OBJ_LEFT_ANCHOR_IDS:
        col = o["x"] // SUBPX
    else:
        col = o["x"] // SUBPX - w // 2
    row = o["y"] // SUBPX
    return col, row


# Object ids whose rectangular w x h bounding box does not represent a solid
# wall/ceiling/floor -- excluded from occupied_cells so they don't create
# false "solid" terrain signals for build_cannons.
_NON_SOLID_IDS = {
    87,  # Slight Slope -- diagonal, not a solid rectangle
    88,  # Steep Slope -- diagonal, not a solid rectangle
    16,  # Semisolid Platform -- walk-through from below/sides
}


def occupied_cells(j, *, exclude_id=None, extra_ground=None):
    """Set of (col, row_from_bottom) tile cells covered by S2 ground terrain
    or by any object's footprint (other than `exclude_id`), plus any cells in
    `extra_ground` (e.g. slope footprints filled in by build_world). Used by
    build_cannons to detect adjacent floor/ceiling/wall terrain."""
    occ = {(g["x"], g["y"]) for g in j.get("ground", [])}
    if extra_ground:
        occ |= extra_ground
    for o in j.get("objects", []):
        oid = o.get("id")
        if oid == exclude_id or oid in _NON_SOLID_IDS:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        col, row = object_cell(o)
        for dx in range(w):
            for dy in range(h):
                occ.add((col + dx, row + dy))
    return occ


def skewer_footprint_cells(objects):
    """Cells covered by each Skewer's (id 83) w x h bounding box.

    MM2 lets a Skewer share its cell with a ground/block tile (it's embedded
    in the surface it extends from), but SMMWE's obj_skewer_res can't overlap
    S2 ground -- the skewer's own body acts as the solid surface there, so
    these cells are dropped from S2 (see build_world)."""
    cells = set()
    for o in objects:
        if o.get("id") != 83:
            continue
        col, row = object_cell(o)
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        for dx in range(w):
            for dy in range(h):
                cells.add((col + dx, row + dy))
    return cells


# MM2 object ids for slope terrain, which SWE has no diagonal equivalent for.
# build_world fills these as a solid ascending/descending staircase of S2
# ground instead of dropping them. Value is the slope's "run" per 1-tile
# rise: Steep Slope (88) rises 1 tile per column (slope 1), Slight Slope (87)
# rises 1 tile per 2 columns (slope 1/2).
_SLOPE_STEPS = {88: 1, 87: 2}
_SLOPE_IDS = set(_SLOPE_STEPS)


def slope_fill_cells(objects):
    """Set of (col, row_from_bottom) ground cells forming a staircase for
    every Slight/Steep Slope (87/88) object, to be merged into S2 ground.

    Every one of the object's w columns gets a step of height
    min(ceil(run/step), h) tiles (run = 1..w from the low end), filled from
    the object's base row upward -- a "slope of 1" (steep, step=1) or "slope
    of 1/2" (slight, step=2) staircase capped at the slope's own height h, so
    the last two columns at the high end both sit flush at height h. flag &
    0x100000 flips which end (left/right) is the low end, matching
    slope_tiles() in mm2_viewer_json.py.

    Verified against 3000065_1_overworld.json: with no x/y adjustment, this
    exactly fills the ground tiles that Goombas standing at the low end, the
    seam between two chained ascending slopes, and the peak all stand on.

    Slopes anchor at their LEFT edge (col = x // SUBPX, no -w//2), per
    obj_anchor() in mm2_viewer_json.py.
    """
    cells = set()
    for o in objects:
        step = _SLOPE_STEPS.get(o.get("id"))
        if step is None:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        base_col = o["x"] // SUBPX
        base_row = o["y"] // SUBPX
        descending = (o.get("flag", 0) & 0x100000) != 0
        for x in range(w):
            run = (w - x) if descending else (x + 1)
            height = min((run + step - 1) // step, h)  # ceil(run/step), capped at h
            for y in range(height):
                cells.add((base_col + x, base_row + y))
    return cells


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
    """MM2 pipes (id 9) -> SWE S5 entries. Returns (s5_list, consumed_plants).

    consumed_plants is a set of id() values for Piranha Plant (id=2) objects
    that have been folded into an upward pipe via msk=1 and must not be
    emitted as standalone S4 objects by build_objects.

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
    # Piranha plant (id=2) lookup by tile cell for msk=1 detection.
    plant_by_cell = {}
    for o in objects:
        if o.get("id") == 2:
            plant_by_cell.setdefault((o["x"] // SUBPX, o["y"] // SUBPX), o)

    out = []
    consumed_plants = set()
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
        # Piranha plant inside an upward pipe: any id=2 at or one tile above
        # the pipe mouth (cols base_col/base_col+1, rows row_top/row_top+1)
        # becomes msk=1 instead of a separate S4 obj_pplant_res.
        if direction == "U":
            for dc in (0, 1):
                for dr in (0, 1):
                    plant = plant_by_cell.get((base_col + dc, row_top + dr))
                    if plant is not None:
                        entry["msk"] = 1
                        consumed_plants.add(id(plant))
        out.append(entry)
    return out, consumed_plants


# MM2 object ids handled by dedicated builders, not the generic S4 path.
_NON_S4_IDS = {9, 11, 13, 14, 16, 17, 47, 49, 55, 91}  # 9=Pipe (S5), 55=Door (S8), 11/13/14/16/17/47/49/91 -> S6/S7 (see below)


def build_objects(objects, gamestyle=3, consumed_plants=None):
    """MM2 objects[] -> SWE S4. Returns (s4_list, dropped_counts).
    Pipes (id 9) and the S6/S7 special cases (13/16/47) are handled by
    build_pipes / build_cannons / build_platform_objects and
    skipped here. consumed_plants is a set of id() values for Piranha
    Plants already folded into pipe S5 entries (msk=1) by build_pipes."""
    # Pre-scan: for blocks, cid stores the MM2 item id directly (NOT an index
    # into objects[]). Map cid -> sprout value for each block that has a known item.
    # Items inside blocks are NOT separate objects[] entries; nothing is skipped.
    block_sprouts = {}  # id(block_obj) -> sprout int
    for o in objects:
        if o.get("id") in _BLOCK_IDS:
            sprout = BLOCK_SPROUT_MAP.get(o.get("cid", -1))
            if sprout is not None:
                block_sprouts[id(o)] = sprout

    out = []
    dropped = {}
    for o in objects:
        oid = o.get("id")
        if oid in _NON_S4_IDS or oid in _SLOPE_IDS:
            continue
        if consumed_plants and oid == 2 and id(o) in consumed_plants:
            continue
        if oid == 44:
            swe_id = STYLE_POWERUP_A_MAP.get(gamestyle)
        elif oid == 81:
            swe_id = STYLE_POWERUP_B_MAP.get(gamestyle)
        elif oid in ONOFF_SWE_IDS:
            off_id, on_id = ONOFF_SWE_IDS[oid]
            swe_id = on_id if o.get("flag", 0) & ONOFF_COLOR_FLAG else off_id
        else:
            swe_id = OBJ_ID_MAP.get(oid)
        if swe_id is None:
            name = o.get("name", f"id={oid}")
            dropped[name] = dropped.get(name, 0) + 1
            continue
        col, row = object_cell(o)
        scl = 1
        yy = (FIELD_HEIGHT_TILES - 1 - row) * PX - OBJ_Y_OFFSET_PX.get(oid, 0)
        if oid in OBJ_H_ANCHOR_TOP_IDS:
            h = max(1, o.get("h", 1))
            yy -= (h - 1) * PX
        if o.get("flag", 0) & BIG_ENEMY_FLAG and swe_id in BIG_OBJ_ID_MAP:
            yy -= BIG_Y_OFFSET.get(swe_id, _BIG_Y_DEFAULT)
            swe_id = BIG_OBJ_ID_MAP[swe_id]
        entry = dict(S4_TEMPLATE)
        entry.update({
            "ID": swe_id,
            "xx": col * PX,
            "yy": yy,
            "scl": scl,
            "dir": 0,   # best-effort: MM2 flag bitfields aren't carried over
        })
        if oid == 12:
            entry["dir"] = THWOMP_DIR
        elif oid == 67:
            entry["dir"] = ONEWAY_DIR_MAP.get((o.get("flag", 0) >> 22) & 0x3, 1)
            entry["rot"] = (entry["dir"] - 1) * 90
        elif oid == 32:
            entry["dir"] = BANZAI_DIR_MAP.get((o.get("flag", 0) >> 22) & 0x3, 1)
            entry["rot"] = (entry["dir"] - 1) * 90
            entry["scl"] = BANZAI_SCL[entry["dir"]]
        elif oid == 83:
            # Skewer anchors one column right of and one row above the
            # generic centered-bottom cell (col + w//2, row + 1) -- derived
            # by hand-placing all 4 of this level's skewers correctly in the
            # SMMWE editor and comparing the resulting .swe (see SKEWER_DIR).
            w = max(1, o.get("w", 1))
            entry["xx"] = (col + w // 2) * PX
            entry["yy"] = yy - PX
            entry["dir"] = SKEWER_DIR.get(o.get("flag", 0) & SKEWER_CEILING_FLAG, 1)
            entry["rot"] = SKEWER_ROT.get(entry["dir"], 0)
        if id(o) in block_sprouts:
            entry["sprout"] = block_sprouts[id(o)]
        out.append(entry)
    return out, dropped


def build_doors(objects):
    """MM2 Door (id 55) -> SWE S8 "obj_door_res" entries.

    SMMWE stores a connected door PAIR as a single S8 entry holding both
    doors' positions: {"xx","yy"} for one door and {"dr_x","dr_y"} for its
    linked partner (reverse-engineered from "3000621 but with the doors.swe").
    Position uses the same (col*PX, (FIELD_HEIGHT_TILES-1-row)*PX) formula as
    the generic S4 path (object_cell, no offsets).

    MM2's JSON has no cid/lid/sid pair link for doors (always -1), but bits
    19-22 of `flag`, i.e. (flag >> 19) & 0xF, hold a per-door pairing order
    (confirmed against the reference file's 3 S8 entries and the user's
    explicit pairing for all 4 pairs in this level). Doors are sorted by
    that order and paired consecutively (1st&2nd, 3rd&4th, ...); within each
    pair the higher-order door's position becomes "xx/yy" and the
    lower-order door's becomes "dr_x/dr_y", matching the reference exactly.
    A trailing, unpaired door is dropped.
    """
    doors = sorted(
        (o for o in objects if o.get("id") == 55),
        key=lambda o: (o.get("flag", 0) >> 19) & 0xF,
    )
    out = []
    for lo, hi in zip(doors[0::2], doors[1::2]):
        c_lo, r_lo = object_cell(lo)
        c_hi, r_hi = object_cell(hi)
        out.append({
            "xx": c_hi * PX,
            "yy": (FIELD_HEIGHT_TILES - 1 - r_hi) * PX,
            "dr_x": c_lo * PX,
            "dr_y": (FIELD_HEIGHT_TILES - 1 - r_lo) * PX,
            "ID": "obj_door_res",
        })
    return out


def build_cannons(objects, occ):
    """MM2 Cannon (id 47) -> SWE S6 "obj_cannon_res" entries.

    Position uses the same centered-column / top-of-footprint formula as
    build_objects' OBJ_H_ANCHOR_TOP_IDS path. The S6 entry has no size
    field, so MM2's `h` cannot be represented and is dropped.

    up/dwn/dir are not stored in MM2's `flag` in a reverse-engineered form,
    so they're inferred from adjacent terrain/objects in `occ` (see
    occupied_cells), verified against the 21 cannons in a real level:

      * A solid cell directly above the footprint means the cannon is
        ceiling-mounted (up=1, dwn=0).
      * A ceiling-mounted cannon with open space directly below shoots
        straight down (dir=6) -- this is the common "turret in an open
        room" case (confirmed: 3 single-tile + 1 big ceiling cannon in an
        open alcove all shoot down, regardless of side walls).
      * A ceiling-mounted cannon with something solid directly below (e.g.
        embedded in the face of a pillar) instead faces away from an
        adjacent side wall (dir=0 right / dir=4 left).
      * Everything else is floor-mounted, facing right (dwn=1, dir=0) --
        side-wall-based left/right facing for floor cannons was tried and
        produced a false positive, so floor cannons always face right.

    A floor-mounted "big" (h>1) cannon's MM2 bounding box is embedded in
    the ground it's mounted on -- only the row above the box (row+h, the
    same row the ceiling-check above confirms is open) is clear, so that's
    where the single S6 tile is placed; h==1 cannons keep the existing
    top-of-footprint placement (row+h-1 == row).

    `flag` bit 26 (1<<26) is set on every cannon whose dir mapping above
    matches the reference saves; the one cannon missing it renders with
    left/right swapped, so dir 0/4 are flipped when that bit is absent.

    rht/lft are always 0, per the reference save."""
    # dir=3 (right) verified against "300361_2 correct cannon.swe": the
    # floor-mounted cannon there (bit26=SET, no flip) has dir=3. dir=7 is the
    # symmetric left counterpart (3+4). dir=6 (ceiling-down) is unchanged.
    # bit26=CLEAR triggers the flip (left-facing cannons); bit26=SET keeps the
    # base direction (right-facing). Ceiling-sideways follows the same pair.
    HFLIP_DIR = {3: 7, 7: 3}
    out = []
    for o in objects:
        if o.get("id") != 47:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        col, row = object_cell(o)

        ceiling = any((col + dx, row + h) in occ for dx in range(w))
        if ceiling:
            up, dwn = 1, 0
            below_open = not any((col + dx, row - 1) in occ for dx in range(w))
            if below_open:
                cannon_dir = 6
            else:
                wall_left = any((col - 1, row + dy) in occ for dy in range(h))
                wall_right = any((col + w, row + dy) in occ for dy in range(h))
                cannon_dir = 7 if (wall_right and not wall_left) else 3
            top_row = row + h - 1
        else:
            up, dwn = 0, 1
            cannon_dir = 3
            top_row = row + h if h > 1 else row

        if not (o.get("flag", 0) & (1 << 26)):
            cannon_dir = HFLIP_DIR.get(cannon_dir, cannon_dir)

        entry = dict(S6_CANNON_TEMPLATE)
        entry.update({
            "ID": "obj_cannon_res",
            "xx": col * PX,
            "yy": ground_yy(top_row),
            "up": up,
            "dwn": dwn,
            "dir": cannon_dir,
        })
        out.append(entry)
    return out


def build_platform_objects(objects, *, gamestyle, theme):
    """MM2 Bullet Bill Blaster (13) / Semisolid Platform (16) / Seesaw (91) /
    Bridge (17) / Castle Bridge (49, approx) / Mushroom Platform (14)
    -> SWE S7 entries.

    S7 is a "stretchy sprite" object: `spr` selects the themed sprite and
    `wth`/`hht` size it (in MM2 tile units). Position reuses the
    centered/left-anchor + top-of-footprint formula from object_cell /
    OBJ_H_ANCHOR_TOP_IDS. wth/hht = w,h directly -- verified against two
    8-entry hand-placed reference grids (a "staircase" of 3x3/3x4/3x5/3x6
    and a "different width" set of 3x3/4x3/5x3/6x3 Semisolid Platforms),
    which gave wth==w and hht==h exactly in all 8 cases. `dph` doesn't
    appear to affect rendering and is left at the constant from the
    original single-object reference of each type (255 / 0).

    Seesaw (91) and Lift (11) both use "obj_platform_res" (a vertically-
    moving platform). Seesaw was verified via a hand-placed reference save
    ("platform example.swe") containing wth=4/6/8 examples:
    spr="spr_<gamestyle>_platform", dph=-1, dir=2, hht=3 (constant, the
    sprite's fixed vertical travel height, independent of the seesaw's h).
    wth = max(4, w) -- the seesaw's width, with a minimum of 4 (per the
    reference's "default" size) -- giving a best-effort "platform as big as
    the seesaw, that goes up and down".

    Lift's S4 mapping to the same "obj_platform_res" ID was missing the
    dph/wth/hht/spr fields obj_platform_res requires, crashing
    instance_create_depth on entering play mode ("argument 2 ... expecting a
    Number") in any level containing a Lift. Routed here with the same
    dph=-1/dir=2/spr as the (working) seesaw mapping, but hht=1 to match
    Lift's actual h=1 footprint (seesaw's hht=3 was seesaw-specific).

    Bridge (17) / Castle Bridge (49) -> "obj_puente_res", left-anchored
    (OBJ_LEFT_ANCHOR_IDS) like the other stretchy platforms. wth=w (MM2
    enforces a minimum bridge width of 3, matching the reference's
    smallest example); hht=3/dph=8/dir=0/spr=PUENTE_SPRITE are constant,
    verified against all 5 widths (3-7) in "bridge example .swe".

    Mushroom Platform (14) -> "obj_mushroom_platform_res", a single S7
    entry per platform (the stalk below it is drawn by the sprite itself,
    not a separate object). Left-anchored, wth=w/hht=h with no yy nudge
    (the same plain (h-1)*PX-shifted anchor as everything else) -- verified
    against two 4-entry reference grids ("mushroom platform example diff
    width/length .swe") spanning w=3-6 and h=3-6: wth==w and hht==h exactly
    in all 8 cases, and yy matched the unmodified formula assuming all 4
    platforms in each grid share the same MM2 row. `dph = 245 + hht`
    (248/249/250/251 for hht=3/4/5/6); `spr` follows mp1_sprite_name.
    """
    out = []
    for o in objects:
        oid = o.get("id")
        if oid not in PLATFORM_S7_IDS:
            continue
        w = max(1, o.get("w", 1))
        h = max(1, o.get("h", 1))
        col, row = object_cell(o)
        if oid == 91:
            wth, hht = max(4, w), 3
        elif oid == 11:
            wth, hht = max(4, w), 1
        elif oid in (17, 49):
            wth, hht = w, 3
        else:
            wth, hht = w, h
        yy = ground_yy(row) - (hht - 1) * PX
        if oid == 91:
            yy += 2 * PX  # nudge the seesaw replacement down 2 tiles
        elif oid in (17, 49):
            yy += 2 * PX  # bridge sprite sits 2 tiles above the generic anchor
        entry = dict(S7_TEMPLATE)
        entry.update({
            "ID": PLATFORM_S7_IDS[oid],
            "xx": col * PX,
            "yy": yy,
            "dir": 0,
            "wth": wth, "hht": hht,
        })
        if oid == 16:
            entry.update({"spr": ssp_sprite_name(gamestyle, theme), "dph": 255})
        elif oid == 13:
            entry.update({"spr": bullebill_sprite_name(gamestyle), "dph": 0})
        elif oid in (17, 49):
            entry.update({"spr": PUENTE_SPRITE, "dph": 8})
        elif oid == 14:
            entry.update({"spr": mp1_sprite_name(gamestyle, theme), "dph": 245 + hht})
        else:  # 91 (Seesaw) / 11 (Lift) -- moving platform
            entry.update({"spr": platform_sprite_name(gamestyle), "dph": -1, "dir": 2})
        out.append(entry)
    return out


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
    gamestyle = GAMESTYLE_MAP.get(j.get("gamestyle_raw", 0), 3)
    theme = THEME_MAP.get(j.get("theme_raw", 0), "overworld")
    s5, consumed_plants = build_pipes(objects)
    s4, dropped = build_objects(objects, gamestyle, consumed_plants)
    s8 = build_doors(objects)

    # Slopes have no SWE equivalent -- fill their footprint in as solid ground.
    slope_cells = slope_fill_cells(objects)
    ground = list(j.get("ground", []))
    existing_ground = {(g["x"], g["y"]) for g in ground}
    for x, y in slope_cells:
        if (x, y) not in existing_ground:
            ground.append({"x": x, "y": y})
            existing_ground.add((x, y))

    occ = occupied_cells(j, exclude_id=47, extra_ground=slope_cells)
    s6 = build_cannons(objects, occ)
    s7 = build_platform_objects(objects, gamestyle=gamestyle, theme=theme)

    # Drop ground tiles under any Skewer -- SMMWE forbids the overlap (see
    # skewer_footprint_cells).
    skewer_cells = skewer_footprint_cells(objects)
    ground = [g for g in ground if (g["x"], g["y"]) not in skewer_cells]

    world = {
        "S1": [build_metadata(j, user=user, name=name, desc=desc,
                              date_str=date_str, time_str=time_str)],
        "S2": build_ground(ground),
        "S3": [],
        "S4": s4,
        "S5": s5,
        "S6": s6,
        "S7": s7,
        "S8": s8,
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
        description="Convert an MM2 level JSON (or a folder of JSONs) into "
                    "Super Mario Maker Worldwide Engine (.swe) save file(s)."
    )
    p.add_argument("json_path", help="Path to a *_overworld.json, a plain level JSON, or a folder of JSONs")
    p.add_argument("-o", "--output", help="Output .swe path (single-file) or output folder (folder mode)")
    p.add_argument("--user", default="patwick", help="Author name stored in the level")
    p.add_argument("--name", default=None, help="Level name (default: JSON 'name'); ignored in folder mode")
    p.add_argument("--desc", default=None, help="Description (default: JSON 'description')")
    p.add_argument("--height", type=int, default=FIELD_HEIGHT_TILES,
                   help=f"Playfield height in tiles (default {FIELD_HEIGHT_TILES})")
    return p.parse_args()


def convert_one(ow_path, sub_path, base, args, out_dir=None):
    """Convert one overworld+optional subworld pair to a .swe file."""
    overworld_json = load_json(ow_path)
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
        sb1 = {"S1": []}

    level = {"S0": s0, "SB1": sb1}
    data = encode_swe(level)

    if out_dir is not None:
        out_path = Path(out_dir) / (base + ".swe")
    elif args.output:
        out_path = Path(args.output)
    else:
        out_path = ow_path.with_name(base + ".swe")
    out_path.write_bytes(data)

    print(f"Wrote {out_path} ({len(data)} bytes)")
    print(f"  ground tiles : {len(s0['S2'])}")
    print(f"  objects      : {len(s0['S4'])}")
    print(f"  pipes        : {len(s0['S5'])}")
    print(f"  cannons      : {len(s0['S6'])}")
    print(f"  platforms    : {len(s0['S7'])}")
    print(f"  doors        : {len(s0['S8'])}")
    if dropped:
        total = sum(dropped.values())
        print(f"  dropped {total} object(s) with no SMMWE equivalent:")
        for name_, n in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"      {n:4d}  {name_}")


def main():
    global FIELD_HEIGHT_TILES
    args = parse_args()
    FIELD_HEIGHT_TILES = args.height

    in_path = Path(args.json_path)

    if in_path.is_dir():
        out_dir = Path(args.output) if args.output else in_path
        out_dir.mkdir(parents=True, exist_ok=True)
        seen_bases = set()
        errors = 0
        for json_file in sorted(in_path.glob("*.json")):
            if json_file.stem.endswith("_subworld"):
                continue  # picked up as companion of overworld
            ow_path, sub_path, base = find_companion(json_file)
            if base in seen_bases:
                continue
            seen_bases.add(base)
            if ow_path is None:
                print(f"Skipping {json_file.name}: no overworld JSON found")
                continue
            try:
                convert_one(ow_path, sub_path, base, args, out_dir=out_dir)
            except Exception as exc:
                print(f"ERROR converting {json_file.name}: {exc}")
                errors += 1
        print(f"\nDone: {len(seen_bases)} level(s) processed, {errors} error(s).")
    else:
        ow_path, sub_path, base = find_companion(in_path)
        if ow_path is None and sub_path is None:
            raise SystemExit(f"Could not find JSON data at {in_path}")
        convert_one(ow_path, sub_path, base, args)


if __name__ == "__main__":
    main()
