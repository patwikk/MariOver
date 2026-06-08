#!/usr/bin/env python3
"""
ascii_to_vglc.py
================
Converts Mario Maker 2 ASCII text arrays into clean VGLC format rows.
Retains all tile maps and canonical character transitions; does not crop,
pad, or window the level — selection is the dataset builder's job.
"""

# Tiles that become solid ground  X
# Covers every MM2_ascii_guide.txt TERRAIN and PLATFORMS/MOVING tile that is
# actually solid (Mario can stand on / collide with it). Tiles with their own
# dedicated VGLC category (breakable brick, question block, starting brick,
# pipes, enemies, coins) are intentionally excluded here.
MM2_GROUND = set(
    "#"   # ground
    "H"   # hard block
    "S"   # stone (solid terrain)
    "h"   # hidden block (solid once revealed)
    "I"   # ice block (solid; slipperiness not representable in VGLC)
    "C"   # crate
    "T"   # tree
    "="   # castle bridge
    "N"   # note block
    "d"   # donut block (solid platform that falls)
    "p"   # p block (togglable solid)
    "O"   # on/off block (togglable solid)
    "."   # dotted-line block (togglable solid)
    "*"   # blinking block (togglable solid)
    "^"   # spike block (solid hazard block)
    "³"   # mushroom platform (solid platform)
    "´"   # semisolid platform (solid from above)
    "·"   # bridge platform (solid)
    "¸"   # lava lift (solid moving platform)
    "¹"   # snake block (solid moving platform)
    "º"   # track block (solid moving platform)
    "»"   # conveyor belt (solid)
    "¼"   # fast conveyor belt (solid)
    "½"   # sprint platform (solid)
    "¾"   # seesaw (solid moving platform)
    "¿"   # swinging claw (solid moving platform)
    "À"   # on/off trampoline (solid)
    "Á"   # mushroom trampoline (solid)
    "J"   # jumping machine (solid)
    "Â"   # half-collision platform (solid from above)
    "Ã"   # donut platform (solid platform that falls)
    "Ù"   # exclamation block (solid)
    "Ç"   # spikes — solid hazard; treated as solid ground tile in VGLC
    "É"   # skewer — solid moving hazard
    "Ë"   # icicle
    "Ø"   # cannon (shooter, solid) — handled separately below, but kept
           # here as fallback solid
)

# Starting brick: the safety platform Mario spawns on. It isn't part of the
# drawn level geometry, so it converts to air rather than solid ground.
MM2_STARTING_BRICK = set("{")

# Moving/damaging hazards that are NOT solid (Mario can't stand on them, just
# takes damage from contact) — closer to "enemy" semantics in VGLC (E =
# enemy/damaging/hazard/moving) than to ground or empty air.
MM2_HAZARD_AS_ENEMY = set(
    "Ä"   # fire bar
    "Å"   # saw
    "Æ"   # burner
    "È"   # spike ball
    "Ê"   # twister
)

MM2_BREAKABLE = set("B")
MM2_QUESTION = set("?")

# Pipes: MM2 marks orientation explicitly, so classify_pipe_cell() can pick
# the correct VGLC pipe glyph (< > [ ]) instead of flattening to ground.
MM2_PIPE_UPRIGHT  = set("|↑")   # mouth faces up (standard pipe)
MM2_PIPE_DOWN     = set("↓")    # mouth faces down (ceiling pipe)
MM2_PIPE_SIDEWAYS = set("←→")  # horizontal pipe — VGLC has no equivalent, treat as ground
MM2_PIPE = MM2_PIPE_UPRIGHT | MM2_PIPE_DOWN | MM2_PIPE_SIDEWAYS

# Door / warp box tiles are solid structures, not pipes -> ground
MM2_PIPE_AS_GROUND = set("DW")

MM2_CANNON_EMITTER = set("V")
MM2_ENEMIES = set(
    "g K P m t o s b L Z y < u x X @ ~ q w Y e F % & r , a n R ! 9 j + ¡ ; A v [ 1 2 3 4 5 6 7 µ"
    .split()
)
MM2_COINS = set("¢$£")

VGLC_EMPTY     = "-"
VGLC_GROUND    = "X"
VGLC_BREAK     = "S"
VGLC_QUESTION  = "?"
VGLC_ENEMY     = "E"
VGLC_COIN      = "o"
VGLC_CANNON_T  = "B"
VGLC_CANNON_B  = "b"
PIPE_TOP_LEFT  = "<"
PIPE_TOP_RIGHT = ">"
PIPE_BOT_LEFT  = "["
PIPE_BOT_RIGHT = "]"

def normalize_grid(lines: list[str]) -> list[list[str]]:
    if not lines:
        return []
    width = max(len(l) for l in lines)
    grid = []
    for l in lines:
        row = list(l)
        row += [" "] * (width - len(row))
        grid.append(row)
    return grid

def find_cannon_positions(grid: list[list[str]]) -> dict[tuple[int,int], str]:
    result = {}
    height = len(grid)
    for r in range(height):
        for c in range(len(grid[r])):
            if grid[r][c] == "V":
                result[(r, c)] = VGLC_CANNON_T
                if r + 1 < height:
                    below_ch = grid[r + 1][c]
                    if below_ch == " " or below_ch not in MM2_GROUND:
                        result[(r + 1, c)] = VGLC_CANNON_B
    return result

def is_pipe_tile(ch: str) -> bool:
    return ch in MM2_PIPE

def classify_pipe_cell(grid: list[list[str]], row: int, col: int) -> str:
    """
    Map an MM2 pipe tile to its VGLC pipe character.

    VGLC pipe convention (SMB) — pipes always open upward:
        < >   top halves (the enterable mouth)
        [ ]   body halves

    MM2 marks pipe orientation explicitly:
      - '|' / '↑'  upright pipe, mouth at the top
      - '↓'        ceiling pipe, mouth at the bottom
      - '←' / '→' sideways pipe — no VGLC equivalent (handled by the caller,
                    which routes these to solid ground before reaching here)

    For ceiling pipes we flip the cap test to look at the cell below, so the
    mouth (where Mario actually enters/exits) still maps onto < / > rather
    than being buried in the middle of a body run.
    """
    ch     = grid[row][col]
    height = len(grid)
    width  = len(grid[0]) if height > 0 else 0

    right = grid[row][col + 1] if col + 1 < width  else " "
    left  = grid[row][col - 1] if col - 1 >= 0     else " "
    is_right_half = is_pipe_tile(left) and not is_pipe_tile(right)

    if ch in MM2_PIPE_DOWN:
        below  = grid[row + 1][col] if row + 1 < height else " "
        is_cap = not is_pipe_tile(below)
    else:
        above  = grid[row - 1][col] if row > 0 else " "
        is_cap = not is_pipe_tile(above)

    if is_cap:
        return PIPE_TOP_RIGHT if is_right_half else PIPE_TOP_LEFT
    else:
        return PIPE_BOT_RIGHT if is_right_half else PIPE_BOT_LEFT

def convert_cell(ch: str, grid: list[list[str]], row: int, col: int, cannon_map: dict) -> str:
    if (row, col) in cannon_map:
        return cannon_map[(row, col)]

    # Empty / air (the spawn platform converts to air too — it's not part of
    # the drawn level geometry)
    if ch == " " or ch in MM2_STARTING_BRICK:
        return VGLC_EMPTY

    if ch in MM2_BREAKABLE:
        return VGLC_BREAK

    if ch in MM2_QUESTION:
        return VGLC_QUESTION

    # Enemies (and non-solid moving hazards, which share VGLC's "damaging
    # / hazard / moving" semantics with enemies)
    if ch in MM2_ENEMIES or ch in MM2_HAZARD_AS_ENEMY:
        return VGLC_ENEMY

    if ch in MM2_COINS:
        return VGLC_COIN

    # Pipes -> proper VGLC pipe glyphs (sideways pipes have no VGLC
    # equivalent and fall through to solid ground)
    if ch in MM2_PIPE:
        if ch in MM2_PIPE_SIDEWAYS:
            return VGLC_GROUND
        return classify_pipe_cell(grid, row, col)

    # Door / warp box -> ground
    if ch in MM2_PIPE_AS_GROUND:
        return VGLC_GROUND

    if ch in MM2_GROUND or ch in "/\\":
        return VGLC_GROUND

    return VGLC_EMPTY

def convert_level(lines: list[str]) -> list[str]:
    """
    Convert a full MM2 ASCII level to VGLC characters, preserving its entire
    height and width. This does not crop, pad, or window the level — that
    selection step belongs to the dataset builder, which already slides a
    window across the converted level to pick scenes.
    """
    sanitized = [l for l in lines if l.strip() not in ("?", "{", "}")]
    grid = normalize_grid(sanitized)
    if not grid:
        return []

    cannon_map = find_cannon_positions(grid)

    height = len(grid)
    width  = len(grid[0])

    out_rows = []
    for r in range(height):
        row_chars = []
        for c in range(width):
            vglc_ch = convert_cell(grid[r][c], grid, r, c, cannon_map)
            row_chars.append(vglc_ch)
        out_rows.append("".join(row_chars))

    # Fill start-ground gap: the bottom ground rows in MM2 often start at col 7
    # because the 7-tile spawn platform is implicit (not drawn as tiles).
    # Find the leftmost X in the bottom two rows and fill those same rows
    # leftward to col 0 — leaving all other (sky) rows untouched.
    if len(out_rows) >= 2:
        leftmost_x = None
        for row in out_rows[-2:]:
            for i, ch in enumerate(row):
                if ch == VGLC_GROUND:
                    if leftmost_x is None or i < leftmost_x:
                        leftmost_x = i
                    break
        if leftmost_x and leftmost_x > 0:
            new_rows = list(out_rows)
            for ri in range(len(new_rows) - 2, len(new_rows)):
                row = list(new_rows[ri])
                if VGLC_GROUND not in row:
                    continue
                for i in range(min(leftmost_x, len(row))):
                    if row[i] == VGLC_EMPTY:
                        row[i] = VGLC_GROUND
                new_rows[ri] = ''.join(row)
            out_rows = new_rows

    return out_rows
