#!/usr/bin/env python3
"""
mm2_to_vglc.py
Converts a Mario Maker 2 ASCII level (.txt) to VGLC format.

Usage:
    python mm2_to_vglc.py input_level.txt [output_level.txt]

If no output path is given, the result is printed to stdout.
"""

import sys
import json
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# VGLC output height (the standard VGLC SMB format is 14 rows)
VGLC_HEIGHT = 14

# ---------------------------------------------------------------------------
# MM2 tile categories (derived from mm2_tileset.json semantics)
# We hard-code the mapping here so the script is self-contained, but the
# logic mirrors the tileset JSON that ships with the dataset.

# Tiles that become solid ground  X
MM2_GROUND = set(
    "#"   # ground
    "H"   # hard block
    # "S" stone -> air (treated as spawn marker placeholder, not solid)
    "I"   # ice block (solid; slipperiness not representable in VGLC)
    "C"   # crate
    "T"   # tree
    "{"   # starting brick
    "="   # castle bridge
    "N"   # note block
    "p"   # p block (togglable solid)
    "O"   # on/off block (togglable solid)
    "*"   # blinking block (togglable solid)
    "³"   # mushroom platform (solid platform)
    "·"   # bridge platform (solid)
    "»"   # conveyor belt (solid)
    "¼"   # fast conveyor belt (solid)
    "J"   # jumping machine (solid)
    "Ù"   # exclamation block (solid)
    "Ç"   # spikes — solid hazard; treated as solid ground tile in VGLC
    "É"   # skewer — solid moving hazard
    "Ë"   # icicle
    "Ø"   # cannon (shooter, solid) — handled separately below, but kept
           # here as fallback solid
)

# Tiles that become breakable brick  S
MM2_BREAKABLE = set("B")  # breakable brick

# Tiles that become question block  ?
# All question-block variants collapse to the full/active version.
MM2_QUESTION = set("?")

# Pipes and warp tiles -> ground for now (no VGLC pipe distinction needed)
MM2_PIPE      = set()   # disabled: pipes treated as ground
MM2_WARPSOLID = set()   # disabled: warp tiles treated as ground
MM2_PIPE_AS_GROUND = set("|DW")  # these all map to X

# Cannon emitter tiles (small airship cannons, etc.) — just place cannon-top
MM2_CANNON_EMITTER = set("V")  # shooter/cannon block

# Enemy tiles  E
# Everything tagged ["enemy", ...] in the tileset.
# Note: 'X' (uppercase) is listed as a single-occurrence boss in a comment in
# the tileset JSON and must be included explicitly.
MM2_ENEMIES = set(
    "g K P m t o s b L Z y < u x X @ ~ q w Y e F % & r , a n R ! 9 j + ¡ ; A v [ 1 2 3 4 5 6 7 µ"
    .split()
)
# Note: 'o' is both enemy (Bob-omb) and coin in smb.json — in MM2 tileset 'o'
# is clearly tagged enemy (explosive). Coins use '¢', '$', '£'.

# Coin tiles  o
MM2_COINS = set("¢$£")  # regular coin, red coin, big coin

# Tiles to delete (replace with '-'):
# Passable non-solid things that don't map to any VGLC concept.
# This is the catch-all for everything not listed above.
# Explicit passable/interactive/decoration tiles:
MM2_DELETE = set(
    "h d . ^ f k c l Z ¦ ¯ ± ] } ) ° ² "  # hidden block, donut, dotted, spike,
    "- ´ À Á Â Ã ¿ Ð "                    # lifts, semisolids, trampolines, one-way
    "Ì Í Î Ï Ñ Ò Ó Ô Õ Ö × "              # decorations, vines, vehicles, tracks
    "U i ¤ M ¶ § ¬ "                       # power-ups
    "¸ ¹ º ½ ¾ "                           # moving platforms (lava lift, snake, track, sprint, seesaw)
    "Ä Å Æ È Ê "                           # fire bar, saw, burner, spike ball, twister
    "G "                                   # goal / flagpole (single-occurrence passable)
    " "                                    # air / empty
    .split()
)

# ---------------------------------------------------------------------------
# VGLC pipe characters
PIPE_TOP_LEFT  = "<"
PIPE_TOP_RIGHT = ">"
PIPE_BOT_LEFT  = "["
PIPE_BOT_RIGHT = "]"
VGLC_EMPTY     = "-"
VGLC_GROUND    = "X"
VGLC_BREAK     = "S"
VGLC_QUESTION  = "?"
VGLC_ENEMY     = "E"
VGLC_COIN      = "o"
VGLC_CANNON_T  = "B"
VGLC_CANNON_B  = "b"

# ---------------------------------------------------------------------------

def load_level(path: str) -> list[str]:
    """Read the level file, preserving exact characters (no rstrip)."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Strip only the trailing newline, keep internal spaces
    lines = [l.rstrip("\n") for l in lines]
    return lines


def normalize_grid(lines: list[str]) -> list[list[str]]:
    """Return a rectangular grid of single characters; pad short rows with spaces."""
    if not lines:
        return []
    width = max(len(l) for l in lines)
    grid = []
    for l in lines:
        row = list(l)
        row += [" "] * (width - len(row))
        grid.append(row)
    return grid


def is_pipe_tile(ch: str) -> bool:
    return ch in MM2_PIPE or ch in MM2_WARPSOLID


def classify_pipe_cell(grid: list[list[str]], row: int, col: int) -> str:
    """
    Given a pipe/warp tile at (row, col), return the VGLC pipe character.

    VGLC pipe convention (SMB):
        < >   ← top of pipe (left half, right half)
        [ ]   ← body/bottom of pipe (left half, right half)

    MM2 pipe columns are 1 cell wide (single '|'), so we manufacture
    left/right halves by treating odd columns as "left" and even as "right" —
    except we check the neighbouring cell first: if the neighbour is also a
    pipe tile we split normally; if not, we just emit a single-width top.

    Simpler heuristic that works well in practice:
      - If there is NO pipe tile directly above → it's a top cell.
      - If there IS a pipe tile above → it's a body cell.
    Then for left/right we check whether the right neighbour is also pipe:
      - If right neighbour is pipe → this cell is the LEFT half.
      - Else if left neighbour is pipe → this cell is the RIGHT half.
      - Else (single-cell pipe column) → emit both halves? No — VGLC uses
        two-cell-wide pipes. We'll emit the single cell as a left half top
        and let the caller handle width. Actually: just emit < or [ for
        single-cell columns; the VGLC reference uses two-cell pipes but
        single-cell pipe columns in MM2 are common, so we just pick one
        representative character.
    """
    height = len(grid)
    width  = len(grid[0]) if height > 0 else 0

    above = grid[row - 1][col] if row > 0 else " "
    right = grid[row][col + 1] if col + 1 < width else " "
    left  = grid[row][col - 1] if col - 1 >= 0  else " "

    is_top  = not is_pipe_tile(above)
    is_left_half = is_pipe_tile(right)    # right neighbour is also pipe → we are the left side
    is_right_half = is_pipe_tile(left) and not is_pipe_tile(right)

    if is_top:
        if is_right_half:
            return PIPE_TOP_RIGHT
        else:
            return PIPE_TOP_LEFT   # single-cell or left half of two-cell
    else:
        if is_right_half:
            return PIPE_BOT_RIGHT
        else:
            return PIPE_BOT_LEFT


def find_cannon_positions(grid: list[list[str]]) -> dict[tuple[int,int], str]:
    """
    Scan for cannon emitter tiles ('V') and return a mapping of
    (row, col) → VGLC character.

    Rule: 'V' shooter tiles are 2 cells tall. The tile itself is the barrel
    (top), so place VGLC_CANNON_T there and VGLC_CANNON_B one row below.
    If the cell below is already solid ground, only place the top.
    """
    result = {}
    height = len(grid)
    width  = len(grid[0]) if height > 0 else 0
    for r in range(height):
        for c in range(width):
            if grid[r][c] == "V":
                result[(r, c)] = VGLC_CANNON_T
                # Add bottom tile if possible and not already occupied
                if r + 1 < height:
                    below_ch = grid[r + 1][c]
                    if below_ch == " " or below_ch not in MM2_GROUND:
                        result[(r + 1, c)] = VGLC_CANNON_B
    return result


def convert_cell(ch: str, grid: list[list[str]], row: int, col: int,
                 cannon_map: dict) -> str:
    """Map a single MM2 character to its VGLC equivalent."""

    # Cannon cells resolved in pre-pass
    if (row, col) in cannon_map:
        return cannon_map[(row, col)]

    # Empty / air
    if ch == " ":
        return VGLC_EMPTY

    # Breakable brick
    if ch in MM2_BREAKABLE:
        return VGLC_BREAK

    # Question block (all variants)
    if ch in MM2_QUESTION:
        return VGLC_QUESTION

    # Enemies
    if ch in MM2_ENEMIES:
        return VGLC_ENEMY

    # Coins (all coin variants)
    if ch in MM2_COINS:
        return VGLC_COIN

    # Pipe / warp tile -> ground
    if ch in MM2_PIPE_AS_GROUND:
        return VGLC_GROUND

    # Solid ground (all remaining solid, hard, platform-solid tiles)
    if ch in MM2_GROUND:
        return VGLC_GROUND

    # Slopes: treat as ground for VGLC
    if ch in "/\\":
        return VGLC_GROUND

    # Anything else passable / decorative / interactive → delete
    return VGLC_EMPTY


def crop_and_pad_to_vglc_height(grid: list[list[str]]) -> list[list[str]]:
    """
    VGLC levels are exactly VGLC_HEIGHT rows tall.

    Strategy:
      1. Drop completely empty rows from the top until we have VGLC_HEIGHT rows
         or exhaust them.
      2. If the grid is taller than VGLC_HEIGHT, keep the bottom VGLC_HEIGHT rows
         (the ground is always at the bottom in SMB-style levels).
      3. If shorter, pad the top with empty rows.
    """
    # Remove leading all-space rows
    while len(grid) > VGLC_HEIGHT and all(c == " " for c in grid[0]):
        grid = grid[1:]

    if len(grid) > VGLC_HEIGHT:
        grid = grid[-VGLC_HEIGHT:]

    width = len(grid[0]) if grid else 0
    while len(grid) < VGLC_HEIGHT:
        grid.insert(0, [" "] * width)

    return grid


def convert_level(lines: list[str]) -> list[str]:
    grid = normalize_grid(lines)

    if not grid:
        return [VGLC_EMPTY * 10] * VGLC_HEIGHT

    # Pre-pass: resolve cannon emitter positions
    cannon_map = find_cannon_positions(grid)

    # Crop/pad to VGLC height before conversion so row indexing is consistent
    grid = crop_and_pad_to_vglc_height(grid)
    # Rebuild cannon map after crop (positions may have shifted)
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
            for ri in range(len(new_rows) - 2, len(new_rows)):  # only bottom 2 rows
                row = list(new_rows[ri])
                # Only fill if this row already has some ground (it is a ground row,
                # not a pure empty/sky row that happens to be at the bottom)
                if VGLC_GROUND not in row:
                    continue
                for i in range(min(leftmost_x, len(row))):
                    if row[i] == VGLC_EMPTY:
                        row[i] = VGLC_GROUND
                new_rows[ri] = ''.join(row)
            out_rows = new_rows

    # Trim trailing all-dash columns so the level width matches actual content
    if out_rows:
        max_content_col = 0
        for row in out_rows:
            for i in range(len(row) - 1, -1, -1):
                if row[i] != VGLC_EMPTY:
                    if i > max_content_col:
                        max_content_col = i
                    break
        out_rows = [row[:max_content_col + 1] for row in out_rows]

    return out_rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Mario Maker 2 ASCII level to VGLC format."
    )
    parser.add_argument("input", help="Path to the MM2 ASCII level .txt file")
    parser.add_argument(
        "output", nargs="?", default=None,
        help="Output path (default: print to stdout)"
    )
    args = parser.parse_args()

    lines = load_level(args.input)
    vglc_rows = convert_level(lines)

    output_text = "\n".join(vglc_rows) + "\n"

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Saved VGLC level to: {args.output}")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()