#!/usr/bin/env python3
"""
mm2view_to_extended.py
Converts a Mario Maker 2 ASCII level (.txt) to an extended simplified tile format.

Like mm2view_to_vglc.py but with ~20 granular tile types instead of collapsing
everything to generic "enemy" / "ground". VGLC characters are reused where they
apply so the two formats stay comparable.

Output characters
-----------------
    -   air / empty
    X   ground (all solid terrain)
    S   brick (breakable brick)
    ?   question block
    o   coin
    E   goomba / generic enemy
    K   koopa
    P   piranha plant (piranha flower, piranha creeper, muncher)
    ^   spike block / spikes
    N   breakable non-brick block (hidden block, donut block)
    M   mushroom platform
    B   cannon top
    b   cannon bottom
    =   bridge (passable bridge platform)
    _   semisolid platform
    W   stone block
    F   fire flower
    T   thwomp

    Upright pipe (↑ / |):
    <   pipe top-left        >   pipe top-right
    [   pipe body-left       ]   pipe body-right

    Upside-down pipe (↓, ceiling-mounted):
    (   pipe cap-left        )   pipe cap-right
    {   pipe body-left       }   pipe body-right

    Sideways pipes (← / →) are treated as solid ground (X).

Usage:
    python mm2view_to_extended.py input_level.txt [output_level.txt]
"""

import sys
import argparse
from pathlib import Path

VGLC_HEIGHT = 20

# ---------------------------------------------------------------------------
# MM2 source character sets

MM2_GROUND = {
    "#",   # ground
    "H",   # hard block
    "I",   # ice block
    "C",   # crate
    "T",   # tree
    "=",   # castle bridge
    "N",   # note block
    "p",   # p block (togglable solid)
    "O",   # on/off block (togglable solid)
    "*",   # blinking block (togglable solid)
    "»",   # conveyor belt
    "¼",   # fast conveyor belt
    "J",   # jumping machine
    "Ù",   # exclamation block
    "É",   # skewer (solid moving hazard)
    "Ë",   # icicle
    "Ø",   # wall cannon (solid structure)
}

MM2_BREAKABLE      = {"B"}           # breakable brick → S
MM2_QUESTION       = {"?"}           # question block → ?
MM2_COINS          = {"¢", "$", "£"} # coin, red coin, big coin → o
MM2_PIPE_UPRIGHT   = {"|", "↑"}      # standard pipe (mouth up)   → <>[]]
MM2_PIPE_DOWN      = {"↓"}           # ceiling pipe (mouth down)  → (){}
MM2_PIPE_SIDEWAYS  = {"←", "→"}     # sideways pipe              → X (ground)
MM2_PIPE           = MM2_PIPE_UPRIGHT | MM2_PIPE_DOWN | MM2_PIPE_SIDEWAYS
MM2_WARP_AS_GROUND = {"D", "W"}      # door / warp box → X

MM2_CANNON         = {"V"}           # bullet bill blaster → B / b

MM2_KOOPA          = {"K"}                    # koopa → K
MM2_PIRANHA        = {"P", "e", ","}          # piranha flower / creeper / muncher → P
MM2_SPIKE          = {"^", "Ç"}              # spike block, spikes → ^
MM2_MUSHROOM_PLAT  = {"³"}                   # mushroom platform → M
MM2_BRIDGE         = {"·"}                   # bridge → =
MM2_SEMISOLID      = {"´", "Â"}             # semisolid / half-collision platform → _
MM2_STONE          = {"S"}                   # stone block → W
MM2_FIRE_FLOWER    = {"i"}                   # fire flower → F
MM2_BREAKABLE_NB   = {"d", "Ã"}             # donut block, falling platform → N

MM2_GOOMBA         = {"g"}                   # goomba → E (explicit so it's documented)
MM2_THWOMP         = {"t"}                   # thwomp → T

# All remaining enemies → E (generic fallback)
MM2_ENEMIES_GENERIC = set(
    "m o s b L Z y < u X x @ ~ q w Y F % & r a n R ! 9 j + ¡ ; A v [ 1 2 3 4 5 6 7 µ"
    .split()
)

# ---------------------------------------------------------------------------
# Output characters — must match the chars in extended_tiles.json exactly
OUT_EMPTY    = " "
OUT_GROUND   = "#"
OUT_BRICK    = "B"
OUT_QUESTION = "?"
OUT_COIN     = "¢"
OUT_ENEMY    = "g"
OUT_KOOPA    = "K"
OUT_PIRANHA  = "P"
OUT_SPIKE    = "^"
OUT_MUSHROOM = "³"
OUT_BREAK_NB = "N"
OUT_CANNON_T = "V"
OUT_CANNON_B = "V"
OUT_BRIDGE   = "·"
OUT_SEMISOLID= "´"
OUT_STONE    = "S"
OUT_FIRE_FL  = "i"
OUT_THWOMP   = "t"
PIPE_UPRIGHT = "↑"   # upright pipe (cap and body collapse to one char)
PIPE_DOWN    = "↓"   # ceiling pipe (cap and body)

# ---------------------------------------------------------------------------

def load_level(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return [l.rstrip("\n") for l in lines]


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


def is_pipe_tile(ch: str) -> bool:
    return ch in MM2_PIPE


def classify_pipe_cell(grid: list[list[str]], row: int, col: int) -> str:
    ch = grid[row][col]
    if ch in MM2_PIPE_SIDEWAYS:
        return OUT_GROUND
    if ch in MM2_PIPE_DOWN:
        return PIPE_DOWN
    return PIPE_UPRIGHT


def find_cannon_positions(grid: list[list[str]]) -> dict[tuple[int, int], str]:
    result = {}
    height = len(grid)
    width  = len(grid[0]) if height > 0 else 0
    for r in range(height):
        for c in range(width):
            if grid[r][c] in MM2_CANNON:
                result[(r, c)] = OUT_CANNON_T
                if r + 1 < height:
                    below = grid[r + 1][c]
                    if below == " " or below not in MM2_GROUND:
                        result[(r + 1, c)] = OUT_CANNON_B
    return result


def convert_cell(ch: str, grid: list[list[str]], row: int, col: int,
                 cannon_map: dict) -> str:
    if (row, col) in cannon_map:
        return cannon_map[(row, col)]

    if ch == " ":
        return OUT_EMPTY

    # Specific enemies (check before generic fallback)
    if ch in MM2_KOOPA:
        return OUT_KOOPA
    if ch in MM2_PIRANHA:
        return OUT_PIRANHA
    if ch in MM2_THWOMP:
        return OUT_THWOMP
    if ch in MM2_GOOMBA or ch in MM2_ENEMIES_GENERIC:
        return OUT_ENEMY

    # Coins
    if ch in MM2_COINS:
        return OUT_COIN

    # Pipe → geometry-aware classification
    if ch in MM2_PIPE:
        return classify_pipe_cell(grid, row, col)

    # Brick
    if ch in MM2_BREAKABLE:
        return OUT_BRICK

    # Question block
    if ch in MM2_QUESTION:
        return OUT_QUESTION

    # Spike blocks / spikes
    if ch in MM2_SPIKE:
        return OUT_SPIKE

    # Mushroom platform
    if ch in MM2_MUSHROOM_PLAT:
        return OUT_MUSHROOM

    # Bridge
    if ch in MM2_BRIDGE:
        return OUT_BRIDGE

    # Semisolid / pass-through platform
    if ch in MM2_SEMISOLID:
        return OUT_SEMISOLID

    # Stone block
    if ch in MM2_STONE:
        return OUT_STONE

    # Fire flower
    if ch in MM2_FIRE_FLOWER:
        return OUT_FIRE_FL

    # Breakable non-brick blocks
    if ch in MM2_BREAKABLE_NB:
        return OUT_BREAK_NB

    # Warp structures → solid ground
    if ch in MM2_WARP_AS_GROUND:
        return OUT_GROUND

    # Solid ground
    if ch in MM2_GROUND:
        return OUT_GROUND

    # Slopes
    if ch in "/\\":
        return OUT_GROUND

    return OUT_EMPTY


def crop_and_pad(grid: list[list[str]], target: int = VGLC_HEIGHT) -> list[list[str]]:
    while len(grid) > target and all(c == " " for c in grid[0]):
        grid = grid[1:]
    if len(grid) > target:
        grid = grid[-target:]
    width = len(grid[0]) if grid else 0
    while len(grid) < target:
        grid.insert(0, [" "] * width)
    return grid


def convert_level(lines: list[str]) -> list[str]:
    grid = normalize_grid(lines)
    if not grid:
        return [OUT_EMPTY * 10] * VGLC_HEIGHT

    grid = crop_and_pad(grid)
    cannon_map = find_cannon_positions(grid)

    height = len(grid)
    width  = len(grid[0])

    out_rows = []
    for r in range(height):
        out_rows.append("".join(
            convert_cell(grid[r][c], grid, r, c, cannon_map)
            for c in range(width)
        ))

    # Fill implicit spawn-platform gap at the left edge of the bottom two rows
    if len(out_rows) >= 2:
        leftmost_x = None
        for row in out_rows[-2:]:
            for i, ch in enumerate(row):
                if ch == OUT_GROUND:
                    if leftmost_x is None or i < leftmost_x:
                        leftmost_x = i
                    break
        if leftmost_x and leftmost_x > 0:
            new_rows = list(out_rows)
            for ri in range(len(new_rows) - 2, len(new_rows)):
                row = list(new_rows[ri])
                if OUT_GROUND not in row:
                    continue
                for i in range(min(leftmost_x, len(row))):
                    if row[i] == OUT_EMPTY:
                        row[i] = OUT_GROUND
                new_rows[ri] = "".join(row)
            out_rows = new_rows

    # Trim trailing all-empty columns
    if out_rows:
        max_col = 0
        for row in out_rows:
            for i in range(len(row) - 1, -1, -1):
                if row[i] != OUT_EMPTY:
                    if i > max_col:
                        max_col = i
                    break
        out_rows = [row[:max_col + 1] for row in out_rows]

    return out_rows


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Mario Maker 2 ASCII level to the extended tile format."
    )
    parser.add_argument("input", help="Path to the MM2 ASCII level .txt file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output path (default: print to stdout)")
    args = parser.parse_args()

    lines = load_level(args.input)
    out_rows = convert_level(lines)

    output_text = "\n".join(out_rows) + "\n"

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"Saved to: {args.output}")
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
