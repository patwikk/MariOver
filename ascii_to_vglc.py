#!/usr/bin/env python3
"""
ascii_to_vglc.py
================
Converts Mario Maker 2 ASCII text arrays into clean VGLC format rows.
Retains all tile maps, vertical cropping, and canonical character transitions.
"""

VGLC_HEIGHT = 14

MM2_GROUND = set(
    "#" "H" "I" "C" "T" "{" "=" "N" "p" "O" "*" "³" "·" "»" "¼" "J" "Ù" "Ç" "É" "Ë" "Ø"
)
MM2_BREAKABLE = set("B")
MM2_QUESTION = set("?")
MM2_PIPE_AS_GROUND = set("|DW")
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

def convert_cell(ch: str, row: int, col: int, cannon_map: dict) -> str:
    if (row, col) in cannon_map:
        return cannon_map[(row, col)]
    if ch == " ":
        return VGLC_EMPTY
    if ch in MM2_BREAKABLE:
        return VGLC_BREAK
    if ch in MM2_QUESTION:
        return VGLC_QUESTION
    if ch in MM2_ENEMIES:
        return VGLC_ENEMY
    if ch in MM2_COINS:
        return VGLC_COIN
    if ch in MM2_PIPE_AS_GROUND:
        return VGLC_GROUND
    if ch in MM2_GROUND or ch in "/\\":
        return VGLC_GROUND
    return VGLC_EMPTY

def crop_and_pad_to_vglc_height(grid: list[list[str]]) -> list[list[str]]:
    while len(grid) > VGLC_HEIGHT and all(c == " " for c in grid[0]):
        grid = grid[1:]
    if len(grid) > VGLC_HEIGHT:
        grid = grid[-VGLC_HEIGHT:]
    width = len(grid[0]) if grid else 0
    while len(grid) < VGLC_HEIGHT:
        grid.insert(0, [" "] * width)
    return grid

def convert_level(lines: list[str]) -> list[str]:
    sanitized = [l for l in lines if l.strip() not in ("?", "{", "}")]
    grid = normalize_grid(sanitized)
    if not grid:
        return [VGLC_EMPTY * 10] * VGLC_HEIGHT

    grid = crop_and_pad_to_vglc_height(grid)
    cannon_map = find_cannon_positions(grid)

    height = len(grid)
    width  = len(grid[0])

    out_rows = []
    for r in range(height):
        row_chars = []
        for c in range(width):
            vglc_ch = convert_cell(grid[r][c], r, c, cannon_map)
            row_chars.append(vglc_ch)
        out_rows.append("".join(row_chars))

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