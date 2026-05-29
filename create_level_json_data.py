import json
import argparse
from pathlib import Path

MM2_TILESET = 'mm2_tileset.json'
MM2_HEIGHT = 20 # full mm2 levels are 240x27
MM2_WIDTH = 20
MM2_EXTRA_TILE = '_'  # not a valid MM2 ASCII tile; used for void/padding
MM2_DEFAULT_TILE = ' '  # space is the default "empty air" tile in MM2 ASCII

"""
Loads a tileset JSON file (which defines what each tile character means).
Sorts the tile characters and assigns each a unique integer ID.
Adds a special "extra tile" for padding if not already present.
Returns a mapping from tile character to integer ID.
"""
def load_tileset(tileset_path, extra_tile=MM2_EXTRA_TILE):
    with open(tileset_path, 'r', encoding='utf-8') as f:
        tileset_data = json.load(f)
    tile_chars = sorted(tileset_data['tiles'].keys())
    if extra_tile not in tile_chars:
        tile_chars.append(extra_tile)
    tile_to_id = {char: idx for idx, char in enumerate(tile_chars)}
    return tile_to_id

"""
Reads all .txt files in the given directory.
Each file is a level, each line is a row of tiles.
Space characters are preserved (they represent empty air in MM2 ASCII).
Trailing blank lines are stripped; interior empty rows are kept.
"""
def load_levels(levels_dir):
    levels = []
    for file in sorted(Path(levels_dir).glob("*.txt")):
        print(f"Loading level file: {file}")
        with open(file, 'r', encoding='latin-1') as f:
            level = [line.rstrip('\n') for line in f]
        while level and level[-1] == '':
            level.pop()
        levels.append(level)
    return levels

"""
Extracts room clusters from the level based on the specified room dimensions and window size.
Pads the dungeon grid with void rooms if needed.
Returns a list of room clusters (each is a 3D grid of tile IDs).
"""
def room_cluster_samples(
    level,
    tile_to_id,
    room_width=11,
    room_height=16,
    window_rooms_w=2,
    window_rooms_h=2,
    extra_tile=MM2_EXTRA_TILE
):
    dungeon_rows = len(level) // room_height
    dungeon_cols = len(level[0]) // room_width

    padded_rows = dungeon_rows + (window_rooms_h - 1)
    padded_cols = dungeon_cols + (window_rooms_w - 1)
    padded_level = [row.ljust(padded_cols * room_width, extra_tile) for row in level]
    while len(padded_level) < padded_rows * room_height:
        padded_level.append(extra_tile * (padded_cols * room_width))

    print(f"Level size: {len(level)}x{len(level[0])}")
    print(f"Room size: {room_height}x{room_width}")
    print(f"Dungeon grid: {dungeon_rows} rows x {dungeon_cols} cols")
    print(f"Padded grid: {padded_rows} rows x {padded_cols} cols")
    print(f"Padded level height: {len(padded_level)}, width: {len(padded_level[0])}")

    samples = []
    for grid_y in range(padded_rows - window_rooms_h + 1):
        for grid_x in range(padded_cols - window_rooms_w + 1):
            cluster = []
            for wy in range(window_rooms_h):
                for wx in range(window_rooms_w):
                    room_top = (grid_y + wy) * room_height
                    room_left = (grid_x + wx) * room_width
                    room = []
                    for y in range(room_height):
                        row = padded_level[room_top + y][room_left:room_left + room_width]
                        room.append([tile_to_id.get(c, tile_to_id[extra_tile]) for c in row])
                    cluster.append(room)
            if any(any(tile != tile_to_id[extra_tile] for row in room for tile in row) for room in cluster):
                samples.append(cluster)
            else:
                print("Discarded empty cluster")

    print(f"Extracted {len(samples)} clusters")
    if samples:
        print("Sample cluster shape:", len(samples[0]), "rooms,", len(samples[0][0]), "rows per room,", len(samples[0][0][0]), "cols per room")
    return samples

"""
Pads the level to target_height and slides a window of size target_height x target_width
across the level horizontally, extracting all possible samples.
Converts each character to its tile ID.
Returns a list of samples (each is a 2D grid of tile IDs).
"""
def pad_and_sample(
    level,
    tile_to_id,
    target_height,
    target_width,
    stride=1,
    scan_mode="platformer",
    room_width=11,
    room_height=16,
    window_rooms_w=2,
    window_rooms_h=2,
    extra_tile=MM2_EXTRA_TILE,
):
    if scan_mode == "room_cluster":
        return room_cluster_samples(
            level,
            tile_to_id,
            room_width=room_width,
            room_height=room_height,
            window_rooms_w=window_rooms_w,
            window_rooms_h=window_rooms_h,
            extra_tile=extra_tile,
        )
    else:
        height = len(level)
        width = max(len(row) for row in level)
        pad_rows = target_height - height
        padded_level = [extra_tile * width] * pad_rows + list(level)
        padded_level = [row.ljust(target_width, extra_tile) for row in padded_level]
        samples = []
        for x in range((width - target_width + stride) // stride):
            sample = []
            for y in range(target_height):
                window_row = padded_level[y][x * stride:(x * stride) + target_width]
                sample.append([tile_to_id.get(c, tile_to_id[extra_tile]) for c in window_row])
            samples.append(sample)
        return samples

"""
Loads the tileset and levels.
For each level, extracts all possible samples.
Collects all samples into a dataset.
Writes the dataset to a JSON file.
"""
def main(
    tileset_path,
    levels_dir,
    output_path,
    target_height,
    target_width,
    stride=1,
    scan_mode="platformer",
    room_width=11,
    room_height=16,
    window_rooms_w=2,
    window_rooms_h=2,
    extra_tile=MM2_EXTRA_TILE
):
    tile_to_id = load_tileset(tileset_path, extra_tile=extra_tile)
    levels = load_levels(levels_dir)
    dataset = []
    for level in levels:
        samples = pad_and_sample(
            level,
            tile_to_id,
            target_height,
            target_width,
            stride=stride,
            scan_mode=scan_mode,
            room_width=room_width,
            room_height=room_height,
            window_rooms_w=window_rooms_w,
            window_rooms_h=window_rooms_h,
            extra_tile=extra_tile
        )
        dataset.extend(samples)
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tileset', default=MM2_TILESET, help='Path to the MM2 tileset JSON')
    parser.add_argument('--levels', default='levels', help='Directory containing MM2 ASCII level .txt files')
    parser.add_argument('--output', required=True, help='Path to the output JSON file')
    parser.add_argument('--target_height', type=int, default=MM2_HEIGHT, help='Target sample height in tiles')
    parser.add_argument('--target_width', type=int, default=MM2_WIDTH, help='Target sample width in tiles')
    parser.add_argument('--extra_tile', default=MM2_EXTRA_TILE, help='Void/padding tile (must not be a real MM2 tile)')
    parser.add_argument('--stride', type=int, default=1, help='Horizontal stride for the sliding window')
    parser.add_argument('--scan_mode', default='platformer', choices=['platformer', 'room_cluster'], help='Sampling mode')
    parser.add_argument('--room_width', type=int, default=11, help='Room width (room_cluster mode)')
    parser.add_argument('--room_height', type=int, default=16, help='Room height (room_cluster mode)')
    parser.add_argument('--window_rooms_w', type=int, default=2, help='Window width in rooms (room_cluster mode)')
    parser.add_argument('--window_rooms_h', type=int, default=2, help='Window height in rooms (room_cluster mode)')

    args = parser.parse_args()

    main(
        args.tileset,
        args.levels,
        args.output,
        args.target_height,
        args.target_width,
        args.stride,
        scan_mode=args.scan_mode,
        room_width=args.room_width,
        room_height=args.room_height,
        window_rooms_w=args.window_rooms_w,
        window_rooms_h=args.window_rooms_h,
        extra_tile=args.extra_tile
    )
