from datasets import load_dataset
from kaitaistruct import KaitaiStream
from io import BytesIO
from level import Level
import zlib

# Loads dataset from Hugging Face   
ds = load_dataset("TheGreatRambler/mm2_level", streaming=True, split="train")

# Adds a SEARCH_KEYWORD to filter datasets to get specific levels for later training purposes
SEARCH_KEYWORD = "kaizo"
filtered_ds = ds.filter(lambda example: SEARCH_KEYWORD in example["name"].lower())


match = next(iter(filtered_ds))

# Get metadata and print it
level_id = match["data_id"]
level_name = match["name"]
print("Level ID: %s | Name: %s" % (level_id, level_name))



level_data = match["level_data"]
level = Level(KaitaiStream(BytesIO(zlib.decompress(level_data))))

# NOTE level.overworld.objects is a fixed size (limitation of Kaitai struct)
# must iterate by object_count or null objects will be included
print("=== OBJECTS ===")
for i in range(level.overworld.object_count):
    obj = level.overworld.objects[i]
    print("X: %d Y: %d ID: %s" % (obj.x, obj.y, obj.id))

print("=== GROUND TILES ===")
for i in range(level.overworld.ground_count):
    tile = level.overworld.ground[i]
    # Convert to same pixel space as objects: multiply by 160
    print("X: %d Y: %d TileID: %d BgID: %d" % (
        tile.x * 160, tile.y * 160, tile.id, tile.background_id
    ))