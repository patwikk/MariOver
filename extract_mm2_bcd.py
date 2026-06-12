"""
Extract .bcd level files from the TheGreatRambler/mm2_level HuggingFace dataset,
formatted correctly for use with Toost (https://github.com/TheGreatRambler/toost).

Background
----------
The on-disk .bcd course file is exactly 0x5C000 (376,832) bytes:
  [0x000–0x00F]  16-byte header (unknown / padding)
  [0x010–0x5BFD0-ish]  AES-128-CBC encrypted course payload
  [0x5BFD0–0x5BFFF]  48-byte trailer:
      [+0x00..+0x0F]  IV  (16 bytes)
      [+0x10..+0x1F]  seed words s0..s3 (4 × uint32 LE) used to derive the key
      [+0x20..+0x2F]  CMAC placeholder / zeros

The dataset's `level_data` column is  zlib( decrypted_payload )  where
decrypted_payload starts at offset 0x10 and is 0x5C000-0x40 = 0x5BFC0 bytes.

To produce a valid .bcd we must:
  1. Decompress level_data → get the 0x5BFC0-byte plaintext.
  2. Choose random IV + seed words.
  3. Derive the AES key from those seed words using the course_key_table
     (same algorithm as simontime/SMM2CourseDecryptor).
  4. AES-128-CBC encrypt the plaintext.
  5. Build the 0x5C000-byte file:
       [0x000..0x00F]  zeros  (the original 16-byte header; we don't have it
                              so we use zeros — toost skips it)
       [0x010..0x5BFCF]  ciphertext
       [0x5BFD0..0x5BFFF]  IV || s0..s3 || zeros×16

Key derivation (from simontime's main.c)
-----------------------------------------
    rand_state = [s0, s1, s2, s3]   (or defaults if all zero)
    For i in 0..3:
        key_word[i] = 0
        For j in 0..3:
            key_word[i] <<= 8
            key_word[i] |= (course_key_table[rand_gen() >> 26]
                             >> ((rand_gen() >> 27) & 24)) & 0xFF

The XORSHIFT128 PRNG:
    n = state[0] ^ (state[0] << 11)
    state[0] = state[1]
    n ^= (n >> 8) ^ state[3] ^ (state[3] >> 19)
    state[1] = state[2]
    state[2] = state[3]
    state[3] = n
    return n

Usage
-----
    # Stream a small sample (recommended for testing):
    python extract_mm2_bcd.py --output_dir ./bcd_levels --limit 100

    # Extract specific data_ids:
    python extract_mm2_bcd.py --ids 3000004 3000007

    # Extract everything (streaming, ~100 GB):
    python extract_mm2_bcd.py --output_dir ./bcd_levels

Requirements:
    pip install datasets pycryptodome
"""

import argparse
import os
import struct
import zlib
from pathlib import Path

# ---------------------------------------------------------------------------
# AES-128-CBC  (requires pycryptodome: pip install pycryptodome)
# ---------------------------------------------------------------------------

def aes_cbc_encrypt(key_bytes: bytes, iv_bytes: bytes, plaintext: bytes) -> bytes:
    from Crypto.Cipher import AES
    cipher = AES.new(key_bytes, AES.MODE_CBC, iv_bytes)
    return cipher.encrypt(plaintext)


# ---------------------------------------------------------------------------
# Key table from simontime/SMM2CourseDecryptor  (keys.h → course_key_table)
# ---------------------------------------------------------------------------

COURSE_KEY_TABLE = [
    0x7AB1C9D2, 0xCA750936, 0x3003E59C, 0xF261014B,
    0x2E25160A, 0xED614811, 0xF1AC6240, 0xD59272CD,
    0xF38549BF, 0x6CF5B327, 0xDA4DB82A, 0x820C435A,
    0xC95609BA, 0x19BE08B0, 0x738E2B81, 0xED3C349A,
    0x045275D1, 0xE0A73635, 0x1DEBF4DA, 0x9924B0DE,
    0x6A1FC367, 0x71970467, 0xFC55ABEB, 0x368D7489,
    0x0CC97D1D, 0x17CC441E, 0x3528D152, 0xD0129B53,
    0xE12A69E9, 0x13D1BDB7, 0x32EAA9ED, 0x42F41D1B,
    0xAEA5F51F, 0x42C5D23C, 0x7CC742ED, 0x723BA5F9,
    0xDE5B99E3, 0x2C0055A4, 0xC38807B4, 0x4C099B61,
    0xC4E4568E, 0x8C29C901, 0xE13B34AC, 0xE7C3F212,
    0xB67EF941, 0x08038965, 0x8AFD1E6A, 0x8E5341A3,
    0xA4C61107, 0xFBAF1418, 0x9B05EF64, 0x3C91734E,
    0x82EC6646, 0xFB19F33E, 0x3BDE6FE2, 0x17A84CCA,
    0xCCDF0CE9, 0x50E4135C, 0xFF2658B2, 0x3780F156,
    0x7D8F5D68, 0x517CBED1, 0x1FCDDF0D, 0x77A58C94,
]

MASK32 = 0xFFFFFFFF


def rand_init(s0, s1, s2, s3):
    cond = s0 | s1 | s2 | s3
    if cond:
        return [s0, s1, s2, s3]
    return [1, 0x6C078967, 0x714ACB41, 0x48077044]


def rand_gen(state):
    n = (state[0] ^ ((state[0] << 11) & MASK32)) & MASK32
    state[0] = state[1]
    n = (n ^ (n >> 8) ^ state[3] ^ (state[3] >> 19)) & MASK32
    state[1] = state[2]
    state[2] = state[3]
    state[3] = n
    return n


def gen_key(key_table, state):
    """Produce a 16-byte AES key (4 × uint32 LE) from the PRNG state."""
    out = [0, 0, 0, 0]
    for i in range(4):
        for _ in range(4):
            out[i] = (out[i] << 8) & MASK32
            idx   = rand_gen(state) >> 26          # 6-bit index into table (64 entries)
            shift = (rand_gen(state) >> 27) & 24   # shift ∈ {0, 8, 16, 24}
            out[i] |= (key_table[idx] >> shift) & 0xFF
    return struct.pack("<4I", *out)


# ---------------------------------------------------------------------------
# Build a valid encrypted .bcd from a raw decrypted payload
# ---------------------------------------------------------------------------

COURSE_FILE_SIZE  = 0x5C000   # 376,832 bytes
HEADER_SIZE       = 0x10      # 16-byte header we skip / zero-pad
TRAILER_SIZE      = 0x30      # 48-byte trailer: IV(16) + seed(16) + zeros(16)
PAYLOAD_SIZE      = COURSE_FILE_SIZE - HEADER_SIZE - TRAILER_SIZE  # 0x5BFC0

# Offset of the gamestyle field (s2le) within the decompressed payload,
# per level.ksy: sum of the fixed header fields before it (52 bytes) plus
# the 189-byte unk1 padding = 241.
GAMESTYLE_OFFSET  = 241
GAMESTYLE_SM3DW   = 22323   # level.ksy enum gamestyle: sm3dw


def get_gamestyle_raw(plaintext: bytes) -> int:
    """Read the raw gamestyle enum value from a decoded level payload."""
    return struct.unpack_from("<h", plaintext, GAMESTYLE_OFFSET)[0]


def build_bcd(plaintext: bytes) -> bytes:
    """
    Given the decrypted course payload (0x5BFC0 bytes), return the full
    0x5C000-byte encrypted .bcd file that Toost can load.
    """
    if len(plaintext) != PAYLOAD_SIZE:
        raise ValueError(
            f"Plaintext must be exactly {PAYLOAD_SIZE} bytes, got {len(plaintext)}"
        )

    # Choose a fixed-but-valid seed and IV.
    # Using the same seed every time is fine — encryption is deterministic
    # and toost only cares about decrypting correctly.
    # We use a reproducible non-zero seed so rand_init doesn't fall back to defaults.
    s0, s1, s2, s3 = 0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0
    iv_bytes = bytes([
        0x4E, 0x69, 0x6E, 0x74, 0x65, 0x6E, 0x64, 0x6F,
        0x4D, 0x61, 0x6B, 0x65, 0x72, 0x32, 0x30, 0x31,
    ])  # "NintendoMaker201" — arbitrary, consistent

    state = rand_init(s0, s1, s2, s3)
    key_bytes = gen_key(COURSE_KEY_TABLE, state)

    ciphertext = aes_cbc_encrypt(key_bytes, iv_bytes, plaintext)

    # Trailer layout  (0x30 bytes):
    #   [0x00..0x0F]  IV
    #   [0x10..0x13]  s0 LE  (seed used to derive key)
    #   [0x14..0x17]  s1 LE
    #   [0x18..0x1B]  s2 LE
    #   [0x1C..0x1F]  s3 LE
    #   [0x20..0x2F]  zeros (CMAC placeholder)
    trailer = (
        iv_bytes
        + struct.pack("<4I", s0, s1, s2, s3)
        + b"\x00" * 16
    )

    bcd = b"\x00" * HEADER_SIZE + ciphertext + trailer
    assert len(bcd) == COURSE_FILE_SIZE, f"BCD size mismatch: {len(bcd)}"
    return bcd


# ---------------------------------------------------------------------------
# Dataset extraction
# ---------------------------------------------------------------------------

def decompress_level_data(raw) -> bytes:
    if isinstance(raw, (list, bytearray)):
        raw = bytes(raw)
    try:
        return zlib.decompress(raw)
    except zlib.error:
        pass
    try:
        return zlib.decompress(raw, 16 + zlib.MAX_WBITS)
    except zlib.error as e:
        raise ValueError(f"Cannot decompress: {e}") from e


def extract_levels(
    output_dir: str,
    limit=None,
    streaming: bool = True,
    data_id_filter=None,
    name_filter=None,
    name_count=None,
    skip_3dworld: bool = False,
):
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset (streaming={streaming}) …")
    ds = load_dataset("TheGreatRambler/mm2_level", streaming=streaming, split="train")

    saved = skipped = errors = 0
    skipped_3dw = 0
    name_saved = 0

    for row in ds:
        data_id = row["data_id"]

        if data_id_filter is not None and data_id not in data_id_filter:
            continue
        
        if name_filter is not None:
            level_name = str(row.get("name", ""))
            if name_filter.lower() not in level_name.lower():
                continue

        raw = row["level_data"]
        if raw is None:
            skipped += 1
            continue

        try:
            plaintext = decompress_level_data(raw)
        except ValueError as e:
            print(f"  [WARN] data_id={data_id}: decompress error: {e}")
            errors += 1
            continue

        if len(plaintext) != PAYLOAD_SIZE:
            print(f"  [WARN] data_id={data_id}: unexpected size {len(plaintext)}, skipping")
            errors += 1
            continue

        if skip_3dworld and get_gamestyle_raw(plaintext) == GAMESTYLE_SM3DW:
            print(f"  [SKIP] data_id={data_id}: skipped because level is 3D World")
            skipped_3dw += 1
            continue

        try:
            bcd = build_bcd(plaintext)
        except Exception as e:
            print(f"  [WARN] data_id={data_id}: encrypt error: {e}")
            errors += 1
            continue

        if name_filter is not None:
            name_saved += 1
            filename = f"{data_id}_{name_saved}.bcd"
        else:
            filename = f"{data_id}.bcd"

        (out / filename).write_bytes(bcd)
        saved += 1

        if name_filter is not None and name_count is not None:
            if name_saved >= name_count:
                break

        if saved % 500 == 0 or saved == 1:
            print(f"  Saved {saved} levels …  (last: {data_id}.bcd)")

        if limit is not None and saved >= limit:
            break

    print(
        f"\nDone. Saved: {saved}  |  Skipped (null): {skipped}  |  "
        f"Skipped (3D World): {skipped_3dw}  |  Errors: {errors}"
    )
    print(f"Output dir: {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Extract properly-formatted .bcd level files from the "
            "TheGreatRambler/mm2_level HuggingFace dataset for use with Toost."
        )
    )
    p.add_argument("--output_dir", "-o", default="./bcd_levels")
    p.add_argument("--limit", "-n", type=int, default=None,
                   help="Max levels to extract (default: all)")
    p.add_argument("--no_stream", action="store_true",
                   help="Download full dataset first (~100 GB)")
    p.add_argument("--ids", nargs="+", type=int, default=None,
                   metavar="DATA_ID", help="Only extract these data_ids")
    p.add_argument("--name", type=str, default=None, help="Extract levels whose name contains this text")
    p.add_argument("--name_count", type=int, default=None, help="Number of matching levels to extract")
    p.add_argument("--skip_3dworld", action="store_true",
                   help="Skip levels whose gamestyle is Super Mario 3D World")
    return p.parse_args()

# python extract_mm2_bcd.py --name "mario" --name_count 25

if __name__ == "__main__":
    args = parse_args()
    extract_levels(
        output_dir=args.output_dir,
        limit=args.limit,
        streaming=not args.no_stream,
        data_id_filter=set(args.ids) if args.ids else None,
        name_filter=args.name,
        name_count=args.name_count,
        skip_3dworld=args.skip_3dworld,
    )