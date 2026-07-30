#!/usr/bin/env python3
"""Compress a .pcbank memory bank by deterministic row sampling."""
import argparse
import shutil
import struct
from pathlib import Path

MAGIC = b"PCBNK01\0"
HEADER = struct.Struct("<8sIIIff")


def sample_indices(total_rows, target_rows):
    if target_rows >= total_rows:
        return list(range(total_rows))
    if target_rows <= 0:
        raise ValueError("--rows must be positive")
    if target_rows == 1:
        return [0]
    return [
        round(index * (total_rows - 1) / (target_rows - 1))
        for index in range(target_rows)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source .pcbank")
    parser.add_argument("--output", required=True, help="Compressed .pcbank")
    parser.add_argument("--rows", type=int, required=True, help="Target memory-bank rows")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    data = source.read_bytes()
    if len(data) < HEADER.size:
        raise RuntimeError("Invalid .pcbank: header is missing")

    magic, version, rows, cols, mean, stddev = HEADER.unpack(data[:HEADER.size])
    if magic != MAGIC or version != 1:
        raise RuntimeError("Unsupported .pcbank format")

    row_bytes = cols * 2
    expected = HEADER.size + rows * row_bytes
    if len(data) != expected:
        raise RuntimeError("Invalid .pcbank: file size does not match rows/cols")

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.rows >= rows:
        shutil.copyfile(source, output)
        print(f"Copied {source} -> {output}; rows={rows}, cols={cols}")
        return

    body = data[HEADER.size:]
    indices = sample_indices(rows, args.rows)
    with output.open("wb") as handle:
        handle.write(HEADER.pack(MAGIC, version, len(indices), cols, mean, stddev))
        for index in indices:
            start = index * row_bytes
            handle.write(body[start:start + row_bytes])
    print(f"Compressed {source} -> {output}; rows={rows}->{len(indices)}, cols={cols}")


if __name__ == "__main__":
    main()
