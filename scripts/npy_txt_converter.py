"""
Convert .npy <-> .txt with shape/dtype metadata for round-trip restoration.

Usage examples:
  python scripts/npy_txt_converter.py npy2txt --input data --recursive
  python scripts/npy_txt_converter.py txt2npy --input data --recursive

  python scripts/npy_txt_converter.py npy2txt --input data/smd/train.npy
  python scripts/npy_txt_converter.py txt2npy --input data/smd/train.txt
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable

import numpy as np


def _iter_files(input_path: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    if recursive:
        yield from input_path.rglob(pattern)
    else:
        yield from input_path.glob(pattern)


def _format_for_dtype(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer):
        return "%d"
    if np.issubdtype(dtype, np.bool_):
        return "%d"
    return "%.18e"


def npy_to_txt(npy_path: Path, txt_path: Path) -> None:
    arr = np.load(npy_path, allow_pickle=False)

    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"Complex dtype not supported for txt conversion: {npy_path}")

    flat = arr.reshape(-1)
    header = f"shape={arr.shape};dtype={arr.dtype}"
    fmt = _format_for_dtype(arr.dtype)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(txt_path, flat, fmt=fmt, header=header, comments="# ")


def _parse_header(txt_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with txt_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()

    if not first.startswith("#"):
        raise ValueError(
            f"Missing metadata header in {txt_path}. Expected '# shape=(...);dtype=...'."
        )

    meta = first.lstrip("#").strip()
    parts = [p.strip() for p in meta.split(";")]
    kv = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip()

    if "shape" not in kv or "dtype" not in kv:
        raise ValueError(
            f"Invalid metadata header in {txt_path}. Expected shape and dtype entries."
        )

    shape = ast.literal_eval(kv["shape"])
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, tuple):
        raise ValueError(f"Invalid shape in header: {shape}")

    dtype = np.dtype(kv["dtype"])
    return shape, dtype


def txt_to_npy(txt_path: Path, npy_path: Path) -> None:
    shape, dtype = _parse_header(txt_path)

    # loadtxt returns scalar for single-value files; normalize to 1-D array.
    data = np.loadtxt(txt_path, comments="#")
    data = np.atleast_1d(data)

    expected = int(np.prod(shape)) if len(shape) > 0 else 1
    if data.size != expected:
        raise ValueError(
            f"Data length mismatch in {txt_path}: expected {expected}, got {data.size}"
        )

    if len(shape) == 0:
        arr = np.array(data[0], dtype=dtype)
    else:
        arr = data.reshape(shape).astype(dtype, copy=False)

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, arr)


def run_npy2txt(input_path: Path, recursive: bool) -> None:
    files = list(_iter_files(input_path, "*.npy", recursive))
    if not files:
        print("No .npy files found.")
        return

    converted = 0
    for npy_file in files:
        txt_file = npy_file.with_suffix(".txt")
        npy_to_txt(npy_file, txt_file)
        converted += 1
        print(f"[OK] {npy_file} -> {txt_file}")

    print(f"Converted {converted} file(s) from npy to txt.")


def run_txt2npy(input_path: Path, recursive: bool) -> None:
    files = list(_iter_files(input_path, "*.txt", recursive))
    if not files:
        print("No .txt files found.")
        return

    converted = 0
    for txt_file in files:
        npy_file = txt_file.with_suffix(".npy")
        txt_to_npy(txt_file, npy_file)
        converted += 1
        print(f"[OK] {txt_file} -> {npy_file}")

    print(f"Converted {converted} file(s) from txt to npy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert .npy <-> .txt with metadata")
    parser.add_argument("mode", choices=["npy2txt", "txt2npy"], help="Conversion mode")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories",
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if args.mode == "npy2txt":
        run_npy2txt(input_path, args.recursive)
    else:
        run_txt2npy(input_path, args.recursive)


if __name__ == "__main__":
    main()
