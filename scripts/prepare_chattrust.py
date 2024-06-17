# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob
import json
import os
import sys
from pathlib import Path
import polars as pl
import numpy as np
from tqdm import tqdm

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
import glob
from transformers import AutoTokenizer

filename_sets = {
    "qa": "*.parquet",
    "core": "*.parquet",
    "math": "*.parquet",
    "instruct": "*.parquet",
    "dictionary": "*.parquet",
    "code": "*.parquet"
}

def prepare_full(
    source_path: Path, tokenizer_repo: Path, destination_path: Path, chunk_size: int, match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset using the original tokenizer."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)

    count_tokens = 0
    for pattern, glob_pattern in filename_sets.items():
        if match and match not in pattern:
            continue

        print(str(source_path / pattern / glob_pattern))
        filepaths = glob.glob(str(source_path / pattern / glob_pattern))

        if not filepaths:
            raise RuntimeError(
                f"No files found matching pattern {glob_pattern} in {source_path}."
            )

        for filepath in filepaths:
            print("filepath:", filepath)

            prefix = pattern#os.path.splitext(os.path.basename(filepath))

            builder = packed_dataset.PackedDatasetBuilder(
                outdir=destination_path,
                prefix=prefix,
                chunk_size=chunk_size,
                sep_token=tokenizer.eos_id,
                dtype="auto",
                vocab_size=tokenizer.vocab_size,
            )

            print(f"Processing {filepath}")

            df = pl.read_parquet(filepath)
            texts = df["text"].to_list()
            texts_tokenized = tokenizer.tokenizer.tokenize(texts)
            for text_ids in tqdm(texts_tokenized):
                x = np.array(text_ids, dtype=builder.dtype)
                count_tokens += x.size
                builder.add_array(x)

            builder.write_reminder()
            print("Total tokens: ", count_tokens)
    print("Total tokens final: ", count_tokens)


# def prepare_full(
#     source_path: Path, tokenizer_repo: Path, destination_path: Path, chunk_size: int, match: str = ""
# ) -> None:
#     """Prepare the "Red Pajama" dataset using the original tokenizer."""
#     destination_path.mkdir(parents=True, exist_ok=True)

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo, trust_remote_code=True)

#     for name in tqdm(filename_sets):
#         if match and match not in name:
#             continue

#         filepath = source_path / name

#         print("filepath:", filepath)
#         if not filepath.is_file():
#             raise RuntimeError(
#                 f"Input file not found at {filepath}."
#             )

#         prefix, _ = os.path.splitext(name)

#         builder = packed_dataset.PackedDatasetBuilder(
#             outdir=destination_path,
#             prefix=prefix,
#             chunk_size=chunk_size,
#             sep_token=tokenizer.eos_id,
#             dtype="auto",
#             vocab_size=tokenizer.vocab_size,
#         )

#         print(f"Processing {name}")

#         df = pl.read_ndjson(filepath)
#         texts = df["text"].to_list()
#         texts_tokenized = tokenizer.tokenizer.tokenize(texts)
#         for text_ids in tqdm(texts_tokenized):
#             x = np.array(text_ids, dtype=builder.dtype)
#             builder.add_array(x)

#         builder.write_reminder()


def prepare(
    source_path: Path = Path("data/input/_pretrain"),
    tokenizer_repo: Path = Path("nampdn-ai/tknz-20k"),
    destination_path: Path = Path("data/output"),
    chunk_size: int = 2048,
    match: str = "",
) -> None:
    
    prepare_full(
        source_path=source_path,
        tokenizer_repo=tokenizer_repo,
        destination_path=destination_path,
        chunk_size=(chunk_size + 1) * 1024,  # block size + 1 for causal, 1024 blocks
        match=match,
    )

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
