import argparse
import numpy as np
from datasets import load_from_disk, Dataset
from itertools import chain
import pyarrow as pa
import os
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Chunk tokenized dataset")
    parser.add_argument(
        "--tokenized_dataset_path",
        type=str,
        default="/proj/checkpoints/songlin/data_cache/tokenized_tokens.arrow",
        help="Path to the saved tokenized dataset (from save_to_disk)"
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=16384,
        help="Context length for each chunk"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_cache/",
        help="Output directory for chunked dataset"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint32",
        choices=["uint16", "uint32", "int32"],
        help="Storage dtype for concatenated tokens"
    )
    parser.add_argument(
        "--npy_cache_path",
        type=str,
        default="data_cache/tokenized_tokens_all.npy",
        help="Optional path to .npy cache file for concatenated tokens"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.npy_cache_path and os.path.exists(args.npy_cache_path):
        print(f"Loading token array from cache: {args.npy_cache_path}")
        all_tokens = np.load(args.npy_cache_path, mmap_mode='r')
        all_tokens = all_tokens.astype(args.dtype, copy=False)
        print(f"Loaded {len(all_tokens):,} tokens from cache")
    else:
        print("Loading tokenized dataset...")
        dataset = load_from_disk(args.tokenized_dataset_path)
        print(f"Dataset has {len(dataset):,} examples")
        chunked_array = dataset.data.column("input_ids")  # pyarrow.ChunkedArray
        # Flatten each chunk and collect
        flattened_chunks = [chunk.flatten() for chunk in chunked_array.chunks]  # list of pyarrow arrays
        flat_array = pa.concat_arrays(flattened_chunks)  # single pyarrow array
        all_tokens = flat_array.to_numpy(zero_copy_only=False).astype(args.dtype, copy=False)
        if args.npy_cache_path:
            print(f"Saving concatenated tokens to cache: {args.npy_cache_path}")
            np.save(args.npy_cache_path, all_tokens)

    print(f"Processing chunks with context length {args.context_length}...")
    total_len = (len(all_tokens) // args.context_length) * args.context_length
    all_tokens = all_tokens[:total_len]
    chunks = all_tokens.reshape(-1, args.context_length)

    print(f"Total tokens: {len(all_tokens):,}, Num chunks: {len(chunks):,}")

    print("Converting to Arrow format (Fast path)...")
    arrow_scalar_type = {
        "uint16": pa.uint16(),
        "uint32": pa.uint32(),
        "int32": pa.int32()
    }[args.dtype]
    flat_array = pa.array(chunks.flatten(), type=arrow_scalar_type)
    arrow_array = pa.FixedSizeListArray.from_arrays(flat_array, args.context_length)
    table = pa.Table.from_arrays([arrow_array], names=["input_ids"])
    new_dataset = Dataset(table)

    output_path = os.path.join(args.output_dir, f"chunked_context{args.context_length}")
    print(f"Saving chunked dataset to {output_path}...")
    new_dataset.save_to_disk(output_path)

    print(f"✅ Saved chunked dataset to {output_path}")
    print(f"📊 Statistics:")
    print(f"   - Total chunks: {len(chunks):,}")
    print(f"   - Context length: {args.context_length}")
    print(f"   - Data type: {args.dtype}")

if __name__ == "__main__":
    main()
