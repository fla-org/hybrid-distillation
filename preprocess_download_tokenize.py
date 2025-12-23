import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser(description="Download a HF dataset, tokenize text, and save_to_disk.")
    p.add_argument("--dataset_name", type=str, required=True,
                   help="HF dataset name")
    p.add_argument("--dataset_config", type=str, default=None,
                   help="Optional HF dataset config name.")
    p.add_argument("--split", type=str, default="train",
                   help="Dataset split to use (train/validation/test or custom).")
    p.add_argument("--text_field", type=str, default="text",
                   help="Which column contains raw text.")
    p.add_argument("--tokenizer", type=str, required=True,
                   help="Tokenizer name/path, e.g. 'Qwen/Qwen2.5-3B-Instruct'.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save the tokenized dataset (save_to_disk).")

    p.add_argument("--max_length", type=int, default=None,
                   help="Optional truncation length per example BEFORE concatenation. "
                        "Usually leave None for pretraining corpora.")
    p.add_argument("--num_proc", type=int, default=8,
                   help="Num processes for dataset.map tokenization.")
    p.add_argument("--batch_size", type=int, default=1024,
                   help="Batch size for tokenization.")
    p.add_argument("--remove_columns", action="store_true",
                   help="Remove non-token columns after tokenization.")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ds_kwargs = {}
    if args.dataset_config:
        ds_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset_name, **ds_kwargs, split=args.split)

    if args.text_field not in dataset.column_names:
        raise ValueError(f"text_field='{args.text_field}' not in columns: {dataset.column_names}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize_batch(batch):
        texts = batch[args.text_field]
        out = tok(
            texts,
            add_special_tokens=False,
            truncation=(args.max_length is not None),
            max_length=args.max_length,
        )
        return {"input_ids": out["input_ids"]}

    keep_cols = None if not args.remove_columns else [args.text_field]
    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        remove_columns=(dataset.column_names if args.remove_columns else None),
        desc="Tokenizing",
    )

    if "input_ids" not in tokenized.column_names:
        raise RuntimeError("Tokenization did not produce 'input_ids' column.")

    tokenized.save_to_disk(args.output_dir)
    print(f"✅ Saved tokenized dataset to: {args.output_dir}")
    print(f"Columns: {tokenized.column_names}")

if __name__ == "__main__":
    main()
