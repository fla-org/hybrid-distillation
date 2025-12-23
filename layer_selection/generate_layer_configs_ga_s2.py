#!/usr/bin/env python3
from pathlib import Path
import argparse
from textwrap import dedent


STAGE2_TEMPLATE = """\
data:
  cache_dir: '{cache_stage2}'

teacher_model:
  name: '{teacher}'

student_model:
  name: '{student}'
  keep_full_attention_layers: [{idx}]

train:
  target_tokens: 600_000_000
  batch_size: 32
  micro_batch_size: 1
  train_seq_len: 4096
  lr_scheduler_type: constant
  lr: 0.000007 # all but attention layers
  lr_attn: 0.0001 # attention layers
  max_grad_norm: 1.0
  output_dir: '{out_prefix}/stage2'
  student_init_ckpt: 'checkpoints/qwen2_7b_gdn_v4_hybrid_0_25_uniform/stage1/converted-hf'
  max_length: 4096
  # tokenizer
  add_eos_token: False
  resume_from_checkpoint: None

stage: 2
"""

def main():
    p = argparse.ArgumentParser(description="Generate per-layer hybrid selection configs.")
    p.add_argument("--num-layers", type=int, default=28)
    p.add_argument("--config-root", default="config/qwen2_7b_gdn_v4_hybrid_layer_selection")
    p.add_argument("--teacher", default="fla-hub/Qwen2.5-7B-Instruct")
    p.add_argument("--student", default="gdn_v4")
    p.add_argument("--cache-stage1", default="data_cache/chunked_context512")
    p.add_argument("--cache-stage2", default="data_cache/chunked_context4096")
    p.add_argument(
        "--outputs-prefix",
        default="checkpoints/qwen2_7b_gdn_v4_hybrid_layer_selection_{index}",
        help="Prefix for output_dir; {index} will be replaced with layer index."
    )
    args = p.parse_args()

    root = Path(args.config_root)
    root.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_layers):
        layer_dir = root / f"layer_{idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = args.outputs_prefix.format(index=idx)

        stage2_yaml = STAGE2_TEMPLATE.format(
            cache_stage2=args.cache_stage2,
            teacher=args.teacher,
            student=args.student,
            idx=idx,
            out_prefix=out_prefix,
        )

        (layer_dir / "stage2.yaml").write_text(dedent(stage2_yaml))

    print(f"Done. Wrote {args.num_layers} layers under {root.resolve()}")

if __name__ == "__main__":
    main()
