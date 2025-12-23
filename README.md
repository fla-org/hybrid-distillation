# Distilling to Hybrid Attention Models via KL-Guided Layer Selection

This repository contains the official code for the paper ***“Distilling to Hybrid Attention Models via KL-Guided Layer Selection.”***

It also includes a reimplementation of **RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale** ([arXiv:2505.03005](https://arxiv.org/abs/2505.03005)), which we use as the backbone of our distillation pipeline.


-----

## Environment Setup

```bash
git clone git@github.com:fla-org/hybrid-distillation.git
cd hybrid-distillation

conda create -n your_env_name python=3.12
conda activate your_env_name

pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Set your enviromental variable:

```bash

export HF_TOKEN="YOUR_HF_TOKEN"
export HF_HOME="YOUR_HF_HOME"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export TRITON_CACHE_DIR=...

export WANDB_ENTITY="YOUR_WANDB_ENTITY"
export WANDB_PROJECT="YOUR_WANDB_PROJECT"

```


## Preprocess corpus

1) **Download + tokenize** a Hugging Face dataset and save it to disk (`save_to_disk`).

```bash
python preprocess_download_tokenize.py \
  --dataset_name <hf_dataset_name_or_path> \
  --split train \
  --text_field text \
  --tokenizer fla-hub/Qwen2.5-3B-Instruct \
  --output_dir data_cache/tokenized_dataset
```

2) **Chunk** the concatenated token stream into fixed-length blocks for each stage.

```bash
# for stage1
python preprocess_chunk.py \
  --tokenized_dataset_path data_cache/tokenized_dataset \
  --context_length 512 \
  --output_dir data_cache

# for stage2
python preprocess_chunk.py \
  --tokenized_dataset_path data_cache/tokenized_dataset \
  --context_length 4096 \
  --output_dir data_cache
```


## Training

Training has **two stages**, driven by YAML configs under `configs/`. You typically run **Stage 1 once per (teacher, student variant)**, then reuse its checkpoint for any number of **Stage 2** runs (different ratios / selectors / layer sets).

### Stage 1: Attention Output Alignment (all-linear)

Stage 1 aligns the student’s linear-attention outputs. It is **always** run with an all-linear student:
`keep_full_attention_layers: []`.

Example (Qwen2.5-3B teacher, `gdn_v4` student):

```bash
deepspeed train.py --cfg configs/qwen2_3b_gdn_v4_hybrid_0_125_uniform/stage1.yaml
````

After Stage 1 finishes, the HF-format checkpoint is saved to:

* `<train.output_dir>/converted-hf/`

Default Stage-1 settings in the provided configs:

* Target tokens: 100M
* Sequence length: 512
* Peak LR: 1e-3
* Scheduler: cosine
* Batch size: 96

### Stage 2: Logits Distillation (hybrid)

Stage 2 distills logits from the teacher into a **hybrid** student. This is where you specify the softmax layers via
`student_model.keep_full_attention_layers`.

```bash
deepspeed train.py --cfg configs/qwen2_3b_gdn_v4_hybrid_0_125_uniform/stage2.yaml
```

Before launching Stage 2:

* Set `train.student_init_ckpt` to the Stage-1 checkpoint:

  * `<stage1_output_dir>/converted-hf/`
* Set `student_model.keep_full_attention_layers: [...]` for your hybrid.

Recommended Stage-2 settings:

* Target tokens: 600M
* Sequence length: 4096

---

## Configs

All configs live under `configs/`. A config folder usually contains a `stage2.yaml` for a specific hybrid run, and may also include a `stage1.yaml` for convenience (but you **do not** need to run Stage 1 for every folder).

### One Stage 1, many Stage 2s

For a fixed pair:

* `teacher_model.name`
* `student_model.name`

you run Stage 1 once and reuse the resulting:

* `<stage1_output_dir>/converted-hf/`

Then any Stage-2 config for that same pair should point to it via:

* `train.student_init_ckpt: <stage1_output_dir>/converted-hf/`

### Naming convention

Config directories typically follow:

`{teacher_family}_{size}_{student_variant}_hybrid_{ratio}_{selector}/`

Examples:

* `configs/qwen2_3b_gdn_v4_hybrid_0_125_uniform/`
* `configs/llama3_3b_gdn_v4_hybrid_0_25_ga_s2/`

### Common knobs

* `data.cache_dir`

  * Stage 1: usually `data_cache/chunked_context512`
  * Stage 2: usually `data_cache/chunked_context4096`
* `student_model.keep_full_attention_layers` (Stage 2): the chosen softmax layers
* `train.output_dir`: output path for checkpoints/logs
* `train.student_init_ckpt` (Stage 2): shared Stage-1 `converted-hf/`
* `train.target_tokens`, `train.train_seq_len`, batch sizes, etc.

### Adding a new Stage-2 config

1. Choose (or create) a canonical Stage-1 run for your (teacher, student) pair.
2. Run Stage 1 once and keep `<output_dir>/converted-hf/`.
3. Create new Stage-2 folders for different ratios / selectors / layer sets.
4. Point all of them to the same `train.student_init_ckpt`.


---

## Layer Selection

This part includes scripts to reproduce **KL-guided layer selection** (our GA-S2 selector) and several **baseline layer-importance heuristics**.

### GA-S2 (our method)

Our main selector measures each layer’s *marginal utility* by restoring **exactly one** layer to full (softmax) attention in an otherwise linear-attention student, running **Stage 2** distillation, then ranking layers by how much the distillation objective improves (we use the logged training loss / KL proxy).

#### 1) Run Stage 1 once (all-linear student)
First run **Stage 1** to align the student’s linear-attention outputs. This produces a Stage-1 checkpoint, which you then convert to a unified HF-style checkpoint (the `converted-hf` folder described above in the Stage 1 section).

You should end up with something like:
```

checkpoints/<exp_name>/stage1/converted-hf/

```

#### 2) Generate per-layer Stage-2 configs
Use `generate_layer_configs_ga_s2.py` to generate **N** Stage-2 YAML configs (one per layer). Each config sets:
- `keep_full_attention_layers: [idx]` for that layer index
- `output_dir` uniquely per layer (important for W&B retrieval)

Example:

```bash
python layer_selection/generate_layer_configs_ga_s2.py \
  --num-layers 28 \
  --config-root config/qwen2_7b_gdn_v4_hybrid_layer_selection \
  --teacher fla-hub/Qwen2.5-7B-Instruct \
  --student gdn_v4 \
  --cache-stage2 data_cache/chunked_context4096 \
  --outputs-prefix checkpoints/qwen2_7b_gdn_v4_hybrid_layer_selection_{index}
```

**Important:** `generate_layer_configs_ga_s2.py` contains a `STAGE2_TEMPLATE` with a `student_init_ckpt:` field.
Before launching runs, make sure it points to your **Stage-1 converted checkpoint**, e.g.

```
student_init_ckpt: 'checkpoints/<your_stage1_exp>/stage1/converted-hf'
```

Also note: the template uses `target_tokens: 600_000_000`, but for *ranking* runs, **~200M tokens is typically enough**. You can edit `target_tokens` in the template (or in the generated YAMLs) to speed up selection.

#### 3) Launch Stage 2 for every layer

Run Stage 2 once per layer (these runs are independent and can be launched in parallel on a cluster):

```bash
for i in $(seq 0 27); do
  deepspeed train.py --cfg config/qwen2_7b_gdn_v4_hybrid_layer_selection/layer_${i}/stage2.yaml
done
```

Each run logs to W&B, and **the run display name is expected to match `output_dir`** (see retrieval script below).

#### 4) Pull loss curves from Weights & Biases

After all runs finish, fetch the logged loss curves from W&B:

1. Edit these constants in `layer_selection/retrieve_loss_log_from_wandb.py`:

* `ENTITY`
* `PROJECT`
* (optionally) `OUTPUT_FILE`
* (optionally) the `RUN_NAME` pattern if your `output_dir` differs

2. Then run:

```bash
python layer_selection/retrieve_loss_log_from_wandb.py
```

This produces a JSON file mapping each run name to sampled `(train/global_step, train/loss)` points.

**Note:** the retrieval script assumes:

* runs are `finished`
* the W&B run `display_name` equals the training `output_dir`
  If your W&B naming differs, update the script accordingly.

#### 5) Convert the W&B loss log into a layer ranking

Point `FILE_TO_LOAD` in `layer_selection/get_ranking_from_wandb_loss_log.py` to the JSON produced above, then run:

```bash
python layer_selection/get_ranking_from_wandb_loss_log.py
```

This prints **layer rankings (best → worst)** at each logged step. In practice, you can take the ranking at a later step (or aggregate across steps) and then select the top-K layers as your final `keep_full_attention_layers`.

#### 6) Train the final hybrid with the selected top-K layers

Create a final Stage-2 config (or modify an existing one) with:

```yaml
student_model:
  keep_full_attention_layers: [l1, l2, ..., lK]
```

Then run the standard distillation pipeline for your final hybrid model (Stage 2 only since you can reuse Stage-1 checkpoint).

---

### Baselines: layer importance via ablation on synthetic tasks

For baseline layer-importance heuristics, we provide `evaluate_layer_importance.py`, which:

1. evaluates a model on a chosen synthetic task,
2. **ablates each layer** by zeroing its attention output (via a forward hook),
3. reports the **performance drop** per layer (larger drop ⇒ more “important”).

Supported tasks (`--task`):

* `synthetic_retrieval`
* `associative_recall`
* `associative_recall_mutihop`
* `variable_tracking`
* `common_words_extraction`
* `frequent_words_extraction`
* `phone_book`

Example (Variable Tracking):

```bash
python layer_selection/evaluate_layer_importance.py \
  --task variable_tracking \
  --model_name fla-hub/Qwen2.5-3B-Instruct \
  --output_dir layer_selection_baselines \
  --num_samples 500 \
  --batch_size 8 \
  --num_chains 2 \
  --num_hops 2 \
  --noise_ratio 1.0
```

Artifacts saved under `--output_dir`:

* `results/baseline_*.json` (baseline accuracy)
* `results/ablation_*.csv` (layer-wise accuracy + drop)
* `plots/ablation_*.png` (visualization)

---

## Evaluation

Evaluation is performed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). To install:

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install "lm_eval[ruler]" # ruler-specific install
```

To evaluate a checkpoint: 


```bash

bash eval.sh /path/to/your/checkpoint

```

Important: Use the HF-converted checkpoint format, which is automatically saved to the `converted-hf` directory inside your `output_dir`.

---

## Acknowledgements

We use the triton-implemented linear attention kernels from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention). We refer to [HazyResearch/lolcats](https://github.com/HazyResearch/lolcats) to construct our training process. The evaluation is supported by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Thank you for these excellent open-source efforts.

## Citation

TBD
