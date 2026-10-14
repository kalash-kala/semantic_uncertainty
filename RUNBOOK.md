# Run Book — semantic_uncertainty pipeline

How to run the NQ (text, 6 LLMs) and VQAv2/AdVQA (vision, 3 VLMs) generation
jobs on the A100 box and the H200/A100 SLURM cluster, individually or in
combination, and where each one reads/writes data.

## 1. The two execution environments

| | This box (`/home/kalashkala/semantic_uncertainty`) | SLURM cluster (`/home/sriramg/kalashabhayk/semantic_uncertainty`) |
|---|---|---|
| How you run things | Direct: `python generate_answers_combined.py ...` in the background (`setsid nohup ... &`) | `sbatch` — queued, dispatched when a node frees up |
| GPUs | 2× A100 80GB, shared with other users' jobs — check `nvidia-smi` before launching | `a100` and `h200` partitions, shared cluster-wide — check `sinfo`/`squeue` |
| Used for | VQAv2 / AdVQA (vision) runs so far | NQ (text, 6-model) runs |
| conda env | `semantic_uncertainty` (`/root/miniconda3/envs/semantic_uncertainty/bin/python`) | same env name, activated via `conda activate semantic_uncertainty` inside the SLURM script |

These are genuinely different machines with different filesystems — paths on
one do not exist on the other. Section 4 spells out exactly where each script
reads/writes.

---

## 2. NQ text-model runs (SLURM: A100 + H200)

Files (repo root, `/home/sriramg/kalashabhayk/semantic_uncertainty/` on the cluster):

- `submit_nq_all.sh` — driver, loops over all 6 models and calls `sbatch`
- `run_nq_a100.sh` — SLURM template, `#SBATCH --partition=a100 --gres=gpu:1`
- `run_nq_h200.sh` — SLURM template, `#SBATCH --partition=h200 --gres=gpu:h200:1`
- `manifests/nq_50k_seed10.json` — pins the exact 50,000 row ids so every
  server/partition scores the identical set (read via `--selection_manifest`,
  bypasses RNG sampling — verified working)

### 2.1 Check GPU availability before submitting

```bash
sinfo -p a100 -o "%P %a %D %T"
sinfo -p h200 -o "%P %a %D %T"
squeue -p a100 -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
squeue -p h200 -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

Look for `idle`/`mix` node states. `alloc` = fully occupied.

### 2.2 Run all 6 models

```bash
cd /home/sriramg/kalashabhayk/semantic_uncertainty
./submit_nq_all.sh a100   # submit all 6 A100 jobs
./submit_nq_all.sh h200   # submit all 6 H200 jobs
./submit_nq_all.sh all    # submit all 12 (only if both partitions are free)
```

Each `sbatch` call is an **independent** job — its own GPU, its own 24h wall
clock. They do not run sequentially and do not block each other.

### 2.3 Run one model individually

```bash
# A100
sbatch --export=MODEL_NAME="Qwen/Qwen2.5-7B-Instruct",GENERATION_BATCH_SIZE=128 run_nq_a100.sh

# H200
sbatch --export=MODEL_NAME="google/gemma-3-27b-it",GENERATION_BATCH_SIZE=36 run_nq_h200.sh
```

To run a subset instead of all 6, comment out the models you don't want in
`submit_nq_all.sh`'s `MODELS=(...)` array and rerun it, or issue the
individual `sbatch` lines above directly.

### 2.4 Confirmed batch sizes (from the completed probe, `logs/probe_nq/results.txt`)

| model | A100 bs | H200 bs (multiplier) |
|---|---|---|
| qwen25_7b | 128 | 179 (1.40x) |
| llama31_8b | 128 | 179 (1.40x) |
| mistral7b | 128 | 179 (1.40x) |
| gemma3_12b | 32 | 46 (1.45x) |
| qwen3_14b | 64 | 99 (1.55x) |
| gemma3_27b | 16 | 36 (2.30x) |

H200 sizes are A100 size × multiplier, rounded down. Multipliers came from
measured VRAM-scaling `(140-W)/(79-W)` with a safety discount, not a flat
1.5x.

### 2.5 Resuming a job that hit the 24h wall

```bash
sbatch --export=MODEL_NAME="google/gemma-3-27b-it",GENERATION_BATCH_SIZE=16,RESUME_DIR="<run dir>" run_nq_a100.sh
```

### 2.6 Storage locations (NQ / SLURM cluster)

`HF_HOME` and `SCRATCH_DIR` are **derived from `$HOME`** inside both
templates (not hardcoded), so they automatically resolve correctly under
whichever SLURM account actually runs the job — no edit needed if you submit
from a different account:

```bash
HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
SCRATCH_DIR=${SCRATCH_DIR:-"/storage/users${HOME#/home}"}
```

For the current account (`$HOME=/home/sriramg/kalashabhayk`) this resolves
to exactly the values below. Both remain overridable via
`--export=HF_HOME=...,SCRATCH_DIR=...` if you ever want to point somewhere
else deliberately.

| what | path (current account) | notes |
|---|---|---|
| HF model + dataset cache | `/home/sriramg/kalashabhayk/.cache/huggingface` (`HF_HOME`) | Both `run_nq_a100.sh` and `run_nq_h200.sh` derive this identically from `$HOME`, so all 12 jobs share one cache — no re-download across partitions. **This is a different cache from the one used on the direct-access A100 box (see §4.1).** `MODEL_NAME` is always an HF repo id (e.g. `Qwen/Qwen2.5-7B-Instruct`), never a filesystem path — `transformers` maps it to `$HF_HOME/hub/models--<org>--<name>/` automatically; you never construct that folder name yourself. |
| Run outputs (`combined_generations.jsonl`, uncertainty PKLs) | `/storage/users/sriramg/kalashabhayk/semantic_uncertainty_data/uncertainty/<run_tag>/` (`SCRATCH_DIR`) | Derived from `$HOME` as above; the pipeline's own default (`/data/kalashkala`) is **not** used here — `/data/kalashkala` doesn't exist on the cluster. |
| Selection manifest | `manifests/nq_50k_seed10.json`, relative to `--chdir` (repo root) | Committed to git, so identical on every clone. |
| SLURM stdout/stderr | `slurm_logs/logs/%x_%j.out`, `slurm_logs/errors/%x_%j.err` | **Still hardcoded** absolute paths in the `#SBATCH` directives, unlike `HF_HOME`/`SCRATCH_DIR` — `#SBATCH` lines are parsed by SLURM before the shell runs, so they cannot reference `$HOME` or any shell variable. If the repo ever lives at a different path (different account, moved repo), these two lines plus `--chdir` are the ones that need manual editing before submitting. |

---

## 3. VQAv2 / AdVQA vision-model runs (this box, direct — no SLURM)

File: `semantic_uncertainty/scripts/run_vqav2_serial.sh`

Runs the 3 VLMs **serially** on one GPU (they don't coexist on one card
alongside other traffic). Models: `Qwen/Qwen2.5-VL-7B-Instruct`,
`google/gemma-3-12b-it`, `mistral-community/pixtral-12b`.

### 3.1 Check GPU availability before launching

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

The script targets GPU 1 (`CUDA_VISIBLE_DEVICES=1`) — check GPU 1 specifically
is free (or edit that line) before launching, since other users share this
box's 2 A100s.

### 3.2 Run all remaining models

```bash
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty
setsid nohup ./scripts/run_vqav2_serial.sh < /dev/null > logs/vqav2_serial_driver.log 2>&1 &
```

Always use `setsid` — plain `nohup` alone dies when the parent shell/session
exits. Progress: `tail -f logs/vqav2_serial_driver.log` (start/finish
markers) and `logs/vqav2_<model>.log` (per-model generation + uncertainty
computation).

### 3.3 Run one model individually

Edit `run_model` calls in `run_vqav2_serial.sh`, or invoke the underlying
command directly:

```bash
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty
export HF_HOME=/data/.cache/huggingface
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/semantic_uncertainty/bin/python \
  generate_answers_combined.py \
  --model_name "google/gemma-3-12b-it" \
  --dataset vqav2 \
  --model_max_new_tokens 50 \
  --num_generations 10 \
  --num_samples 40000 \
  --num_few_shot 0 \
  --generation_batch_size 12 \
  --metric squad \
  --no-compute_p_true \
  --no-save_hidden_states \
  --no-use_context \
  --entity vqav2_run \
  >> logs/vqav2_gemma3_12b.log 2>&1
```

### 3.4 Confirmed batch sizes (from `logs/probe/results.txt`)

| model | working batch size | note |
|---|---|---|
| qwen25vl | 16 | ran successfully at the original default; never re-probed |
| gemma3_12b | **12** | OOM'd at 16 (iteration 26/2187); probe confirmed 12 works |
| pixtral12b | **8** | script originally had 4; probe confirmed 8 works, doubling throughput |

### 3.5 Storage locations (VQAv2 / this box)

| what | path | notes |
|---|---|---|
| HF model cache | `/data/.cache/huggingface` (`HF_HOME`) | **Different from the SLURM cluster's `HF_HOME`** (`/home/sriramg/kalashabhayk/.cache/huggingface`, §2.6) — these two are on different machines/filesystems and do not share downloads. If you ever run a text model on this box too, it will populate `/data/.cache/huggingface`, separate from the cluster's cache. |
| VQAv2 images + question/annotation JSON | `/data/kalashkala/vqa_data/vqav2/sampled/*.json`, `/data/kalashkala/vqa_data/coco/val2014/` | Local files, read directly in `uncertainty/data/data_utils.py` — **not** affected by `HF_HOME` at all, since this data isn't pulled from the HF hub. |
| AdVQA images + JSON (if run) | `/data/kalashkala/vqa_data/{advqa,coco/val2017}/` | Same story — local, not `HF_HOME`. |
| Run outputs | `/data/kalashkala/semantic_uncertainty_data/uncertainty/<run_tag>/` (`SCRATCH_DIR`, defaults to `/data/kalashkala` since unset here) | **This is the pipeline's built-in default** — the NQ/SLURM scripts override it explicitly to a cluster path instead (§2.6); this script does not override it, so it falls through to the default. |

---

## 4. Storage-location summary — where the discrepancies actually are

There is **no discrepancy within a single environment** — every script in a
given environment (SLURM cluster, or this box) points at one shared
`HF_HOME` and one shared `SCRATCH_DIR`, so jobs on the same machine never
re-download or write to divergent caches.

The discrepancies are **across** the two environments, because they're
different machines:

1. **`HF_HOME` differs by environment**, not by dataset or model:
   - SLURM cluster (NQ jobs): `/home/sriramg/kalashabhayk/.cache/huggingface`
   - This box (VQAv2/AdVQA jobs): `/data/.cache/huggingface`
   - These are separate filesystems. A model downloaded on one is not visible
     on the other — if you ever run e.g. `Qwen/Qwen2.5-7B-Instruct` on both,
     it downloads twice.

2. **`SCRATCH_DIR` (run outputs) differs by environment**:
   - SLURM cluster: `/storage/users/sriramg/kalashabhayk/semantic_uncertainty_data/...` (explicit override in `run_nq_{a100,h200}.sh`)
   - This box: `/data/kalashkala/semantic_uncertainty_data/...` (the pipeline's built-in default, unoverridden)

3. **Image/VQA data is never in `HF_HOME` at all** — `vqav2` and `advqa` are
   loaded from local JSON + JPEG files under `/data/kalashkala/vqa_data/`,
   hardcoded in `uncertainty/data/data_utils.py`. This path only exists on
   this box; it would need to exist (and be populated) on the SLURM cluster
   too before a vision job could run there.

4. **Repo path itself differs**: this box has the repo at
   `/home/kalashkala/semantic_uncertainty`; the SLURM templates hardcode
   `/home/sriramg/kalashabhayk/semantic_uncertainty` in their `#SBATCH
   --chdir/--output/--error` lines (required, since `#SBATCH` directives
   can't reference shell variables). If the cluster repo ever moves, these
   three lines in both `run_nq_a100.sh` and `run_nq_h200.sh` need manual
   updates — nothing else in the pipeline cares about absolute repo path.

**Net effect if you're running everything from
`/home/sriramg/kalashabhayk/semantic_uncertainty` on the SLURM cluster only**
(as you described): no discrepancy to worry about — `HF_HOME` and
`SCRATCH_DIR` are consistent across all 12 NQ jobs (6 models × 2 partitions),
since both templates hardcode the same values. The discrepancy only becomes
relevant if you also run vision jobs on this A100 box, or move the repo to a
new path.

---

## 5. Quick reference — all commands

```bash
# NQ, SLURM cluster — check availability, then submit
sinfo -p a100 -o "%P %a %D %T"; sinfo -p h200 -o "%P %a %D %T"
cd /home/sriramg/kalashabhayk/semantic_uncertainty
./submit_nq_all.sh a100          # all 6 models, A100
./submit_nq_all.sh h200          # all 6 models, H200
./submit_nq_all.sh all           # all 12 jobs
sbatch --export=MODEL_NAME="...",GENERATION_BATCH_SIZE=N run_nq_a100.sh   # one model, A100
sbatch --export=MODEL_NAME="...",GENERATION_BATCH_SIZE=N run_nq_h200.sh   # one model, H200

# VQAv2, this box — check GPU, then launch
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
cd /home/kalashkala/semantic_uncertainty/semantic_uncertainty
setsid nohup ./scripts/run_vqav2_serial.sh < /dev/null > logs/vqav2_serial_driver.log 2>&1 &
tail -f logs/vqav2_serial_driver.log
```
