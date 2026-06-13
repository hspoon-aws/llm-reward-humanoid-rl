# Backup & Restore — surviving EC2 termination (capacity block end)

The B200 is on a **capacity block** that ends; the instance will be terminated.
This doc says exactly what is at risk, what is already safe, and how to back up
before termination and rehydrate a fresh box afterward.

Scripts:
- `scripts/backup_to_s3.sh` — run BEFORE termination (on the box).
- `scripts/restore_from_s3.sh` — run on the NEW box AFTER termination.

---

## 1. What is already safe (no action needed)

| Asset | Where it lives | Why it's safe |
|---|---|---|
| Source code, configs, docs, specs | **GitHub** (`YOUR_REPO`, branch `main`) | Committed + pushed (commit `8f4b2fa`+). |
| Completed-iteration artifacts (reward.py, metrics.json, policy `.pkl`+netcfg, best/worst mp4) | **S3** `runs-mujoco/<run-id>/iteration-NN/` | The loop uploads these per iteration. |
| Best-policy export | **S3** `runs-mujoco/<run-id>/` (best_policy path) | Exported at run end. |
| Qwen-3-Coder weights (57 GB) | **S3** `models/qwen3-coder-30b/` | Re-downloadable; not unique. |
| MJX venv (8 GB) | n/a | Rebuildable from `scripts/b200_setup_mjx.sh` (pinned versions). |

## 2. What is LOCAL-ONLY and lost on termination (the backup target)

| Asset | Path on box | Risk |
|---|---|---|
| **Run logs** | `/data/mujoco/runs/*.log` | **HIGH** — training curves, `eval_reward` history, `[iter]`/`[warm-start]` lines. NEVER auto-uploaded. Primary blog record. |
| In-progress run's local checkpoints | `/data/mujoco/runs/<run-id>/iter_*.pkl` not yet uploaded | MEDIUM — the iteration still running at termination is lost; completed ones are already in S3. |
| Earlier ad-hoc run dirs (`full`, `mgpu`, `smoke`, …) | `/data/mujoco/runs/...` | LOW–MEDIUM — historical, may not all be in S3. |

`backup_to_s3.sh` syncs all of `/data/mujoco/runs/` (logs + checkpoints) to
**`s3://humanoid-from-scratch-123456789012/runs-mujoco/_box_backup/`**.

> Why `_box_backup/` under `runs-mujoco/`: the instance role's `s3:PutObject` is
> scoped to `runs/*` and `runs-mujoco/*` only. A top-level `runs-mujoco-backup/`
> prefix is NOT writable and returns AccessDenied. Keep backups inside
> `runs-mujoco/`.

## 3. Backup procedure (BEFORE termination)

From your workstation (drives the box via SSM):
```bash
scripts/ssm_run.sh 'bash /data/mujoco/scripts/backup_to_s3.sh'
```
Verify it reports a non-zero log count, e.g. `done: 112 objects in backup (13 logs)`.

Also confirm (one-time, already true here):
- `git status` is clean and pushed: `git -P log -n1 --oneline` on your workstation.
- Model weights still present: `aws s3 ls s3://<bucket>/models/qwen3-coder-30b/`.

A snapshot of the actual backup taken at capacity-block end is recorded in §6.

## 4. Restore procedure (on the NEW box)

DLAMI GPU instance assumed (git + python3 + NVIDIA driver present), with an
instance role that can read the bucket (and, for Bedrock runs, `bedrock:InvokeModel`).

```bash
# copy restore_from_s3.sh onto the new box (or clone the repo and use it in place)
bash restore_from_s3.sh
```
It performs: (1) git clone the project into `/data/mujoco`, (2) recreate the MJX
venv via `b200_setup_mjx.sh`, (3) pull model weights from S3 (skip with
`SKIP_WEIGHTS=1` for Bedrock-only runs), (4) restore `runs/` + logs from
`_box_backup/`, (5) sanity-check imports + config load.

### Resuming a run
The loop checkpoints to `loop_checkpoint.json` and resumes at the next iteration
(Req 16.2). To continue where a run left off, relaunch with the SAME `--run-id`
and `--checkpoint-dir`. Warm-start (`warm_start: true`) then re-seeds training
from the best policy so far.

### Bedrock vs local vLLM after restore
- **Bedrock run** (`llm.provider: bedrock`): no GPU reserved, no weights needed
  (`SKIP_WEIGHTS=1`). Ensure the new instance role has `bedrock:InvokeModel` on
  `us.anthropic.claude-opus-4-8` (see `infra/instance-role-policy.json`, the
  `InvokeBedrockClaude` statement).
- **Local vLLM run** (`llm.provider: vllm`): download weights and restart vLLM
  (command printed at the end of `restore_from_s3.sh`); GPU 0 is reserved for it.

## 5. What is NOT backed up on purpose
- `mjxvenv` (rebuildable), system packages, `__pycache__`, `*.lock`.
- Model weights are not re-copied by the backup (already durable in S3).

## 6. Backup snapshot at capacity-block end
- Destination: `s3://humanoid-from-scratch-123456789012/runs-mujoco/_box_backup/`
- Contents at last sync: 112 objects, 13 logs (run-1gpu-v4, run-6gpu-v2,
  run-6gpu-v3-bedrock, run-8gpu-v4-warmstart, plus earlier ad-hoc runs) +
  local checkpoints/metrics + smoke demo videos.
- Re-run `backup_to_s3.sh` immediately before termination to capture the latest
  logs/checkpoints (the in-flight run advances after this snapshot).
