# Lesson: self-hosted vLLM vs. Amazon Bedrock for the reward-gen LLM

**Date:** 2026-06-12 (capacity-block sprint, pre-launch planning)
**Where it came from:** evaluating Amazon Bedrock as a *backup* LLM backend in case the
self-hosted Qwen3-Coder-30B underperforms at generating reward code. The whole LLM surface is
one method (`QwenClient._chat()` in `src/llm/qwen_client.py`) talking to an OpenAI-compatible
`/chat/completions` endpoint, so the backend is swappable without touching the orchestrator,
reward executor, or prompts.
**Blog section:** "Self-hosting the language model" → when to self-host vs. call a managed API

## TL;DR

For a low-volume, code-generation workload like this Eureka-style loop, the per-token cost of a
managed API is **rounding error**, and the real comparison is **GPU allocation**, not dollars
spent on inference. Self-hosting Qwen on vLLM costs $0 in API charges but consumes a whole B200
(GPU 0) for the full sprint; Bedrock moves the LLM off-box for a few dollars and hands that GPU
back to training. The counterintuitive takeaway: at this scale the "free" self-hosted option is
the *more expensive* one once you price the GPU it occupies.

## The workload is tiny (this is the whole point)

The cost intuition flips once you look at how few LLM calls the loop actually makes. From
`config/run_config.yaml`:

- `max_iterations: 12` — at most 12 full iterations.
- A few calls per iteration: one `generate_reward`, up to `max_invalid_retries: 3` re-prompts on
  invalid code, one `analyze_failure` on failure.
- `max_tokens: 4096` per call.

Generous upper bound: ~100 calls over 24h, ~4K input + ~4K output tokens each →
**~0.4M input + ~0.4M output tokens for the entire sprint.** This is not a high-throughput
inference service; it is a handful of careful calls.

## Option 1 — Bedrock (Claude Opus 4.8, on-demand)

At us-west-2 on-demand pricing of $5 / 1M input and $25 / 1M output tokens:

| | tokens | rate | cost |
|---|---|---|---|
| Input | 0.4M | $5/M | $2.00 |
| Output | 0.4M | $25/M | $10.00 |
| **Total** | | | **~$12** |

Even tripling the call count, or falling back to Bedrock for the *entire* run, lands at
**$25–40 total**. A cheaper model (e.g. Claude Sonnet at ~$3/$15 per 1M, or Llama 3.1
70B at ~$2.65/$3.50 per 1M) is a few dollars. No infrastructure on the instance, no GPU
consumed.

## Option 2 — Qwen on vLLM (self-hosted)

Zero incremental API charge — but not free. vLLM occupies **GPU 0**, which the run config
explicitly reserves (`training.llm_gpu: 0`, leaving GPUs 1–7 for PPO). On a `p6-b200.48xlarge`
at ~$113.93/hr on-demand, one of eight GPUs is ~$14.24/hr → **~$340 of instance capacity over a
24h sprint dedicated to serving the LLM**, whether or not Qwen performs well. That cost is real;
it is just bundled into the instance bill instead of appearing on a per-token invoice.

## The real comparison: GPUs, not dollars

Both dollar figures are small next to the ~$2,700 instance bill for a 24h sprint. The meaningful
difference is **how many GPUs train**:

| | LLM API spend | Training GPUs | Notes |
|---|---|---|---|
| vLLM (self-host) | $0 | **7** | GPU 0 reserved for the model |
| Bedrock | ~$12–40 | **8** | LLM off-box; ~14% more training throughput |

So at this volume Bedrock is effectively free, and switching to it doesn't just de-risk a Qwen
failure — it *gives back a training GPU* (~14% more PPO throughput on the same instance).

## When the math would flip

Self-hosting wins decisively when **token volume is high and sustained**. If this were a
production service doing millions of calls, the per-token API cost would dominate and a
dedicated GPU (or Bedrock Provisioned Throughput) becomes the economical choice. The break-even
is roughly "is the GPU-hours cost of hosting less than the per-token cost of the same volume?"
For a 12-iteration research sprint, volume is nowhere near that line. The lesson generalizes:

> **Price the resource the "free" option silently consumes.** Self-hosting trades a per-token
> bill for a fixed GPU-hours bill. At low volume the fixed cost loses; at high volume it wins.
> The crossover is set by call volume, not by which option *looks* free.

## Cost of having the backup ready (do this in pre-work, not mid-sprint)

The only real spend for the Bedrock fallback is one-time plumbing, and it must be staged before
launch so it isn't debugged under pressure at hour 14:

- **VPC interface endpoint** `com.amazonaws.us-west-2.bedrock-runtime` (the instances run in a
  private subnet with no internet egress — Req 22). Interface endpoints bill per-hour + per-GB,
  pennies for this run.
- **IAM**: add `bedrock:InvokeModel` scoped to the chosen fallback model ARN(s) on the
  Instance_Role (least-privilege; do not broaden to `*`).
- **Model access**: confirm the model is enabled in `us-west-2` (Bedrock model access
  is granted per-model, per-account) and pick the model now. This run used Claude Opus 4.8
  for reward generation; Claude Sonnet is a cheaper alternative for the same workload.

## Caveats / things this estimate glosses over

- **Capacity-block pricing differs from on-demand.** The ~$113.93/hr figure is on-demand
  us-east-1; the actual capacity-block rate changes the GPU opportunity-cost number but not the
  conclusion (Bedrock spend stays trivial at 12 iterations).
- **Cross-region inference / model availability.** If the fallback model isn't in `us-west-2` or
  needs a cross-region inference profile, per-call routing and data-egress assumptions change.
- **The Req 16 recovery path means something different on Bedrock.** The
  `ServiceUnavailableError` wait-and-resume logic probes a loopback TCP socket to wait out a
  vLLM crash. On Bedrock, "unavailable" shows up as `ThrottlingException`, not a dead socket, so
  that branch effectively no-ops — fine for a backup, but the auto-recovery behavior is not
  identical across backends.

## Follow-ups

- [ ] Stage the Bedrock backup plumbing in pre-work (VPC endpoint + `bedrock:InvokeModel` IAM +
      confirm model access in us-west-2).
- [ ] Implement a config-selectable provider in `QwenClient._chat()` (`provider: vllm` default,
      `provider: bedrock` opt-in via `run_config.yaml`) so the fallback is a one-line config
      flip; keep offline tests green and add a Bedrock-branch test.
- [ ] If adopting Bedrock as primary, reclaim GPU 0 for training (drop `llm_gpu`, extend
      `training_gpus` to all 8) and remap the Req 16 recovery path to Bedrock throttling
      semantics.
- [ ] Reference this lesson in the assembled Blog (Req 21) under LLM self-hosting.
