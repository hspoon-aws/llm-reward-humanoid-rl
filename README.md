# Teaching a humanoid to walk by writing its reward function with an LLM

An **Eureka-style loop** where a large language model writes and iteratively
refines the *reward function* for a reinforcement-learning agent, and the agent
(a Unitree H1 humanoid) learns to walk to a goal and stop there, upright. No
human hand-tunes the reward math; the control surface is a plain-English task
description.

> Built and explored over a single 24-hour 8-GPU reservation. This repo is a
> **reference / write-up**, not a turnkey product: the code is shared to
> illustrate the approach and the lessons, not to be run as-is.

## The story

Read the write-up first, it's the point of this repo:

- **[Read the article online](https://hspoon-aws.github.io/llm-reward-humanoid-rl/blog/blog.html)** — the styled version with embedded demo videos and diagrams (best experience).
- **[blog/blog.md](blog/blog.md)** — the same article as Markdown (also as raw
  [blog.html](blog/blog.html)).

## What's here

| Path | What it is |
|---|---|
| `blog/` | The article (md + html) and demo-video / metric assets. |
| `docs/lessons/` | The engineering lessons in detail (hardware fit, vLLM vs managed API, the hour-zero bring-up gotchas, prompting for reward code, sim-to-real, GPU scaling, training budget). |
| `isaac-track/` | Reference code for the **NVIDIA Isaac Lab** track (PyTorch / RSL-RL), plus the self-hosted vLLM serving scripts and the AWS CDK infra. |
| `mujoco-track/` | Reference code for the **MuJoCo MJX** track (JAX / Brax PPO). |

The two tracks run the same task with the same LLM-driven loop on two different
simulators and two different GPU classes, which makes them a genuine
Isaac-vs-MuJoCo comparison. See each track's `README.md`.

## How the loop works

![The Eureka loop: the Unitree H1 environment and a plain-English task combine to prompt an LLM, which writes a reward function; PPO trains a policy on it; metrics and video feed back so the model refines the reward, and round again.](blog/assets/eureka-loop.svg)

The LLM writes a `compute_reward(...)` function; PPO trains a policy with it; an
evaluator scores the result on metrics the LLM didn't choose; those metrics feed
back so the model refines the reward. Repeat.

## How it runs on AWS

![AWS architecture: a no-public-IP GPU host in a private subnet runs the LLM-to-reward-to-RL loop, reaching S3 for weights and artifacts and Amazon Bedrock for reward generation, with operator access only through SSM Session Manager and egress via NAT plus VPC endpoints.](blog/assets/aws-architecture.svg)

A single GPU host runs the loop inside a locked-down VPC: no public IP,
operator access only through SSM Session Manager, weights and run artifacts in
S3, and (optionally) Amazon Bedrock for reward generation. See
[`isaac-track/infra/`](isaac-track/infra/) for the CDK stack.

## A note on the code

This is reference material extracted from a research sprint. It is **not**
maintained, has no guarantees, and was scrubbed of account-specific values
before publishing:

- AWS account IDs appear as the placeholder `123456789012`.
- Instance / snapshot IDs appear as `i-EXAMPLE...` / `snap-EXAMPLE...`.
- The S3 bucket `humanoid-from-scratch-123456789012` is a placeholder; substitute your own.
- AWS profile names appear as `your-aws-profile`.

You will need to supply your own AWS account, bucket, model weights, and GPU
hardware to run anything here. Test suites and internal planning specs were
intentionally left out to keep the repo focused on the reference implementation.

## Stack

- **Simulators:** NVIDIA Isaac Lab (Isaac Sim) and MuJoCo MJX (MuJoCo Playground).
- **RL:** RSL-RL PPO (PyTorch) and Brax PPO (JAX).
- **Reward LLM:** self-hosted Qwen3-Coder-30B via vLLM, and Amazon Bedrock (Claude Opus 4.8).
- **Robot:** Unitree H1 humanoid.

## License

[MIT](LICENSE).
