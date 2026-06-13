# Humanoid GPU Host — CDK

Provisions the secure EC2 foundation (Requirement 22) for the `p6-b200.48xlarge`
capacity-block instance that runs the humanoid LLM→RL sprint. Synthesizes a
launch template you fire at **21:30** (inside the capacity-block window) plus the
VPC, endpoints, security group, and least-privilege role it depends on.

## What it builds

| Resource | Purpose | Req |
|----------|---------|-----|
| Dedicated VPC `10.20.0.0/16`, single AZ `us-west-2b` | matches the reservation AZ | 22.1 |
| Isolated private subnet, `AssociatePublicIpAddress: false` | no inbound route, no public IP | 22.1 |
| Default-deny security group, **zero inbound rules** | admin via SSM only | 22.2, 22.4 |
| NAT gateway + S3 gateway endpoint + `ssm`/`ec2messages`/`ssmmessages` interface endpoints | no-public-IP egress | 22.6 |
| Instance role: `models/*` read, `runs/*` write, `AmazonSSMManagedInstanceCore` | least privilege | 22.5 |
| Launch template: DLAMI, IMDSv2, capacity-reservation target, weights volume from snapshot | warm vLLM start | — |

Loopback-only service binding (vLLM/TensorBoard `--host 127.0.0.1`, Req 22.3) is
enforced by `scripts/launch_vllm.sh`; the default-deny SG is the network backstop.

All inputs are in `cdk.json` context (reservation id, AMI, snapshot, AZ, CIDR,
volume sizes). Override with `-c key=value` at synth/deploy time.

## Pinned facts (cdk.json)

- Reservation: `cr-EXAMPLE000000000` — `p6-b200.48xlarge`, `us-west-2b`, capacity-block window (24h)
- AMI: `ami-EXAMPLE0000000000` (Deep Learning Base OSS Nvidia, Ubuntu 22.04)
- Weights snapshot: `snap-EXAMPLE00000000` (150 GB xfs → `/data/models/qwen3-coder-30b`)
- Bucket: `humanoid-from-scratch-<ACCOUNT_ID>`

## Usage

```bash
cd infra
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export AWS_PROFILE=your-aws-profile

# one-time per account/region
npx cdk bootstrap --app ".venv/bin/python app.py"

# review, then deploy the foundation (VPC/endpoints/SG/role/launch template)
npx cdk synth --app ".venv/bin/python app.py"
npx cdk deploy --app ".venv/bin/python app.py"
```

Deploying the stack creates the launch template but **does not launch the GPU
instance** (the capacity reservation is consumed only when you run it). At 21:30:

```bash
LT=$(aws cloudformation describe-stacks --stack-name HumanoidGpuStack \
  --query "Stacks[0].Outputs[?OutputKey=='LaunchTemplateId'].OutputValue" --output text)
aws ec2 run-instances --region us-west-2 \
  --launch-template LaunchTemplateId=$LT \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=project,Value=humanoid-llm-rl-poc}]'
```

The instance boots, mounts the weights volume at `/data`, and is reachable only
via SSM:

```bash
aws ssm start-session --target <instance-id>
# port-forward TensorBoard / vLLM (no inbound SG rule needed):
aws ssm start-session --target <instance-id> \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["6006"],"localPortNumber":["6006"]}'
```

## Notes / decisions

- **Dedicated VPC over reuse.** Any pre-existing VPCs in the account belonged to
  unrelated CloudFormation stacks (not project-owned, no clean private+NAT path), so the
  stack builds its own to keep the security posture self-contained and tear-down clean.
- **NAT + endpoints (both).** Endpoints alone cover S3 and SSM, but the host also
  needs general egress (pip, apt, Isaac Lab/Isaac Sim assets, vLLM wheels). NAT
  provides that; the S3 gateway endpoint still keeps weight/artifact traffic
  in-region and off the NAT. Set `-c use_nat_gateway=false` for an
  endpoints-only, no-general-egress posture (you must then pre-bake all deps).
- **Weights via block device, not restore script.** The launch template creates
  the weights volume directly from the snapshot and attaches it, so the host has
  weights at boot without running `restore_and_launch.sh`. That script remains
  the fallback if you launch outside this template.
- CDK library is pinned to `2.159.x` to match the installed CLI (`2.1106`).
