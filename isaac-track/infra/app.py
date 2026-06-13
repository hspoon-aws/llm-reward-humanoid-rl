#!/usr/bin/env python3
"""CDK app for the humanoid-locomotion GPU sprint host.

Provisions the secure EC2 foundation (Requirement 22) for the
`p6-b200.48xlarge` capacity-block instance: a dedicated VPC with an isolated
private subnet pinned to the reservation's AZ, default-deny security group,
least-privilege SSM-managed instance role, VPC endpoints (S3 gateway + SSM
interface), and a launch template that targets the capacity reservation and
attaches the Qwen3-Coder weights volume restored from the model snapshot.

All tunables are read from cdk.json context so the same app serves
`cdk synth`/`deploy` at 21:30 without code edits.
"""
import aws_cdk as cdk

from humanoid_gpu_stack import HumanoidGpuStack

app = cdk.App()

# Account is resolved from the active AWS profile/credentials at synth time;
# region is pinned to the capacity-reservation region from context.
region = app.node.try_get_context("region") or "us-west-2"

HumanoidGpuStack(
    app,
    "HumanoidGpuStack",
    env=cdk.Environment(region=region),
    description="Secure p6-b200 capacity-block host for the humanoid LLM->RL sprint (Req 22)",
)

app.synth()
