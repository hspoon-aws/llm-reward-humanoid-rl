# Lesson: AWS bring-up gotchas (CDK bootstrap + Capacity Block launch)

**Date:** 2026-06-11 (Phase 0/2 infra bring-up)
**Where it bit us:** standing up the CDK foundation and launching the `p6-b200.48xlarge`
Capacity Block instance.
**Blog section:** "Provisioning the GPU host on AWS" → launch gotchas

Two infra problems that cost real time during bring-up and weren't obvious from the docs.
Captured here so they aren't lost to git history.

## 1. CDK bootstrap stuck in `ROLLBACK_FAILED`

**Symptom:** `cdk bootstrap` / `cdk deploy` failed; `describe-stacks CDKToolkit` showed
`StackStatus = ROLLBACK_FAILED`.

**Root cause:** a *prior* bootstrap attempt had run under a **limited (non-Admin) role** (an
existing instance role on the account), which lacked `iam:DeleteRole` / `iam:GetRole` etc. The
toolkit stack half-created its roles, failed, and could not roll back — leaving it wedged. The
`ROLLBACK_FAILED` resources were only the standard CDK staging roles/bucket/ECR — **no project
data**.

**Resolution:**
1. Inspect what's left: `aws cloudformation list-stack-resources --stack-name CDKToolkit` (confirm
   it's only CDK staging resources — roles, staging bucket, ECR repo, SSM param).
2. Delete under Admin: `aws cloudformation delete-stack --stack-name CDKToolkit` + `wait
   stack-delete-complete`.
3. Re-bootstrap cleanly under the Admin profile: `cdk bootstrap aws://<acct>/<region>`.

**Takeaways:**
- A wedged `CDKToolkit` is almost always safe to delete + recreate — it holds no app state.
- Bootstrap with the **right (Admin) credentials the first time**; a half-bootstrap under a
  limited role is the usual cause of `ROLLBACK_FAILED`.

## 2. Launching a Capacity Block for ML instance

The `p6-b200.48xlarge` came from a **Capacity Block** reservation, which has launch requirements a
normal on-demand run-instances doesn't.

**Symptom A:** `RunInstances ... InvalidParameterValue: The market type (purchasing) option is not
valid.`
**Cause/Fix:** Capacity Block instances must be launched with
`--instance-market-options 'MarketType=capacity-block'`. (We first tried the default/no market
type, then the standard on-demand path — both rejected.)

**Symptom B:** `InvalidParameterCombination: Network interfaces and an instance-level security
groups may not be specified on the same request.`
**Cause/Fix:** the CDK L2 `LaunchTemplate` put the SG at the top level (`SecurityGroupIds`) AND we
also set it inside the `NetworkInterfaces` override (needed for `AssociatePublicIpAddress=false`).
Those two are mutually exclusive at run-instances time. Fix: attach the SG **only** inside the
network interface (drop the top-level `security_group=`).

**Symptom C:** `run-instances` used the launch template's **default version**, not `$Latest`, so a
fixed template version kept failing.
**Fix:** pass `LaunchTemplateId=...,Version=N` explicitly, and set the corrected version as the
template default (`modify-launch-template --default-version N`).

**Takeaways:**
- Capacity Block launch = on-demand `run-instances` **plus** `MarketType=capacity-block`.
- When forcing no-public-IP via a `NetworkInterfaces` override, the SG belongs in the interface,
  never also at the top level.
- `run-instances --launch-template` defaults to the template's **default version**; pin
  `Version=` or update the default after editing.

## 3. Capacity Block billing (decision lesson, not a bug)

Stopping or terminating a **Capacity Block** instance saves **nothing** — you pre-paid for the
whole reserved window. So:
- Don't stop a Capacity Block host "to save money"; there is no per-hour burn to stop. Use it for
  whatever it *can* do until the window ends.
- Conversely, an **on-demand** GPU instance (the g6/L4 we used for Isaac Sim) *is* stoppable to
  save money — but stopping does **not** reserve its capacity, so it may fail to restart with
  `InsufficientInstanceCapacity` (see `lesson-isaac-lab-bringup.md`). Trade-off: on-demand is
  cheap-when-stopped but not guaranteed-on-restart.
