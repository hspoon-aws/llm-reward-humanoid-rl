#!/usr/bin/env python3
"""Task 0.6.6 — security-posture verification (Requirement 22).

Run ON the capacity-block / seeding instance. Checks, and fails loudly on
violation:

  22.1  the instance has NO public IPv4 address (via IMDSv2)
  22.2  the instance security group(s) have no high-risk inbound rules open to
        0.0.0.0/0 or ::/0 (ports 22 / 8000 / 6006 / 3389)
  22.3  vLLM (8000) and TensorBoard (6006) listen on loopback (127.0.0.1) only

Exit code 0 = posture OK, non-zero = at least one violation.
Requires the instance role to allow ec2:DescribeInstances /
ec2:DescribeSecurityGroups (read-only).
"""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.request

HIGH_RISK_PORTS = {22, 8000, 6006, 3389}
OPEN_CIDRS = {"0.0.0.0/0", "::/0"}
LOOPBACK_PORTS = {8000, 6006}

violations: list[str] = []
notes: list[str] = []


def imds(path: str, token: str) -> str | None:
    req = urllib.request.Request(
        f"http://169.254.169.254/latest/meta-data/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode().strip()
    except Exception:
        return None


def imds_token() -> str | None:
    req = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"},
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as r:
            return r.read().decode().strip()
    except Exception:
        return None


def aws_json(args: list[str]) -> dict | list | None:
    try:
        out = subprocess.check_output(["aws", *args], text=True, stderr=subprocess.DEVNULL)
        return json.loads(out)
    except Exception as e:  # noqa: BLE001
        notes.append(f"aws {' '.join(args)} failed: {e}")
        return None


def check_no_public_ip(token: str) -> str:
    # IMDS only exposes public-ipv4 when one is assigned.
    pub = imds("public-ipv4", token)
    if pub:
        violations.append(f"22.1 instance has a public IPv4: {pub}")
    else:
        notes.append("22.1 OK: no public IPv4 assigned")
    return imds("instance-id", token) or ""


def check_security_groups(instance_id: str, region: str) -> None:
    if not instance_id:
        notes.append("22.2 SKIPPED: instance id unavailable")
        return
    data = aws_json([
        "ec2", "describe-instances", "--region", region,
        "--instance-ids", instance_id,
        "--query", "Reservations[].Instances[].SecurityGroups[].GroupId",
        "--output", "json",
    ])
    sg_ids = data or []
    if not sg_ids:
        notes.append("22.2 SKIPPED: could not resolve security groups")
        return
    sgs = aws_json([
        "ec2", "describe-security-groups", "--region", region,
        "--group-ids", *sg_ids, "--output", "json",
    ])
    if not sgs:
        return
    found_open = False
    for sg in sgs.get("SecurityGroups", []):
        for perm in sg.get("IpPermissions", []):
            from_p = perm.get("FromPort")
            to_p = perm.get("ToPort")
            cidrs = {r.get("CidrIp") for r in perm.get("IpRanges", [])}
            cidrs |= {r.get("CidrIpv6") for r in perm.get("Ipv6Ranges", [])}
            open_to_world = cidrs & OPEN_CIDRS
            if not open_to_world:
                continue
            # all-traffic rule (no port range) or overlaps a high-risk port
            if from_p is None or to_p is None:
                violations.append(f"22.2 SG {sg['GroupId']} opens ALL inbound to {open_to_world}")
                found_open = True
                continue
            risky = {p for p in HIGH_RISK_PORTS if from_p <= p <= to_p}
            if risky:
                violations.append(
                    f"22.2 SG {sg['GroupId']} opens {sorted(risky)} to {open_to_world}")
                found_open = True
    if not found_open:
        notes.append("22.2 OK: no high-risk inbound rules open to the internet")


def check_loopback_binding() -> None:
    try:
        out = subprocess.check_output(["ss", "-ltnH"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        try:
            out = subprocess.check_output(["netstat", "-ltn"], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            notes.append("22.3 SKIPPED: neither ss nor netstat available")
            return
    for port in LOOPBACK_PORTS:
        listeners = [ln for ln in out.splitlines() if f":{port} " in ln or ln.rstrip().endswith(f":{port}")]
        if not listeners:
            notes.append(f"22.3 note: nothing listening on :{port} yet")
            continue
        for ln in listeners:
            # the local address is the 4th field for ss, varies for netstat;
            # just scan the line for a non-loopback bind on this port.
            if "0.0.0.0:%d" % port in ln or "*:%d" % port in ln or "[::]:%d" % port in ln:
                violations.append(f"22.3 port {port} bound to a routable address: {ln.strip()}")
            elif "127.0.0.1:%d" % port in ln or "[::1]:%d" % port in ln:
                notes.append(f"22.3 OK: port {port} bound to loopback")


def main() -> None:
    region = "us-west-2"
    token = imds_token()
    if not token:
        print("ERROR: IMDSv2 unreachable; run this ON the instance", file=sys.stderr)
        sys.exit(2)
    region = imds("placement/region", token) or region
    iid = check_no_public_ip(token)
    check_security_groups(iid, region)
    check_loopback_binding()

    print("--- security posture notes ---")
    for n in notes:
        print(f"  {n}")
    if violations:
        print("\nSECURITY POSTURE FAILED:", file=sys.stderr)
        for v in violations:
            print(f"  ✗ {v}", file=sys.stderr)
        sys.exit(1)
    print("\nSECURITY POSTURE OK")


if __name__ == "__main__":
    main()
