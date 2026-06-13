"""HumanoidGpuStack — secure capacity-block host for the LLM->RL sprint.

Implements the design's "Security Foundation" and Requirement 22:

  22.1  private subnet, no public IPv4, no Elastic IP
  22.2  default-deny security group (zero inbound rules)
  22.3  loopback-only services (enforced by launch scripts; SG is the backstop)
  22.4  admin access via SSM Session Manager only (no SSH/bastion)
  22.5  least-privilege instance role: models/* read, runs/* write, SSM core
  22.6  no-public-IP egress via NAT gateway + S3 gateway endpoint and
        ssm/ec2messages/ssmmessages interface endpoints

The instance targets the capacity reservation by ID, runs on the DLAMI, and
boots with the Qwen3-Coder weights volume (created from the model snapshot)
attached so vLLM serves without re-downloading.
"""
from pathlib import Path

from aws_cdk import (
    Aws,
    CfnOutput,
    Stack,
    Tags,
    aws_ec2 as ec2,
    aws_iam as iam,
)
from constructs import Construct


class HumanoidGpuStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context
        project_tag = ctx("project_tag")
        bucket_name = ctx("bucket_name")
        az = ctx("az")
        vpc_cidr = ctx("vpc_cidr")
        instance_type = ctx("instance_type")
        cr_id = ctx("capacity_reservation_id")
        ami_id = ctx("ami_id")
        root_volume_gb = int(ctx("root_volume_gb"))
        model_dir = ctx("model_dir")
        use_nat = bool(ctx("use_nat_gateway"))

        bucket_arn = f"arn:aws:s3:::{bucket_name}"

        # ----------------------------------------------------------------- #
        # Network (Req 22.1, 22.6)
        # ----------------------------------------------------------------- #
        # One private subnet pinned to the reservation AZ. PRIVATE_WITH_EGRESS
        # routes outbound through a NAT gateway (no inbound route, no public IP
        # on the instance). A PUBLIC subnet exists only to host the NAT gateway
        # itself; the GPU host never lands there.
        subnet_config = [
            ec2.SubnetConfiguration(
                name="private-egress",
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                cidr_mask=20,
            )
        ]
        if use_nat:
            subnet_config.insert(
                0,
                ec2.SubnetConfiguration(
                    name="nat-public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
            )

        vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr(vpc_cidr),
            availability_zones=[az],  # pin every subnet to the reservation AZ
            nat_gateways=1 if use_nat else 0,
            subnet_configuration=subnet_config,
            restrict_default_security_group=True,
        )

        # Gateway endpoint keeps S3 weight/artifact traffic in-region and works
        # even with NAT disabled (Req 22.6, preferred path).
        vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
        )

        # Interface endpoints so SSM Session Manager works without internet
        # egress (Req 22.4, 22.6).
        private_selection = ec2.SubnetSelection(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        )
        for name, svc in (
            ("Ssm", ec2.InterfaceVpcEndpointAwsService.SSM),
            ("Ec2Messages", ec2.InterfaceVpcEndpointAwsService.EC2_MESSAGES),
            ("SsmMessages", ec2.InterfaceVpcEndpointAwsService.SSM_MESSAGES),
        ):
            vpc.add_interface_endpoint(
                f"{name}Endpoint",
                service=svc,
                subnets=private_selection,
                private_dns_enabled=True,
            )

        # ----------------------------------------------------------------- #
        # Security group (Req 22.2): default-deny, ZERO inbound rules.
        # ----------------------------------------------------------------- #
        sg = ec2.SecurityGroup(
            self,
            "HostSg",
            vpc=vpc,
            description="Humanoid GPU host: default-deny inbound, egress only (Req 22.2)",
            allow_all_outbound=True,
        )
        # Intentionally NO add_ingress_rule calls. Admin arrives over SSM.

        # ----------------------------------------------------------------- #
        # Instance role (Req 22.5): least privilege, SSM core only.
        # ----------------------------------------------------------------- #
        role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="Least-privilege role for the humanoid GPU host (Req 22.5)",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                )
            ],
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ListProjectBucket",
                actions=["s3:ListBucket", "s3:GetBucketLocation"],
                resources=[bucket_arn],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ReadModelWeights",
                actions=["s3:GetObject"],
                resources=[f"{bucket_arn}/models/*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="WriteRunArtifacts",
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                resources=[f"{bucket_arn}/runs/*"],
            )
        )
        # Self-discovery + tag own volumes (used by restore/verify scripts).
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DescribeAndTagProjectResources",
                actions=["ec2:DescribeVolumes", "ec2:CreateTags"],
                resources=["*"],
            )
        )

        # ----------------------------------------------------------------- #
        # Launch template (Req 22.1 no public IP; capacity-reservation target)
        # ----------------------------------------------------------------- #
        user_data_raw = (
            Path(__file__).parent / "user_data.sh"
        ).read_text(encoding="utf-8")
        # Weights hydrate from S3 -> local NVMe instance store at boot (NOT from
        # an EBS snapshot, which lazy-hydrates at ~4 MB/s without FSR and stalls
        # vLLM for hours). See docs/lesson-ebs-snapshot-hydration.md.
        user_data_raw = (
            user_data_raw.replace("__MODEL_DIR__", model_dir)
            .replace("__REGION__", self.region)
            .replace("__BUCKET__", bucket_name)
        )
        user_data = ec2.UserData.custom(user_data_raw)

        machine_image = ec2.MachineImage.generic_linux({self.region: ami_id})

        # Only a root volume: sized for the DLAMI + Isaac Lab/Sim + the model
        # weights and compile caches hydrated to the local NVMe instance store.
        # No EBS-from-snapshot weights volume — the snapshot hydration penalty is
        # exactly what we are avoiding by pulling weights from S3 to NVMe.
        block_devices = [
            ec2.BlockDevice(
                device_name="/dev/sda1",  # DLAMI Ubuntu root
                volume=ec2.BlockDeviceVolume.ebs(
                    root_volume_gb,
                    volume_type=ec2.EbsDeviceVolumeType.GP3,
                    delete_on_termination=True,
                    encrypted=True,
                ),
            ),
        ]

        launch_template = ec2.LaunchTemplate(
            self,
            "GpuLaunchTemplate",
            launch_template_name=f"{project_tag}-p6b200",
            machine_image=machine_image,
            instance_type=ec2.InstanceType(instance_type),
            role=role,
            # SG is attached via the NetworkInterfaces override below (no
            # public IP). Specifying it here too would put SecurityGroupIds at
            # the top level, which conflicts with interface-level Groups at
            # run-instances time.
            user_data=user_data,
            block_devices=block_devices,
            require_imdsv2=True,
            detailed_monitoring=True,
        )

        # Target the capacity reservation by ID and force no-public-IP placement
        # in the private subnet. These are set on the CfnLaunchTemplate data
        # because the L2 LaunchTemplate doesn't surface them directly.
        private_subnet = vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
        ).subnets[0]

        cfn_lt = launch_template.node.default_child
        cfn_lt.add_property_override(
            "LaunchTemplateData.CapacityReservationSpecification",
            {
                "CapacityReservationTarget": {
                    "CapacityReservationId": cr_id,
                }
            },
        )
        cfn_lt.add_property_override(
            "LaunchTemplateData.NetworkInterfaces",
            [
                {
                    "DeviceIndex": 0,
                    "AssociatePublicIpAddress": False,  # Req 22.1
                    "SubnetId": private_subnet.subnet_id,
                    "Groups": [sg.security_group_id],
                }
            ],
        )

        Tags.of(self).add("project", project_tag)

        # ----------------------------------------------------------------- #
        # Outputs — everything the 21:30 launch step needs.
        # ----------------------------------------------------------------- #
        CfnOutput(self, "LaunchTemplateId", value=launch_template.launch_template_id)
        CfnOutput(self, "PrivateSubnetId", value=private_subnet.subnet_id)
        CfnOutput(self, "SecurityGroupId", value=sg.security_group_id)
        CfnOutput(self, "InstanceRoleArn", value=role.role_arn)
        CfnOutput(self, "VpcId", value=vpc.vpc_id)
        CfnOutput(
            self,
            "RunInstancesHint",
            value=(
                f"aws ec2 run-instances --region {self.region} "
                f"--launch-template LaunchTemplateId={launch_template.launch_template_id} "
                f"--instance-market-options '{{}}' "
                f"--profile your-aws-profile"
            ),
        )
