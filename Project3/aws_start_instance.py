# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Lukas Hoyer, lhoyer@student.ethz.ch

import os
import time
import json
import subprocess
import argparse
import tarfile
import boto3

AWS = 'aws'   # path to `aws` CLI executable

PERMISSION_FILE_PATH = '~/.ssh/dlad-aws.pem'
AMI = 'ami-0b64362b8113b27fd' # Pre-setup AMI based on Deep Learning AMI (Ubuntu 18.04) Version 41.0 AMI 07f83f2fb8212ce3b
REGION = 'us-east-2'
NON_ROOT = 'ubuntu'
TIMEOUT = {'train': 48, 'devel': 4}  # in hours
RSYNC_EXCLUDE = "--exclude 'wandb/' --exclude 'doc/'"
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))


class color:
   GREEN = '\033[32m'
   END = '\033[0m'

def build_ssh_cmd(hostname):
    ssh_options = f"-q -o StrictHostKeyChecking=no -o ConnectTimeout=180 -i {PERMISSION_FILE_PATH}"
    return f'ssh {ssh_options} {NON_ROOT}@{hostname}'

def build_rsync_cmd(hostname):
    ssh_options = f"-q -o StrictHostKeyChecking=no -o ConnectTimeout=180 -i {PERMISSION_FILE_PATH}"
    return f"rsync -av -e 'ssh {ssh_options}' . {RSYNC_EXCLUDE} {NON_ROOT}@{hostname}:~/code/"

def setup_s3_bucket():
    if not os.path.exists("aws_configs/default_s3_bucket.txt"):
        print("You currently have no AWS S3 bucket specified. These are your existing buckets:\n")
        os.system("aws s3 ls")
        print("\nThis list is empty for a new account.")
        print("Choose an existing or new name for your bucket according to the naming rule (https://docs.aws.amazon.com"
              "/awscloudtrail/latest/userguide/cloudtrail-s3-bucket-naming-requirements.html).")
        bucket_name = input("Bucket name (without s3://): ")
        print(f"Create bucket {bucket_name}...")
        if os.system(f"aws s3 mb s3://{bucket_name}") != 0:
            quit()
        if os.system(f'aws s3api put-public-access-block --bucket {bucket_name} --public-access-block-configuration '
                         f'"BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,'
                         f'RestrictPublicBuckets=true"') != 0:
            quit()
        with open("aws_configs/default_s3_bucket.txt", "w") as fh:
            fh.write(bucket_name)

def setup_group_id():
    if not os.path.exists("aws_configs/group_id.txt"):
        group_id = input("Please enter your DLAD group ID as raw number: ")
        try:
            int(group_id)  # test if conversion is valid
        except ValueError:
            print("Your group ID is not a valid integer.")
            quit()
        assert 0 <= int(group_id) < 100, "Your group ID should be between 0 and 100."
        with open("aws_configs/group_id.txt", "w") as fh:
            fh.write(group_id)

def setup_wandb():
    if not os.path.exists("aws_configs/wandb.key"):
        wandb_key = input("Please enter your wandb key (https://wandb.ai/authorize): ")
        with open("aws_configs/wandb.key", "w") as fh:
            fh.write(wandb_key)

def code_archive_filter(x):
    if 'wandb/' not in x.name and 'doc/' not in x.name and 'instance_state.txt' not in x.name and 'pycache' not in x.name and '.tar.gz' not in x.name:
        return x
    else:
        return None

def gen_code_archive(file):
    with tarfile.open(file, mode='w:gz') as tar:
        tar.add('.', filter=code_archive_filter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--mode", choices=["train", "devel"],
        required=True,
        help="Mode of the instance setup.\n"
             "Devel: With code and setup\n"
             "Train: With code, setup and automatic training"
    )
    parser.add_argument(
        "--instance", choices=["m5n.xlarge", "p2.xlarge", "p3.2xlarge"],
        required=True,
        help="Instance type: m5n.xlarge, p2.xlarge, or p3.2xlarge"
    )
    parser.add_argument(
        "--on-demand", action='store_true',
        help="Use a more expensive on-demand instance."
    )
    args = parser.parse_args()
    if args.mode == 'devel':
        assert args.instance == 'm5n.xlarge', 'Please use an m5n.xlarge instance for development.'

    setup_wandb()
    setup_s3_bucket()
    setup_group_id()

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    tag = f'{timestamp}'

    # Generate and upload code tar.gz to AWS S3
    print('Upload code to AWS S3...')
    code_archive = f'code_{tag}.tar.gz'
    gen_code_archive(code_archive)
    s3 = boto3.client('s3')
    with open('aws_configs/default_s3_bucket.txt', 'r') as fh:
        S3_BUCKET_NAME = fh.read()
    response = s3.upload_file(code_archive, S3_BUCKET_NAME,
                              f'code/{code_archive}')
    os.remove(code_archive)

    # Create user_data.sh that will be executed when instance is starting for the first time
    print(f'Set timeout to {TIMEOUT[args.mode]} hours.')
    with open('aws_configs/user_data.sh', 'r') as fh:
        user_data = fh.read().format(
            bucket=S3_BUCKET_NAME,
            code_archive=code_archive,
            timeout=TIMEOUT[args.mode],
            script=f'{args.mode}.sh'
        )

    instance_tag = 'ResourceType=instance,Tags=[{Key=Name,Value=' + tag + '}]'
    spot_tag = 'ResourceType=spot-instances-request,Tags=[{Key=Name,Value=' + tag + '}]'


    # Refer to https://docs.aws.amazon.com/cli/latest/reference/ec2/run-instances.html
    my_cmd = [AWS, 'ec2', 'run-instances',
              '--tag-specifications', instance_tag,
              '--instance-type', args.instance,
              '--image-id', AMI,
              '--key-name', 'dlad-aws',
              '--security-groups', 'dlad-sg',
              '--iam-instance-profile', 'Name="dlad-instance-profile"',
              '--ebs-optimized',
              '--block-device-mappings', f'DeviceName="/dev/sda1",Ebs={{VolumeSize=250}}',
              '--user-data', user_data,
    ]

    # Spot options
    if not args.on_demand:
        # Managed spot train
        if args.mode == 'train':
            my_cmd.extend([
                '--tag-specifications', spot_tag,
                '--instance-market-options', f'file://{TOOLS_DIR}/aws_configs/persistent-spot-options.json',
            ])
        # One-time development spot instance (does not spawn again)
        else:
            my_cmd.extend([
                '--tag-specifications', spot_tag,
                '--instance-market-options', f'file://{TOOLS_DIR}/aws_configs/spot-options.json',
            ])


    print("Launch instance...")
    response = None
    successful = False
    while not successful:
        try:
            response = json.loads(subprocess.check_output(my_cmd))
            successful = True
        except subprocess.CalledProcessError:
            wait_seconds = 120
            print(f'launch unsuccessfull, retrying in {wait_seconds} seconds...')
            time.sleep(wait_seconds)


    instance_id = response['Instances'][0]['InstanceId']
    dns_response = json.loads(subprocess.check_output([AWS,
                                                       'ec2',
                                                       'describe-instances',
                                                       '--region',
                                                       REGION,
                                                       '--instance-ids',
                                                       instance_id]))
    instance_dns = dns_response['Reservations'][0]['Instances'][0]['PublicDnsName']
    ssh_command = build_ssh_cmd(instance_dns)
    print('AWS instance was launched.')

    print('Wait for AWS instance to initialize...')
    successful = False
    while not successful:
        try:
            subprocess.run([f"{ssh_command} echo 'SSH connection initialized'"], shell=True, check=True)
            successful = True
        except subprocess.CalledProcessError:
            print(f'Wait for instance...')

    print(f'Sucessfully started instance {instance_id} with tag {tag}')
    print('Connect to instance using ssh:')
    print(color.GREEN + ssh_command + color.END)
    print('Rsync file updates:')
    print(color.GREEN + build_rsync_cmd(instance_dns) + color.END)
    print('Connect to tmux session using ssh:')
    print(color.GREEN + f"{ssh_command} -t tmux attach-session -t dlad" + color.END)

    with open('aws/aws.log', 'a') as file_name:
        file_name.write(f'{tag}\n')
        file_name.write(f'{ssh_command}\n\n')
