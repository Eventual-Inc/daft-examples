# AWS S3 Files: end-to-end mount tutorial

[AWS S3 Files](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-files.html) (launched 2026-04-07) gives any S3 bucket a POSIX filesystem interface. Mount the bucket on an EC2 instance and use it like a local directory — `open()`, `read()`, `write()`, `chmod`, `ls`. Underneath, the data is still regular S3 objects, accessible through the S3 API at the same time.

This walkthrough is end-to-end. Replace `my-bucket`, `123456789012`, `us-east-1`, and the subnet/security group IDs with your own values.

## Why mount an S3 bucket?

- **Strong consistency for writers** — agents, CI jobs, and humans can `write()` files and have them durably reflected to S3 without eventual-consistency surprises.
- **POSIX semantics** — `fopen()`, file locking, `chmod`/`chown`, and directory operations work for any tool that expects a real filesystem.
- **Dual access** — files written via the mount appear in S3 within ~60 seconds; objects added via the S3 API appear in the mount within seconds. Use whichever path fits your tool.

When this matters most: AI knowledge bases, agent memory, and lakehouses where you want the same bucket to serve structured tables (read via S3/Glue catalog) and unstructured files (read or written via POSIX).

## Prerequisites

S3 Files has strict bucket requirements. Verify these before continuing:

- General-purpose bucket (not directory bucket)
- Versioning **enabled** (required for sync)
- SSE-S3 or SSE-KMS encryption (no SSE-C)
- EC2 instance in the same region as the bucket
- `amazon-efs-utils` v3.0.0+ on the EC2 instance

```bash
# Verify bucket meets requirements
aws s3api get-bucket-versioning --bucket my-bucket
aws s3api get-bucket-encryption --bucket my-bucket
```

## Step 1: install the S3 Files client on your EC2 instance

```bash
# Amazon Linux 2 / 2023
sudo yum -y install amazon-efs-utils

# Other distros
curl https://amazon-efs-utils.aws.com/efs-utils-installer.sh | sudo sh -s -- --install

# Verify version (must be >=3.0.0)
mount.s3files --version
```

## Step 2: create the S3 Files service role

S3 Files needs an IAM role it can assume to read and write your bucket on your behalf. Save the trust policy and inline policy as JSON files first.

`s3files-trust-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "elasticfilesystem.amazonaws.com" },
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": { "aws:SourceAccount": "123456789012" },
      "ArnLike": { "aws:SourceArn": "arn:aws:s3files:us-east-1:123456789012:file-system/*" }
    }
  }]
}
```

`s3files-bucket-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:ListBucket", "s3:ListBucketVersions"],
      "Resource": "arn:aws:s3:::my-bucket"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:AbortMultipartUpload",
        "s3:DeleteObject*",
        "s3:GetObject*",
        "s3:List*",
        "s3:PutObject*"
      ],
      "Resource": "arn:aws:s3:::my-bucket/*"
    }
  ]
}
```

Then create the role:

```bash
aws iam create-role \
  --role-name S3FilesServiceRole \
  --assume-role-policy-document file://s3files-trust-policy.json

aws iam put-role-policy \
  --role-name S3FilesServiceRole \
  --policy-name S3FilesBucketAccess \
  --policy-document file://s3files-bucket-policy.json
```

## Step 3: attach the client policy to your EC2 instance role

The EC2 instance needs permission to mount the file system. Attach the managed policy to whichever IAM role is on your instance profile.

```bash
aws iam attach-role-policy \
  --role-name <your-ec2-instance-role> \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FilesClientFullAccess
```

Use `AmazonS3FilesClientReadOnlyAccess` instead if the instance should only read the bucket.

## Step 4: open NFS traffic between EC2 and the mount target

S3 Files uses NFS over TCP port 2049. Create a dedicated security group for the mount target and authorize traffic from your EC2 instance's security group.

```bash
# Create a security group for the mount target
aws ec2 create-security-group \
  --group-name s3files-mount-target \
  --description "S3 Files NFS access" \
  --vpc-id <your-vpc-id>
# returns the new sg-xxxxx ID

# Allow NFS from the EC2 instance's security group
aws ec2 authorize-security-group-ingress \
  --group-id <mount-target-sg-id> \
  --protocol tcp --port 2049 \
  --source-group <ec2-instance-sg-id>

# Allow outbound NFS from the EC2 instance to the mount target
aws ec2 authorize-security-group-egress \
  --group-id <ec2-instance-sg-id> \
  --protocol tcp --port 2049 \
  --source-group <mount-target-sg-id>
```

## Step 5: create the file system

```bash
aws s3files create-file-system \
  --region us-east-1 \
  --bucket arn:aws:s3:::my-bucket \
  --role-arn arn:aws:iam::123456789012:role/S3FilesServiceRole
```

The response includes a `FileSystemId` (e.g. `fs-0123456789abcdef0`). Save it. Creation finishes in a few minutes — poll with `aws s3files get-file-system --file-system-id <id>` until `LifecycleState: AVAILABLE`.

## Step 6: create a mount target

One mount target per Availability Zone. Use a subnet in the same VPC and AZ as your EC2 instance.

```bash
aws s3files create-mount-target \
  --region us-east-1 \
  --file-system-id fs-0123456789abcdef0 \
  --subnet-id subnet-abc123 \
  --security-groups <mount-target-sg-id>
```

Mount targets take up to ~5 minutes. Poll with `aws s3files describe-mount-targets --file-system-id <id>` until `LifecycleState: AVAILABLE`.

## Step 7: mount on the EC2 instance

```bash
sudo mkdir -p /mnt/my-bucket
sudo mount -t s3files fs-0123456789abcdef0:/ /mnt/my-bucket

# Verify
ls /mnt/my-bucket/
mount | grep s3files
```

## Step 8: persist the mount across reboots (optional)

Add an `/etc/fstab` entry so the mount comes back after reboot:

```
fs-0123456789abcdef0:/ /mnt/my-bucket s3files _netdev,noresvport 0 0
```

## Step 9: verify dual access

Confirm your data is accessible through both paths:

```bash
# Write via the mount (POSIX)
echo "hello s3 files" > /mnt/my-bucket/test.txt

# Read via the S3 API (within ~60s, after sync)
aws s3 cp s3://my-bucket/test.txt -

# Or read with Daft
python -c "
import daft
from daft.functions import file as daft_file
df = (daft.from_glob_path('s3://my-bucket/test.txt')
        .with_column('file', daft_file(daft.col('path'))))
df.show()
"
```

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| `mount.s3files: command not found` | `amazon-efs-utils` not installed or older than 3.0.0 |
| `mount.nfs4: Connection timed out` | Security group doesn't allow TCP 2049 between EC2 and mount target |
| `mount.s3files: access denied` | EC2 instance role missing `AmazonS3FilesClientFullAccess` |
| `LifecycleState: ERROR` on file system | Bucket missing versioning, encryption, or service role can't assume |
| Files written to mount don't appear in S3 | Wait up to 60s — write batching aggregates changes before pushing to S3 |
| `.s3files-lost+found-*/` directory appears | Conflicting writes between mount and S3 API; S3 wins, your local change went to lost+found |

For the full reference, see the [S3 Files documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-files.html).
