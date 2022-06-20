#!/bin/bash

S3BUCKET=s3://$(cat "aws_configs/default_s3_bucket.txt")

mkdir -p s3_sync/
cd s3_sync/
echo "Sync tensorboards from" $S3BUCKET
aws s3 sync $S3BUCKET . --exact-timestamps --exclude "*" --include "*event*" --include "*.yaml"
#find . -type d -empty -delete

echo "Start tensorboard"
tensorboard --logdir=.