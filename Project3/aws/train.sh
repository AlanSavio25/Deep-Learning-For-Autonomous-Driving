#!/bin/bash

export WANDB_API_KEY=$(cat "aws_configs/wandb.key")

# Run training
echo "Start training"
mkdir -p /home/ubuntu/results/
cd /home/ubuntu/code/
python train.py

# Wait a moment before stopping the instance to give a chance to debug
echo "Terminate instance in 10 minutes. Use Ctrl+C to cancel the termination..."
sleep 10m && bash aws/stop_self.sh
