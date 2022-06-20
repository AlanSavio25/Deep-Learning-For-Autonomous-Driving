Content-Type: multipart/mixed; boundary="//"
MIME-Version: 1.0

--//
Content-Type: text/cloud-config; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="cloud-config.txt"

#cloud-config
cloud_final_modules:
- [scripts-user, always]

--//
Content-Type: text/x-shellscript; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="userdata.txt"

#!/bin/bash
if [ ! -d "/home/ubuntu/code" ]; then
  su - ubuntu -c "mkdir /home/ubuntu/code"
  su - ubuntu -c "aws s3 cp s3://{bucket}/code/{code_archive} /home/ubuntu/"
  su - ubuntu -c "tar -xzf /home/ubuntu/{code_archive} -C /home/ubuntu/code/"
fi
su - ubuntu -c "tmux new-session -d -s dlad -n train"
su - ubuntu -c "tmux send-keys -t dlad:train 'nohup bash /home/ubuntu/code/aws/timeout.sh {timeout}h > code/aws/timeout.log 2>&1 &' Enter"
su - ubuntu -c "tmux send-keys -t dlad:train 'source activate pytorch_latest_p37' Enter"
su - ubuntu -c "tmux send-keys -t dlad:train 'cd ~/code && bash aws/{script}' Enter"
--//--
