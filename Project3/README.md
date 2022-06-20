# DLAD Exercise 3

### AWS Setup

If not already done, please follow https://gitlab.ethz.ch/dlad21/aws-tools/ to setup your AWS access.

### AWS Development and Testing

You can launch a development AWS EC2 CPU instance using:

```shell script
python aws_start_instance.py --mode devel --instance m5n.xlarge
```

It'll have the requirements installed and dataset available. Your local code is uploaded during the initialization as
well.

You can attach to the tmux session of the instance using the last printed ssh command.
To test your source code for task 1, 2, 4, and 5, you can run following script on the EC2 instance:

```shell script
python tests/test.py --task X  # , where X is the task number.
```

As the instance has to warm up, the **first call will take several minutes** until it runs. If you want to 
update the source code on the AWS instance, you can run the rsync command printed by aws_start_instance.py. After you 
finished testing, please **manually stop the instance** using:

```shell script
bash aws/stop_self.sh
```

The development instance is only intended for task 1-5, which do no require a GPU. 
If you want to launch the tests automatically on the instance, refer to 
[aws/devel.sh](aws/devel.sh).

### AWS Training

You can launch a training on an AWS GPU spot or on-demand instance using:

```shell script
# Spot instance:
python aws_start_instance.py --mode train --instance p2.xlarge
# On-demand instance:
python aws_start_instance.py --mode train --instance p2.xlarge --on-demand
```

During the first run, the script will ask you for some information such as the wandb token for the setup.
You can attach to the launched tmux session by running the last printed command. If you want to close the connection
but keep the script running, detach from tmux using Ctrl+B and D. After that, you can exit the ssh connection, while
tmux and the training keep running. You can enter the scroll mode using Ctrl+B and [ and exit it with Q. 
In the scroll mode, you can scroll using the arrow keys or page up and down. Tmux has also some other nice features
such as multiple windows or panels (https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/). Please note
that there is a **timeout** of 48 hours to the instance. If you find that not sufficient, please adjust it
in [aws_start_instance.py](aws_start_instance.py). To check if you are unintentionally using AWS resources, you can
have a look at the AWS cost explorer: https://console.aws.amazon.com/cost-management/home?region=us-east-1#/dashboard.

You can change the training hyperparameters in [config.yaml](config.yaml). 

Please note that the first epoch will train considerably slower than the following ones as the required parts of the
AWS volume are downloaded from S3 on demand.

When using spot instances for training (`--mode train`), the spot request is persistent. This means that after a spot interrupt has happened,
a new spot instance will be launched as soon as sufficient capacity is available and the training is resumed from
the last checkpoint. To stop the persistent spot request for the instance manually, please cancel the spot request
[https://us-east-2.console.aws.amazon.com/ec2sp/v2/home?region=us-east-2#/spot](https://us-east-2.console.aws.amazon.com/ec2sp/v2/home?region=us-east-2#/spot).
When you just terminate the instance, the persistent spot request will launch another instance to continue the training.

### AWS Interactive Development

During developing your own code, you'll often run into the problem that the training crashes briefly after the start due
to some typo. In order to avoid the overhead of waiting until AWS allows you to start a new instance as well as the
instance setup, you can continue using the same instance for further development. For that purpose cancel the automatic
termination using Ctrl+C. Fix the bug in your local environment and update your AWS files by running the rsync command, 
which was printed by aws_start_instance.py, on your local machine. After that, you can start the training on the AWS 
instance by running:
```shell script
cd ~/code/ && bash aws/train.sh
```

Remember, that you are now responsible for manually terminating the instance using 

```bash aws/stop_self.sh```

Please, avoid long idle times for GPU instances, especially when they are on-demand instances.

### Weights and Biases Monitoring

You can monitor the training via the wandb web interface https://wandb.ai/home. If you have lost the ec2 instance 
information for a particular (still running) experiment, you can view it by choosing the 
Table panel on the left side and horizontally scroll the columns until you find the EC2 columns.

In the workspace panel, we recommend switching the x-axis to epoch (x icon in the top right corner) for
visualization.
The logged histograms, you can only view if you click on a single run.

### AWS S3 Checkpoints and Submission Zip

To avoid exceeding the free wandb quota, the checkpoints and submission zips are saved to AWS S3. The link is logged
to wandb. You can find it on the dash board (https://wandb.ai/home) in the Table panel (available on the left side)
in the column S3_Link. 

Use the following command to download a submission archive to the local machine:

```shell script
aws s3 cp <s3_link> <local_destination>
```

### AWS Budget and Instances

You have an **AWS budget of 350 USD** available for exercise 3.

In order to monitor your spending, you can use the AWS cost explorer:
https://us-east-1.console.aws.amazon.com/cost-management/home?region=us-east-2#/dashboard
From this page, you can click on "View in Cost Explorer" to see more details and choose
the date range. Please make sure to visualize the correct date range. We recommend that
you regularly monitor your expenses. If you exceed your budget, we might disable the AWS
account without previous warning or deduct points from this exercise. So, please be careful.

We give you the freedom to choose whether you want to use spot or on-demand instances.
When AWS is not too crowded, still using spot instances can be a way to get more training
time out of your budget.

The AWS prices are:

* m5n.xlarge spot instance (CPU): 0.04 USD/h
* p2.xlarge spot instance (K80 GPU): 0.27 USD/h
* p3.2xlarge spot instance (V100 GPU): 0.92 USD/h
* m5n.xlarge on-demand instance (CPU): 0.24 USD/h
* p2.xlarge on-demand instance (K80 GPU): 0.90 USD/h
* p3.2xlarge on-demand instance (V100 GPU): 3.06 USD/h

Your GPU instance quota is:

* 16 vCPU cores for P spot instances 
* 8 vCPU cores for P on-demand instances

A p2.xlarge instance has 4 vCPU cores and a p3.2xlarge instance has 8 vCPU cores.
With this quota, you could spend your entire budget within 71 hours.
 
The expected training times for the baseline are roughly:

* 43 hours on p2.xlarge instances (120 min for first epoch; 70 min for each following epoch)
* 18 hours on p3.2xlarge instances (60 min for first epoch; 30 min for each following epoch)