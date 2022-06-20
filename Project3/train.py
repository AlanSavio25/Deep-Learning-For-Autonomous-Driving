# Course: Deep Learning for Autonomous Driving, ETH Zurich
# Material for Project 3
# For further questions contact Ozan Unal, ozan.unal@vision.ee.ethz.ch

import os
import sys
import argparse
import uuid
from datetime import datetime

import boto3
import requests
import yaml
import numpy as np
import shutil
import wandb
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

from model import Model
from dataset import DatasetLoader

from utils.task4 import RegressionLoss, ClassificationLoss
from utils.eval import generate_final_predictions, save_detections, generate_submission, compute_map
from utils.vis import point_scene

from aws_start_instance import build_ssh_cmd, build_rsync_cmd


class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dir = self.config['eval']['output_dir']
        if os.path.exists(self.output_dir): shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        self.model = Model(config['model'])
        self.reg_loss = RegressionLoss(config['loss'])
        self.cls_loss = ClassificationLoss(config['loss'])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, assinged_target, iou = batch['input'], batch['assinged_target'], batch['iou']
        pred_box, pred_class = self(x)
        loss = self.reg_loss(pred_box, assinged_target, iou) \
               + self.cls_loss(pred_class, iou)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, assinged_target, iou = batch['input'], batch['assinged_target'], batch['iou']
        pred_box, pred_class = self(x)

        loss = self.reg_loss(pred_box, assinged_target, iou) \
               + self.cls_loss(pred_class, iou)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        nms_pred, nms_score = generate_final_predictions(pred_box, pred_class, config['eval'])
        save_detections(os.path.join(self.output_dir, 'pred'), batch['frame'], nms_pred, nms_score)

        # Visualization
        if batch_idx == 0:
            scene = point_scene(batch['points'], nms_pred, batch['target'], name=f'e{self.current_epoch}')
            self.logger.experiment[0].log(scene, commit=False)

    def validation_epoch_end(self, outputs):
        easy, moderate, hard = compute_map(self.valid_dataset.hf,
                                           os.path.join(self.output_dir, 'pred'),
                                           self.valid_dataset.frames)
        shutil.rmtree(self.output_dir, 'pred')
        self.log('e_map', easy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('m_map', moderate, on_step=False, on_epoch=True, prog_bar=True)
        self.log('h_map', hard, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        frame, x = batch['frame'], batch['input']
        pred_box, pred_class = self(x)
        nms_pred, nms_score = generate_final_predictions(pred_box, pred_class, config['eval'])
        save_detections(os.path.join(self.output_dir, 'test'), frame, nms_pred, nms_score)

    @property
    def submission_file(self):
        return os.path.join(self.output_dir, 'submission.zip')

    def test_epoch_end(self, outputs):
        generate_submission(os.path.join(self.output_dir, 'test'), self.submission_file)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), **self.config['optimizer'])
        scheduler = lrs.MultiStepLR(optimizer, **self.config['scheduler'])
        return [optimizer], [scheduler]

    def setup(self, stage):
        self.train_dataset = DatasetLoader(config=config['data'], split='train')
        self.valid_dataset = DatasetLoader(config=config['data'], split='val')
        self.test_dataset = DatasetLoader(config=config['data'], split='test')

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.config['data']['batch_size'],
                          shuffle=True,
                          pin_memory=True,
                          num_workers=os.cpu_count(),
                          collate_fn=self.train_dataset.collate_batch)

    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=os.cpu_count(),
                          collate_fn=self.train_dataset.collate_batch)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset,
                          batch_size=1,
                          shuffle=False,
                          pin_memory=True,
                          num_workers=os.cpu_count(),
                          collate_fn=self.test_dataset.collate_batch)

def train(config, run_name, s3_log_path):
    with open('aws_configs/group_id.txt', 'r') as fh:
        GROUP_ID = int(fh.read())

    wandb_logger = WandbLogger(
        name=run_name,
        project='DLAD-Ex3',
        save_dir=os.path.join(config["trainer"]["default_root_dir"])
    )
    tb_s3_logger = TensorBoardLogger(
        name="tb",
        version="",
        save_dir=s3_log_path,
    )

    checkpoint_local_callback = ModelCheckpoint(
        dirpath=os.path.join(config["trainer"]["default_root_dir"], 'checkpoints'),
    )
    checkpoint_s3_callback = ModelCheckpoint(
        dirpath=s3_log_path,
        verbose=True,
    )

    # Log AWS instance information to wandb
    ec2_hostname = requests.get('http://169.254.169.254/latest/meta-data/public-hostname').text
    ec2_meta = {
        "EC2_Hostname": ec2_hostname,
        "EC2_Instance_ID": requests.get('http://169.254.169.254/latest/meta-data/instance-id').text,
        "EC2_SSH": build_ssh_cmd(ec2_hostname),
        "EC2_SSH_Tmux": f"{build_ssh_cmd(ec2_hostname)} -t tmux attach-session -t dlad",
        "EC2_Rsync": build_rsync_cmd(ec2_hostname),
        "S3_Path": s3_log_path,
        "S3_Link": f"https://s3.console.aws.amazon.com/s3/buckets/{S3_BUCKET_NAME}?region=us-east-2&prefix={run_name}/",
        "Group_Id": GROUP_ID
    }
    wandb_logger.log_hyperparams({**ec2_meta, **config})
    tb_s3_logger.log_hyperparams({**ec2_meta, **config})

    # Setup training framework
    if config["trainer"]["resume_from_checkpoint"] is not None and "s3://" in config["trainer"]["resume_from_checkpoint"]:
        s3 = boto3.resource('s3')
        _, _, resume_bucket_name, resume_bucket_local_path = config["trainer"]["resume_from_checkpoint"].split('/', 3)
        resume_bucket = s3.Bucket(resume_bucket_name)
        checkpoints = list(resume_bucket.objects.filter(Prefix=resume_bucket_local_path))
        checkpoints = [c for c in checkpoints if c.key.endswith(".ckpt")]
        if len(checkpoints) != 1:
            print("Your s3 path specification did not match a single checkpoint. Please be more specific:")
            for c in checkpoints:
                print(f"s3://{c.bucket_name}/{c.key}")
            exit()
        else:
            config["trainer"]["resume_from_checkpoint"] = f"s3://{checkpoints[0].bucket_name}/{checkpoints[0].key}"
            print(f'Resume from checkpoint S3 {config["trainer"]["resume_from_checkpoint"]}')

    print("Start training", run_name)

    trainer = pl.Trainer(
        logger=[wandb_logger, tb_s3_logger],
        callbacks=[checkpoint_local_callback, checkpoint_s3_callback],
        gpus=-1 if torch.cuda.is_available() else None,
        accelerator='ddp' if torch.cuda.is_available() else None,
        **config['trainer']
    )
    litModel = LitModel(config)
    trainer.fit(litModel)
    trainer.test(litModel)

    ret_code = os.system(f"aws s3 cp {litModel.submission_file} {s3_log_path}")
    assert ret_code == 0
    wandb.finish()

def get_newest_ckpt(s3_path):
    s3 = boto3.resource('s3')
    _, _, resume_bucket_name, resume_bucket_local_path = s3_path.split('/', 3)
    resume_bucket = s3.Bucket(resume_bucket_name)
    checkpoints = list(
        resume_bucket.objects.filter(Prefix=resume_bucket_local_path))
    checkpoints = [c for c in checkpoints if c.key.endswith(".ckpt")]
    if len(checkpoints) == 0:
        return None
    else:
        checkpoints = sorted(checkpoints, key=lambda x: x.last_modified, reverse=True)
        print(checkpoints)
        return f"s3://{checkpoints[0].bucket_name}/{checkpoints[0].key}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config_path', default='config.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, 'r'))

    if os.path.isfile('instance_state.txt'):
        with open('instance_state.txt', 'r') as fh:
            states = fh.readlines()
        for s in reversed(states):
            last_resume_ckpt = get_newest_ckpt(s.replace('\n',''))
            if last_resume_ckpt is not None:
                config["trainer"]['resume_from_checkpoint'] = last_resume_ckpt
                print(f'Resume from {last_resume_ckpt}.')
                break
            else:
                print(f'There is no checkpoint for {s}.')

    with open('aws_configs/default_s3_bucket.txt', 'r') as fh:
        S3_BUCKET_NAME = fh.read()
    with open('aws_configs/group_id.txt', 'r') as fh:
        GROUP_ID = int(fh.read())
    timestamp = datetime.now().strftime('%m%d-%H%M')
    run_name = f'G{GROUP_ID}_{timestamp}_{config["name"]}_{str(uuid.uuid4())[:5]}'
    s3_log_path = f"s3://{S3_BUCKET_NAME}/{run_name}/"
    with open('instance_state.txt', 'a') as fh:
        fh.write(f'{s3_log_path}\n')
    train(config, run_name, s3_log_path)
    os.remove('instance_state.txt')