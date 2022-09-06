import click, os, glob, os.path as osp
import json
import numpy as np
import cv2
from natsort import natsorted
from termcolor import colored
import albumentations as A
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models.efficientnet import efficientnet_b0

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


class FFHQ_Dataset(Dataset):
    test_scenes = ['00000', '01000', '02000', '03000', '04000']

    def __init__(self, data_path="/mnt/datasets/ffhq128", train=True):
        self.train = train
        if train:
            self.fps = []
            self.train_scenes = [s for s in natsorted(os.listdir(data_path)) if s not in self.test_scenes]
            for scene in self.train_scenes:
                for fp in glob.glob(osp.join(data_path, scene, "*.png")):
                    self.fps.append(fp)
        else:
            self.fps = []
            for scene in self.test_scenes:
                for fp in glob.glob(osp.join(data_path, scene, "*.png")):
                    self.fps.append(fp)
            self.angles = np.random.rand(len(self.fps)).astype('f4') * 360

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, i):
        if self.train:
            return self.get_train_data(i)
        else:
            return self.get_test_data(i)

    def get_train_data(self, i):
        im = cv2.imread(self.fps[i])
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # (128, 128)

        angle = np.random.rand(1) * 360
        M = cv2.getRotationMatrix2D((im.shape[0] // 2, im.shape[1] // 2), angle[0], 1)
        im = cv2.warpAffine(im, M, (im.shape[0], im.shape[1]))

        im = im.astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))
        return im, angle.astype('f4'), self.fps[i]

    def get_test_data(self, i):
        im = cv2.imread(self.fps[i])  # (128, 128, 3)
        # im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # (128, 128)

        # rotate image by angle
        angle = self.angles[i]
        M = cv2.getRotationMatrix2D((im.shape[0] // 2, im.shape[1] // 2), angle, 1)
        im = cv2.warpAffine(im, M, (im.shape[0], im.shape[1]))

        im = im.astype(np.float32) / 255.0
        im = np.transpose(im, (2, 0, 1))
        return im, self.angles[[i]], self.fps[i]


class LitModel(pl.LightningModule):

    def __init__(self, batch_size, lr, num_workers):
        super().__init__()
        self.save_hyperparameters()
        self.net = efficientnet_b0(num_classes=1)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        X, y, fp = batch
        pred = self(X)
        loss = F.l1_loss(pred, y)
        self.log("loss", loss, batch_size=self.hparams['batch_size'])
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, fp = batch
        pred = self(X)
        loss = F.l1_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters())
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [schedular]

    def train_dataloader(self):
        return DataLoader(FFHQ_Dataset(train=True), batch_size=self.hparams.batch_size,
                          num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(FFHQ_Dataset(train=False), batch_size=self.hparams.batch_size,
                          num_workers=self.hparams['num_workers'], shuffle=False, pin_memory=True)


@click.command()
@click.option('--epoch', default=1000)
@click.option('--batch_size', default=32)
@click.option('--lr', default=1e-3)
@click.option('--num_workers', default=0)
def run(**kwargs):
    print(colored(json.dumps(kwargs, indent=2), 'blue'))

    pl.seed_everything(42)

    # logger
    version = 0
    logger = TensorBoardLogger("work_dir", name="demo", version=version)

    # trainer
    debug = False
    debug_args = {'limit_train_batches': 10, "limit_val_batches": 10} if debug else {}

    model = LitModel(kwargs['batch_size'], kwargs['lr'], kwargs['num_workers'])
    callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(logger, accelerator='gpu', max_epochs=kwargs["epoch"], callbacks=[callback], **debug_args)

    # fit
    trainer.fit(model)

    # results = trainer.test()
    # print(results)


if __name__ == '__main__':
    run()
