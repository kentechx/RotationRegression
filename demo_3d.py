import click, os, glob, os.path as osp
import json
import numpy as np
import trimesh
import igl
from scipy.spatial.transform import Rotation
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.my_dgcnn import MyDGCNN_Cls


class ToothDataset(Dataset):
    test_scenes = ['1801_2000', '2001_2175']

    def __init__(self, data_path="/home/kent/Datasets/step_data/data2000_20200810",
                 train=True):
        self.train = train
        if train:
            self.fps = []
            for scene in os.listdir(data_path):
                if scene in self.test_scenes:
                    continue
                for fp in glob.glob(osp.join(data_path, scene, "*/3*._Crown.stl")):
                    self.fps.append(fp)
        else:
            self.fps = []
            for scene in self.test_scenes:
                for fp in glob.glob(osp.join(data_path, scene, "*/3*._Crown.stl")):
                    self.fps.append(fp)
            self.quaternion = np.random.rand(len(self.fps), 4)  # (N, 4)
            self.quaternion /= np.linalg.norm(self.quaternion, axis=1, keepdims=True)

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, i):
        if self.train:
            return self.get_train_data(i)
        else:
            return self.get_test_data(i)

    def get_train_data(self, i):
        fp = self.fps[i]
        m: trimesh.Trimesh = trimesh.load(fp)
        m.vertices -= m.vertices.mean(0)
        x = np.concatenate([m.vertices, m.vertex_normals], axis=1).astype('f4')
        idx = np.random.choice(np.arange(len(x)), 1024, replace=True)
        x = x[idx]
        tid = int(osp.basename(fp)[:2])

        quaternion = np.random.rand(4)  # (N, 4)
        quaternion /= np.linalg.norm(quaternion)
        mat = Rotation.from_quat(quaternion).as_matrix().astype('f4')
        x[:, :3] = x[:, :3] @ mat.T
        x[:, 3:] = x[:, 3:] @ mat.T

        return x.T, quaternion, tid, fp

    def get_test_data(self, i):
        fp = self.fps[i]
        m: trimesh.Trimesh = trimesh.load(fp)
        m.vertices -= m.vertices.mean(0)
        x = np.concatenate([m.vertices, m.vertex_normals], axis=1).astype('f4')
        idx = np.random.choice(np.arange(len(x)), 1024, replace=True)
        x = x[idx]
        tid = int(osp.basename(fp)[:2])

        mat = Rotation.from_quat(self.quaternion[i]).as_matrix().astype('f4')
        x[:, :3] = x[:, :3] @ mat.T
        x[:, 3:] = x[:, 3:] @ mat.T

        return x.T, self.quaternion[i], tid, fp


class LitModel(pl.LightningModule):

    def __init__(self, batch_size, lr, num_workers):
        super().__init__()
        self.save_hyperparameters()
        args = type('', (), {})()
        args.k = 64
        args.dynamic = True
        args.use_stn = False
        args.input_channels = 6
        args.output_channels = 4
        args.n_edgeconvs_backbone = 3
        args.edgeconv_channels = [64, 64, 64]
        args.emb_dims = 1024
        args.norm = "instance"
        args.dropout = 0.
        self.net = MyDGCNN_Cls(args)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        X, y, tid, fp = batch
        pred, _ = self(X)
        loss = F.l1_loss(pred, y)
        self.log("loss", loss, batch_size=self.hparams['batch_size'])
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, tid, fp = batch
        pred, _ = self(X)
        loss = F.l1_loss(pred, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters())
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return [optimizer], [schedular]

    def train_dataloader(self):
        return DataLoader(ToothDataset(train=True), batch_size=self.hparams.batch_size,
                          num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(ToothDataset(train=False), batch_size=self.hparams.batch_size,
                          num_workers=self.hparams['num_workers'], shuffle=False, pin_memory=True)


@click.command()
@click.option('--epoch', default=1000)
@click.option('--batch_size', default=16)
@click.option('--lr', default=1e-3)
@click.option('--num_workers', default=4)
def run(**kwargs):
    print(colored(json.dumps(kwargs, indent=2), 'blue'))

    pl.seed_everything(42)

    # logger
    version = 'demo_3d'
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