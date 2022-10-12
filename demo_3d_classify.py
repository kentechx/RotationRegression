import click, os, glob, os.path as osp
import json
import numpy as np
import trimesh
import trimesh.sample
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

from my_dgcnn import MyDGCNN_Cls


class ToothDataset(Dataset):
    test_scenes = ['1801_2000', '2001_2175']

    def __init__(self, data_path="/home/kent/Datasets/step_data/data2000_20200810",
                 train=True, num_pts=1024):
        self.train = train
        self.num_pts = num_pts
        if train:
            self.fps = []
            for scene in os.listdir(data_path):
                if scene in self.test_scenes:
                    continue
                for fp in glob.glob(osp.join(data_path, scene, "*/35._Crown.stl")):
                    tid = int(osp.basename(fp)[:2])
                    if tid % 10 >= 8:
                        continue
                    self.fps.append(fp)
        else:
            self.fps = []
            for scene in self.test_scenes:
                for fp in glob.glob(osp.join(data_path, scene, "*/35._Crown.stl")):
                    tid = int(osp.basename(fp)[:2])
                    if tid % 10 >= 8:
                        continue
                    self.fps.append(fp)
            self.mat = Rotation.random(len(self.fps), random_state=42).as_matrix().astype('f4')

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
        pts, fi = trimesh.sample.sample_surface_even(m, self.num_pts)
        x = np.concatenate([pts, m.face_normals[fi]], axis=1).astype('f4')
        # idx = np.random.choice(np.arange(len(x)), 1024, replace=True)
        # x = x[idx]
        tid = int(osp.basename(fp)[:2])

        mat = Rotation.random().as_matrix().astype('f4')
        x[:, :3] = x[:, :3] @ mat.T
        x[:, 3:] = x[:, 3:] @ mat.T

        return x.T, mat.T.reshape(-1), tid, fp

    def get_test_data(self, i):
        fp = self.fps[i]
        m: trimesh.Trimesh = trimesh.load(fp)
        m.vertices -= m.vertices.mean(0)
        pts, fi = trimesh.sample.sample_surface_even(m, self.num_pts)
        x = np.concatenate([pts, m.face_normals[fi]], axis=1).astype('f4')
        # idx = np.random.choice(np.arange(len(x)), 1024, replace=True)
        # x = x[idx]
        tid = int(osp.basename(fp)[:2])

        mat = self.mat[i]
        x[:, :3] = x[:, :3] @ mat.T
        x[:, 3:] = x[:, 3:] @ mat.T

        return x.T, mat.T.reshape(-1), tid, fp


def angle_diff(pred, y):
    rot_pred = torch.zeros((len(pred), 3, 3), device=pred.device)
    rot_pred[:, :, 0] = pred[:, :3]
    rot_pred[:, :, 1] = pred[:, 3:]
    rot_pred[:, :, 2] = torch.cross(pred[:, :3], pred[:, 3:], dim=1)
    rot_pred[:, :, 1] = torch.cross(rot_pred[:, :, 2], rot_pred[:, :, 0], dim=1)

    rot_pred[:, :, 0] /= torch.norm(rot_pred[:, :, 0], dim=1, keepdim=True) + 1e-8
    rot_pred[:, :, 1] /= torch.norm(rot_pred[:, :, 1], dim=1, keepdim=True) + 1e-8
    rot_pred[:, :, 2] /= torch.norm(rot_pred[:, :, 2], dim=1, keepdim=True) + 1e-8

    rot_y = torch.zeros((len(y), 3, 3), device=y.device)
    rot_y[:, :, 0] = y[:, :3]
    rot_y[:, :, 1] = y[:, 3:6]
    rot_y[:, :, 2] = y[:, 6:]
    rot = torch.bmm(rot_pred, rot_y.transpose(1, 2))
    angle = torch.clip((rot.diagonal(dim1=1, dim2=2).sum(-1) - 1.) / 2., -1, 1).acos() * 180 / np.pi  # (n, )
    return angle


class AngleGeodesic(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("angle_geodesic", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        angles = angle_diff(preds, target)
        self.angle_geodesic += angles.sum()
        self.total += len(preds)

    def compute(self):
        return self.angle_geodesic / self.total


class LitModel(pl.LightningModule):

    def __init__(self, epochs, batch_size, lr, num_pts, num_workers):
        super().__init__()
        self.save_hyperparameters()
        args = type('', (), {})()
        args.k = 64
        args.dynamic = True
        args.use_stn = False
        args.input_channels = 6
        args.output_channels = 6
        args.n_edgeconvs_backbone = 3
        args.edgeconv_channels = [64, 64, 64]
        args.emb_dims = 1024
        args.norm = "batch"
        args.dropout = 0.
        self.net = MyDGCNN_Cls(args)

        self.train_angle_geodesic = AngleGeodesic()
        self.val_angle_geodesic = AngleGeodesic()

    def forward(self, x, pts):
        return self.net(x, pts)

    def training_step(self, batch, batch_idx):
        X, y, tid, fp = batch
        pred, _ = self(X, X[:, :3].clone())
        loss = F.mse_loss(pred, y[:, :6])
        angles = angle_diff(pred, y)
        self.train_angle_geodesic(pred, y)
        self.log("angle_diff", self.train_angle_geodesic, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("angle_diff_max", angles.max(), prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("loss", loss, batch_size=self.hparams['batch_size'])
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        X, y, tid, fp = batch
        pred, _ = self(X, X[:, :3].clone())
        loss = F.mse_loss(pred, y[:, :6])
        angles = angle_diff(pred, y)
        self.val_angle_geodesic(pred, y)
        self.log("val_angle_diff", self.val_angle_geodesic, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("val_angle_diff_max", angles.max(), prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.net.parameters())
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, total_steps=self.hparams.epochs * len(self.train_dataloader()), max_lr=self.hparams.lr,
            pct_start=0.1, div_factor=10, final_div_factor=100)
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def train_dataloader(self):
        return DataLoader(ToothDataset(train=True, num_pts=self.hparams.num_pts), batch_size=self.hparams.batch_size,
                          num_workers=self.hparams['num_workers'], shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(ToothDataset(train=False, num_pts=self.hparams.num_pts), batch_size=self.hparams.batch_size,
                          num_workers=self.hparams['num_workers'], shuffle=False, pin_memory=True)


@click.command()
@click.option('--epoch', default=100)
@click.option('--batch_size', default=20)
@click.option('--lr', default=1e-3)
@click.option('--num_pts', default=1024)
@click.option('--num_workers', default=4)
@click.option('--version', default='demo_3d_5')
def run(**kwargs):
    print(colored(json.dumps(kwargs, indent=2), 'blue'))

    pl.seed_everything(42)

    # logger
    version = kwargs['version']
    logger = TensorBoardLogger("work_dir", name="demo", version=version)

    # trainer
    debug = False
    debug_args = {'limit_train_batches': 10, "limit_val_batches": 10} if debug else {}

    model = LitModel(kwargs['epoch'], kwargs['batch_size'], kwargs['lr'], kwargs['num_pts'], kwargs['num_workers'])
    callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(logger, accelerator='gpu', max_epochs=kwargs["epoch"], callbacks=[callback], **debug_args)

    # fit
    trainer.fit(model)

    # results = trainer.test()
    # print(results)


if __name__ == '__main__':
    run()
