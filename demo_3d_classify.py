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

from my_dgcnn import MyDGCNN_Cls, MyDGCNN_Seg


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.identity(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def spiral_sphere(delta_angle=10.):
    """
    n     angle diff (u, sigma)
    100   (18.7, 1.0)
    200   (13.3, 0.8)
    300   (10.8, 0.6)
    400   (9.3, 0.5)
    500   (8.5, 0.5)
    1000  (6, 0.3)
    2000  (4.2, 0.3)
    3000  (3.45, 0.22)
    4000  (3, 0.2)
    5000  (2.7, 0.2)
    6000  (2.44, 0.16)
    7000  (2.26, 0.147)
    10000 (1.9, 0.13)
    :param n:
    :return:
    """
    # estimate n
    delta_radius = delta_angle * np.pi / 180
    rhs = (1 - np.cos(delta_radius)) / (1 - np.cos(np.pi * (1 + 5 ** 0.5)))
    n = int(1.5 / (1 - np.cos(np.arcsin(rhs ** 0.5))))

    # generate points on a sphere
    indices = np.arange(0, n, dtype=float) + 0.5

    theta = np.arccos(1 - 2 * indices / n)
    phi = np.pi * (1 + 5 ** 0.5) * indices

    x, y, z = np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)
    xyz = np.stack([x, y, z], axis=1)
    return xyz


def generate_rotations(delta_angle=10., angle_thresh=None):
    ref_z_axis = np.array([0, 0, 1.])
    ref_x_axis = Rotation.from_euler('z', np.arange(np.ceil(360 / delta_angle).astype('i4')) * delta_angle,
                                     degrees=True).as_matrix() @ np.array([1., 0, 0])

    rotations = []
    z_axis = spiral_sphere(delta_angle)
    for i in range(len(z_axis)):
        rot = rotation_matrix_from_vectors(ref_z_axis, z_axis[i])
        x_axis = ref_x_axis @ rot.T
        y_axis = np.cross(z_axis[i], x_axis)
        rots = np.stack([x_axis, y_axis, np.tile(z_axis[i], (len(x_axis), 1))], axis=1)
        rotations.append(rots)

    rotations = np.concatenate(rotations, axis=0)

    if angle_thresh is not None:
        angles = np.arccos(((rotations.diagonal(axis1=1, axis2=2).sum(-1) - 1.) / 2.).clip(-1.,
                                                                                           1.)) * 180 / np.pi  # (n, )
        rotations = rotations[angles < angle_thresh]
    return rotations


def sample_rotations(n, angle_thre=None):
    if angle_thre is None:
        return Rotation.random(n).as_matrix()

    rots = np.zeros((0, 3, 3))
    while len(rots) < n:
        _rots = Rotation.random(int(n * 180 / angle_thre)).as_matrix()
        angles = np.arccos(((_rots.diagonal(axis1=1, axis2=2).sum(-1) - 1.) / 2.).clip(-1., 1.)) * 180 / np.pi  # (n, )
        rots = np.concatenate([_rots[angles < angle_thre], rots], axis=0)
    return rots[:n]


class ToothDataset(Dataset):
    test_scenes = ['1801_2000', '2001_2175']

    def __init__(self, data_path="/mnt/datasets/tooth/step_tooth",
                 train=True, num_pts=1024):
        self.train = train
        self.num_pts = num_pts
        if train:
            self.fps = []
            for scene in os.listdir(data_path):
                if scene in self.test_scenes:
                    continue
                fps = glob.glob(osp.join(data_path, scene, "*/35._Crown.stl"))
                self.fps.extend(fps)
        else:
            self.fps = []
            for scene in self.test_scenes:
                fps = glob.glob(osp.join(data_path, scene, "*/35._Crown.stl"))
                self.fps.extend(fps)
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
        xyz = np.array(pts).astype('f4')
        normals = np.array(m.face_normals[fi]).astype('f4')
        tid = int(osp.basename(fp)[:2])

        mat = Rotation.random().as_matrix().astype('f4')
        xyz = xyz @ mat.T
        normals = normals @ mat.T

        return xyz, normals, mat.T, tid, fp

    def get_test_data(self, i):
        fp = self.fps[i]
        m: trimesh.Trimesh = trimesh.load(fp)
        m.vertices -= m.vertices.mean(0)
        pts, fi = trimesh.sample.sample_surface_even(m, self.num_pts)
        xyz = np.array(pts).astype('f4')
        normals = np.array(m.face_normals[fi]).astype('f4')
        tid = int(osp.basename(fp)[:2])
        mat = self.mat[i]
        xyz = xyz @ mat.T
        normals = normals @ mat.T

        return xyz, normals, mat.T, tid, fp


def angle_diff(rot_pred, rot_y):
    rot = torch.bmm(rot_pred, rot_y.transpose(1, 2))
    angle = torch.clip((rot.diagonal(dim1=1, dim2=2).sum(-1) - 1.) / 2., -1, 1).acos() * 180 / np.pi  # (n, )
    return angle


class AngleGeodesic(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("angle_geodesic", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, rot_pred: torch.Tensor, rot_y: torch.Tensor):
        angles = angle_diff(rot_pred, rot_y)
        self.angle_geodesic += angles.sum()
        self.total += len(rot_pred)

    def compute(self):
        return self.angle_geodesic / self.total


class Encoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.net(x)
        x = torch.max(x, dim=2)[0]
        return x


class LitModel(pl.LightningModule):

    def __init__(self, input_channels, n_encode_embed, epochs, batch_size, lr, num_pts, num_workers):
        super().__init__()
        self.save_hyperparameters()

        # rotations
        self.rot_list_probs = [0.25, 0.25, 0.5]
        self.rot_list = nn.ParameterList([
            nn.Parameter(torch.tensor(generate_rotations(20.), dtype=torch.float32), requires_grad=False),
            nn.Parameter(torch.tensor(generate_rotations(5., 40), dtype=torch.float32), requires_grad=False),
            nn.Parameter(torch.tensor(generate_rotations(1.5, 10), dtype=torch.float32), requires_grad=False)
        ])
        self.train_rot_idx = np.random.choice(len(self.rot_list), 1000000, p=self.rot_list_probs)
        self.val_rot_idx = np.random.RandomState(42).choice(len(self.rot_list), 1000000, p=self.rot_list_probs)

        # augmentation
        self.rand_rot_list = [[], sample_rotations(1000, 20), sample_rotations(1000, 5)]

        # network
        self.encoder = Encoder(input_channels, n_encode_embed)

        args = type('', (), {})()
        args.k = 64
        args.dynamic = True
        args.use_stn = False
        args.input_channels = n_encode_embed
        args.output_channels = 1
        args.n_edgeconvs_backbone = 3
        args.edgeconv_channels = [64, 64, 64]
        args.emb_dims = 1024
        args.global_pool_backbone = 'max'
        args.norm = "instance"
        args.dropout = 0.
        self.net = MyDGCNN_Seg(args)

        # metrics
        self.train_angle_geodesic = AngleGeodesic()
        self.train_acc = torchmetrics.Accuracy()
        self.val_angle_geodesic = AngleGeodesic()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, xyz, normals, rots):
        if self.hparams.input_channels == 3:
            x = torch.tile(xyz.unsqueeze(1), (1, rots.shape[1], 1, 1)) @ rots.transpose(2, 3)  # (B, n_rots, n_pts, 3)
        elif self.hparams.input_channels == 6:
            xyz = torch.tile(xyz.unsqueeze(1), (1, rots.shape[1], 1, 1)) @ rots.transpose(2, 3)  # (B, n_rots, n_pts, 3)
            normals = torch.tile(normals.unsqueeze(1), (1, rots.shape[1], 1, 1)) @ rots.transpose(2, 3)
            x = torch.cat([xyz, normals], dim=-1)
        else:
            raise ValueError

        x = self.encoder(x)
        x = x.transpose(1, 2)  # (B, n_encode_embed, n_rots)
        # x = xyz.view(xyz.shape[0], xyz.shape[1], -1).transpose(1, 2).contiguous()
        return self.net(x)

    def forward_test(self, xyz, normals):
        ret = torch.eye(3, device=xyz.device).unsqueeze(0).repeat(xyz.shape[0], 1, 1)
        for rots in self.rot_list:
            rots = torch.tile(rots, (xyz.shape[0], 1, 1, 1))
            out, _ = self(xyz, normals, rots)
            out = out.squeeze(1)
            rot_pred = rots[torch.arange(xyz.shape[0]), out.argmax(dim=1)]
            xyz = xyz @ rot_pred.transpose(1, 2)
            normals = normals @ rot_pred.transpose(1, 2)

            ret = rot_pred @ ret

        return ret

    def get_y(self, rots, rot_y):
        ys = []
        for bi in range(rot_y.shape[0]):
            angles = angle_diff(rots[bi], torch.tile(rot_y[[bi]], (rots.shape[1], 1, 1)))
            # y = torch.exp(-(angles - angles.min()) ** 2 / (1. ** 2))
            # y = y / y.sum()
            y = torch.zeros_like(angles)
            y[angles.argmin()] = 1.
            ys.append(y)
        ys = torch.stack(ys)
        return ys

    def get_rots(self, rot_y, batch_idx, train=True):
        if train:
            rot_i = self.train_rot_idx[batch_idx % len(self.train_rot_idx)]
        else:
            rot_i = self.val_rot_idx[batch_idx % len(self.val_rot_idx)]
        rots = torch.tile(self.rot_list[rot_i], (rot_y.shape[0], 1, 1, 1))  # (B, N, 3, 3)

        if rot_i > 0:
            # rotate rots
            aug_rot = torch.tensor(self.rand_rot_list[rot_i][batch_idx % len(self.rand_rot_list[rot_i])],
                                   device=rot_y.device, dtype=torch.float32)
            rots = rots @ rot_y[:, None] @ aug_rot
        return rots

    def training_step(self, batch, batch_idx):
        xyz, normals, rot_y, tid, fp = batch
        rots = self.get_rots(rot_y, batch_idx, True)
        ys = self.get_y(rots, rot_y)
        pred, _ = self(xyz, normals, rots)
        pred = pred[:, 0]
        loss = F.cross_entropy(pred, ys)

        rot_pred = rots[torch.arange(pred.shape[0]), pred.argmax(1)]
        # rot_y_ = rots[:, ys.argmax(1)][:, 0]
        angles = angle_diff(rot_pred, rot_y)
        self.train_angle_geodesic(rot_pred, rot_y)
        self.train_acc(pred.argmax(1), ys.argmax(1))
        self.log("angle_diff", self.train_angle_geodesic, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("angle_diff_max", angles.max(), prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("acc", self.train_acc, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("loss", loss, batch_size=self.hparams['batch_size'])
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        xyz, normals, rot_y, tid, fp = batch
        rot_pred = self.forward_test(xyz, normals)

        angles = angle_diff(rot_pred, rot_y)
        self.val_angle_geodesic(rot_pred, rot_y)
        self.log("val_angle_diff", self.val_angle_geodesic, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("val_angle_diff_max", angles.max(), prog_bar=True, batch_size=self.hparams['batch_size'])

    def _validation_step(self, batch, batch_idx):
        xyz, normals, rot_y, tid, fp = batch
        rots = self.get_rots(rot_y, batch_idx, False)
        ys = self.get_y(rots, rot_y)
        pred, _ = self(xyz, normals, rots)
        pred = pred[:, 0]
        loss = F.cross_entropy(pred, ys)

        rot_pred = rots[torch.arange(pred.shape[0]), pred.argmax(1)]
        angles = angle_diff(rot_pred, rot_y)
        self.val_angle_geodesic(rot_pred, rot_y)
        self.val_acc(pred.argmax(1), ys.argmax(1))
        self.log("val_angle_diff", self.val_angle_geodesic, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("val_angle_diff_max", angles.max(), prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("val_acc", self.val_acc, prog_bar=True, batch_size=self.hparams['batch_size'])
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams['batch_size'])

    def on_train_epoch_end(self) -> None:
        self.train_rot_idx = np.random.choice(len(self.rot_list), 1000000, p=self.rot_list_probs)
        self.rand_rot_list = [[], sample_rotations(1000, 20), sample_rotations(1000, 5)]

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
@click.option('--input_channels', default=6, type=int)
@click.option('--n_encode_embed', default=256, type=int)
@click.option('--epoch', default=10)
@click.option('--batch_size', default=2)
@click.option('--lr', default=1e-3)
@click.option('--num_pts', default=1024)
@click.option('--num_workers', default=4)
@click.option('--version', default='0')
def run(**kwargs):
    print(colored(json.dumps(kwargs, indent=2), 'blue'))

    pl.seed_everything(42)

    # logger
    version = kwargs['version']
    logger = TensorBoardLogger("work_dir", name="demo", version=version)

    # trainer
    debug = False
    debug_args = {'limit_train_batches': 10, "limit_val_batches": 10} if debug else {}

    model = LitModel(kwargs['input_channels'], kwargs['n_encode_embed'], kwargs['epoch'], kwargs['batch_size'],
                     kwargs['lr'], kwargs['num_pts'], kwargs['num_workers'])
    callback = ModelCheckpoint(save_last=True)
    trainer = pl.Trainer(logger, accelerator='gpu', max_epochs=kwargs["epoch"], callbacks=[callback], **debug_args)

    # fit
    trainer.fit(model)

    # results = trainer.test()
    # print(results)


if __name__ == '__main__':
    run()
