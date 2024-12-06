import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
import lightning as L
from models.VWM import VisualWorldModel
from utils.plucker_ray import plucker_embedding
import numpy as np
from lightning.pytorch.loggers import WandbLogger
import os
import warnings

warnings.filterwarnings(
    "ignore", "You are using `torch.load` with `weights_only=False`*."
)


class VWMTrainer(L.LightningModule):
    def __init__(
        self,
        mae_ckpt_path,
        action_dim,
        save_dir,
        intrinsic: torch.tensor,
        img_size=224,
        mask_ratio=0.5,
        lr=1e-5,
        device="cuda",
    ):
        super().__init__()
        self.net = VisualWorldModel(mae_ckpt_path, action_dim)
        uv = (
            np.array(
                np.meshgrid(np.arange(img_size), np.arange(img_size), indexing="ij")
            )
            .reshape(2, -1)
            .T
        )
        self.uv = torch.tensor(uv, dtype=torch.float32).unsqueeze(0).to(device)

        self.intr = intrinsic.unsqueeze(0).to(device)

        self.mask_ratio = mask_ratio
        self.lr = lr
        self.img_size = img_size
        self.save_dir = save_dir

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        img1, img2, action, src_view, tgt_view = batch
        bs, _, img_size, __ = img1.shape
        assert img_size == self.img_size

        imgs = torch.stack([img1, img2], dim=1)
        src_plucker = (
            plucker_embedding(
                src_view, self.uv.repeat(bs, 1, 1), self.intr.repeat(bs, 1, 1)
            )
            .reshape(bs, img_size, img_size, 6)
            .permute(0, 3, 1, 2)
        )
        tgt_plucker = (
            plucker_embedding(
                tgt_view, self.uv.repeat(bs, 1, 1), self.intr.repeat(bs, 1, 1)
            )
            .reshape(bs, img_size, img_size, 6)
            .permute(0, 3, 1, 2)
        )
        loss, pred, mask = self.net(
            imgs, action, src_plucker, tgt_plucker, mask_ratio=self.mask_ratio
        )

        self.log("train/loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, action, src_view, tgt_view = batch
        bs, _, img_size, __ = img1.shape
        assert img_size == self.img_size

        imgs = torch.stack([img1, img2], dim=1)
        src_plucker = (
            plucker_embedding(
                src_view, self.uv.repeat(bs, 1, 1), self.intr.repeat(bs, 1, 1)
            )
            .reshape(bs, img_size, img_size, 6)
            .permute(0, 3, 1, 2)
        )
        tgt_plucker = (
            plucker_embedding(
                tgt_view, self.uv.repeat(bs, 1, 1), self.intr.repeat(bs, 1, 1)
            )
            .reshape(bs, img_size, img_size, 6)
            .permute(0, 3, 1, 2)
        )
        loss, pred, mask = self.net(
            imgs, action, src_plucker, tgt_plucker, mask_ratio=self.mask_ratio
        )

        self.log("val/loss", loss, on_epoch=True, on_step=False)

        pred = self.net.mae.unpatchify(pred)

        if batch_idx < 10 and self.current_epoch > 0:
            save_dir = os.path.join(self.save_dir, f"imgs/Epoch{self.current_epoch}")
            os.makedirs(save_dir, exist_ok=True)
            save_img_path = lambda name: os.path.join(
                save_dir, f"{name}_{batch_idx}.jpg"
            )
            pred_img = to_pil_image(pred[0].detach().cpu())
            pred_img.save(save_img_path("pred"))
            gt_img = to_pil_image(img2[0].detach().cpu())
            gt_img.save(save_img_path("gt"))
            error_map = torch.abs(pred - img1).detach().cpu()
            error_map_img = to_pil_image(error_map[0])
            error_map_img.save(save_img_path("error_map"))

            cat_img = torch.cat([img2[0], pred[0], torch.abs(pred - img2)[0]], dim=-1)
            cat_img = to_pil_image(cat_img.detach().cpu())
            cat_img.save(save_img_path("log"))

        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch > 0:
            save_dir = os.path.join(self.save_dir, f"imgs/Epoch{self.current_epoch}")
            imgs = os.listdir(save_dir)
            n_imgs = len(imgs) // 4

            if len(imgs) > 0 and isinstance(self.logger, WandbLogger):
                self.logger.log_image(
                    "val/vis",
                    [os.path.join(save_dir, f"log_{i}.jpg") for i in range(n_imgs)],
                )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
