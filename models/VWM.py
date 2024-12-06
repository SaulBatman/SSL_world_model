import os
import torch
import torch.nn as nn
from models.MAE import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14
import copy
from timm.models.vision_transformer import PatchEmbed


def PE(x, L):
    pe = list()
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2.0**i * x))
    return torch.cat(pe, -1)


class VisualWorldModel(nn.Module):
    def __init__(self, mae_ckpt_path, action_dim=3, n_register=0):
        super().__init__()
        mae_type = os.path.basename(mae_ckpt_path).split(".")[0].split("_")[-1]
        if mae_type == "base":
            self.mae = mae_vit_base_patch16()
        elif mae_type == "large":
            self.mae = mae_vit_large_patch16()
        elif mae_type == "huge":
            self.mae = mae_vit_huge_patch14()
        else:
            raise NotImplementedError

        self.load_mae(mae_ckpt_path)
        embed_dim = self.mae.embed_dim
        img_size = self.mae.img_size
        patch_size = self.mae.patch_size
        decoder_embed_dim = self.mae.decoder_embed_dim
        self.src_ray_PE_encoder = self.init_zero_module(
            PatchEmbed(img_size, patch_size, 6, embed_dim)
        )

        self.action_encoder_proj = nn.Linear(action_dim, embed_dim)
        self.action_decoder_proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.tgt_ray_PE_decoder = self.init_zero_module(
            PatchEmbed(img_size, patch_size, 6, decoder_embed_dim)
        )

        self.n_register = n_register
        if n_register > 0:
            self.encoder_register = nn.Parameter(torch.zeros((n_register, embed_dim)))
            self.decoder_register = nn.Parameter(
                torch.zeros((n_register, decoder_embed_dim))
            )

        self.l2_loss = nn.MSELoss()

    @staticmethod
    def init_zero_module(module):
        for p in module.parameters():
            p.data.zero_()
        return module

    def load_mae(self, mae_ckpt_path):
        checkpoint = torch.load(mae_ckpt_path, map_location="cpu")
        msg = self.mae.load_state_dict(checkpoint["model"], strict=False)
        print("Load MAE pretrained ckpt: ", msg)

    def forward(self, imgs, action, src_ray_img, tgt_ray_img, mask_ratio=0.75):
        """
        imgs: input image pair tensor [B, 2, C, H, W]
        action: action tensor [B, dim_action] => end effector position and rotation + gripper state
        ray_img: view tensor [B, 6, H, W]
        """
        src_img = imgs[:, 0]
        tgt_img = imgs[:, 1]
        latent, mask, ids_restore = self.forward_encoder(
            src_img, action, src_ray_img, mask_ratio
        )
        pred = self.forward_decoder(latent, ids_restore, tgt_ray_img)  # [N, L, p*p*3]
        loss = self.forward_loss(tgt_img, pred, mask)
        return loss, pred, mask

    def forward_loss(self, imgs, pred, mask):
        target = self.mae.patchify(imgs)

        loss = (pred - target) ** 2
        return loss.mean()

    def forward_decoder(self, x, ids_restore, tgt_ray_img):
        # split cls+img tokens and action, view tokens
        n_tokens = x.shape[1]
        x, action = x.split([n_tokens - 1, 1], dim=1)

        # embed tokens
        x = self.mae.decoder_embed(x)
        action = self.action_decoder_proj(action)
        plucker_embed = self.tgt_ray_PE_decoder(tgt_ray_img)

        # append mask tokens to sequence
        mask_tokens = self.mae.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_ + plucker_embed], dim=1)  # append cls token

        # add pos embed
        x = x + self.mae.decoder_pos_embed

        if self.n_register > 0:
            x = torch.cat([x, action, self.decoder_register[None]], dim=1)
        else:
            x = torch.cat([x, action], dim=1)

        # apply Transformer blocks
        for blk in self.mae.decoder_blocks:
            x = blk(x)
        x = self.mae.decoder_norm(x)

        # predictor projection
        x = self.mae.decoder_pred(x)

        # remove cls token, action, view tokens
        x = x[:, 1 : -1 - self.n_register, :]

        return x

    def forward_encoder(self, x, action, ray_img, mask_ratio=0.75):
        # embed patches
        x = self.mae.patch_embed(x)
        plucker_embed = self.src_ray_PE_encoder(ray_img)

        # add pos embed w/o cls token
        x = x + self.mae.pos_embed[:, 1:, :] + plucker_embed

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.mae.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.mae.cls_token + self.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # project action
        action = self.action_encoder_proj(action).unsqueeze(1)

        if self.n_register > 0:
            x = torch.cat([x, action, self.encoder_register[None]], dim=1)
        else:
            x = torch.cat([x, action], dim=1)

        # apply Transformer blocks
        for blk in self.mae.blocks:
            x = blk(x)
        x = self.mae.norm(x)

        # drop the registers
        if self.n_register > 0:
            x = x[:, : -self.n_register, :]

        return x, mask, ids_restore
