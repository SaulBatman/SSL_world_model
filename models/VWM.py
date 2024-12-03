import os
import torch
import torch.nn as nn
from models.MAE import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14


def PE(x, L):
    pe = list()
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn(2.0**i * x))
    return torch.cat(pe, -1)


class VisualWorldModel(nn.Module):
    def __init__(self, mae_ckpt_path, action_dim=3, view_dim=2, n_register=0):
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
        self.embed_dim = self.mae.embed_dim
        self.decoder_embed_dim = self.mae.decoder_embed_dim

        self.action_encoder_proj = nn.Linear(action_dim, self.embed_dim)
        self.view_src_encoder_proj = nn.Linear(view_dim, self.embed_dim)
        self.view_tgt_encoder_proj = nn.Linear(view_dim, self.embed_dim)

        self.action_decoder_proj = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.view_src_decoder_proj = nn.Linear(self.embed_dim, self.decoder_embed_dim)
        self.view_tgt_decoder_proj = nn.Linear(self.embed_dim, self.decoder_embed_dim)

        self.learnable_pe_encoder = nn.Parameter(torch.zeros((3, self.embed_dim)))
        self.learnable_pe_decoder = nn.Parameter(
            torch.zeros((3, self.decoder_embed_dim))
        )

        self.n_register = n_register
        if n_register > 0:
            self.encoder_register = nn.Parameter(
                torch.zeros((n_register, self.embed_dim))
            )
            self.decoder_register = nn.Parameter(
                torch.zeros((n_register, self.decoder_embed_dim))
            )

    def load_mae(self, mae_ckpt_path):
        checkpoint = torch.load(mae_ckpt_path, map_location="cpu")
        msg = self.mae.load_state_dict(checkpoint["model"], strict=False)
        print("Load MAE pretrained ckpt: ", msg)

    def forward(self, imgs, action, view_src, view_tgt, mask_ratio=0.75):
        """
        imgs: input image pair tensor [B, 2, C, H, W]
        action: action tensor [B, dim_action] => end effector position and rotation + gripper state
        view_src: view tensor [B, dim_view]
        view_tgt: view tensor [B, dim_view]
        """
        latent, mask, ids_restore = self.forward_encoder(
            imgs, action, view_src, view_tgt, mask_ratio
        )
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def forward_loss(self, imgs, pred, mask):
        return 0

    def forward_decoder(self, x, ids_restore):
        # split cls+img tokens and action, view tokens
        n_tokens = x.shape[1]
        x, action, view_src, view_tgt = x.split([n_tokens - 3, 1, 1, 1], dim=1)

        # embed tokens
        x = self.mae.decoder_embed(x)
        action = self.action_decoder_proj(action)
        view_src = self.view_src_decoder_proj(view_src)
        view_tgt = self.view_tgt_decoder_proj(view_tgt)

        # append mask tokens to sequence
        mask_tokens = self.mae.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.mae.decoder_pos_embed

        # expand action, view_src, view_tgt to all patches
        pad_tokens = (
            torch.cat([action, view_src, view_tgt], dim=1)
            + self.learnable_pe_decoder[None]
        )
        if self.n_register > 0:
            x = torch.cat([x, pad_tokens, self.decoder_register[None]], dim=1)
        else:
            x = torch.cat([x, pad_tokens], dim=1)

        # apply Transformer blocks
        for blk in self.mae.decoder_blocks:
            x = blk(x)
        x = self.mae.decoder_norm(x)

        # predictor projection
        x = self.mae.decoder_pred(x)

        # remove cls token, action, view tokens
        x = x[:, 1 : -3 - self.n_register, :]

        return x

    def forward_encoder(self, x, action, view_src, view_tgt, mask_ratio=0.75):
        # embed patches
        x = self.mae.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.mae.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.mae.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.mae.cls_token + self.mae.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # project action, view_src, view_tgt
        action = self.action_encoder_proj(action).unsqueeze(1)
        view_src = self.view_src_encoder_proj(view_src).unsqueeze(1)
        view_tgt = self.view_tgt_encoder_proj(view_tgt).unsqueeze(1)

        # expand action, view_src, view_tgt to all patches
        pad_tokens = (
            torch.cat([action, view_src, view_tgt], dim=1)
            + self.learnable_pe_encoder[None]
        )
        if self.n_register > 0:
            x = torch.cat([x, pad_tokens, self.encoder_register[None]], dim=1)
        else:
            x = torch.cat([x, pad_tokens], dim=1)

        # apply Transformer blocks
        for blk in self.mae.blocks:
            x = blk(x)
        x = self.mae.norm(x)

        # drop the registers
        if self.n_register > 0:
            x = x[:, : -self.n_register, :]

        return x, mask, ids_restore
