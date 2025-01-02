import torch
from world_model.datasetmultiview import DatasetMultiview
from torch.utils.data import DataLoader
from trainer import VWMTrainer
from tqdm import tqdm
import lightning as L
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import os

parser = ArgumentParser()
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--n_epoch", type=int, default=500)
parser.add_argument("--mask_ratio", type=float, default=0.5)
parser.add_argument("--n_epoch_save", type=int, default=50)
parser.add_argument("--wandb_project", type=str, default="SSL_final_project")
parser.add_argument("--exp_name", type=str, default="baseline")
parser.add_argument("--multi_view", action="store_true")

parser.add_argument(
    "--mae_type", type=str, default="base", choices=["base", "large", "huge"]
)
args = parser.parse_args()

mae_ckpt_path = f"/users/zli419/data/users/zli419/SSL/SSL_world_model/pretrained/MAE/mae_visualize_vit_{args.mae_type}.pth"
action_dim = 7

train_dataset = DatasetMultiview(
    dataset_root="/users/zli419/data/users/zli419/SSL/SSL_world_model",
    mode="train",
    enable_multi_view=args.multi_view,
)
test_dataset = DatasetMultiview(
    dataset_root="/users/zli419/data/users/zli419/SSL/SSL_world_model",
    mode="val",
    enable_multi_view=True
)


train_loader = DataLoader(
    train_dataset,
    batch_size=args.bs,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
)
test_dataset = DataLoader(test_dataset)

callbacks = [
    ModelCheckpoint(
        filename="{epoch:04d}",
        every_n_epochs=args.n_epoch_save,
        monitor="epoch",
        mode="max",
        save_top_k=5,
        save_on_train_epoch_end=True,
    ),
]

wandb_logger = WandbLogger(
    project=args.wandb_project,
    name=args.exp_name,
    log_model=False,
)
hparams = vars(args)
wandb_logger.log_hyperparams(hparams)

os.makedirs(f"logs/{args.exp_name}", exist_ok=True)
VWM_trainer = VWMTrainer(
    mae_ckpt_path,
    action_dim,
    f"logs/{args.exp_name}",
    train_dataset.intrinsic,
    mask_ratio=args.mask_ratio,
)
trainer = L.Trainer(
    max_epochs=args.n_epoch,
    accelerator="gpu",
    devices=[0],
    deterministic=False,
    detect_anomaly=False,
    benchmark=True,
    check_val_every_n_epoch=10,
    logger=wandb_logger,
    callbacks=callbacks,
)
trainer.fit(
    model=VWM_trainer, train_dataloaders=train_loader, val_dataloaders=test_dataset
)

# breakpoint()

# for epoch in range(args.n_epoch):
#     total_loss = 0.0
#     for x in tqdm(my_loader, ncols=100):
#         img1, img2, action, src_view, tgt_view = x
#         imgs = torch.stack([img1, img2], dim=1).cuda()
#         action = action.cuda()

#         src_plucker = (
#             plucker_embedding(src_view.cuda(), uv, intr)
#             .reshape(bs, img_size, img_size, 6)
#             .permute(0, 3, 1, 2)
#         )
#         tgt_plucker = (
#             plucker_embedding(tgt_view.cuda(), uv, intr)
#             .reshape(bs, img_size, img_size, 6)
#             .permute(0, 3, 1, 2)
#         )
#         loss, pred, mask = net(imgs, action, src_plucker, tgt_plucker, mask_ratio=0.5)

#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         total_loss += loss.item()

#     writer.add_scalar("Loss/train", total_loss / len(my_loader), epoch)

#     if (epoch + 1) % 100 == 0:
#         torch.save(
#             net.state_dict(),
#             f"/users/zli419/data/users/zli419/SSL/SSL_world_model/results/vwm_{epoch + 1}.pth",
#         )

#     if (epoch + 1) % 20 == 0:
#         writer.add_image("Image/Current_Frame", img1[0], global_step=epoch)

#         pred = net.mae.unpatchify(pred).detach().cpu()
#         writer.add_image("Image/Next_Frame", pred[0], global_step=epoch)

#         writer.add_image("Image/Error", pred[0] - img2[0], global_step=epoch)

# writer.flush()
# writer.close()
