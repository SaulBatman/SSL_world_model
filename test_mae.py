import torch
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from models.MAE import mae_vit_base_patch16
import warnings

warnings.filterwarnings(
    "ignore", "You are using `torch.load` with `weights_only=False`*."
)

mae_model = mae_vit_base_patch16()
chkpt_dir = "pretrained/MAE/mae_visualize_vit_base.pth"
checkpoint = torch.load(chkpt_dir, map_location="cpu")
msg = mae_model.load_state_dict(checkpoint["model"], strict=False)
print(msg)

mae_model = mae_model.cuda()
mae_model.eval()

test_img = Image.open("test_img.jpg")
test_img = test_img.resize((224, 224))
test_img = np.array(test_img) / 255.0

assert test_img.shape == (224, 224, 3)

# normalize by ImageNet mean and std
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
test_img = test_img - imagenet_mean
test_img = test_img / imagenet_std

# run the reconstruction
x = torch.tensor(test_img).float().cuda()

# make it a batch-like
x = x.unsqueeze(dim=0)
x = torch.einsum("nhwc->nchw", x)

# run MAE
loss, y, mask = mae_model(x.float(), mask_ratio=0.75)
y = mae_model.unpatchify(y)
y = torch.einsum("nchw->nhwc", y).detach().cpu()

# visualize the mask
mask = mask.detach()
mask = mask.unsqueeze(-1).repeat(
    1, 1, mae_model.patch_embed.patch_size[0] ** 2 * 3
)  # (N, H*W, p*p*3)
mask = mae_model.unpatchify(mask)  # 1 is removing, 0 is keeping
mask = torch.einsum("nchw->nhwc", mask).detach().cpu()

x = torch.einsum("nchw->nhwc", x)

# masked image
im_masked = x.detach().cpu() * (1 - mask)

# MAE reconstruction pasted with visible patches
im_paste = x.detach().cpu() * (1 - mask) + y * mask


def show_image(image, title=""):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


# make the plt figure larger
plt.rcParams["figure.figsize"] = [24, 24]

plt.subplot(1, 4, 1)
show_image(x[0].detach().cpu(), "original")

plt.subplot(1, 4, 2)
show_image(im_masked[0], "masked")

plt.subplot(1, 4, 3)
show_image(y[0], "reconstruction")

plt.subplot(1, 4, 4)
show_image(im_paste[0], "reconstruction + visible")

plt.savefig("test_vis.png")
