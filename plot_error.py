from glob import glob
from PIL import Image
import numpy as np

no_ar_loss_list = list()
ar_loss_list = list()

for i in range(60):
    ar_imgs = glob(f"/users/zli419/data/users/zli419/SSL/SSL_world_model/logs/single_view/test/ar/camera{i}/demo_16/*.jpg")
    no_ar_imgs = glob(f"/users/zli419/data/users/zli419/SSL/SSL_world_model/logs/single_view/test/no_ar/camera{i}/demo_16/*.jpg")
    gt_imgs = glob(f"/users/zli419/data/users/zli419/SSL/SSL_world_model/square_d0/demo_16/obs/camera{i}_image/*.jpg")

    ar_imgs = sorted(ar_imgs, key=lambda x: int(x.split("/")[-1].split(".")[0]))[:-1]
    no_ar_imgs = sorted(no_ar_imgs, key=lambda x: int(x.split("/")[-1].split(".")[0]))[:-1]
    gt_imgs = sorted(gt_imgs, key=lambda x: int(x.split("/")[-1].split(".")[0]))[1:]

    ar_imgs = [np.array(Image.open(img)) / 255. for img in ar_imgs]
    no_ar_imgs = [np.array(Image.open(img)) / 255. for img in no_ar_imgs]
    gt_imgs = [np.array(Image.open(img)) / 255. for img in gt_imgs]

    no_ar_loss = list()
    ar_loss = list()
    for i in range(len(ar_imgs)):
        no_ar_loss.append((np.sqrt((gt_imgs[i] - no_ar_imgs[i])**2)).mean())
        ar_loss.append((np.sqrt((gt_imgs[i] - ar_imgs[i])**2)).mean())
    
    no_ar_loss_list.append(np.array(no_ar_loss))
    ar_loss_list.append(np.array(ar_loss))

mean_no_ar_loss = np.mean(np.stack(no_ar_loss_list), axis=0)
mean_ar_loss = np.mean(np.stack(ar_loss_list), axis=0)
import matplotlib.pyplot as plt
plt.plot(mean_no_ar_loss, label="Single Step")
plt.plot(mean_ar_loss, label="AR")
plt.xlabel("Frame")
plt.ylabel("Pixel RMSE")
plt.legend()
plt.savefig("error.png")
