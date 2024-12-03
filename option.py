import argparse


def training_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--mae_ckpt_dir",
        type=str,
        default="pretrained/MAE/mae_visualize_vit_base.pth",
        help="path to MAE checkpoint",
    )

    parser.add_argument("--n_register", type=int, default=0, help="number of register")
