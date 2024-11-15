# SSL world modeling

## installation
1. initialiate conda
```
conda create -n videopo python==3.10.13
conda activate videopo
pip install -r requirements.txt
```
2. download dataset
Please download camera_meta.json, TASKNAME_spaceview_rel.hdf5, TASKNAME_spaceview_rgbd_rel.hdf5.zarr.zip into YOURDATAPATH. The google link is [here](https://drive.google.com/drive/folders/1FZTe1qSjcA7PnY0heno1bujCQ3FYRyjI?usp=sharing). For example, TASKNAME could be square_d0.
2. run exp
```
python train.py --dataset_root=YOURDATAPATH --results_root=./pretrained --main_camera=spaceview --task=square_d0 --modality=rgbd --num_input_frame=2 --num_output_frame=2 --train_num_epochs=2 --batch_size=16 --frame_skip=1 --training_diffusion_steps=100 --eval_diffusion_steps=16 --num_train_demo=1000 --save_and_sample_every_epoch=1 --num_eval_frame=80
```