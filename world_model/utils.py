import os
import glob
import cv2
import wandb
import imageio


def get_paths(root="../berkeley"):
    f = []
    for dirpath, dirname, filename in os.walk(root):
        if "image" in dirpath:
            f.append(dirpath)
    print(f"Found {len(f)} sequences")
    return f

def get_paths_from_dir(dir_path):
    paths = glob.glob(os.path.join(dir_path, 'im*.jpg'))
    try:
        paths = sorted(paths, key=lambda x: int((x.split('/')[-1].split('.')[0])[3:]))
    except:
        print(paths)
    return paths

def render_video(path, frames, fps, codec):
    height, width, channels = frames[0].shape
    isColor = True if channels==3 else False
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=isColor)
    # Write each frame to the video file
    for frame in frames:
        # Convert from RGB (NumPy format) to BGR (OpenCV format)
        if isColor:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    # Release everything if job is finished
    out.release()

def render_video_with_imageio(path, frames, fps):
    writer = imageio.get_writer(path, fps=fps)
    # Write each frame to the video file
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def save_videos(folder_path, seqs, fps=10, codec='avc1', save_depth=True):
    """
    Save a numpy array of frames as a video.

    Args:
        seqs (np.ndarray): A numpy array of shape (B, T, C, H, W), where T is the number of frames,
                             C is the number of channels (should be 3 for RGB), H is the height, and W is the width.
        folder_path (str): The folder path where the video will be saved.
        fps (int): Frames per second of the output video.
        codec (str): FourCC code for the video codec (e.g., 'mp4v' for .mp4, 'XVID' for .avi).
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    video_logs = dict()
    for i, frames in enumerate(seqs):
        # Ensure frames are in the shape (T, H, W, C) with 3 channels (RGB)
        if frames.shape[1] not in [3, 4]:
            raise ValueError("The input frames must have 3 or 4 channels (RGB, RGBD).")
        
        # Rearrange the shape from (T, C, H, W) to (T, H, W, C)
        frames = frames.transpose(0, 2, 3, 1)
        
        rgb_video_name = f'rgb-video-{i}.mp4'
        rgb_path = os.path.join(folder_path, rgb_video_name)
        render_video_with_imageio(rgb_path, frames[...,:3], fps)
        video_logs[rgb_video_name] = wandb.Video(rgb_path)

        if save_depth:
            depth_video_name = f'depth-video-{i}.mp4'
            depth_path = os.path.join(folder_path, depth_video_name)
            render_video_with_imageio(depth_path, frames[...,3:4], fps)
            video_logs[depth_video_name] = wandb.Video(depth_path)
        
    return video_logs

