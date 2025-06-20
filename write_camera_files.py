from threestudio.data.uncond import RandomCameraDataset
from threestudio.data.uncond import RandomCameraDataModuleConfig
import numpy as np
from extern.sugar.gaussian_splatting.scene.colmap_loader import rotmat2qvec
import math
from tqdm import tqdm
import torch 

# Config and dataset
cfg = RandomCameraDataModuleConfig
# You may want to set cfg fields here if needed
cfg.height= 1024
cfg.width=1024
cfg.eval_camera_distance=4.
cfg.eval_height=1024
cfg.eval_width=1024
cfg.eval_elevation_deg = 0

dataset = RandomCameraDataset(cfg, "test")


def opengl_to_colmap_c2w(c2w):
    """
    Convert OpenGL-style c2w (camera looks -Z) to COLMAP c2w (camera looks +Z).
    c2w: (4, 4) torch.Tensor or np.ndarray
    Returns: (4, 4) torch.Tensor in COLMAP convention
    """
    # Axis conversion matrix: flips Y and Z
    M = torch.tensor([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=c2w.dtype, device=c2w.device)
    return c2w @ M

def c2w_to_colmap_pose(c2w):
    """
    Given OpenGL-style c2w, returns COLMAP quaternion and translation.
    """
    # 1. Convert to COLMAP convention
    c2w_colmap = opengl_to_colmap_c2w(c2w)
    # 2. Invert to get w2c
    w2c_colmap = torch.linalg.inv(c2w_colmap)
    # 3. Extract rotation and translation
    R = w2c_colmap[:3, :3].T.cpu().numpy()
    t = w2c_colmap[:3, 3].cpu().numpy()
    # print(-np.linalg.inv(R)@t)
    # 4. Convert rotation to quaternion (qw, qx, qy, qz)
    qvec = rotmat2qvec(R)
    return qvec, t


# Prepare lists to store camera/image info
image_entries = []
camera_id = 1  # single camera model for all images
model = "PINHOLE"

# Get intrinsics from the first batch (assuming all are the same)
batch0 = dataset[0]
width = int(batch0['width'])
height = int(batch0['height'])
print(width)
fovy = float(batch0['fovy'])
focal_length = 0.5 * height / math.tan(0.5 * fovy)
cx = width / 2.0
cy = height / 2.0

# Write cameras.txt
with open("../../Data/scene_info/cameras.txt", "w") as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(f"# Number of cameras: 1\n")
    f.write(f"{camera_id} {model} {width} {height} {focal_length} {focal_length} {cx} {cy}\n")

# Write images.txt
with open("../../Data/scene_info/images.txt", "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f.write(f"# Number of images: {len(dataset)}, mean observations per image: 0\n")
    for i, batch in tqdm(enumerate(dataset)):

        c2w = batch['c2w']
        qvec, T = c2w_to_colmap_pose(c2w)
        image_name = f"image_{i:05d}.png"
        # Write image line
        f.write(f"{i} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {T[0]} {T[1]} {T[2]} {camera_id} {image_name}\n")
        f.write("\n")  # Empty 2D-3D correspondences