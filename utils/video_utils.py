import numpy as np
import torch
import sys
sys.path.append(".")
sys.path.append("..")
from models.eg3d.camera_utils import LookAtPoseSampler
from models.eg3d.shape_utils import convert_sdf_samples_to_ply
import imageio
import scipy
from tqdm import tqdm
import os


def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size


def gen_interp_video(G, latent, mp4: str,  w_frames=60 * 4, kind='cubic', grid_dims=(1, 1),
                     num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image',
                     gen_shapes=False, device=torch.device('cuda'), triplane_off=None, **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]
    
    # change if needed
    # G.rendering_kwargs['depth_resolution'] = 512
    # G.rendering_kwargs['depth_resolution_importance'] = 512

    name = mp4[:-4]
    if num_keyframes is None:

        num_keyframes = 1 // (grid_w * grid_h)

    camera_lookat_point = torch.tensor([0, 0, 0.2], device=device) if cfg == 'FFHQ' else torch.tensor([0, 0, 0],
                                                                                                      device=device)

    # zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device)
    cam2world_pose = LookAtPoseSampler.sample(3.14 / 2, 3.14 / 2, camera_lookat_point, radius=2.7, device=device)
    intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    c = c.repeat(latent.shape[0], 1)
    ws = latent # 1, 14, 512

    if ws.shape[1] != G.backbone.mapping.num_ws:
        ws = ws.repeat([1,G.backbone.mapping.num_ws, 1])


    _ = G.synthesis(ws[:1], c[:1], noise_mode='const', triplane_offsets=triplane_off)  # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])
    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1]) # (5, 14, 512)
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)
    # Render video.
    max_batch = 10000000
    voxel_resolution = 512
    video_out = imageio.get_writer(mp4, mode='I', fps=120, codec='libx264', **video_kwargs)

    if gen_shapes:
        outdir = 'interpolation_{}/'.format(name)
        os.makedirs(outdir, exist_ok=True)
    all_poses = []
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                pitch_range = 0.25
                yaw_range = 0.35
                cam2world_pose = LookAtPoseSampler.sample(
                    3.14 / 2 + yaw_range * np.sin(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    3.14 / 2 - 0.05 + pitch_range * np.cos(2 * 3.14 * frame_idx / (num_keyframes * w_frames)),
                    camera_lookat_point, radius=2.7, device=device)
                all_poses.append(cam2world_pose.squeeze().cpu().numpy())
                intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
                c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)

                img = G.synthesis(ws=w.unsqueeze(0), c=c[0:1], triplane_offsets=triplane_off, noise_mode='const')[image_mode][0]

                if image_mode == 'image_depth':
                    img = -img
                    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1

                imgs.append(img)

                if gen_shapes:
                    # generate shapes
                    print('Generating shape for frame %d / %d ...' % (frame_idx, num_keyframes * w_frames))

                    samples, voxel_origin, voxel_size = create_samples(N=voxel_resolution, voxel_origin=[0, 0, 0],
                                                                       cube_length=G.rendering_kwargs['box_warp'])
                    samples = samples.to(device)
                    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
                    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
                    transformed_ray_directions_expanded[..., -1] = -1

                    head = 0
                    with tqdm(total=samples.shape[1]) as pbar:
                        with torch.no_grad():
                            while head < samples.shape[1]:
                                torch.manual_seed(0)
                                sigma = G.sample_mixed(samples[:, head:head + max_batch],
                                                       transformed_ray_directions_expanded[:, :samples.shape[1] - head],
                                                       w.unsqueeze(0), truncation_psi=psi, noise_mode='const')['sigma']
                                sigmas[:, head:head + max_batch] = sigma
                                head += max_batch
                                pbar.update(max_batch)

                    sigmas = sigmas.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).cpu().numpy()
                    sigmas = np.flip(sigmas, 0)

                    pad = int(30 * voxel_resolution / 256)
                    pad_top = int(38 * voxel_resolution / 256)
                    sigmas[:pad] = 0
                    sigmas[-pad:] = 0
                    sigmas[:, :pad] = 0
                    sigmas[:, -pad_top:] = 0
                    sigmas[:, :, :pad] = 0
                    sigmas[:, :, -pad:] = 0

                    output_ply = True
                    if output_ply:
                        convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1,
                                                   os.path.join(outdir, f'{frame_idx:04d}_shape.ply'), level=10)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()