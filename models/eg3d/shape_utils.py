import time
import plyfile
import glob
import numpy as np
import os
import torch
import torch.utils.data
import skimage.measure
import argparse
import mrcfile
from tqdm import tqdm

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
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

def convert_sdf_samples_to_ply(
    numpy_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
    level=0.0
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    start_time = time.time()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    # try:
    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=[voxel_size] * 3
    )
    # except:
    #     pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)


def extract_shape(G, ws, triplaneoffsets=None, triplanes=None, device="cuda"):
    max_batch=1000000
    shape_res = 256
    samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)
    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    with torch.no_grad():
        while head < samples.shape[1]:
            torch.manual_seed(0)
            planes = G.backbone.synthesis(ws, update_emas=False)
            if triplaneoffsets is not None:
                planes = planes + triplaneoffsets
            elif triplanes is not None:
                planes = triplanes
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
            sigma = G.renderer.run_model(planes, G.decoder, samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], G.rendering_kwargs)['sigma']

            sigmas[:, head:head+max_batch] = sigma
            head += max_batch

    sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    # Trim the border of the extracted cube
    pad = int(30 * shape_res / 256)
    pad_value = -1000
    sigmas[:pad] = pad_value
    sigmas[-pad:] = pad_value
    sigmas[:, :pad] = pad_value
    sigmas[:, -pad:] = pad_value
    sigmas[:, :, :pad] = pad_value
    sigmas[:, :, -pad:] = pad_value
    return sigmas