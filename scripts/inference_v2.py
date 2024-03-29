import os
from argparse import Namespace

from tqdm import tqdm
import time
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("..")

from datasets.dataset_v2 import ImageFolderDataset
from utils.common import tensor2im, log_input_image
from metrics.metrics import Metrics
from options.test_options_v2 import TestOptions
from models.triplanenet_v2 import TriPlaneNet

from models.eg3d.camera_utils import LookAtPoseSampler
from models.eg3d.shape_utils import extract_shape
from models.eg3d.shape_utils import convert_sdf_samples_to_ply
from utils.video_utils import gen_interp_video

def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'inference_coupled')
    out_path_shapes = os.path.join(test_opts.exp_dir, 'inference_shapes')
    out_path_depths = os.path.join(test_opts.exp_dir, 'inference_depths')
    out_path_videos = os.path.join(test_opts.exp_dir, 'inference_videos')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)
    os.makedirs(out_path_shapes, exist_ok=True)
    os.makedirs(out_path_depths, exist_ok=True)
    os.makedirs(out_path_videos, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    net = TriPlaneNet(opts)

    # change if needed
    net.decoder.rendering_kwargs['depth_resolution'] = int(net.decoder.rendering_kwargs['depth_resolution'] * 2)
    net.decoder.rendering_kwargs['depth_resolution_importance'] = int(
        net.decoder.rendering_kwargs['depth_resolution_importance'] * 2)

    net.eval()
    net.cuda()

    if opts.calculate_metrics:
        metrics = Metrics()
        scores = {}
        scores = {
            'mse': [],
            'lpips': [],
            'ms-ssim': [],
            'id-sim_same-view': [],

        }
        for angle_y in opts.novel_view_angles:
            scores[f'id-sim_{angle_y}'] = []
        

    print("Loading Dataset...")
    dataset = ImageFolderDataset(path=opts.data_path,
			                               resolution=None, use_labels=True)
    
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    global_time = []
    global_i = 0
    for x, _, camera_param, x_mirror, _, camera_param_mirror, _, fname in tqdm(dataloader):

        with torch.no_grad():
            x, camera_param, x_mirror, camera_param_mirror = x.cuda().float(), camera_param.cuda().float(), x_mirror.cuda().float(), camera_param_mirror.cuda().float()
            tic = time.time()
            results_batch, depth_batch, sigmas = run_on_batch(x, x_mirror, camera_param, camera_param_mirror, net, opts, fname)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(results_batch[0].shape[0]):
            res = None
            for j in range(len(results_batch)):
                result = tensor2im(results_batch[j][i])

                if opts.couple_outputs:
                    input_im = log_input_image(x[i], opts)
                    resize_amount = (256, 256)
                    if res is None:
                        res = np.concatenate([np.array(input_im.resize(resize_amount)),
                                                np.array(result.resize(resize_amount))], axis=1)
                    else:
                        res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)
                if j == 0:
                    im_save_path = os.path.join(out_path_results, f'{fname[i]}_same-view.png')
                    depth_save_path = os.path.join(out_path_depths, f'{fname[i]}_same-view.npy')
                    with open(depth_save_path, 'wb') as f:
                        np.save(f, depth_batch[i].cpu().numpy()[0])
                    if opts.calculate_metrics:
                        scores['mse'].append(metrics.mse(x[i], results_batch[j][i]))
                        scores['lpips'].append(metrics.lpips(x[i], results_batch[j][i]))
                        scores['ms-ssim'].append(metrics.ms_ssim(x[i], results_batch[j][i]))
                        id_sim = metrics.id_similarity(x[i], results_batch[j][i])
                        if id_sim is not None:
                            scores['id-sim_same-view'].append(id_sim)
                else:
                    im_save_path = os.path.join(out_path_results, f'{fname[i]}_{opts.novel_view_angles[j-1]}.png')
                    if opts.calculate_metrics:
                        id_sim = metrics.id_similarity(x[i], results_batch[j][i])
                        if id_sim is not None:
                            scores[f'id-sim_{opts.novel_view_angles[j-1]}'].append(id_sim)
                Image.fromarray(np.array(result)).save(im_save_path)
            input_im = (log_input_image(x[i], opts)).save(os.path.join(out_path_results, f'{fname[i]}_original.png'))
            if opts.couple_outputs:
                Image.fromarray(res).save(os.path.join(out_path_coupled, f'{fname[i]}.png'))

            if opts.shapes:
                convert_sdf_samples_to_ply(np.transpose(sigmas[i], (2, 1, 0)), [0, 0, 0], 1, os.path.join(out_path_shapes, f'{fname[i]}.ply'), level=10)
            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    if opts.calculate_metrics:
        scores = {key: float(np.nanmean(np.array(value))) for key, value in scores.items()}
        result_str += '\n' + f'{str(scores)}'
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)


def run_on_batch(x, x_mirror, camera_param, camera_param_mirror, net, opts, fname):
    outputs_batch = net(x, camera_param, camera_param_mirror, novel_view_camera_params=None, x_mirror=x_mirror)
    results_batch, depth_batch, latents_batch, triplaneoff_batch = [outputs_batch["y_hat_resized"]], outputs_batch["depth"], outputs_batch["latent_codes"], outputs_batch["triplane_offsets"]
    if opts.gen_video:
        for i in range(results_batch[0].size(0)):
            gen_interp_video(G=net.decoder, latent=latents_batch[i:i+1, :],
                            mp4=os.path.join(opts.exp_dir, f'inference_videos/{fname[i]}.mp4'), 
                            bitrate='10M', 
                            grid_dims=(1,1), 
                            num_keyframes=None,
                            w_frames=120*4,
                            psi=None,
                            truncation_cutoff=None, 
                            cfg='FFHQ', 
                            image_mode='image',
                            triplane_off=triplaneoff_batch[i:i+1, :, :, :])

    for angle_y in opts.novel_view_angles:
        angle_p = 0
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, 
                                                torch.tensor([0, 0, 0.2]).cuda(), 
                                                radius=2.7, batch_size=opts.test_batch_size, device="cuda")

        novel_view_camera_params = camera_param.clone()
        novel_view_camera_params[:, :16] = cam2world_pose.view(cam2world_pose.size(0), -1)
        results_batch += [net(x, camera_param, camera_param_mirror, novel_view_camera_params=novel_view_camera_params, x_mirror=x_mirror)["y_hat_novel_resized"]]
    sigmas = []
    if opts.shapes:
        for i in range(latents_batch.shape[0]):
            sigmas += [extract_shape(net.decoder, latents_batch[i:i+1, :], triplanes=triplaneoff_batch[i:i+1, :, :, :])]

    return results_batch, depth_batch, sigmas


if __name__ == '__main__':
    run()
