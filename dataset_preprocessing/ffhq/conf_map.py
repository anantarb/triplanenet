import argparse
import numpy as np
import torch
import PIL.Image
import os
import torch.nn as nn
from PIL import Image, ImageOps
from tqdm import tqdm
import sys
sys.path.append("./unsup3d/")
from unsup3d.networks import ConfNet

def _file_ext(fname):
    return os.path.splitext(fname)[1].lower()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument("--dest", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load('./unsup3d/pretrained/pretrained_celeba/checkpoint030.pth')
    conf_net = ConfNet(cin=3, cout=2, nf=64, zdim=128).to("cuda").eval()
    conf_net.load_state_dict(ckpt['netC'])

    _all_fnames = {os.path.relpath(os.path.join(root, fname), start=args.source) for root, _dirs, files in os.walk(args.source) for fname in files}
    PIL.Image.init()
    _image_fnames = sorted(fname for fname in _all_fnames if _file_ext(fname) in PIL.Image.EXTENSION)

    for i, p in tqdm(enumerate(_image_fnames)):
        pil_im = Image.open(os.path.join(args.source, p)).convert('RGB')
        image_name = p.split('/')[-1][:-4]
        im = np.uint8(pil_im)
        im = torch.FloatTensor(im /255.).permute(2,0,1).unsqueeze(0)
        im = nn.functional.interpolate(im, (256, 256), mode='bilinear', align_corners=False)
        im = nn.functional.interpolate(im, (64, 64), mode='bilinear', align_corners=False)
        input_im = im.to("cuda") *2.-1.
        with torch.no_grad():
            conf_sigma_l1, conf_sigma_percl = conf_net(input_im)
        conf_sigma_l1_flip = conf_sigma_l1[:,1:]
        conf_sigma_l1_flip = nn.functional.interpolate(conf_sigma_l1_flip, (256, 256), mode='bilinear', align_corners=False)
        conf_sigma_l1_flip = conf_sigma_l1_flip.cpu().squeeze().numpy()
        os.makedirs(args.dest, exist_ok=True)
        with open(f'{os.path.join(args.dest, image_name)}.npy', 'wb') as f:
            np.save(f, conf_sigma_l1_flip)


