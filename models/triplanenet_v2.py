import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.eg3d.triplane_v2 import TriPlaneGenerator
from configs.paths_config import model_paths
import pickle
from models.encoders.triplane_encoders_v2 import TriPlane_Encoder
from torch_utils import misc

class TriPlaneNet(nn.Module):
    def __init__(self, opts):
        super(TriPlaneNet, self).__init__()
        self.opts = opts
        self.set_psp_encoder()
        self.set_triplane_encoder()
        self.set_eg3d_generator()

        self.load_weights()

    def forward(self, x, camera_params, mirror_camera_params, novel_view_camera_params=None, x_mirror=None):

        x_clone = x.clone().detach()
        if novel_view_camera_params is None:
            novel_view_camera_params = mirror_camera_params
        initial_outputs = self.get_initial_inversion(x, camera_params.clone().detach(), mirror_camera_params)
        
        new_codes = initial_outputs["codes"].clone().detach()
        y_hat_initial_clone = initial_outputs["y_hat_initial_resized"].clone().detach()
        y_hat_initial_mirror_clone = initial_outputs["y_hat_initial_novel_resized"].clone().detach()

        x_input = torch.cat([y_hat_initial_clone, x_clone - y_hat_initial_clone, x_mirror - y_hat_initial_mirror_clone], dim=1)
        
        triplane_offsets = self.triplanenet_encoder(x_input, initial_outputs["triplanes"].clone().detach())

        outs = self.decoder.synthesis(new_codes, camera_params.clone().detach(), triplane_offsets=triplane_offsets, noise_mode='const')
        y_hat = outs["image"]
        y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))
        outputs = {"y_hat": y_hat, "y_hat_resized": y_hat_resized, "triplane_offsets": triplane_offsets, "latent_codes": initial_outputs["codes"], "depth": outs["image_depth"], }
        if novel_view_camera_params is not None:
            outs_novel = self.decoder.synthesis(new_codes, novel_view_camera_params.clone().detach(), triplane_offsets=triplane_offsets, noise_mode='const')
            y_hat_novel = outs_novel["image"]
            y_hat_novel_resized = F.adaptive_avg_pool2d(y_hat_novel, (256, 256))
            outputs["y_hat_novel"] = y_hat_novel
            outputs["y_hat_novel_resized"] = y_hat_novel_resized
            outputs["depth_novel"] = outs_novel["image_depth"]
            return outputs
        
        return outputs
    

    def get_initial_inversion(self, x, camera_params, novel_view_camera_params):
        codes = self.psp_encoder(x)
        # add to average latent code
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        outs = self.decoder.synthesis(codes, camera_params, noise_mode='const')
        y_hat = outs['image']
        y_hat_resized = F.adaptive_avg_pool2d(y_hat, (256, 256))
        outputs = {"codes": codes, "y_hat_initial": y_hat, "y_hat_initial_resized": y_hat_resized, "triplanes": outs["triplanes"], "depth": outs["image_depth"]}
        if novel_view_camera_params is not None:
            y_hat_novel = self.decoder.synthesis(codes, novel_view_camera_params, noise_mode='const')['image']
            y_hat_novel_resized =  F.adaptive_avg_pool2d(y_hat_novel, (256, 256))
            outputs["y_hat_initial_novel"] = y_hat_novel
            outputs["y_hat_initial_novel_resized"] = y_hat_novel_resized
            return outputs

        return outputs

    def set_psp_encoder(self):
        self.psp_encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)

    def set_triplane_encoder(self):
        self.triplanenet_encoder = TriPlane_Encoder(self.opts)

    def set_eg3d_generator(self):
        ckpt = torch.load(model_paths["eg3d_ffhq"])
        self.latent_avg = ckpt['latent_avg'].to(self.opts.device).repeat(self.opts.n_styles, 1)

        init_args = ()
        init_kwargs = ckpt['init_kwargs']
        self.decoder = TriPlaneGenerator(*init_args, **init_kwargs).eval().requires_grad_(False).to(self.opts.device)
        self.decoder.neural_rendering_resolution = 128
        self.decoder.load_state_dict(ckpt['G_ema'], strict=False)
        self.decoder.requires_grad_(False)

    def load_weights(self):

        if self.opts.checkpoint_path is None:
            print('Loading encoders weights from irse50!')
            encoder_ckpt = torch.load(model_paths['ir_se50'])
            self.psp_encoder.load_state_dict(encoder_ckpt, strict=False)

            # alter cuz triplane encoder works with concatenated inputs
            shape = encoder_ckpt['input_layer.0.weight'].shape
            altered_input_layer = torch.randn(shape[0], 9, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
            encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            self.triplanenet_encoder.load_state_dict(encoder_ckpt, strict=False)
        else:
            checkpoint = torch.load(self.opts.checkpoint_path, map_location='cpu')['state_dict']
            self.load_state_dict(checkpoint, strict=False)
            print(f"Loading encoders weights from {self.opts.checkpoint_path}")