import torch
from torch import nn
import torch.nn.functional as F
from models.encoders import psp_encoders
from models.eg3d.triplane import TriPlaneGenerator
from configs.paths_config import model_paths
import pickle
from models.encoders.triplane_encoders import TriPlane_Encoder
from torch_utils import misc

class TriPlaneNet(nn.Module):
    def __init__(self, opts):
        super(TriPlaneNet, self).__init__()
        self.opts = opts
        self.set_psp_encoder()
        self.set_triplane_encoder()
        self.set_eg3d_generator()

        self.load_weights()

    def forward(self, x, camera_params, novel_view_camera_params=None, resize=True, return_latents=False, return_yhat_psp=False, return_triplaneoffsets=False, CTTR=False):
        if novel_view_camera_params is None:
            novel_view_camera_params = camera_params
        x_clone = x.clone().detach()
        y_hat_psp, codes = self.__get_initial_inversion(x, camera_params)
        
        new_codes = codes.clone().detach()
        y_hat_psp_clone = y_hat_psp.clone().detach()
        x_input = torch.cat([y_hat_psp_clone, x_clone - y_hat_psp_clone], dim=1)
        
        triplane_offsets = self.triplanenet_encoder(x_input)

        if CTTR:
            images = self.decoder.synthesis(new_codes, camera_params, triplane_offsets=triplane_offsets, noise_mode='const')
        else:
            images = self.decoder.synthesis(new_codes, novel_view_camera_params, triplane_offsets=triplane_offsets, noise_mode='const')
        y_hat = images['image']
        if resize or CTTR:
           y_hat = F.adaptive_avg_pool2d(y_hat, (256, 256))

        if CTTR:
            x_input = torch.cat([y_hat, x_clone - y_hat], dim=1)
            triplane_offsets = self.triplanenet_encoder(x_input)
        
            images = self.decoder.synthesis(new_codes, novel_view_camera_params, triplane_offsets=triplane_offsets, noise_mode='const')
            y_hat = images['image']
            if resize:
                y_hat = F.adaptive_avg_pool2d(y_hat, (256, 256))
        
        if not return_latents and not return_yhat_psp and not return_triplaneoffsets:
            return y_hat
        outputs = [y_hat]
        if return_latents:
            outputs += [codes]
        if return_yhat_psp:
            outputs += [y_hat_psp]
        if return_triplaneoffsets:
            outputs += [triplane_offsets]
        return outputs
    

    def __get_initial_inversion(self, x, camera_params, resize=True):
        codes = self.psp_encoder(x)
        # add to average latent code
        codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
        y_hat = self.decoder.synthesis(codes, camera_params, noise_mode='const')
        y_hat = F.adaptive_avg_pool2d(y_hat['image'], (256, 256))
        return y_hat, codes

    def set_psp_encoder(self):
        self.psp_encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)

    def set_triplane_encoder(self):
        self.triplanenet_encoder = TriPlane_Encoder()

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
            altered_input_layer = torch.randn(shape[0], 6, shape[2], shape[3], dtype=torch.float32)
            altered_input_layer[:, :3, :, :] = encoder_ckpt['input_layer.0.weight']
            encoder_ckpt['input_layer.0.weight'] = altered_input_layer
            self.triplanenet_encoder.load_state_dict(encoder_ckpt, strict=False)
        else:
            checkpoint = torch.load(self.opts.checkpoint_path, map_location='cpu')['state_dict']
            self.load_state_dict(checkpoint, strict=True)
            print(f"Loading encoders weights from {self.opts.checkpoint_path}")

