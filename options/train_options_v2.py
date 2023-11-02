from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--device', default='cuda:0', type=str, help='CUDA device to train the model on')
        self.parser.add_argument('--n_styles', default=14, type=int, help='Number of style inputs to the generator')
        self.parser.add_argument('--use_pixelshuffle', action="store_true", help='Whether to use Pixel Shuffle during upsampling.')

        self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--batch_size', default=3, type=int, help='Batch size for first branch training only (first phase)')
        self.parser.add_argument('--batch_size_after_triplanenet', default=3, type=int, help='Batch size after triplanenet kicks in (second phase)')
        self.parser.add_argument('--batch_size_after_discriminator', default=3, type=int, help='Batch size after discriminator kicks in (third phase)')
        self.parser.add_argument('--batch_size_after_pspstop', default=3, type=int, help='Batch size after stopping first branch training (fourth phase)')

        self.parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')

        self.parser.add_argument('--use_discriminator', action="store_true", help='Whether to use Discriminator.')
        self.parser.add_argument('--learning_rate_dis', default=0.0001, type=float, help='Optimizer learning rate for discriminator')

        self.parser.add_argument('--learning_rate_psp', default=0.0001, type=float, help='Optimizer learning rate for psp')
        self.parser.add_argument('--learning_rate_triplane', default=0.0001, type=float, help='Optimizer learning rate for triplanenet')

        self.parser.add_argument('--optim_name_psp', default='ranger', type=str, help='Which optimizer to use for psp')
        self.parser.add_argument('--optim_name_triplane', default='ranger', type=str, help='Which optimizer to use for triplanenet')

        self.parser.add_argument("--add_triplanenet", type=int, default=20000, help='Steps after which triplanenet training starts')
        self.parser.add_argument("--add_discriminator", type=int, default=600000, help='Steps after which discriminator training starts')
        self.parser.add_argument("--stop_psp", type=int, default=500000, help='Steps after which psp training stops')

        self.parser.add_argument('--lpips_lambda_psp', default=0.8, type=float, help='LPIPS loss multiplier factor for psp')
        self.parser.add_argument('--lpips_lambda_psp_mirror', default=0.8, type=float, help='LPIPS loss multiplier factor for mirror loss')

        self.parser.add_argument('--lpips_lambda_triplane', default=0.8, type=float, help='LPIPS loss multiplier factor for triplanenet')
        self.parser.add_argument('--lpips_lambda_triplane_mirror', default=0.8, type=float, help='LPIPS loss multiplier factor for mirror loss')

        self.parser.add_argument('--id_lambda_psp', default=0.1, type=float, help='ID loss multiplier factor for psp')
        self.parser.add_argument('--id_lambda_psp_mirror', default=0.1, type=float, help='ID loss multiplier factor for mirror loss')

        self.parser.add_argument('--id_lambda_triplane', default=0.1, type=float, help='ID loss multiplier factor for triplanenet')
        self.parser.add_argument('--id_lambda_triplane_mirror', default=0.1, type=float, help='ID loss multiplier factor for mirror loss')

        self.parser.add_argument('--l2_lambda_psp', default=1.0, type=float, help='L2 loss multiplier factor for psp')
        self.parser.add_argument('--l2_lambda_psp_mirror', default=1.0, type=float, help='L2 loss multiplier factor for mirror loss')

        self.parser.add_argument('--l1_lambda_triplane', default=1.0, type=float, help='L1 smooth loss multiplier factor for triplanenet')
        self.parser.add_argument('--l1_lambda_triplane_mirror', default=1.0, type=float, help='L1 smooth loss multiplier factor for mirror loss')

        self.parser.add_argument('--mirror_lambda', default=0.1, type=float, help='Mirror loss multiplier factor')
        self.parser.add_argument('--normalize_mirror', action="store_true", help='Normalize mirror loss by conf map mean')
        self.parser.add_argument('--adv_lambda', default=0.005, type=float, help='Adversarial loss multiplier factor')
        self.parser.add_argument('--d_r1_gamma', default=10.0, type=float, help='Weight of the r1 regularization')
        self.parser.add_argument('--d_reg_every', default=16, type=int, help='Interval for applying r1 regularization on discriminator')

        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint to continue training')

        self.parser.add_argument('--max_steps', default=1500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=10000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=100000, type=int, help='Model checkpoint interval')

        # arguments for weights & biases support
        self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
