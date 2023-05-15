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

		self.parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=4, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=8, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=8, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate_psp', default=0.0001, type=float, help='Optimizer learning rate for psp')
		self.parser.add_argument('--learning_rate_triplane', default=0.0001, type=float, help='Optimizer learning rate for triplanenet')
		self.parser.add_argument('--optim_name_psp', default='ranger', type=str, help='Which optimizer to use for psp')
		self.parser.add_argument('--optim_name_triplane', default='ranger', type=str, help='Which optimizer to use for triplanenet')
		self.parser.add_argument("--add_triplanenet", type=int, default=20000, help='Steps after which triplanenet training starts')
		self.parser.add_argument("--stop_psp", type=int, default=500000, help='Steps after which psp training stops')

		self.parser.add_argument('--lpips_lambda_psp', default=0.8, type=float, help='LPIPS loss multiplier factor for psp')
		self.parser.add_argument('--lpips_lambda_triplane', default=0.1, type=float, help='LPIPS loss multiplier factor for triplanenet')
		self.parser.add_argument('--id_lambda_psp', default=0.1, type=float, help='ID loss multiplier factor for psp')
		self.parser.add_argument('--id_lambda_triplane', default=0.1, type=float, help='ID loss multiplier factor for triplanenet')
		self.parser.add_argument('--l2_lambda_psp', default=1.0, type=float, help='L2 loss multiplier factor for psp')
		self.parser.add_argument('--l1_lambda_triplane', default=1.0, type=float, help='L2 loss multiplier factor for triplanenet')
		
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint to continue training')

		self.parser.add_argument('--max_steps', default=1000000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=100, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=None, type=int, help='Model checkpoint interval')

		# arguments for weights & biases support
		self.parser.add_argument('--use_wandb', action="store_true", help='Whether to use Weights & Biases to track experiment.')

	def parse(self):
		opts = self.parser.parse_args()
		return opts
