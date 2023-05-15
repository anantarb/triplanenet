from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to TriPlaneNet model checkpoint')
		self.parser.add_argument('--data_path', type=str, default='gt_images', help='Path to directory of images to evaluate')
		self.parser.add_argument('--novel_view_angles', nargs="*", type=float, default=[], help='Novel view angles from the frontal for novel view rendering')
		self.parser.add_argument('--frame_rate', type=int, default=30, help='Video Frame rate')
		self.parser.add_argument('--CTTR', action='store_true', help='Whether to apply CTTR')


		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

	def parse(self):
		opts = self.parser.parse_args()
		return opts