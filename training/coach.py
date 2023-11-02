import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss
from criteria.lpips.lpips import LPIPS
from models.triplanenet import TriPlaneNet
from training.ranger import Ranger
from datasets.dataset import ImageFolderDataset
from configs.paths_config import dataset_paths

class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0
		self.device = self.opts.device

		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		# Initialize network
		self.net = TriPlaneNet(self.opts).to(self.device)

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		if self.opts.lpips_lambda_psp > 0 or self.opts.lpips_lambda_triplane > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda_psp > 0 or self.opts.id_lambda_triplane > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer_triplane = self.configure_triplane_optimizers()
		self.optimizer_psp = self.configure_psp_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

		# Resume training process from checkpoint path
		if self.opts.checkpoint_path is not None:
			ckpt = torch.load(self.opts.checkpoint_path, map_location="cpu")

			print("Load encoder optimizer from checkpoint")
			if "triplane_optimizer" in ckpt:
				self.optimizer_triplane.load_state_dict(ckpt["triplane_optimizer"])
			if "psp_optimizer" in ckpt:
				self.optimizer_psp.load_state_dict(ckpt["psp_optimizer"])

			if "step" in ckpt:
				self.global_step = ckpt["step"]
				print(f"Resuming training process from step {self.global_step}")

			if "best_val_loss" in ckpt:
				self.best_val_loss = ckpt["best_val_loss"]
				print(f"Current best val loss: {self.best_val_loss }")

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer_triplane.zero_grad()
				self.optimizer_psp.zero_grad()
				x, camera_param, _ = batch
				x, camera_param = x.to(self.device).float(), camera_param.to(self.device).float()
				x_clone = x.clone()
				if self.global_step >= self.opts.add_triplanenet:
					self.net.triplanenet_encoder.requires_grad_(True)
				else:
					self.net.triplanenet_encoder.requires_grad_(False)
				if self.global_step < self.opts.stop_psp:
					self.net.psp_encoder.requires_grad_(True)
				else:
					self.net.psp_encoder.eval().requires_grad_(False)
				outputs = self.net.forward(x, camera_params=camera_param, resize=True, return_yhat_psp=True)
				y_hat, y_hat_psp = outputs[0], outputs[1]
				loss_dict = {}
				if self.global_step < self.opts.stop_psp:
					loss_psp, loss_dict, id_logs = self.calc_loss_psp(x_clone, y_hat_psp, loss_dict)
					loss_psp.backward()
					self.optimizer_psp.step()
					self.optimizer_psp.zero_grad()
				if self.global_step >= self.opts.add_triplanenet:
					loss_triplane, loss_dict, id_logs = self.calc_loss_triplane(x, y_hat, loss_dict)
					loss_triplane.backward()
					self.optimizer_triplane.step()
					self.optimizer_triplane.zero_grad()


				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					self.parse_and_log_images(id_logs, x, y_hat, y_hat_psp, title='images/train')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.opts.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x, y_hat, y_hat_psp, id_logs, prefix="train", step=self.global_step, opts=self.opts)

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict is not None:
						if 'loss_triplane' in val_loss_dict:
							new_val_loss = val_loss_dict['loss_triplane']
						else:
							new_val_loss = val_loss_dict['loss_psp']

					if val_loss_dict and (self.best_val_loss is None or new_val_loss < self.best_val_loss):
						self.best_val_loss = new_val_loss
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			x, camera_param, _ = batch
			with torch.no_grad():
				x, camera_param = x.to(self.device).float(), camera_param.to(self.device).float()
				outputs = self.net.forward(x, camera_params=camera_param, resize=True, return_yhat_psp=True)
				y_hat, y_hat_psp = outputs[0], outputs[1]
				cur_loss_dict = {}
				if self.global_step < self.opts.stop_psp:
					_, cur_loss_dict, id_logs = self.calc_loss_psp(x, y_hat_psp, cur_loss_dict)

				if self.global_step >= self.opts.add_triplanenet:
					_, cur_loss_dict, id_logs = self.calc_loss_triplane(x, y_hat, cur_loss_dict)


			agg_loss_dict.append(cur_loss_dict)
			
			# Logging related
			self.parse_and_log_images(id_logs, x, y_hat, y_hat_psp,
									  title='images/test',
									  subscript='{:04d}'.format(batch_idx))

			# Log images of first batch to wandb
			if self.opts.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, y_hat, y_hat_psp, id_logs, prefix="test", step=self.global_step, opts=self.opts)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_triplane_optimizers(self):
		params = list(self.net.triplanenet_encoder.parameters())
		if self.opts.optim_name_triplane == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_triplane)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate_triplane)
		return optimizer
	
	def configure_psp_optimizers(self):
		params = list(self.net.psp_encoder.parameters())
		if self.opts.optim_name_psp == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate_psp)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate_psp)
		return optimizer

	def configure_datasets(self):
		
		train_dataset = ImageFolderDataset(path=dataset_paths['train'],
			                               resolution=None, use_labels=True)
		test_dataset = ImageFolderDataset(path=dataset_paths['test'],
									 resolution=None, use_labels=True)
		
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss_psp(self, x, y_hat, loss_dict):
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda_psp > 0:
			loss_id, sim_improvement, _ = self.id_loss(y_hat, x, x)
			loss_dict['loss_id_psp'] = float(loss_id)
			loss_dict['id_improve_psp'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda_psp
		
		if self.opts.l2_lambda_psp > 0:
			loss_l2 = F.mse_loss(y_hat, x)
			loss_dict['loss_l2_psp'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda_psp

		if self.opts.lpips_lambda_psp > 0:
			loss_lpips = self.lpips_loss(y_hat, x)
			loss_dict['loss_lpips_psp'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda_psp
		
		loss_dict['loss_psp'] = float(loss)
		return loss, loss_dict, id_logs
	
	def calc_loss_triplane(self, x, y_hat, loss_dict):
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda_triplane > 0:
			loss_id, sim_improvement, _ = self.id_loss(y_hat, x, x)
			loss_dict['loss_id_triplane'] = float(loss_id)
			loss_dict['id_improve_triplane'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda_triplane
		
		if self.opts.l1_lambda_triplane > 0:
			loss_l1 = F.smooth_l1_loss(y_hat, x)
			loss_dict['loss_l1_triplane'] = float(loss_l1)
			loss += loss_l1 * self.opts.l1_lambda_triplane
		if self.opts.lpips_lambda_triplane > 0:
			loss_lpips = self.lpips_loss(y_hat, x)
			loss_dict['loss_lpips_triplane'] = float(loss_lpips)
			loss += loss_lpips * self.opts.lpips_lambda_triplane
		
		loss_dict['loss_triplane'] = float(loss)
		return loss, loss_dict, id_logs

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
		if self.opts.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y_hat, y_hat_psp, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'y_hat': common.tensor2im(y_hat[i]),
				'y_hat_psp': common.tensor2im(y_hat_psp[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts),
			'best_val_loss': self.best_val_loss,
			'step': self.global_step,
			'triplane_optimizer': self.optimizer_triplane.state_dict(),
			'psp_optimizer': self.optimizer_psp.state_dict(),
		}
		return save_dict
