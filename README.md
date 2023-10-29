# TriPlaneNet

https://github.com/seva100/triplanenet/assets/5861398/1f1a3f4e-586b-4a1f-a610-691683968e7b

TriPlaneNet inverts an input image into the latent space of 3D GAN for novel view rendering.

This is the official repository that contains source code for the arXiv paper [TriPlaneNet v1](https://anantarb.github.io/triplanenet).

[[Paper v2](https://arxiv.org/abs/2303.13497)] [[Paper v1](https://arxiv.org/abs/2303.13497v1)] [[Project Page](https://anantarb.github.io/triplanenet)] [[Video](https://youtu.be/GpmSswHMeWU)]

If you find TriPlaneNet useful for your work please cite:
```
@preprint{bhattarai2023triplanenet,
    title={TriPlaneNet: An Encoder for EG3D Inversion},
    author={Bhattarai, Ananta R. and Nie{\ss}ner, Matthias and Sevastopolsky, Artem},
    journal={arXiv preprint arXiv:2303.13497},
    year={2023}
}
```

## News

* **29.10.2023**: The second version of the article has been accepted to WACV 2024 and features additional contributions and better results. :fire: Code for TriPlaneNet v2 is coming soon.

## Prepare Environment

1. Clone the repository and run the script to set up the environment:

```
git clone https://github.com/anantarb/triplanenet.git --recursive
source ./scripts/install_deps.sh
```
This will set up a conda environment `triplanenet` with all dependencies. The script will also download the required pre-trained models and places them into the respective folders. For [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) that [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/) uses, get access to the model as described in [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/) and organize in the structure accordingly.

2. (Optional) Install `ffmpeg` on the system if you want to run the model on videos.

## Prepare datasets
Datasets are stored in a directory containing PNG/JPG files and a metadata file `dataset.json` for labels. Each label is a 25-length list of floating point numbers, which is the concatenation of the flattened 4x4 camera extrinsic matrix and flattened 3x3 camera intrinsic matrix. We provide an example of the dataset structure in `dataset_preprocessing/ffhq/example_dataset/`. Training Dataset is pre-processed using the procedure as described in [here](./dataset_preprocessing/ffhq/README.md). For inference, images should be pre-processed in a way that align with the training data following [dataset preprocessing](./dataset_preprocessing/ffhq/README.md). 

## Pretrained Models
Please download the pre-trained model from the following link. TriPlaneNet model contains the entire TriPlaneNet architecture, including the encoder and decoder weights.
| Path | Description
| :--- | :----------
|[EG3D Inversion](https://drive.google.com/file/d/17pzNC00BOFCu0oCqbCybfAiaZF7iwNUU/view?usp=share_link)  | TriPlaneNet trained with the (FFHQ dataset + synthesized EG3D samples) for EG3D inversion.

If you wish to use the pretrained model for training or inference, you may do so using the flag `--checkpoint_path`.

In addition, we provide various auxiliary models needed for training your own TriPlaneNet model from scratch as well as pretrained models needed for evaluation.
| Path | Description
| :--- | :----------
|[FFHQ EG3D](https://drive.google.com/file/d/1xi6C528TybEGyPgzWTUTRqCBbUdbg7Uc/view?usp=share_link) | EG3D model pretrained on FFHQ taken from [NVlabs](https://github.com/NVlabs/eg3d) with 512x512 output resolution.
|[IR-SE50 Model](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view) | Pretrained IR-SE50 model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in our ID loss during TriPlaneNet training.
|[CurricularFace Backbone](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view)  | Pretrained CurricularFace model taken from [HuangYG123](https://github.com/HuangYG123/CurricularFace) for use in ID similarity metric computation.
|[MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view)  | Weights for MTCNN model taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch) for use in ID similarity metric computation. (Unpack the tar.gz to extract the 3 model weights.)

By default, we assume that all auxiliary models are downloaded and saved to the directory `pretrained_models`. However, you may use your own paths by changing the necessary values in `configs/path_configs.py`.

## Training

### Preparing your Data
- Refer to `configs/paths_config.py` to define the necessary data paths and model paths for training and evaluation. For example, the dataset path inside `configs/paths_config.py` should look like this:
```
dataset_paths = {
	'test': '/path/to/dataset/train/',
	'train': '/path/to/dataset/test/',
}
```
We provide an example how the dataset directory should be structured [here](./dataset_preprocessing/ffhq/example_dataset). 

- Don't forget to pre-process the data as described in [here](./dataset_preprocessing/ffhq/README.md).

### Training TriPlaneNet
The main training script can be found in `scripts/train.py`.
Intermediate training results are saved to `opts.exp_dir`. This includes checkpoints, train outputs, and test outputs.
Additionally, if you have tensorboard installed, you can visualize tensorboard logs in `opts.exp_dir/logs`.
Experiments can also be tracked with [Weights & Biases](https://wandb.ai/home). To enable Weights & Biases (`wandb`), first make an account on the platform's webpage and install `wandb` using `pip install wandb`. Then, to train TriPlaneNet using `wandb`, simply add the flag `--use_wandb`.
Training, for example, can be launched with the following command
```
python scripts/train.py \
--exp_dir=/path/to/experiment \
--device=cuda:0 \
--n_styles=14 \
--batch_size=4 \
--test_batch_size=4 \
--workers=8
--test_workers=8 \
--val_interval=2500 \
--save_interval=5000 \
--use_wandb
```
We provide a highly configurable implementation. See `options/train_options.py` for a complete list of the configuration options.

## Testing

### Inference on images
Having trained your model, you can use `scripts/inference.py` to apply the model on a set of images.
```
python scripts/inference.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/test_data \
--test_batch_size=4 \
--test_workers=4 \
--couple_outputs \
--resize_outputs \
--novel_view_angles -0.3 0.3
```
For challenging cases, you can also apply cascaded test-time refinement (CTTR) by simply adding the flag `--CTTR`. For more inference options, see `options/test_options.py`.

### Inference on videos
You can use `scripts/inference_video.py` to apply the model on a video.
```
python scripts/inference_video.py \
--exp_dir=/path/to/experiment \
--checkpoint_path=experiment/checkpoints/best_model.pt \
--data_path=/path/to/extracted_frames \
--test_batch_size=4 \
--test_workers=4 \
--frame_rate=30 \
--novel_view_angles -0.3 0.3 
```
To extract frames from a video, refer to [dataset preprocessing](./dataset_preprocessing/ffhq/README.md).
For more video inference options, see `options/test_options_videos.py`.

## Limitations
- Our method works best on the frontal images due to frontal bias in the training dataset.

## Major changes
- In the paper, we report final ID similarity metric for all methods by performing the normalization `(ID_SIM) * 0.5 + 0.5`. However, we drop the normalization in this code.   

## Acknowledgements

Our work builds on top of amazing open-source networks and codebases. 
We thank the authors for providing them.

- [pSp](https://github.com/eladrich/pixel2style2pixel): a encoder-based approach to embed input images into W+ space of StyleGAN.
- [EG3D](https://github.com/NVlabs/eg3d): a state-of-the-art 3D GAN for geometry aware image synthesis.
