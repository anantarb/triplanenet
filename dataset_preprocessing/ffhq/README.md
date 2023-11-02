# Dataset preprocessing

## Image preprocessing
To pre-process images, do the following:
1. Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/) and [unsup3d](https://elliottwu.com/projects/20_unsup3d/) submodules are properly initialized
```
git submodule update --init --recursive
```
2. Follow [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/) README.md to download the required models and components and place them accordingly. You only need to download [Basel Face Model 2009 (BFM09)](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-0&id=basel_face_model) and place it inside the `BFM/` directory. All the other required models and components will be initialized if you run `source ./scripts/install_deps.sh` (see [README.md](./README.md)).  

3. Run the following commands
```
python preprocess_in_the_wild.py --indir=/path/to/images/ --conf_map
```
Pre-processed images will be saved inside `/path/to/images/preprocessed/`. Also, a metadata file `dataset.json` will be created inside `/path/to/images/preprocessed/` for labels. Each label is a 25-length list of floating point numbers, which is the concatenation of the flattened 4x4 camera extrinsic matrix and flattened 3x3 camera intrinsic matrix. Additionally, the confidence maps given by a pre-trained network from [unsup3d](https://elliottwu.com/projects/20_unsup3d/) will be saved inside `/path/to/images/preprocessed/conf_map/`. The confidence maps are only required to train TriPlaneNet v2 and not needed for training TriPlaneNet v1 and inference purposes. 

## Video preprocessing
To pre-process the video, do the following:
1. Follow 1 and 2 from Image preprocessing. 

2. Run the following commands
```
python preprocess_in_the_wild_videos.py --indir=/path/to/video --frame_rate=30 --out_dir=/path/to/output/dir
```
Pre-processed video frames will be saved inside `/path/to/output/dir/preprocessed/`. In addition, a metadata file `dataset.json` will be created inside `/path/to/output/dir/preprocessed/` for labels. Each label is a 25-length list of floating point numbers, which is the concatenation of the flattened 4x4 camera extrinsic matrix and flattened 3x3 camera intrinsic matrix. Now, you can directly use this `out_dir/preprocessed` directory as an `--data_path` argument to `scripts/inference_video_v2.py` and `scripts/inference_video.py`.