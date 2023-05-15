# Dataset preprocessing

## Image preprocessing
To pre-process images do the following:
1. Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/) submodule is properly initialized
```
git submodule update --init --recursive
```
2. Follow [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/) README.md to download required models and components and place them accordingly.

3. Run the following commands
```
python preprocess_in_the_wild.py --indir=/path/to/images/
```
Pre-processed images will be saved inside `/path/to/images/preprocessed/`. In addition to that, a metadata file `dataset.json` will be created inside `/path/to/images/preprocessed/` for labels. Each label is a 25-length list of floating point numbers, which is the concatenation of the flattened 4x4 camera extrinsic matrix and flattened 3x3 camera intrinsic matrix.

## Video preprocessing
To pre-process video do the following:
1. Follow 1 and 2 from Image preprocessing. 

2. Run the following commands
```
python preprocess_in_the_wild_videos.py --indir=/path/to/video --frame_rate=30 --out_dir=/path/to/output/dir
```
Pre-processed video frames will be saved inside `/path/to/output/dir/preprocessed/`. In addition to that, a metadata file `dataset.json` will be created inside `/path/to/output/dir/preprocessed/` for labels. Each label is a 25-length list of floating point numbers, which is the concatenation of the flattened 4x4 camera extrinsic matrix and flattened 3x3 camera intrinsic matrix. Now, you can directly use this `out_dir/preprocessed` directory as an `--data_path` arugment to `scripts/inference_video.py`.