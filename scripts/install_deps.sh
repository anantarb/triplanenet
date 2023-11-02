#!/bin/bash
conda create -n triplanenet python=3.9
conda activate triplanenet

pip install \
    numpy==1.22.4 \
    click==8.0.4 \
    torch==1.11.0+cu113 \
    torchvision==0.12.0+cu113 \
    Pillow==9.3.0 \
    scipy==1.7.1 \
    requests==2.26.0 \
    tqdm==4.62.2 \
    matplotlib==3.4.2 \
    imageio==2.9.0 \
    imageio-ffmpeg==0.4.3 \
    imgui==1.3.0 \
    glfw==2.2.0 \
    PyOpenGL==3.1.5 \
    pyspng==0.1.1 \
    psutil==5.9.4 \
    mrcfile==1.4.3 \
    tensorboard==2.9.1 \
    gdown==4.7.1 \
    opencv-python==4.6.0.66 \
    kornia==0.6.8 \
    mtcnn==0.1.1 \
    dominate==2.7.0 \
    scikit-image==0.19.3 \
    tensorflow==2.9.2 \
    trimesh==3.16.2 \
    ninja==1.11.1 \
    wandb==0.13.5 \
    pytorch-msssim==0.2.1 \
    plyfile==0.7.4 \
    --extra-index-url https://download.pytorch.org/whl/cu113

mkdir ./pretrained_models/
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1xi6C528TybEGyPgzWTUTRqCBbUdbg7Uc', 'pretrained_models/ffhq512-128.pth', quiet=False)"
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn', 'pretrained_models/model_ir_se50.pth', quiet=False)"
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj', 'pretrained_models/CurricularFace_Backbone.pth', quiet=False)"
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja', 'pretrained_models/mtcnn.tar.gz', quiet=False)"
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=17pzNC00BOFCu0oCqbCybfAiaZF7iwNUU', 'pretrained_models/triplanenet_final.pth', quiet=False)"
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1KET4pHodyTeYsMtJTTKMh0YJ2a-LhgLl', 'pretrained_models/discriminator.pth', quiet=False)"
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1Q7hGZw97nTmRyJ-itVGRaFJn9KyjAuhn', 'pretrained_models/triplanenet_v2_final.pth', quiet=False)"
cd pretrained_models/
tar -xvzf mtcnn.tar.gz

cd ../

git submodule update --init --recursive

rm -rf ./dataset_preprocessing/ffhq/unsup3d/unsup3d/__init__.py
cd ./dataset_preprocessing/ffhq/unsup3d/pretrained/
sh download_pretrained_celeba.sh
cd ../../../../

cd ./dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/
git clone https://github.com/NVlabs/nvdiffrast
cd nvdiffrast
pip install .

cd ..
git clone https://github.com/deepinsight/insightface.git
cp -r ./insightface/recognition/arcface_torch ./models/

python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1bw5Xf8C12pWmcMhNEu6PtsYVZkVucEN6', 'BFM/Exp_Pca.bin', quiet=False)"
mkdir checkpoints/
cd checkpoints/
mkdir pretrained/
cd ..
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1BlDBB4dLLrlN3cJhVL4nmrd_g6Jx6uP0', 'checkpoints/pretrained/epoch_20.pth', quiet=False)"
cd ../../../
