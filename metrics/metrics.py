import torch
from models.encoders.model_irse import IR_101
from models.mtcnn.mtcnn import MTCNN
from criteria.lpips.lpips import LPIPS
from configs.paths_config import model_paths
from pytorch_msssim import MS_SSIM
import torchvision.transforms.functional as F
import torchvision.transforms as trans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Metrics:
    "This class calculates different metrics given original image and reconstructed image."
    "Images are expected to be the tensor of size (3, H, W) and normalized between -1 and 1."

    def __init__(self):

        self.mse_module = torch.nn.MSELoss()
        self.lpips_module = LPIPS(net_type='alex')
        self.ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)

        self.facenet = IR_101(input_size=112)
        self.facenet.load_state_dict(torch.load(model_paths["circular_face"], map_location=device))
        self.facenet.to(device)
        self.facenet.eval()
        self.mtcnn = MTCNN()
        self.id_transform = trans.Compose([
                                trans.ToTensor(),
                                trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.universal_transform = trans.Compose([
                                trans.Resize((256, 256))
        ])


    def mse(self, X, Y):
        X = self.universal_transform(X)
        Y = self.universal_transform(Y)
        with torch.no_grad():
            return float(self.mse_module(Y.to(device), X.to(device)))


    def id_similarity(self, X, Y):
        X = self.universal_transform(X)
        Y = self.universal_transform(Y)
        X = (X + 1) / 2
        Y = (Y + 1) / 2
        X = F.to_pil_image(X)
        Y = F.to_pil_image(Y)
        with torch.no_grad():
            X, _ = self.mtcnn.align(X)

        with torch.no_grad():
            try:
                X_id = self.facenet(self.id_transform(X).unsqueeze(0).to(device))[0]
            except:
                return None

        with torch.no_grad():
            Y, _ = self.mtcnn.align(Y)

        with torch.no_grad():
            try:
                Y_id = self.facenet(self.id_transform(Y).unsqueeze(0).to(device))[0]
            except:
                return None
        return float(X_id.dot(Y_id))

    def lpips(self, X, Y):
        X = self.universal_transform(X)
        Y = self.universal_transform(Y)
        with torch.no_grad():
            return float(self.lpips_module(Y.unsqueeze(0).to(device), X.unsqueeze(0).to(device)))

    def ms_ssim(self, X, Y):
        X = self.universal_transform(X)
        Y = self.universal_transform(Y)
        X = ((X + 1) / 2) 
        Y = ((Y + 1) / 2)

        with torch.no_grad():
            return float(self.ms_ssim_module(X.unsqueeze(0).to(device), Y.unsqueeze(0).to(device)))