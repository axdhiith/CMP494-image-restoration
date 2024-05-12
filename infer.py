import subprocess
import argparse
import os
from deep_options import opt
from deep_models import CC_Module
from funie_datasets import TestDataset, denorm
from funie_models import FUnIEGeneratorV2
import cv2
import numpy as np
import torch
import glob
import shutil
import torch
from torchvision import transforms
import shutil

class Predictor(object):
    def __init__(self, model, test_loader, model_path, save_path, is_cuda):

        self.test_loader = test_loader
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

        self.is_cuda = is_cuda

        self.model = model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found!")
        self.load(model_path)
        if self.is_cuda:
            self.model.cuda()

    def predict(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (paths, images) in enumerate(self.test_loader):
                bs = images.size(0)
                if self.is_cuda:
                    images = images.cuda()
                fake_images = self.model(images)

                fake_images = denorm(fake_images.data)
                fake_images = torch.clamp(fake_images, min=0., max=255.)
                fake_images = fake_images.type(torch.uint8)

                for idx in range(bs):
                    name = os.path.splitext(os.path.basename(paths[idx]))[0]
                    fake_image = fake_images[idx]
                    fake_image = transforms.ToPILImage()(fake_image).convert("RGB")
                    fake_image.save(f"{self.save_path}/{name}.png")
        return

    def load(self, model):
        device = "cuda:0" if self.is_cuda else "cpu"
        ckpt = torch.load(model, map_location=device)
        self.model.load_state_dict(ckpt["state_dict"])
        # print(f"At epoch: {ckpt['epoch']} (loss={ckpt['best_loss']:.3f})")
        print(f">>> Load generator from {model}")

def get_InferA(path):
    filesA = []
    filesA += sorted(glob.glob(path + "/*.*"))
    return filesA

def main():
    parser = argparse.ArgumentParser(description="Inference with the model.")
    parser.add_argument("--data", default="./inference_dataset", type=str, help="path to data")
    parser.add_argument("--num_workers", default=8, type=int, help="number of workers")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument('--deep_path', default='./saved_model/ckpts/netG_100.pt', type=str, help='path to DeepWaveNet model weight')
    parser.add_argument('--funie_path', default='./saved_model/funiegan/best_gen.pth.tar', type=str, help='path to FUnIE-GAN model weight')
    args = parser.parse_args()
    
    print("\n\n------INFERENCING DEEPWAVENET PIPELINE: IN PROGRESS------")
    
    saved_path = './saved_model'
    default_save_path = './inference'
    os.makedirs(default_save_path, exist_ok=True)

    CHECKPOINTS_DIR = opt.checkpoints_dir
    OUTPUT_DIR = os.path.join(default_save_path, 'MiddleDataset/')

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ch = 3

    network = CC_Module()
    checkpoint = torch.load(args.deep_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()
    network.to(device)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_files = get_InferA(args.data)

    for m in total_files:
        img = cv2.imread(m)
        img = img[:, :, ::-1]
        img = np.float32(img) / 255.0
        h, w, c = img.shape
        train_x = np.zeros((1, ch, h, w)).astype(np.float32)
        train_x[0, 0, :, :] = img[:, :, 0]
        train_x[0, 1, :, :] = img[:, :, 1]
        train_x[0, 2, :, :] = img[:, :, 2]
        dataset_torchx = torch.from_numpy(train_x)
        dataset_torchx = dataset_torchx.to(device)
        output = network(dataset_torchx)
        output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        output = output[:, :, ::-1]

        output_filename = os.path.join(OUTPUT_DIR, os.path.basename(m))
        cv2.imwrite(output_filename, output)
    print("\n\n------INFERENCING DEEPWAVENET PIPELINE: COMPLETED------")

    print("\n\n------INFERENCING FUNIE-GAN PIPELINE: IN PROGRESS------")
    np.random.seed(77)
    torch.manual_seed(77)
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        torch.cuda.manual_seed(77)
        
    test_set = TestDataset(OUTPUT_DIR, (256,256))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    net = FUnIEGeneratorV2()
    model = args.funie_path
    predictor = Predictor(net, test_loader, model, default_save_path, is_cuda)
    predictor.predict()
    shutil.rmtree(OUTPUT_DIR)
    print("\n\n------INFERENCING FUNIE-GAN PIPELINE: COMPLETED------")

if __name__ == "__main__":
    print("\n\n------MODEL INFERENCING: STARTED------")
    main()
    print("\n\n------MODEL INFERENCING: COMPLETED------")