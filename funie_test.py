import os
import argparse
import numpy as np
from PIL import Image
from glob import glob
from ntpath import basename
from os.path import join, exists
# pytorch libs
from funie_models import FUnIEGeneratorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
import shutil
from measure_ssim_psnr import SSIMs_PSNRs
import glob

## options
# parser = argparse.ArgumentParser()
# parser.add_argument("--data_dir", type=str, default="./saved_model/EU")
# parser.add_argument("--sample_dir", type=str, default="/home/undrdawatr/Vision/Models_to_select/FUnIE-GAN-pytorch/final_output1/")
# parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
# parser.add_argument("--model_path", type=str, default="/home/undrdawatr/Vision/Models_to_select/FUnIE-GAN-pytorch/saved_model/best_gen.pth.tar")
# opt = parser.parse_args()

def get_TestA(INP_DIR, CLEAN_DIR):
    filesA, filesB = [], []
    filesA += sorted(glob.glob(INP_DIR + "/*.*"))
    filesB += sorted(glob.glob(CLEAN_DIR + "/*.*"))
    return filesA, filesB

CHECKPOINTS_DIR = './saved_model/funiegan/best_gen.pth.tar'

INP_DIR = './saved_model/DeepWaveNetTest/trainA'
CLEAN_DIR = './saved_model/DeepWaveNetTest/trainB'

OUTPUT_DIRA = './saved_model/FunieganTest/trainA'
OUTPUT_DIRB = './saved_model/FunieganTest/trainB'

## checks
assert exists(CHECKPOINTS_DIR), "model not found"
if not os.path.exists(OUTPUT_DIRA):
    os.makedirs(OUTPUT_DIRA)
if not os.path.exists(OUTPUT_DIRB):
    os.makedirs(OUTPUT_DIRB)
is_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor 

model = FUnIEGeneratorV2()

## load weights
device = "cuda:0" if is_cuda else "cpu"
model.load_state_dict(torch.load(CHECKPOINTS_DIR, map_location=device)['state_dict'])
if is_cuda: model.cuda()
model.eval()
print ("Loaded model from %s" % (CHECKPOINTS_DIR))
## data pipeline
img_width, img_height, channels = 256, 256, 3
transforms_ = [transforms.Resize((img_height, img_width), Image.BICUBIC),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
transform = transforms.Compose(transforms_)

if __name__ == '__main__':
    ## testing loop
    # test_files = sorted(glob(join(INP_DIR, "*.*")))
    total_filesA, total_filesB = get_TestA(INP_DIR, CLEAN_DIR)
    for path,s in zip(total_filesA, total_filesB):
        inp_img = transform(Image.open(path))
        inp_img = Variable(inp_img).type(Tensor).unsqueeze(0)
        # generate enhanced image
        gen_img = model(inp_img)
        # save output
        img_sample = gen_img.data
        save_image(img_sample, join(OUTPUT_DIRA, basename(path)), normalize=True)
        shutil.copy(s, os.path.join(OUTPUT_DIRB, os.path.basename(path)))

    
    SSIM_measures, PSNR_measures, UIQM_measures, UICM_measures, UISM_measures, UICONM_measures = SSIMs_PSNRs(OUTPUT_DIRB, OUTPUT_DIRA)
        
    print("\n\n------TESTING FUNIE-GAN PIPELINE: COMPLETED------")
    
    print("\n\n------ MODEL RESULTS ------")
    print(f"SSIM on {len(SSIM_measures)} samples\n")
    print(f"Mean: {np.mean(SSIM_measures):.4f} std: {np.std(SSIM_measures):.4f}\n")
    print(f"PSNR on {len(PSNR_measures)} samples\n")
    print(f"Mean: {np.mean(PSNR_measures):.4f} std: {np.std(PSNR_measures):.4f}\n")
    print(f"UIQM on {len(UIQM_measures)} samples\n")
    print(f"Mean: {np.mean(UIQM_measures):.4f} std: {np.std(UIQM_measures):.4f}\n")
    print(f"UICM on {len(UICM_measures)} samples\n")
    print(f"Mean: {np.mean(UICM_measures):.4f} std: {np.std(UICM_measures):.4f}\n")
    print(f"UISM on {len(UISM_measures)} samples\n")
    print(f"Mean: {np.mean(UISM_measures):.4f} std: {np.std(UISM_measures):.4f}\n")
    print(f"UICONM on {len(UICONM_measures)} samples\n")
    print(f"Mean: {np.mean(UICONM_measures):.4f} std: {np.std(UICONM_measures):.4f}\n")