from deep_models import CC_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from deep_misc import getLatestCheckpointName
import cv2
import os
import numpy as np
import time
from deep_options import opt
import math
from measure_ssim_psnr import SSIMs_PSNRs
import shutil
import glob

def get_TestA(INP_DIR, CLEAN_DIR):
    filesA, filesB = [], []
    filesA += sorted(glob.glob(INP_DIR + "/*.*"))
    filesB += sorted(glob.glob(CLEAN_DIR + "/*.*"))
    return filesA, filesB
   
CHECKPOINTS_DIR = './saved_model/ckpts/'
INP_DIR = opt.testing_dir_inp #trainA
CLEAN_DIR = opt.testing_dir_gt #trainB

OUTPUT_DIRA = './saved_model/DeepWaveNetTest/trainA'
OUTPUT_DIRB = './saved_model/DeepWaveNetTest/trainB'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ch = 3

network = CC_Module()
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR, getLatestCheckpointName()))
network.load_state_dict(checkpoint['model_state_dict'])
network.eval()
network.to(device)

if not os.path.exists(OUTPUT_DIRA):
    os.makedirs(OUTPUT_DIRA)
if not os.path.exists(OUTPUT_DIRB):
    os.makedirs(OUTPUT_DIRB)

if __name__ == '__main__':
    total_filesA, total_filesB = get_TestA(INP_DIR, CLEAN_DIR)

    for m,s in zip(total_filesA, total_filesB):
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

        output_filename = os.path.join(OUTPUT_DIRA, os.path.basename(m))
        cv2.imwrite(output_filename, output)
        shutil.copy(s, os.path.join(OUTPUT_DIRB, os.path.basename(s)))

    
    SSIM_measures, PSNR_measures, UIQM_measures, UICM_measures, UISM_measures, UICONM_measures = SSIMs_PSNRs(OUTPUT_DIRB, OUTPUT_DIRA)
    
    print("\n\n------TESTING DEEPWAVENET PIPELINE - RESULTS------")
    
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
    
    print("\n\n------TESTING DEEPWAVENET PIPELINE: COMPLETED------")  