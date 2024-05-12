import os
import subprocess
import argparse
from deep_options import opt
from deep_models import CC_Module
from deep_misc import getLatestCheckpointName
import cv2
import numpy as np
import torch
import glob
import shutil

def get_TrainA(root):
    filesA, filesB = [], []
    sub_dirs = ['underwater_imagenet', 'underwater_dark', 'underwater_scenes']
    for sd in sub_dirs:
        filesA += sorted(glob.glob(os.path.join(root, sd, 'trainA') + "/*.*"))
        filesB += sorted(glob.glob(os.path.join(root, sd, 'trainB') + "/*.*"))
    return filesA, filesB

def run_deepwavenet(args):
    try:
        subprocess.run(["python", "deep_train.py",
                        "--data_path", str(args.data),
                        "--end_epoch", str(args.epochs),
                        "--batch_size", str(args.batch_size)],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running DeepWaveNet: {e}")

def run_funie_gan(args):
    try:
        subprocess.run(["python", "funie_train.py",
                        "-d", str(args.data),
                        "--epochs", str(args.epochs),
                        "-b", str(args.batch_size),
                        "-j", str(args.num_workers)],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running FUnIE-GAN: {e}")

def main():
    parser = argparse.ArgumentParser(description="Train multiple models.")
    parser.add_argument("--epochs", default=1, type=int, help="number of total epochs to run")
    parser.add_argument("--data", default="./EUVP Dataset/Paired", type=str, help="path to data")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    args = parser.parse_args()
    
    default_save_path = './saved_model'
    os.makedirs(default_save_path, exist_ok=True)
    
    run_deepwavenet(args)
    
    print("\n------TRAINING DEEPWAVENET PIPELINE: COMPLETED------")
    print(f"\n------SAVING DEEPWAVENET RESULTS AT {os.path.join(default_save_path, 'MiddleDataset')} ------")
    CHECKPOINTS_DIR = opt.checkpoints_dir
    OUTPUT_DIRA = os.path.join(default_save_path, 'MiddleDataset/trainA')
    OUTPUT_DIRB = os.path.join(default_save_path, 'MiddleDataset/trainB')

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


    total_filesA, total_filesB = get_TrainA(args.data)

    for m,n in zip(total_filesA, total_filesB):
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
        shutil.copy(n, os.path.join(OUTPUT_DIRB, os.path.basename(n)))


    
    print("\n\n------TRAINING FUNIE-GAN PIPELINE: IN PROGRESS------")
    args.data = os.path.join(default_save_path, 'MiddleDataset/') # middle data containing trainA and trainB
    run_funie_gan(args)

if __name__ == "__main__":
    print("\n\n------MODEL TRAINING: STARTED------")
    main()
    print("\n\n------MODEL TRAINING: COMPLETED------")