import subprocess
# import argparse

# from __future__ import division
# import numpy as np
# import math
# from scipy.ndimage import gaussian_filter
# from PIL import Image
# from glob import glob
# from os.path import join
# from ntpath import basename

# import models
# from measure_ssim_psnr import *


def run_deep_test():
    try:
        subprocess.run(["python", "deep_test.py"],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running DeepWave Test: {e}")

def run_funiegan_test():
    try:
        subprocess.run(["python", "funie_test.py"],
                        #  "--data_dir", str(args.save_model),
                        # "--sample_dir", str(args.sample_dir),
                        # "--model_name", "funiegan",
                        # "--model_path", str(args.model_path)],
                       check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running FUnie-GAN Test: {e}")

def main():
    # parser = argparse.ArgumentParser(description="Test multiple models.")
    
    # parser.add_argument("--save_model", default="saved_model", type=str, metavar="PATH",
                        # help="path to save model checkpoints")
    # parser.add_argument("--data", default="./EUVP Dataset/Paired", type=str, metavar="PATH",
                        # help="path to data")
    # parser.add_argument("--num_workers", default=4, type=int, metavar="N",
    #                     help="number of workers")
    
    # parser.add_argument('--testing_dir_inp', default="./EUVP Dataset/test_samples/Inp/")
    # parser.add_argument('--testing_dir_gt', default="./EUVP Dataset/test_samples/GTr/")
    
    # parser.add_argument("--data_dir", type=str, default="/home/undrdawatr/Vision/Project/MiddleDataset/trainA/")
    # parser.add_argument("--sample_dir", type=str, default="/home/undrdawatr/Vision/Models_to_select/FUnIE-GAN-pytorch/final_output1/")
    # parser.add_argument("--model_name", type=str, default="funiegan") # or "ugan"
    # parser.add_argument("--model_path", type=str, default="/home/undrdawatr/Vision/Models_to_select/FUnIE-GAN-pytorch/saved_model/best_gen.pth.tar")
    # args = parser.parse_args()

    # ----------- NOT SURE WHAT TO DO HERE ------------------------
    # os.makedirs(args.save_model, exists_ok=True)
    # default_save_path = args.save_model
    # args.save_model = os.path.join(default_save_path, 'DeepWaveNet')
    
    run_deep_test()
    
    # after running the DeepWave testing
    # args.save_model = os.path.join(default_save_path, 'FUnIE
    run_funiegan_test()

if __name__ == "__main__":
    main()