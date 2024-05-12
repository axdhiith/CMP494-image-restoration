## Deep WaveNet and FuNIE-GAN 

Our project consists of a modified architecture where Deep WaveNet along with FuNIE-GAN is used to further enhance the image restoration process. 
The DeepWaveNet architecture used in our project is from the paper **Wavelength-based Attributed Deep Neural Network for Underwater Image Restoration**
[**arXiv version**](https://arxiv.org/abs/2106.07910)   

The FuNIE-GAN architecture used the PyTorch implementation from the paper **Fast Underwater Image Enhancement for Improved Visual Perception** [**arXiv version**](https://arxiv.org/pdf/1903.09766.pdf).

- This project deals with the **Underwater Image Restoration**, where it performs image enhancement on hazy underwater images as these images would suffer from low contrast and high color distortions as it propagates through the water.
  - This model performs a low-level vision task such as **image enhancement** along with the enhancement performed by FuNIE-GAN.
  - For further enhancement of the metrics evaluated in DeepWaveNet, the enhanced images from this model is then fed into the FuNIE-GAN to further improve the quality of these images. The objective of this is to improve the model's performance and the metrics and benchmark it with the SOTA models as well.

- For underwater image enhancement (uie), we have utilized publicly available datasets [**EUVP**](http://irvlab.cs.umn.edu/resources/euvp-dataset).
- Below, we provide the detailed instructions for each task in a single README file to reproduce the original results.


### Results
![Block](imgs/teaser.png)

### Prerequisites

The codes work with *minimum* requirements as given below.
```bash
# tested with the following dependencies on Ubuntu 16.04 LTS system:
Python 3.5.2
Pytorch '1.0.1.post2'
torchvision 0.2.2
opencv 4.0.0
scipy 1.2.1
numpy 1.16.2
tqdm
```
To install using linux [env](https://github.com/pksvision/Deep-WaveNet-Underwater-Image-Restoration/blob/main/installation/requirements.txt)

```bash
pip install -r requirements.txt

```

### Datasets Preparation
##### To train and test the combined architecture on the [EUVP](http://irvlab.cs.umn.edu/resources/euvp-dataset) dataset, it is important to know the contents of the dataset. In our project, we will be using the **Paired Dataset**, which consists of three sub directories, known as 'underwater_dark', 'underwater_scenes' and 'underwater_imagenet'. 

|                     | Train | Validation | Total Pairs |
|---------------------|-------|------------|-------------|
| Underwater Imagenet | 3330  | 370        |3700         |
| Underwater Dark     | 4995  | 555        |5550         |
| Underwater Scenes   | 1967  | 218        |2185         |

In the folder `test_samples`, total 515 data are in pairs. Images in `GTr` are in good quality and images in `Inp` are in poor quality.
Define the training-set folders absolute path in the respective `options.py` file wherever required.

## TRAINING 

The train.py file consists of a pipeline where it begins with the training of the DeepWaveNet architecture using trainA (hazy images) along with trainB (ground truth). This requires the following parser arguments:

**--epochs**: Number of total epochs to run
**--data_path**: Path to where the Paired dataset is stored
**--batch_size**: Batch Size 

The output images from this is stored into a directory called 'MiddleDataset/trainA', and consequently this is fed into the FuNIE-GAN train.py for further image enhancement. This requires the following parser arguments:

**-d**: The data path
**-a**: The version of the model, which is 'v2' in this case
**--epochs**: Number of total epochs
**-b**: Batch Size
**-j**: Number of workers

The two train.py files from each model are combined into one train.py.

This is done by the CLI command:
```bash
python train.py --data-path --end-epoch --batch_size -d -a --epochs -b -j
// for example
!python train.py --epochs 1 --data "./EUVP Dataset/Paired" --num_workers 8 --batch_size 8
```

## TESTING

Once the training is complete, the testing can be done in a similar way. The two test.py files are once again pipelined wherein firstly the test.py for DeepWaveNet is called. Here, since the parameters are already set, there are no additional parameters sent in CLI command to test. If needed, it can be changed in the test.py files. After DeepWaveNet is tested on the testsamples called as 'Inp', it is sent to be tested on FuNIE-GAN. The best generator path is saved in the 'saved_models' which can be used to do testing.

This is done by the CLI command:

```bash
python test.py
```

## INFERENCE

Similar to the testing, the infer.py is called to perform inferencing on the combined model using the TestDataset. It would load the best checkpoints for the DeepWaveNet model and the best generator path for the FuNIE-GAN model for the CLI command. The following parameters are required in this case:

**--data**: The data path for the images to be inferenced
**--num_workers**: Total number of workers
**--batch_size**: Batch Size
**--deep_path**: The path for the checkpoints for the DeepWaveNet model
**--funie_path**: The best generator path for the FuNIE-GAN model


This is done by the CLI command:

```bash
python infer.py --data --num_workers --batch_size --deep_path --funie_path
// for example
python infer.py --data "./inference_dataset" --num_workers 8 --batch_size 8 --deep_path "./saved_model/ckpts/netG_100.pt" --funie_path "./saved_model/funiegan/best_gen.pth.tar"
```


### Evaluation Metrics
- Image quality metrics (IQMs) used in this work are the following:
  
1. SSIM
2. PSNR
3. UIQM
4. UISM
5. UICM
6. UICONM
  
- Below are the results for the EUVP dataset from the DeepWaveNet paper:

| **Method**   | `MSE` | `PSNR` | `SSIM` |
| ------------ | :---: | :----: | :----: |
| Deep WaveNet |  .29  | 28.62  |  .83   |


- Below are our results for the EUVP Datasetthe with the combined models:

|        **Method**        | `MSE` | `PSNR` | `SSIM` |
| ------------------------ | :---: | :----: | :----: |
| Deep WaveNet + FuNIE-GAN |  .29  |  28.22 |  .84   | 

### License and Citation
- The usage of this software is only for academic purposes. One can not use it for commercial products in any form. 
- If you use this work or codes (for academic purposes only), please cite the following:
```
@misc{sharma2021wavelengthbased,
      title={Wavelength-based Attributed Deep Neural Network for Underwater Image Restoration}, 
      author={Prasen Kumar Sharma and Ira Bisht and Arijit Sur},
      year={2021},
      eprint={2106.07910},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

 @article{islam2019fast,
     title={Fast Underwater Image Enhancement for Improved Visual Perception},
     author={Islam, Md Jahidul and Xia, Youya and Sattar, Junaed},
     journal={IEEE Robotics and Automation Letters (RA-L)},
     volume={5},
     number={2},
     pages={3227--3234},
     year={2020},
     publisher={IEEE}
}

@ARTICLE{8917818,  
    author={Li, Chongyi 
            and Guo, Chunle 
            and Ren, Wenqi 
            and Cong, Runmin 
            and Hou, Junhui 
            and Kwong, Sam 
            and Tao, Dacheng},  
    journal={IEEE Transactions on Image Processing},   
    title={An Underwater Image Enhancement Benchmark Dataset and Beyond},   
    year={2020},  
    volume={29},  
    number={},  
    pages={4376-4389},  
    doi={10.1109/TIP.2019.2955241}
}

@inproceedings{eriba2019kornia,
  author    = {E. Riba, D. Mishkin, D. Ponsa, E. Rublee and G. Bradski},
  title     = {Kornia: an Open Source Differentiable Computer Vision Library for PyTorch},
  booktitle = {Winter Conference on Applications of Computer Vision},
  year      = {2020},
  url       = {https://arxiv.org/pdf/1910.02190.pdf}
}

@inproceedings{islam2020suim,
  title={{Semantic Segmentation of Underwater Imagery: Dataset and Benchmark}},
  author={Islam, Md Jahidul and Edge, Chelsey and Xiao, Yuyang and Luo, Peigen and Mehtaz, 
              Muntaqim and Morse, Christopher and Enan, Sadman Sakib and Sattar, Junaed},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020},
  organization={IEEE/RSJ}
}

@article{8765346,
  author = {Z. {Cao} and G. {Hidalgo Martinez} and T. {Simon} and S. {Wei} and Y. A. {Sheikh}},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title = {OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
  year = {2019}
}
```

