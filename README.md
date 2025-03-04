# DCT-Diffusion: Depth Completion for Transparent Objects with Diffusion Denoising Approach

PyTorch implementation of paper 'DCT-Diffusion: Depth Completion for Transparent Objects with Diffusion Denoising Approach'"

## Discription
This repository provides the code for our model, which partially references [TODE](https://github.com/yuchendoudou/TODE) and [DiffusionDepth](https://github.com/duanyiqun/DiffusionDepth).Regarding our model, we provide a visual demonstration of the results.

## Dataset Preparation

### ClearGrasp Dataset

You can download the ClearGrasp dataset (including both training and testing datasets) from the [official website](https://sites.google.com/view/cleargrasp/data). After downloading and extracting the zip files, the folder structure should look like this:

```
${DATASET_ROOT_DIR}
├── cleargrasp
│   ├── cleargrasp-dataset-train
│   ├── cleargrasp-dataset-test-val
```

### Omniverse Object Dataset

The Omniverse Object Dataset can be downloaded from [this link](https://drive.google.com/drive/folders/1wCB1vZ1F3up5FY5qPjhcfSfgXpAtn31H?usp=sharing). After downloading and extracting the zip files, the folder structure will be:

```
${DATASET_ROOT_DIR}
├── omniverse
│   ├── train
│   │    ├── 20200904
│   │    ├── 20200910
```

## TransCG Dataset

The TransCG dataset can be downloaded from [here](https://graspnet.net/transcg). After downloading and extracting the zip files, the folder structure will be as follows:

```
${DATASET_ROOT_DIR}
├── transcg
│   ├── data
│   │    ├── scene1
│   │    |      ├── 0
|   |    |      |   ├── corrected_pose
|   |    |      |   ├── depth1.png 
|   |    |      |   ├── depth1-gt.png
|   |    |      |   ├── depth1-gt-mask.png
|   |    |      |   ├── depth1-gt-sn.png
```

## Requirements

The code has been tested under

- Ubuntu 20.04 + NVIDIA GeForce RTX 3090
- PyTorch 1.11.0

    System dependencies can be installed by:

```shell
sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11sudo apt install libopenexr-dev zlib1g-dev openexr
```

Other dependencies can be installed by

```shell
pip install -r requirements.txt
```

## Testing

We provide transcg pretrained checkpoints at checkpoints/. You can use these pretrained models to directly test the model. Ensure that the correct checkpoint file is referenced in the configuration file, and execute the test with the following command:

```shell
python test.py --cfg [Configuration File]
```

## Training

To train the model from scratch, you need to specify the appropriate configuration file that matches your training and validation setup. Below are the training commands for different scenarios:

```
#Train on transcg dataset and test on transcg
python train.py -c ./configs/train_transcg_val_transcg.yaml

#Tran on CGsyn+ood and test on CGreal
python train.py -c ./configs/train_cgsyn+ood_val_cgreal.yaml
#Tran on CGsyn+ood and test on Transcg
python train.py -c ./configs/train_cgsyn+ood_val_transcg.yaml
```

Make sure that the dataset paths and hyperparameters are correctly set in the configuration files before starting the training. If you want to resume training from a checkpoint, specify the checkpoint path in the configuration file under the `checkpoint_path` field.

## Model Results Visualization

The following images showcase the results of our model applied to scenes from the ClearGrasp and TransCG datasets. These include the RGB images, raw depth maps, and the refined depth maps predicted by our method. You can see the improvements in depth prediction as the raw depth maps are refined by our approach.

![](images/2025-03-02-01-14-27-img_v3_02k0_229d8e81-32e4-4f51-9769-7f10be3823ag.jpg)

This figure presents a comparison of reconstructed 3D point clouds from depth and RGB images in four different scenes. The first row displays point clouds generated from the original depth maps, while the second row shows the point clouds reconstructed using our refined depth maps. Our model successfully enhances depth completion, resulting in significantly improved point cloud quality.

![](images/2025-03-02-01-14-37-img_v3_02k0_ab7bc45d-1aff-4c5a-974a-acf05e55754g.jpg)
