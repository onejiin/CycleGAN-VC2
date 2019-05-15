# CycleGAN-VC2
This code is based on "Lei Mao" CycleGAN-VC (Clone to : https://github.com/leimao/Voice_Converter_CycleGAN.git)


## Introduction

CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion, 
Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, and Nobukatsu Hojo, arxiv 2019

Data save as HDF5 format
(world_decompose extracts f0, aperiodicity and spectral envelope. This function is computationally intensive.)


## Dependencies

* Python 3.5
* Numpy 1.14
* TensorFlow 1.8
* ProgressBar2 3.37.1
* LibROSA 0.6
* [PyWorld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder)


## Usage
### Download Dataset

Download and unzip [VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211) dataset to designated directories.

```bash
$ python download.py --help
usage: download.py [-h] [--download_dir DOWNLOAD_DIR] [--data_dir DATA_DIR]
                   [--datasets DATASETS]

Download CycleGAN voice conversion datasets.

optional arguments:
  -h, --help            show this help message and exit
  --download_dir DOWNLOAD_DIR
                        Download directory for zipped data
  --data_dir DATA_DIR   Data directory for unzipped data
  --datasets DATASETS   Datasets available: vcc2016
```

For example, to download the datasets to ``download`` directory and extract to ``data`` directory:

```bash
$ python download.py --download_dir ./download --data_dir ./data --datasets vcc2016
```


### Train Model

There are various models which have original VC2 or VC1

To have a good conversion capability, the training would take at least 1000 epochs, which could take very long time even using a NVIDIA GTX TITAN X graphic card. 

```bash
$ python train.py --help
usage: train.py [-h] [--train_A_dir TRAIN_A_DIR] [--train_B_dir TRAIN_B_DIR]
                [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                [--random_seed RANDOM_SEED]
                [--validation_A_dir VALIDATION_A_DIR]
                [--validation_B_dir VALIDATION_B_DIR]
                [--output_dir OUTPUT_DIR]
                [--tensorboard_log_dir TENSORBOARD_LOG_DIR]
                [--gen_model SELECT_GENERATOR]
                [--MCEPs_dim MEL-FEATURE_DIM]
                [--hdf5A_path SAVE_HDF5] [--hdf5B_path SAVE_HDF5]
                [--lambda_cycle CYCLE_WEIGHT]
                [--lambda_identity IDENTITY_WEIGHT]


Train CycleGAN model for datasets.

optional arguments:
  -h, --help            show this help message and exit
  --train_A_dir TRAIN_A_DIR
                        Directory for A.
  --train_B_dir TRAIN_B_DIR
                        Directory for B.
  --model_dir MODEL_DIR
                        Directory for saving models.
  --model_name MODEL_NAME
                        File name for saving model.
  --random_seed RANDOM_SEED
                        Random seed for model training.
  --validation_A_dir VALIDATION_A_DIR
                        Convert validation A after each training epoch. If set
                        none, no conversion would be done during the training.
  --validation_B_dir VALIDATION_B_DIR
                        Convert validation B after each training epoch. If set
                        none, no conversion would be done during the training.
  --output_dir OUTPUT_DIR
                        Output directory for converted validation voices.
  --tensorboard_log_dir TENSORBOARD_LOG_DIR
                        TensorBoard log directory.
  --gen_model
                        select CycleGAN-VC1 or CycleGAN-VC2 or CycleGAN2_withDeconv
  --MCEPs_dim 
                        Mel-cepstral coefficient dimension
  --hdf5A_path
  --hdf5B_path 
                        save hdf5 db root
  --lambda_cycle
  --lambda_identity
                        generator loss = cycle*lambda + identity*lambda + generator
```
For example,
```bash
$ python train.py --gen_model CycleGAN-VC2
```

### Conversion

```bash
$ python convert.py --help
usage: convert.py [-h] [--model_dir MODEL_DIR] [--model_name MODEL_NAME]
                  [--data_dir DATA_DIR]
                  [--conversion_direction CONVERSION_DIRECTION]
                  [--output_dir OUTPUT_DIR]
                  [--pc PITCH_SHIFT]
                  [--generation_model MODEL_SELECT]

Convert voices using pre-trained CycleGAN model.

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Directory for the pre-trained model.
  --model_name MODEL_NAME
                        Filename for the pre-trained model.
  --data_dir DATA_DIR   Directory for the voices for conversion.
  --conversion_direction CONVERSION_DIRECTION
                        Conversion direction for CycleGAN. A2B or B2A. The
                        first object in the model file name is A, and the
                        second object in the model file name is B.
  --output_dir OUTPUT_DIR
                        Directory for the converted voices.
  --pc PITCH_SHIFT
                        pitch shift or not
  --generation_model MODEL_SELECT
                        select generator model, CycleGAN-VC2
```

To convert voice, put wav-formed speeches into ``data_dir`` and run the following commands in the terminal, the converted speeches would be saved in the ``output_dir``:

```bash
$ python convert.py --model_dir ./model/sf1_tm1 --model_name sf1_tm1.ckpt --data_dir ./data/evaluation_all/SF1 --conversion_direction A2B --output_dir ./converted_voices
```
The convention for ``conversion_direction`` is that the first object in the model filename is A, and the second object in the model filename is B. In this case, ``SF1 = A`` and ``TM1 = B``.

## Reference

* Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, and Nobukatsu Hojo, CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion, 2019. (Voice Conversion CycleGAN-VC2)
* Wenzhe Shi, Jose Caballero, Ferenc Huszár, Johannes Totz, Andrew P. Aitken, Rob Bishop, Daniel Rueckert, Zehan Wang. Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network. 2016. (Pixel Shuffler)
* Yann Dauphin, Angela Fan, Michael Auli, David Grangier. Language Modeling with Gated Convolutional Networks. 2017. (Gated CNN)
* Takuhiro Kaneko, Hirokazu Kameoka, Kaoru Hiramatsu, Kunio Kashino. Sequence-to-Sequence Voice Conversion with Similarity Metric Learned Using Generative Adversarial Networks. 2017. (1D Gated CNN)
* Kun Liu, Jianping Zhang, Yonghong Yan. High Quality Voice Conversion through Phoneme-based Linear Mapping Functions with STRAIGHT for Mandarin. 2007. (Foundamental Frequnecy Transformation)
* [PyWorld and SPTK Comparison](http://nbviewer.jupyter.org/gist/r9y9/ca05349097b2a3926ec77a02e62c6632)
* [Gated CNN TensorFlow](https://github.com/anantzoid/Language-Modeling-GatedCNN)

## Contribution

I modification deconvolution network.
Paper uses pixel shuffle method however general upsample method uses conv2d_transpose layer.
If you want to use deconv layer, --gen_model CycleGAN2_withDeconv
