# IQ-VQA

Python implementation of work presented in https://arxiv.org/abs/2007.04422

### Table of Contents
0. [Citing IQ-VQA](#citing-iq-vqa)
0. [Installing pythia environment](#installing-pythia-environment)
0. [Quick start](#quick-start)
0. [Preprocess dataset](#preprocess-dataset)
0. [Best pretrained models](#best-pretrained-models)
0. [Customize config](#customize-config)
0. [AWS s3 dataset summary](#aws-s3-dataset-summary)
0. [Acknowledgements](#acknowledgements)
0. [References](#references)


### Citing IQ-VQA
If you use this work in your research, please use the following BibTeX entry for reference:

The software:

```
@misc{goel2020iqvqa,
    title={IQ-VQA: Intelligent Visual Question Answering},
    author={Vatsal Goel and Mohit Chandak and Ashish Anand and Prithwijit Guha},
    year={2020},
    eprint={2007.04422},
    archivePrefix={arXiv},
    primaryClass={cs.CV}}
```

### Installing pythia environment

1. Install Anaconda (Anaconda recommended: https://www.continuum.io/downloads).
2. Install cudnn v7.0 and cuda.9.0
3. Create environment for pythia
```bash
conda create --name vqa python=3.6

source activate vqa
pip install demjson pyyaml

pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl

pip install torchvision
pip install tensorboardX

```


### Quick start
We provide preprocessed data files to directly start training and evaluating. Instead of using the original `train2014` and `val2014` splits, we split `val2014` into `val2train2014` and `minival2014`, and use `train2014` + `val2train2014` for training and `minival2014` for validation.

Download data. This step may take some time. Check the sizes of files at the end of readme.
```bash

git clone git@github.com:facebookresearch/pythia.git
cd Pythia

mkdir data
cd data
wget https://dl.fbaipublicfiles.com/pythia/data/vqa2.0_glove.6B.300d.txt.npy
wget https://dl.fbaipublicfiles.com/pythia/data/vocabulary_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/answers_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/imdb.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/features/rcnn_10_100.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/features/detectron_fix_100.tar.gz
wget https://dl.fbaipublicfiles.com/pythia/data/large_vocabulary_vqa.txt
wget https://dl.fbaipublicfiles.com/pythia/data/large_vqa2.0_glove.6B.300d.txt.npy
gunzip imdb.tar.gz 
tar -xf imdb.tar

gunzip rcnn_10_100.tar.gz 
tar -xf rcnn_10_100.tar
rm -f rcnn_10_100.tar

gunzip detectron.tar.gz
tar -xf detectron.tar
rm -f detectron.tar
```

Optional command-line arguments for `train.py`
```bash
python train.py -h

usage: train.py [-h] [--config CONFIG] [--out_dir OUT_DIR] [--seed SEED]
                [--config_overwrite CONFIG_OVERWRITE] [--force_restart]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config yaml file
  --out_dir OUT_DIR     output directory, default is current directory
  --seed SEED           random seed, default 1234, set seed to -1 if need a
                        random seed between 1 and 100000
  --config_overwrite CONFIG_OVERWRITE
                        a json string to update yaml config file
  --force_restart       flag to force clean previous result and restart
                        training
```

If there is a out of memory error, try:
```bash
python train.py --config_overwrite '{data:{image_fast_reader:false}}'
```
Check result for the default run
```bash
cd results/default/1234

```
The results folder contains the following info
```angular2html
results
|_ default
|  |_ 1234 (default seed)
|  |  |_config.yaml
|  |  |_best_model.pth
|  |  |_best_model_predict_test.pkl 
|  |  |_best_model_predict_test.json (json file for predicted results on test dataset)
|  |  |_model_00001000.pth (mpdel snapshot at iter 1000)
|  |  |_result_on_val.txt
|  |  |_ ...
|  |_(other_cofig_setting)
|  |  |_...
|_ (other_config_file)
|

```
The log files for tensorbord are stored under `boards/`


### Preprocess dataset
If you want to start from the original VQA dataset and preprocess data by yourself, use the following instructions in [data_preprocess.md](data_prep/data_preprocess.md). 
***This part is not necessary if you download all data from quick start.***


#### Best Pretrained Models
The best pretrained model can be downloaded as follows:

```bash
mkdir pretrained_models/
cd pretrained_models
wget https://dl.fbaipublicfiles.com/pythia/pretrained_models/detectron_100_resnet_most_data.tar.gz
gunzip detectron_100_resnet_most_data.tar.gz 
tar -xf detectron_100_resnet_most_data.tar
rm -f detectron_100_resnet_most_data.tar
```


Get ResNet152 features and Detectron features with fixed 100 bounding boxes
```bash
cd data
wget https://dl.fbaipublicfiles.com/pythia/features/detectron_fix_100.tar.gz
gunzip detectron_fix_100.tar.gz
tar -xf detectron_fix_100.tar
rm -f detectron_fix_100.tar

wget https://dl.fbaipublicfiles.com/pythia/features/resnet152.tar.gz
gunzip resnet152.tar.gz
tar -xf resnet152.tar
rm -f resnet152.tar
```


Test the best model on the VQA test2015 dataset
```bash

python run_test.py --config pretrained_models/config.yaml \
--model_path pretrained_models/Pythia_IQ.pth \
--out_prefix test_best_model --store_result
```

The results will be saved as a json file `test_best_model.json`, and this file can be uploaded to the evaluation server on EvalAI (https://evalai.cloudcv.org/web/challenges/challenge-page/80/submission).

### Customize config
To change models or adjust hyper-parameters, see [config_help.md](config_help.md)

### AWS s3 dataset summary
Here, we listed the size of some large files in our AWS S3 bucket.

| Description | size  |
| --- | --- | 
|data/rcnn_10_100.tar.gz | 71.0GB |
|data/detectron.tar.gz | 106.2 GB|
|data/detectron_fix_100.tar.gz|162.6GB|
|data/resnet152.tar.gz | 399.6GB|

### Acknowledgements
We would like to thank Manish Borthakur, Aditi Jain and Udita Mittal from Indian Institute of Technology, Delhi for annotating the VQA-Implications dataset.

### References
- V. Goel and M. Chandak and A. Anand and P. Guha. IQ-VQA: Intelligent Visual Question Answering. arXiv preprint arXiv:2007.04422, 2020.
