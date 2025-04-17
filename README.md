# multiband-fusion-all
A generalized multiband signal fusion learning platform based on our paper:  Deep Learning-Based Multiband Signal Fusion for 3-D SAR Super-Resolution ([arXiv](https://arxiv.org/abs/2305.02017), [DOI](https://doi.org/10.1109/TAES.2023.3270111)).

## Installation
First, install the prerequisite Python dependencies:
```bash
pip install -r requirements.txt
```

## Creating a Dataset
Create a small dataset for test purposes
```bash
python create_dataset.py --num_train 1024 --num_val 128 --num_test 128 
```

Learn about the dataset creation options with
```bash
python create_dataset.py --help
```


## Training a Model
Learn about the model training options with
```bash
python train_model.py --help
```

Train using the small test dataset from above with a smaller model size than used in the paper
```bash
python train_model.py --dataset dataset_60GHz_77GHz_1024_128_Nt64.mrd \
    --n_feats 8 --n_res_blocks 2 --epochs 10 --lr 1e-2
```

## Testing a Model
```bash
python test_model.py --checkpoint 2025-04-17_kR-Net_60GHz_77GHz_3ce4/epoch10_2025-04-17_kR-Net_60GHz_77GHz_3ce4.pt \
    --n_feats 8 --n_res_blocks 2
```