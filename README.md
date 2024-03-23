
# KVQ_for 2024 VQA

## Getting Start

### Prepare environment
```bash
conda create -n KVQ python=3.8.8
conda activate KVQ
```

Our codes are compatible with pytorch 2.1.1, you may try newer version.

### Prepare training dataset
Download KVQ dataset from codalab competition [this link] (https://codalab.lisn.upsaclay.fr/)
Please add the path of KVQ and annotation to the items of "data_prefix" and "anno_file" in the config file (i.e. /config/kwai_simpleVQA.yml)

### prepare Slowfast feature 
```bash
python SlowFast_features.py --gpu_ids 0 --video_root yout_path  --video_csv yout_path
```
Please add the path of Slowfast feature to the items of "data_prefix_3D"  in the config file (i.e. /config/kwai_simpleVQA.yml)

### prepare pretrained-weights
```bash
mkdir pretrained_weights
```
Download the .pth at [this link](https://github.com/SwinTransformer/storage/releases/tag/v1.0.4), and put this .pth in the pretrained_weights

### Train 
```bash
nohup python -u train.py  --o config/kwai_simpleVQA.yml --gpu_id 0 > log/kwai_simpleVQA.log 2>&1 &
```
or 
```bash
bash scripts/train.sh
```
### Test
```bash
nohup python -u test.py  --o config/kwai_simpleVQA_test.yml --gpu_id 0 > log/kwai_simpleVQA_test.log 2>&1 &
```
or 
```bash
bash scripts/test.sh
```







