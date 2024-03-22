# kvq_2024_VQAchallenge
This is a rep for the NTIRE Workshop and Challenges @ CVPR 2024
✨ Getting Start
Prepare environment
conda create -n KVQ python=3.8
conda activate KVQ
pip install -r requirements.txt
Our codes are compatible with pytorch1.9.0, you may try newer version.

Prepare training dataset
Download KVQ dataset from codalab competition [this link] (https://codalab.lisn.upsaclay.fr/) Please add the path of KVQ and annotation to the items of "data_prefix" and "anno_file" in the config file (i.e. /config/kwai_simpleVQA.yml)

prepare Slowfast feature
python SlowFast_features.py --gpu_ids 0,1 --video_root yout_path  --video_csv yout_path
Please add the path of Slowfast feature to the items of "data_prefix_3D" in the config file (i.e. /config/kwai_simpleVQA.yml)

Train
nohup python -u train.py  --o config/kwai_simpleVQA.yml --gpu_id 0,1 > log/kwai_simpleVQA.log 2>&1 &
or

bash scripts/train.sh
Test
nohup python -u test.py  --o config/kwai_simpleVQA_test.yml --gpu_id 0 > log/kwai_simpleVQA_test.log 2>&1 &
or

bash scripts/test.sh
