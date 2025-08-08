nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 1 --emb_size 64 --hid_size 64  >./train.log/ratio_0.2-task_1-head_1-hidden_64.log 2>&1 &
nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 2 --emb_size 64 --hid_size 64  >./train.log/ratio_0.2-task_1-head_2-hidden_64.log 2>&1 &
nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 3 --emb_size 64 --hid_size 64  >./train.log/ratio_0.2-task_1-head_3-hidden_64.log 2>&1 &
nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 4 --emb_size 64 --hid_size 64  >./train.log/ratio_0.2-task_1-head_4-hidden_64.log 2>&1 &
nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 5 --emb_size 64 --hid_size 64  >./train.log/ratio_0.2-task_1-head_5-hidden_64.log 2>&1 &

nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 4 --emb_size 64 --hid_size 16  >./train.log/ratio_0.2-task_1-head_4-hidden_16.log 2>&1 &
nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 4 --emb_size 64 --hid_size 32  >./train.log/ratio_0.2-task_1-head_4-hidden_32.log 2>&1 &
nohup python -u train.py --test_ratio 0.2 --dataset 1 --gpus 1 --head_num 4 --emb_size 64 --hid_size 128  >./train.log/ratio_0.2-task_1-head_4-hidden_128.log 2>&1 &