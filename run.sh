nohup python -u pretrain.py >> sh.log 2>&1 &
nohup python -u train.py >> sh.log 2>&1 &
nohup python -u test.py >> sh.log 2>&1 &