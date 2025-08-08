import os
import random
import numpy as np
import argparse
# import wandb
import torch
from tqdm import tqdm
from datetime import datetime
from model import GCMCModel
from utils import generate_data
from logutils import getLogger

parser = argparse.ArgumentParser(description='Specify some parameters!')
parser.add_argument('--gpus', default='1', help='gpus')
parser.add_argument('--test_ratio', default='0.2', help="test_ratio")
parser.add_argument('--test_degree', default='0', help="test_degree")
parser.add_argument('--dataset', default='1', help="dataset")
args = parser.parse_args()
"""
_tgt_CDs_and_Vinyl_src_Books
_tgt_Movies_and_TV_src_Books
_tgt_CDs_and_Vinyl_src_Movies_and_TV
"""
file_flag = args.dataset
file_name =""
if file_flag == "1":
    file_name = f"{args.test_ratio}_tgt_CDs_and_Vinyl_src_Movies_and_TV"
elif file_flag == "2":
    file_name = f"{args.test_ratio}_tgt_Movies_and_TV_src_Books"
elif file_flag == "3":
    file_name = f"{args.test_ratio}_tgt_CDs_and_Vinyl_src_Books"

now = datetime.now().strftime("%Y_%m_%d")
logger = getLogger('log/pretrain/'+file_name+'_pretrain.log')

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.01  # default 0.001
DECAYING_FACTOR = 0.95  # default 0.95
LAMBDA_REG = 0.05  # default 0.05,
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_TEST = 1024  # 原本是2048
n_epochs = 100
key_deta = 20
n_rating = 5
datadir = 'data/ready/' + file_name + '.pkl'


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


fix_seed(1234)


def neg_sampling(train_set_i, num_neg_per=5):
    size = train_set_i.size(0)
    neg_iid = torch.randint(0, n_item, (num_neg_per * size,)).reshape(-1)
    return torch.stack([train_set_i[:, 0].repeat(num_neg_per), neg_iid, torch.zeros(num_neg_per * size)], dim=1)


def auc_calc(score_label):
    fp1, tp1, fp2, tp2, auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for s in score_label:
        fp2 += (1 - s[1])  # noclick
        tp2 += s[1]  # click
        auc += (tp2 - tp1) * (fp2 + fp1) / 2
        fp1, tp1 = fp2, tp2
    try:
        return 1 - auc / (tp2 * fp2)
    except:
        return 0.5


def train(train_set, model, optimizer, i):
    model.train()
    optimizer.zero_grad()

    train_set_i = train_set[i * BATCH_SIZE_TRAIN: (i + 1) * BATCH_SIZE_TRAIN]
    train_set_i_x = train_set_i[:, :2].long().to(device)  # （1024，2）
    train_set_i_y = train_set_i[:, 2].long().to(device)
    train_set_i_x = train_set_i_x.to(device)
    train_set_i_y = train_set_i_y.to(device)
    pred_y = model(train_set_i_x)
    loss_r = torch.sum((train_set_i_y - pred_y) ** 2)
    loss_reg = model.regularization_loss()
    loss = loss_r + LAMBDA_REG * loss_reg
    loss.backward()
    optimizer.step()
    return loss_r.item(), loss_reg.item()


def test(model, test_set):
    model.eval()
    loss_r_test_sum, l1_sum, l2_sum = 0., 0., 0.
    test_size = test_set.size(0)
    for i in tqdm(range(test_size // BATCH_SIZE_TEST + 1), desc="test"):
        with torch.no_grad():
            test_set_i = test_set[i * BATCH_SIZE_TEST: (i + 1) * BATCH_SIZE_TEST].to(device)
            # test_set_neg_i = neg_sampling(test_set_i)
            # test_set_i = torch.cat([test_set_i, test_set_neg_i], dim=0)
            test_set_i_x = test_set_i[:, :2].long()
            test_set_i_y = test_set_i[:, 2].double()
            # pred_y = model( test_set_i_x, edge_UI_i, edge_IU_i)
            pred_y = model(test_set_i_x)
            loss_r = torch.sum((test_set_i_y - pred_y) ** 2)

        y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
        loss_r_test_sum += loss_r.item()
        l1_sum += np.sum(np.abs(y_hat - y))
        l2_sum += np.sum(np.square(y_hat - y))

    TestLoss = loss_r_test_sum / test_size
    MAE = l1_sum / test_size
    RMSE = np.sqrt(l2_sum / test_size)

    return TestLoss, MAE, RMSE


def save_model(model, path):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + file_name + '_model.pkl')


def main(stage, model, optimizer, scheduler):
    if stage == "src":
        train_set = torch.tensor(train_src)
        save_path = f'model/pretrain-{stage}/'
    elif stage == "tgt":
        train_set = torch.tensor(train_tgt)
        save_path = f'model/pretrain-{stage}/'

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    train_set = train_set[torch.randperm(train_set.size(0))]
    train_size, test_size = train_set.size(0), test_set.size(0)
    bestRMSE = 10.0
    step = 0

    for epoch in range(n_epochs):
        train_set = train_set[torch.randperm(train_size)]
        iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
        for i in tqdm(range(train_size // BATCH_SIZE_TRAIN + 1), desc=f"pretrain_epoch{epoch}"):
            loss_r, loss_reg = train(train_set, model, optimizer, i)
            step += 1
            iter_num += 1
            loss_r_sum += loss_r
            loss_reg_sum += loss_reg
        loss_r_train = loss_r_sum / train_size
        loss_reg_train = loss_reg_sum / train_size
        logger.info('Epoch {} Step {}'.format(epoch, step))
        logger.info('PreTrainLoss {:.4f} Reg: {:.4f}'.format(loss_r_train, loss_reg_train))

        loss_r_test, MAE, RMSE = test(model, test_set)
        logger.info('TestLoss: {:.10f} MAE: {:.4f} RMSE: {:.4f} '.format(loss_r_test, MAE, RMSE))

        scheduler.step()
        if RMSE < bestRMSE:
            bestRMSE = RMSE
    save_model(model, path=save_path)


n_user, n_item, test_user_num, train_ov_tgt, train_ov_src, train_src, train_tgt, test_tgt_data, test_src_data, user_his_dict_src, user_his_dict_tgt, user_supp_list, edge_UI_tgt, edge_UI_src, test_edge_UI_tgt = \
    generate_data(device=device,datadir=datadir, sample_graph=False, logger=logger, key_deta=key_deta)

test_set = torch.tensor(test_tgt_data)
model = GCMCModel(n_user=n_user,
                  n_item=n_item,
                  n_rating=n_rating,
                  device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.)  # 5e-2
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAYING_FACTOR)

logger.info("Pretraining src")
main("src", model, optimizer, scheduler)
logger.info("Pretraing tgt")
main("tgt", model, optimizer, scheduler)
