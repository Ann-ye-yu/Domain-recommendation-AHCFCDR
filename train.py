import os
import random
import numpy as np
import argparse
from model import IRMC_GC_Model
from utils import generate_data
from datetime import datetime
import torch
import torch.nn.functional as F
from logutils import getLogger
from tqdm import tqdm


# import wandb
# wandb.init(project='TBD_Pro',
#            entity='jojozhao',
#            name='train_0.8_t-movie_s-book',
#            config={
#                "learning_rate":  0.0001,
#                "batchsize_train": 1024,
#                "batchsize_test": 1024,  # 原本是2048
#                "n_epoches": 100,
#                "head_num": 4,
#                "sample_num": 500,
#            })

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def sequence_adjust(seq):
    seq_new = seq
    if len(seq) <= 0:
        seq_new = [np.random.randint(0, n_item) for i in range(HIS_SAMPLE_NUM)]
    if len(seq) > HIS_MAXLEN:
        random.shuffle(seq)
        seq_new = seq[:HIS_MAXLEN]
    return seq_new


def neg_sampling(train_set_i, num_neg_per=1):
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


def train(model, optimizer, i, train_set, user_his_dict_src, user_his_dict_tgt, edge_UI, edge_IU, alpha):
    model.train()
    optimizer.zero_grad()
    train_set_que_i = train_set[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]  # (1024,3)
    # train_set_que_neg_i = neg_sampling(train_set_que_i)  # (1024,3)
    # train_set_que_i = torch.cat([train_set_que_i, train_set_que_neg_i], dim=0)  # (2048,3)
    train_set_i_x = train_set_que_i[:, :2].long()  # (2048,2)
    train_set_i_y = train_set_que_i[:, 2].float()  # (2048)
    train_set_i_u = train_set_que_i[:, :1].long()
    # edge_UI_i = [edge_UI[n][train_set_i_x[:, 0]].to(device) for n in range(n_rating)]
    # edge_IU_i = [edge_IU[n][train_set_i_x[:, 1]].to(device) for n in range(n_rating)]
    train_set_i_x = train_set_i_x.to(device)
    train_set_i_y = train_set_i_y.to(device)

    # 用户在原来领域的历史交互信息
    src_set_his = [torch.tensor(
        sequence_adjust(user_his_dict_src[train_set_i_u[k][0].item()]),
        dtype=torch.long
    ) for k in range(train_set_i_u.size(0))]  # 2048个list , 存储用户的交互
    src_hl = [src_set_his[k].size(0) for k in range(train_set_i_u.size(0))]  # (1024)
    src_hl = torch.tensor(src_hl, dtype=torch.long).to(device)
    src_his = torch.nn.utils.rnn.pad_sequence(src_set_his, batch_first=True, padding_value=0.).to(
        device)  # (1024,100)

    # 用户在目标领域的历史交互信息
    tgt_set_his = [torch.tensor(
        sequence_adjust(user_his_dict_tgt[train_set_i_u[k][0].item()]),
        dtype=torch.long
    ) for k in range(train_set_i_u.size(0))]  # 2048个list , 存储用户的交互
    tgt_hl = [tgt_set_his[k].size(0) for k in range(train_set_i_u.size(0))]  # (1024)
    tgt_hl = torch.tensor(tgt_hl, dtype=torch.long).to(device)
    tgt_his = torch.nn.utils.rnn.pad_sequence(tgt_set_his, batch_first=True, padding_value=0.).to(
        device)  # (1024,100)

    pred_y, sim_pre, pred_src_domain, pred_tgt_domain, pred_t, emb_s, emb_t = model(train_set_i_x, src_his=src_his,
                                                                                    src_hl=src_hl, tgt_his=tgt_his,
                                                                                    tgt_hl=tgt_hl, alpha=alpha,
                                                                                    train=True)
    loss = torch.sum((train_set_i_y - pred_y) ** 2)

    # loss_t = torch.sum((train_set_i_y - pred_t) ** 2)
    # loss += loss_t
    #
    # sim_label = torch.arange(BATCH_SIZE).to(device)  # (1,batch):[0,1,2,3,4,5,....batch-1]
    # nce_loss = F.cross_entropy(sim_pre, sim_label)
    # loss += nce_loss
    #
    # # 域分类器损失
    # one_vector = torch.ones(pred_tgt_domain.size(0), pred_tgt_domain.size(1)).to(device)
    # domain_label = torch.zeros(BATCH_SIZE).long().to(device)
    # pred_tgt_domain = one_vector - pred_tgt_domain
    # domain_loss = domain_criterion(pred_src_domain, domain_label) + domain_criterion(pred_tgt_domain, domain_label)
    # loss += domain_loss

    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, test_set, user_his_dict_src, edge_UI, edge_IU, i):
    model.eval()
    with torch.no_grad():
        test_set_i = test_set[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        test_set_i_x = test_set_i[:, :2].long().to(device)
        test_set_i_y = test_set_i[:, 2].float().to(device)
        test_set_i_u = test_set_i[:, :1].long()

        src_set_his = [torch.tensor(
            sequence_adjust(user_his_dict_src[test_set_i_u[k][0].item()]),
            dtype=torch.long
        ) for k in range(test_set_i_u.size(0))]  # 2048个list , 存储用户的交互
        src_hl = [src_set_his[k].size(0) for k in range(test_set_i_u.size(0))]
        src_hl = torch.tensor(src_hl, dtype=torch.long).to(device)
        src_his = torch.nn.utils.rnn.pad_sequence(src_set_his, batch_first=True, padding_value=0.).to(
            device)

        # 用户在目标领域的历史交互信息
        tgt_set_his = [torch.tensor(
            sequence_adjust(user_his_dict_tgt[test_set_i_u[k][0].item()]),
            dtype=torch.long
        ) for k in range(test_set_i_u.size(0))]  # 2048个list , 存储用户的交互
        tgt_hl = [tgt_set_his[k].size(0) for k in range(test_set_i_u.size(0))]  # (1024)
        tgt_hl = torch.tensor(tgt_hl, dtype=torch.long).to(device)
        tgt_his = torch.nn.utils.rnn.pad_sequence(tgt_set_his, batch_first=True, padding_value=0.).to(
            device)  # (1024,100)

        pred_y, sim_pre, pred_src_domain, pred_tgt_domain, _, _, _ = model(test_set_i_x, src_his=src_his,
                                                                           src_hl=src_hl, tgt_his=tgt_his,
                                                                           tgt_hl=tgt_hl)
        loss_r = torch.sum((test_set_i_y - pred_y) ** 2)

    y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
    l1 = np.sum(np.abs(y_hat - y))
    l2 = np.sum(np.square(y_hat - y))
    return loss_r.item(), l1, l2


def save_model(model, path):
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), path + file_name + '_model.pkl')


def load_model(model, path):
    model.load_embedding_nn(path + file_name + '_model.pkl')


fix_seed(1234)
parser = argparse.ArgumentParser(description='Specify some parameters!')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--test_ratio', default='0.2', help="test_ratio")
parser.add_argument('--test_degree', default='0', help="test_degree")
parser.add_argument('--dataset', default='1', help="dataset")
parser.add_argument('--head_num', type=int, default=3, help="head_num")
parser.add_argument('--emb_size', type=int, default=64, help="emb_size")
parser.add_argument('--hid_size', type=int, default=128, help="hid_size")
parser.add_argument('--epoch', type=int, default=50, help="epoch")

args = parser.parse_args()
"""
_tgt_CDs_and_Vinyl_src_Books
_tgt_Movies_and_TV_src_Books
_tgt_CDs_and_Vinyl_src_Movies_and_TV
"""
file_flag = args.dataset
file_name = ""
if file_flag == "1":
    file_name = f"{args.test_ratio}_tgt_CDs_and_Vinyl_src_Movies_and_TV"
elif file_flag == "2":
    file_name = f"{args.test_ratio}_tgt_Movies_and_TV_src_Books"
elif file_flag == "3":
    file_name = f"{args.test_ratio}_tgt_CDs_and_Vinyl_src_Books"

now = datetime.now().strftime("%Y_%m_%d")
# logger = getLogger('log/'+now+'_train/'+file_name+'_train.log')
logger = getLogger('log/2024_06_10_train/' + file_name + '_train.log')
logger.info('2024_06_10_train file: ' + file_name)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
datadir = 'data/ready/' + file_name + '.pkl'

LEARNING_RATE = 0.0001
DECAYING_FACTOR = 0.95
LAMBDA_REC = 1.
BATCH_SIZE = 128
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = args.epoch  # 500
n_rating = 5

n_user, n_item, test_user_num, train_ov_tgt, train_ov_src, train_src, train_tgt, test_tgt_data, test_src_data, \
user_his_dict_src, user_his_dict_tgt, user_supp_list, edge_UI_tgt, edge_UI_src, test_edge_UI_tgt = \
    generate_data(device=device, datadir=datadir, sample_graph=False, logger=logger, test_degree=args.test_degree)

supp_users = torch.tensor(user_supp_list, dtype=torch.long)
edge_IU_tgt = []
# for n in range(n_rating):
#     edge_UI_tgt[n] = torch.tensor(edge_UI_tgt[n])
#     edge_IU_n = edge_UI_tgt[n].transpose(1, 0).contiguous()
#     edge_IU_tgt.append(edge_IU_n)
#
test_edge_IU = []
# for n in range(n_rating):
#     test_edge_UI[n] = torch.tensor(test_edge_UI[n])
#     test_edge_IU_n = test_edge_UI[n].transpose(1, 0).contiguous()
#     test_edge_IU.append(test_edge_IU_n)

train_set = torch.tensor(train_ov_tgt)
test_set = torch.tensor(test_tgt_data)
train_set = train_set[torch.randperm(train_set.size(0))]
train_size, test_size = train_set.size(0), test_set.size(0)

n_iter = n_epochs * train_size // BATCH_SIZE
bestRMSE = 10.0
src_pretrain_path = "model/pretrain-src/" + file_name + '_model.pkl'
tgt_pretrain_path = "model/pretrain-tgt/" + file_name + '_model.pkl'

model = IRMC_GC_Model(n_user=n_user,
                      n_item=n_item,
                      n_rating=n_rating,
                      supp_users=supp_users,
                      device=device,
                      src_pretrain_path=src_pretrain_path,
                      tgt_pretrain_path=tgt_pretrain_path,
                      args=args
                      ).to(device)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAYING_FACTOR)
start_time = datetime.now()
step = 0
domain_criterion = torch.nn.CrossEntropyLoss()

for epoch in range(n_epochs):
    # 训练
    train_set = train_set[torch.randperm(train_size)]
    loss_r_sum = 0.
    emb_t = None
    for i in tqdm(range(train_size // BATCH_SIZE), desc=f"epoch{epoch}_train"):
        len_batch = train_size // BATCH_SIZE
        p = float(i + epoch * len_batch) / n_epochs / len_batch
        alpha = 2. / (1. + np.exp(
            -10 * p)) - 1  # 这个计算过程会使得 alpha 在训练初期接近 -1，在训练后期逐渐向1靠近。这样做的目的是为了在训练过程中动态调整一个参数，以便在模型训练的不同阶段应用不同的策略或权重调整。
        loss_r = train(model, optimizer, i, train_set, user_his_dict_src, user_his_dict_tgt, edge_UI_tgt,
                       edge_IU_tgt, alpha)
        loss_r_sum += loss_r
    #     if epoch == args.epoch-1:
    #         if i == 0:
    #             emb_t = emb_t_i
    #         else:
    #             emb_t = torch.cat((emb_t, emb_t_i), dim=0)
    #     step += 1
    # if epoch == args.epoch-1:
    #     np.save('data/'+args.dataset+'_train_emb.npy', emb_t.cpu().detach().numpy())
    loss_r_train = loss_r_sum / train_size
    cost_time = str((datetime.now() - start_time) / (epoch + 1) * (n_epochs - epoch)).split('.')[0]
    logger.info('Epoch {}: TrainLoss {:.4f} (left: {})'.format(epoch, loss_r_train, cost_time))
    scheduler.step()

    # 测试
    ypre_score_dic = {}
    count11 = count22 = count33 = count44 = count55 = 0

    y_score_dic = {}
    count1 = count2 = count3 = count4 = count5 = 0
    test_emb_t = None
    loss_r_test_sum, l1_sum, l2_sum = 0., 0., 0.
    for i in tqdm(range(test_size // BATCH_SIZE), desc=f"epoch{epoch}_test"):
        loss_r_test, l1, l2 = test(model, test_set, user_his_dict_src, test_edge_UI_tgt, test_edge_IU, i)
        loss_r_test_sum += loss_r_test
        l1_sum += l1
        l2_sum += l2
    TestLoss = loss_r_test_sum / test_size
    test_MAE = l1_sum / test_size
    test_RMSE = np.sqrt(l2_sum / test_size)
    # logger.info('user_num: {te}, TestLoss: {:.4f} MAE: {:.4f} RMSE: {:.4f}'.format(TestLoss, test_MAE, test_RMSE))
    logger.info(
        f'user dregree: {args.test_degree:} , user num: {test_user_num}, mae : {test_MAE:.4f}, rmse : {test_RMSE:.4f}, testloss : {TestLoss:.4f}')
    if test_RMSE < bestRMSE:
        bestRMSE = test_RMSE
        save_model(model, 'model/train/')
