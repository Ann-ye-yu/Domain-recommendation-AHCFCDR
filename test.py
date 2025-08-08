import os
import argparse
from model import IRMC_GC_Model
from utils import *
import torch
import torch.nn.functional as F
from datetime import datetime
# import wandb
from logutils import getLogger

# wandb.init(project='TBD_Pro',
#            entity='jojozhao',
#            name='train_0.2_train',
#            config={
#                "learning_rate":  0.0001,
#                "batchsize_train": 1024,
#                "batchsize_test": 1024,  # 原本是2048
#                "n_epoches": 100,
#                "head_num": 4,
#                "sample_num": 500,
#            })
ypre_score_dic = {}
count11 = count22 = count33 = count44 = count55 = 0
y_score_dic = {}
count1 = count2 = count3 = count4 = count5 = 0


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


def neg_sampling(train_set_i, num_neg_per=5):
    size = train_set_i.size(0)
    neg_iid = torch.randint(0, n_item, (num_neg_per * size,)).reshape(-1)
    return torch.stack([train_set_i[:, 0].repeat(num_neg_per), neg_iid, torch.zeros(num_neg_per * size)], dim=1)


def count_score(pred_y, test_set_i_y):
    test_set_i_y1 = test_set_i_y.long()
    pred_y1 = pred_y.long()
    global y_score_dic
    global count1
    global count2
    global count3
    global count4
    global count5
    for i in test_set_i_y1:
        if i == 1:
            count1 += 1
        elif i == 2:
            count2 += 1
        elif i == 3:
            count3 += 1
        elif i == 4:
            count4 += 1
        elif i == 5:
            count5 += 1
    y_score_dic['1'] = count1
    y_score_dic['2'] = count2
    y_score_dic['3'] = count3
    y_score_dic['4'] = count4
    y_score_dic['5'] = count5

    global ypre_score_dic
    global count11
    global count22
    global count33
    global count44
    global count55
    for i in pred_y1:
        if i == 1:
            count11 += 1
        elif i == 2:
            count22 += 1
        elif i == 3:
            count33 += 1
        elif i == 4:
            count44 += 1
        elif i == 5:
            count55 += 1
    ypre_score_dic['1'] = count11
    ypre_score_dic['2'] = count22
    ypre_score_dic['3'] = count33
    ypre_score_dic['4'] = count44
    ypre_score_dic['5'] = count55


def test(model, test_set, user_his_dict_src, user_his_dic_tgt):
    model.eval()
    loss_r_test_sum, l1_sum, l2_sum, ndcg_sum, num = 0., 0., 0., 0., 0
    test_size = test_set.size(0)
    user_score_dict, user_label_dict = {}, {}
    for k in user_his_dic_tgt.keys():
        user_score_dict[k] = []
        user_label_dict[k] = []
    for i in range(test_size // BATCH_SIZE_TEST + 1):
        with torch.no_grad():
            test_set_i = test_set[i * BATCH_SIZE_TEST: (i + 1) * BATCH_SIZE_TEST]
            test_set_i_x = test_set_i[:, :2].long().to(device)
            test_set_i_y = test_set_i[:, 2].float().to(device)
            test_set_i_u = test_set_i[:, :1].float()

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

            pred_y, sim_pre, src_domain_output, tgt_domain_output, _, emb_s_i, emb_t_i = model(test_set_i_x,
                                                                                               src_his=src_his,
                                                                                               src_hl=src_hl,
                                                                                               tgt_his=tgt_his,
                                                                                               tgt_hl=tgt_hl)
            # loss_r = F.binary_cross_entropy_with_logits(pred_y, test_set_i_y, reduction='sum')
            if i == 0:
                test_emb_s = emb_s_i
                test_emb_t = emb_t_i
            else:
                test_emb_s = torch.cat((test_emb_s, emb_s_i), dim=0)
                test_emb_t = torch.cat((test_emb_t, emb_t_i), dim=0)
        y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
        # loss_r_test_sum += loss_r.item()
        l1_sum += np.sum(np.abs(y_hat - y))
        l2_sum += np.sum(np.square(y_hat - y))
        for k in range(test_set_i.size(0)):
            u, s, y = test_set_i_x[k, 0].item(), pred_y[k].item(), test_set_i_y[k].item()
            user_score_dict[u] += [s]
            user_label_dict[u] += [y]

        count_score(pred_y, test_set_i_y)
    logger.info("---------------------------------------------------------------------------")
    logger.info(f"真实评分分布：{y_score_dic}")
    logger.info(f"预测评分分布：{ypre_score_dic}")
    logger.info("---------------------------------------------------------------------------")
    # TestLoss = loss_r_test_sum / test_size

    MAE = l1_sum / test_size
    RMSE = np.sqrt(l2_sum / test_size)
    for u in user_score_dict.keys():
        if len(user_score_dict[u]) <= 1:
            continue
        ndcg_sum += ndcg_k(user_score_dict[u], user_label_dict[u], 10)
        num += 1
    np.save('data/' + args.dataset + '_hybrid_src_emb_no1.npy', test_emb_s.cpu().numpy())
    # np.save('data/' + args.dataset + '_hybrid_tgt_emb.npy', test_emb_t.cpu().numpy())
    return MAE, RMSE, ndcg_sum, num


def load_model_s(model, path):
    model.load_model(path + file_name + '_model.pkl')


def load_model_q(model, path):
    model.load_model(path + file_name + '_model.pkl')


fix_seed(1234)
parser = argparse.ArgumentParser(description='Specify some parameters!')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--test_ratio', default='0.2', help="test_ratio")
parser.add_argument('--test_degree', default='0', help="test_degree")  # 冷启动用户交互数量敏感度测试参数
parser.add_argument('--dataset', default='1', help="dataset")
parser.add_argument('--head_num', type=int, default=3, help="head_num")
parser.add_argument('--emb_size', type=int, default=64, help="emb_size")
parser.add_argument('--hid_size', type=int, default=128, help="hid_size")
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

# 敏感度测试用的
if args.test_degree == '1':
    degree = '[5,9]'
elif args.test_degree == '2':
    degree = '[10,14]'
elif args.test_degree == '3':
    degree = '[15,19]'
elif args.test_degree == '4':
    degree = '(20,)'
elif args.test_degree == '0':
    degree = 'all'

# 定义的一些参数和变量
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.001
DECAYING_FACTOR = 1.
LAMBDA_REG = 0.05
BATCH_SIZE_TEST = 128
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 1  # 500
n_rating = 5
datadir = 'data/ready/' + file_name + '.pkl'
src_pretrain_path = "model/pretrain-src/" + file_name + '_model.pkl'
tgt_pretrain_path = "model/pretrain-tgt/" + file_name + '_model.pkl'

# 读取预处理的数据
n_user, n_item, test_user_num, train_ov_tgt, train_ov_src, train_src, train_tgt, test_tgt_data, test_src_data, \
user_his_dict_src, user_his_dict_tgt, user_supp_list, edge_UI_tgt, edge_UI_src, test_edge_UI_tgt = \
    generate_data(device=device, datadir=datadir, test_degree=args.test_degree, sample_graph=False, logger=logger)

supp_users = torch.tensor(user_supp_list, dtype=torch.long)
test_set = torch.tensor(test_tgt_data)

model_q = IRMC_GC_Model(n_user=n_user,
                        n_item=n_item,
                        n_rating=n_rating,
                        supp_users=supp_users,
                        device=device,
                        src_pretrain_path=src_pretrain_path,
                        tgt_pretrain_path=tgt_pretrain_path,
                        args=args).to(device)
load_model_q(model_q, 'model/train/')
# 测试
MAE_q, RMSE_q, ndcg_sum_q, num_q = test(model_q, test_set, user_his_dict_src, user_his_dict_tgt)
NDCG_q = ndcg_sum_q / num_q
logger.info(f'user dregree: {degree} , user num: {test_user_num}, mae : {MAE_q: .4f}, rmse : {RMSE_q:.4f}')
