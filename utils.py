import pickle
import random
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from sklearn.manifold import TSNE
random.seed(1234)
HIS_MAXLEN = 100

def get_key_nodes(device,train_data, train_users,train_items):
    train_data_array = np.array(train_data)
    users, items = train_data_array[:, 0], train_data_array[:, 1]
    row = np.concatenate([users, items + train_users], axis=0)
    column = np.concatenate([items + train_users, users], axis=0)
    '''
    根据用户交互的度从大到小排序
    '''
    adj_mat = sp.coo_matrix((np.ones(row.shape), np.stack([row, column], axis=0)),
                            shape=(train_users + train_items, train_users + train_items),
                            dtype=np.float32).tocsr()
    # normalized_adj_mat = normalize(adj_mat, axis=1, norm='l1')  # (70846,70846)
    normalized_adj_mat = adj_mat
    user_metrics = np.array(np.sum(normalized_adj_mat[:, :train_users], axis=0)).squeeze()  # (train_users)
    item_metrics = np.array(np.sum(normalized_adj_mat[:, train_users:], axis=0)).squeeze()  # (40988)
    ranked_users = np.argsort(user_metrics)[::-1].copy() # 返回按照user_metrics中数值从大到小排序后的索引
    ranked_items = np.argsort(item_metrics)[::-1].copy()
    '''
    在上面排序的基础上，选取度高的用户，再根据用户评分的方差从大到小排序，方差越大的说明用户评分分布比较离散，更加公平公正
    '''
    # key_users = ranked_users[:1000]
    # bool_index = np.isin(train_data_array[:,0],key_users)
    # key_users_data = train_data_array[bool_index,:]
    # users_score_dic = {}
    # for data in key_users_data:
    #     uid = data[0]
    #     if uid not in users_score_dic:
    #         users_score_dic[uid] = [data[2]]
    #     else:
    #         users_score_dic[uid].append(data[2])
    # users_var_dic = {}
    # for uid,score_list in users_score_dic.items():
    #     var = np.var(np.array( score_list))
    #     users_var_dic[uid] = var
    # sorted_dic = sorted(users_var_dic.items(),key = lambda x:x[1],reverse=True)
    # ranked_users = np.array([uid for uid,li in sorted_dic])
    return ranked_users

def sequence_adjust(seq,n_item):
    seq_new = seq
    if len(seq) <= 0:
        seq_new = [np.random.randint(0, n_item) for i in range(HIS_MAXLEN)]
    if len(seq) > HIS_MAXLEN:
        random.shuffle(seq)
        seq_new = seq[:HIS_MAXLEN]
    if len(seq) <HIS_MAXLEN:
        seq_new = seq_new + [0]*(HIS_MAXLEN-len(seq))
    return seq_new

def generate_data(device,datadir, logger, test_degree='0', sample_graph=False, key_deta=20):
    with open(datadir, 'rb') as f:
        # train_uid的编号是从0到n的话，test_uid的编号是接着从n开始编号的
        # tgt领域的iid从0开始编号，src领域的iid接着从tgt的iid开始编号
        train_src = pickle.load(f)
        train_tgt = pickle.load(f)
        train_ov_src = pickle.load(f)  # user个二维list:[[u,i,r],[u,i,r],[u,i,r]]
        train_ov_tgt = pickle.load(f)
        test_src = pickle.load(f)
        test_tgt = pickle.load(f)
        all_uid_num, all_iid_num = pickle.load(f)

    n_rating = 5
    n_user = all_uid_num
    n_item = all_iid_num

    test_tgt_ui_dic = {}
    for data in test_tgt:
        if data[0] not in test_tgt_ui_dic:
            test_tgt_ui_dic[data[0]] = [data[1]]
        else:
            test_tgt_ui_dic[data[0]].append(data[1])

    test_src_ui_dic = {}
    for data in test_src:
        if data[0] not in test_src_ui_dic:
            test_src_ui_dic[data[0]] = [data[1]]
        else:
            test_src_ui_dic[data[0]].append(data[1])

    test_src_data, test_tgt_data = [], []
    test_user_set = set()
    for data_src in test_src:
        user_src = data_src[0]
        len_iteractions_u_src = len(test_src_ui_dic[user_src])
        if test_degree == '1':
            if 9 >= len_iteractions_u_src >= 5:
                test_src_data.append(data_src)
                test_user_set.add(user_src)
        elif test_degree == '2':
            if 14 >= len_iteractions_u_src >= 10:
                test_src_data.append(data_src)
                test_user_set.add(user_src)
        elif test_degree == '3':
            if 19 >= len_iteractions_u_src >= 15:
                test_src_data.append(data_src)
                test_user_set.add(user_src)
        elif test_degree == '4':
            if 19 >= len_iteractions_u_src > 20:
                test_src_data.append(data_src)
                test_user_set.add(user_src)
        else:
            test_src_data.append(data_src)
            test_user_set.add(user_src)
    for data in test_tgt:
        if data[0] in test_user_set:
            test_tgt_data.append(data)
    test_user_num = len(test_user_set)

    # 获取支持用户
    user_supp_list = get_key_nodes(device,train_tgt, n_user,n_item).tolist()

    # overlap用户在源领域上的历史交互
    user_his_dict_src = {}
    for data in train_ov_src:
        user = data[0]
        if user not in user_his_dict_src:
            user_his_dict_src[user] = [data[1]]
        else:
            user_his_dict_src[user].append(data[1])
    for data in test_src:
        user = data[0]
        if user not in user_his_dict_src:
            user_his_dict_src[user] = [data[1]]
        else:
            user_his_dict_src[user].append(data[1])

    # overlap用户在目标领域上的历史交互
    user_his_dict_tgt = {}
    for data in train_ov_tgt:
        user = data[0]
        if user not in user_his_dict_tgt:
            user_his_dict_tgt[user] = [data[1]]
        else:
            user_his_dict_tgt[user].append(data[1])
    for data in test_tgt:
        user = data[0]
        if user not in user_his_dict_tgt:
            user_his_dict_tgt[user] = [data[1]]
        else:
            user_his_dict_tgt[user].append(data[1])

    for key,value in user_his_dict_tgt.items():
        new_value = sequence_adjust(value, n_item)
        user_his_dict_tgt[key] = new_value
    for key,value in user_his_dict_src.items():
        new_value = sequence_adjust(value, n_item)
        user_his_dict_src[key] = new_value

    # 如果模型采用图的方法的化，需要构建ui图
    use_graph = False
    train_uir_tgt = train_tgt
    edge_array_tgt = np.array(train_uir_tgt, dtype=np.int32)
    edge_array_tgt = np.transpose(edge_array_tgt, (1, 0))  # (3,)
    edge_UI_tgt = []
    if use_graph:
        for i in range(1, n_rating + 1):
            if sample_graph:
                edge_i = edge_array_tgt[:2, edge_array_tgt[2] == i]
                edge_UI_tgt.append(edge_i)
            else:
                edge_i = edge_array_tgt[:2, edge_array_tgt[2] == i]
                edge_UI_i = np.zeros((n_user, n_item), dtype=np.int32)
                edge_UI_i[edge_i[0], edge_i[1]] = 1
                edge_UI_tgt.append(edge_UI_i)

    train_uir_src = train_src
    edge_array_src = np.array(train_uir_src, dtype=np.int32)
    edge_array_src = np.transpose(edge_array_src, (1, 0))
    edge_UI_src = []
    if use_graph:
        for i in range(1, n_rating + 1):
            if sample_graph:
                edge_i = edge_array_src[:2, edge_array_src[2] == i]
                edge_UI_src.append(edge_i)
            else:
                edge_i = edge_array_src[:2, edge_array_src[2] == i]
                edge_UI_i = np.zeros((n_user, n_item), dtype=np.int32)  # (6040,3706)
                edge_UI_i[edge_i[0], edge_i[1]] = 1
                edge_UI_src.append(edge_UI_i)

    test_uir_tgt = test_tgt
    test_edge_array_tgt = np.array(test_uir_tgt, dtype=np.int32)
    test_edge_array_tgt = np.transpose(test_edge_array_tgt, (1, 0))  # (3,)
    test_edge_UI_tgt = []
    if use_graph:
        for i in range(1, n_rating + 1):
            if sample_graph:
                edge_i = test_edge_array_tgt[:2, test_edge_array_tgt[2] == i]
                test_edge_UI_tgt.append(edge_i)
            else:
                edge_i = test_edge_array_tgt[:2, test_edge_array_tgt[2] == i]
                edge_UI_i = np.zeros((n_user, n_item), dtype=np.int32)  # (6040,3706)
                edge_UI_i[edge_i[0], edge_i[1]] = 1
                test_edge_UI_tgt.append(edge_UI_i)

    return n_user, n_item, test_user_num, train_ov_tgt, train_ov_src, train_src, train_tgt, test_tgt_data, test_src_data, user_his_dict_src, user_his_dict_tgt, user_supp_list, edge_UI_tgt, edge_UI_src, test_edge_UI_tgt


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


def recall_calc(score_label):
    num, num_tp = 0, 0
    for s in score_label:
        if s[1] == 1:
            num += 1
            if s[0] >= 0.5:
                num_tp += 1
    return num_tp / num


def precision_calc(score_label):
    num, num_tp = 0, 0
    for s in score_label:
        if s[0] >= 0.5:
            num += 1
            if s[1] == 1:
                num_tp += 1
    return num_tp / num

'''
    dcg_k函数的计算原理：
        我们的模型计算出某个用户对所有候选项目的相似度后(在评分任务中，相似度为评分)，
    根据相似度对项目进行排序，返回前k个最相似的项目作为推荐结果。
    这k个项目维持我们推荐的顺序，为它们标注上它们在数据集中真实的打分，然后拿来计算DCG。
'''
def dcg_k(score_label, k):
    '''
    :param score_label:预测评分和真实评分的（2*n）形状的组合列表(并且是按照预测评分从大到小排好序的)： [[预测评分，真实评分],[预测评分，真实评分],......]
    :param k: 召回参数
    :return: dck
    '''
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            # dcg += (2 ** s[1] - 1) / np.log2(1 + (1+i)) # 用指数计算的方式
            dcg += s[1] / np.log2(1 + (1 + i))
            i += 1
    return dcg


'''
    ndcg_k函数的计算原理：
    1、 ndcg_k = dcg_k/idcg_k,即就是折损累计增益dcg_k除以最大累计增益idcg_k
    2、最大累计增益iDCG的计算方法是：
        模型返回了k个推荐的项目，我们将这k个项目标注上它们在原始数据集上的分数，
        然后再根据分数进行重排序，然后再计算DCG，得到的就是iDCG
'''
def ndcg_k(y_hat, y, k):
    '''
    :param y_hat: 模型的预测分数列表，list类型
    :param y: 真实评分列表, list类型
    :param k: 召回参数
    :return: NDCG@K
    '''
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d: d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d: d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            # norm += (2 ** s[1] - 1) / np.log2(2 + i) # 用指数计算的方式
            norm += s[1] / np.log2(1 + (1 + i))
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm

def visual_emb():
    '''
    : param emb:  shape: (batch,emb_size),type:tensor
    : return:
    '''
    np.random.seed(1)
    # Perform t-SNE on your data subset
    emb0_all = np.load('data/2_hybrid_src_emb_no1.npy')  # "Emb without AL"
    emb1_all = np.load('data/2_hybrid_src_emb.npy')  # Emb with AL
    emb2_all = np.load('data/2_hybrid_tgt_emb.npy')  # Ground Truth Emb

    emb0 = emb0_all[np.random.choice(emb0_all.shape[0], 8000, replace=False), :]
    emb1 = emb1_all[np.random.choice(emb1_all.shape[0], 8000, replace=False), :]
    emb2 = emb2_all[np.random.choice(emb2_all.shape[0], 8000, replace=False), :]

    tsne0 = TSNE(n_components=2, random_state=42).fit_transform(emb0)
    tsne1 = TSNE(n_components=2, random_state=42).fit_transform(emb1)
    tsne2 = TSNE(n_components=2, random_state=42).fit_transform(emb2)

    # Plot the t-SNE transformed data
    # plt.scatter(tsne0[:, 0], tsne0[:, 1], s=2, color="darkorange", cmap='Spectral', label="Emb without AL")  # 1_hybrid_src_emb_no
    plt.scatter(tsne1[:, 0], tsne1[:, 1], s=2, c="forestgreen", label="Emb with AL")
    plt.scatter(tsne2[:, 0], tsne2[:, 1], s=2, c="black", label="Ground Truth Emb")

    plt.legend(loc="upper right")
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title('', fontsize=24)
    plt.tight_layout()
    # plt.savefig('data/data3_noad1.png', dpi=300, bbox_inches='tight')
    plt.savefig('data/data2_ad1.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_hyper():
    # 数据
    h_values = [1, 2, 3, 4, 5]
    d_values = [16, 32, 64, 128, 256]  # 修改为 32, 64, 128, 256

    # 任务1的MAE和RMSE
    mae_task1 = [0.7596, 0.7529, 0.7472, 0.7485, 0.7699]
    rmse_task1 = [0.9982, 0.9971, 0.9968, 1.0000, 1.0118]

    # 对于不同的h值，我们可以选择指定d时的数据
    # 任务2的MAE和RMSE
    mae_task2 = [0.8450, 0.8438, 0.8411, 0.8458, 0.8437]
    rmse_task2 = [1.0953, 1.0941, 1.0909, 1.0939, 1.0936]
    # 任务3的MAE和RMSE
    mae_task3 = [0.7429, 0.7321, 0.7279, 0.7285, 0.7288]
    rmse_task3 = [0.9568, 0.9455, 0.9418, 0.9420, 0.9425]

    # 对于不同的d值，我们可以选择h=3时的数据
    d_mae_task1 = [0.7887, 0.7808, 0.7508, 0.7472, 0.7602]  # 对应 d=16, 32, 64, 128, 256
    d_rmse_task1 = [1.0257, 1.0091, 0.9985, 0.9968, 1.0085]

    d_mae_task2 = [0.8602, 0.8553, 0.8430, 0.8411, 0.8429]
    d_rmse_task2 = [1.1078, 1.1021, 1.0959, 1.0909, 1.1004]

    d_mae_task3 = [0.7718, 0.7508, 0.7384, 0.7279, 0.7334]
    d_rmse_task3 = [0.9835, 0.9750, 0.9623, 0.9418, 0.9553]

    # 设置图片大小一致
    plt.figure(figsize=(8, 6))

    # 绘制MAE
    plt.plot(h_values, mae_task1, marker='o', label='Task 1 MAE')
    plt.plot(h_values, mae_task2, marker='o', label='Task 2 MAE')
    plt.plot(h_values, mae_task3, marker='o', label='Task 3 MAE')

    # 添加数据标注（放在下面两条折线的上方）
    for i, (mae1, mae2, mae3) in enumerate(zip(mae_task1, mae_task2, mae_task3)):
        plt.text(h_values[i], mae1+0.002 , f'{mae1:.4f}', fontsize=8, ha='center')
        plt.text(h_values[i], mae2+0.002 , f'{mae2:.4f}', fontsize=8, ha='center')
        plt.text(h_values[i], mae3+0.002 , f'{mae3:.4f}', fontsize=8, ha='center')

    plt.xticks(h_values, [str(h) for h in h_values])
    plt.xlabel('h values')
    plt.ylabel('MAE')
    plt.legend(loc='center right',bbox_to_anchor=(1, 0.6), fontsize='medium', frameon=True, ncol=1, shadow=False)
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局
    plt.savefig('data/mae_plot.png', bbox_inches='tight')  # 存储图片，确保图例完整
    plt.show()

    # 设置图片大小一致
    plt.figure(figsize=(8, 6))

    # 绘制RMSE
    plt.plot(h_values, rmse_task1, marker='o', label='Task 1 RMSE')
    plt.plot(h_values, rmse_task2, marker='o', label='Task 2 RMSE')
    plt.plot(h_values, rmse_task3, marker='o', label='Task 3 RMSE')

    # 添加数据标注（放在下面两条折线的上方）
    for i, (rmse1, rmse2, rmse3) in enumerate(zip(rmse_task1, rmse_task2, rmse_task3)):
        plt.text(h_values[i], rmse1 +0.002, f'{rmse1:.4f}', fontsize=8, ha='center')
        plt.text(h_values[i], rmse2 +0.002, f'{rmse2:.4f}', fontsize=8, ha='center')
        plt.text(h_values[i], rmse3 +0.002, f'{rmse3:.4f}', fontsize=8, ha='center')

    plt.xticks(h_values, [str(h) for h in h_values])
    plt.xlabel('h values')
    plt.ylabel('RMSE')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.6), fontsize='medium', frameon=True, ncol=1, shadow=False)
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局
    plt.savefig('data/rmse_plot.png', bbox_inches='tight')  # 存储图片，确保图例完整
    plt.show()

    # 设置图片大小一致
    plt.figure(figsize=(8, 6))

    # 绘制MAE（d值）
    plt.plot(d_values, d_mae_task1, marker='o', label='Task 1 MAE')
    plt.plot(d_values, d_mae_task2, marker='o', label='Task 2 MAE')
    plt.plot(d_values, d_mae_task3, marker='o', label='Task 3 MAE')

    # 添加数据标注（放在下面两条折线的上方）
    for i, (mae1, mae2, mae3) in enumerate(zip(d_mae_task1, d_mae_task2, d_mae_task3)):
        plt.text(d_values[i], mae1+0.002, f'{mae1:.4f}', fontsize=8, ha='center')
        plt.text(d_values[i], mae2+0.002, f'{mae2:.4f}', fontsize=8, ha='center')
        plt.text(d_values[i], mae3 +0.002, f'{mae3:.4f}', fontsize=8, ha='center')

    plt.xticks(d_values, [str(d) for d in d_values])
    plt.xlabel('d values')
    plt.ylabel('MAE')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.6), fontsize='medium', frameon=True, ncol=1, shadow=False)
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局
    plt.savefig('data/mae_d_plot.png', bbox_inches='tight')  # 存储图片，确保图例完整
    plt.show()

    # 设置图片大小一致
    plt.figure(figsize=(8, 6))

    # 绘制RMSE（d值）
    plt.plot(d_values, d_rmse_task1, marker='o', label='Task 1 RMSE')
    plt.plot(d_values, d_rmse_task2, marker='o', label='Task 2 RMSE')
    plt.plot(d_values, d_rmse_task3, marker='o', label='Task 3 RMSE')

    # 添加数据标注（放在下面两条折线的上方）
    for i, (rmse1, rmse2, rmse3) in enumerate(zip(d_rmse_task1, d_rmse_task2, d_rmse_task3)):
        plt.text(d_values[i], rmse1+0.002, f'{rmse1:.4f}', fontsize=8, ha='center')
        plt.text(d_values[i], rmse2+0.002, f'{rmse2:.4f}', fontsize=8, ha='center')
        plt.text(d_values[i], rmse3+0.002, f'{rmse3:.4f}', fontsize=8, ha='center')

    plt.xticks(d_values, [str(d) for d in d_values])
    plt.xlabel('d values')
    plt.ylabel('RMSE')
    plt.legend(loc='center right', bbox_to_anchor=(1, 0.6), fontsize='medium', frameon=True, ncol=1, shadow=False)
    plt.grid(True)
    plt.tight_layout()  # 自动调整布局
    plt.savefig('data/rmse_d_plot.png', bbox_inches='tight')  # 存储图片，确保图例完整
    plt.show()

if __name__ == "__main__":
    visual_emb()