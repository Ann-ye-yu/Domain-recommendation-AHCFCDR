import argparse
import json
import pandas as pd
import gzip
import tqdm
import pickle
import random
from logutils import *
import os
from collections import defaultdict

logger = getLogger_preprocess('log/preprocess.log')

class DataPreprocessingMid():
    def __init__(self,
                 root,
                 dealing):
        self.root = root
        self.dealing = dealing

    def main(self):
        logger.info('Parsing ' + self.dealing + ' Mid...')
        re = []
        with gzip.open(self.root + 'raw/reviews_' + self.dealing + '_5.json.gz', 'rb') as f:
            for line in tqdm.tqdm(f, smoothing=0, mininterval=1.0):
                line = json.loads(line)
                re.append([line['reviewerID'], line['asin'], line['overall']])
        re = pd.DataFrame(re, columns=['uid', 'iid', 'y'])
        logger.info(self.dealing + ' Mid Done.')
        re.to_csv(self.root + 'mid/' + self.dealing + '.csv', index=0)
        return re


def read_mid(root, field):
    path = root + 'mid/' + field + '.csv'
    re = pd.read_csv(path)
    return re


def get_data(src, tgt, test_ratio):
    src = src.groupby("uid").filter(lambda x: (len(x) > 5))
    src = src.groupby("iid").filter(lambda x: (len(x) > 5))
    tgt = tgt.groupby("uid").filter(lambda x: (len(x) > 5))
    tgt = tgt.groupby("iid").filter(lambda x: (len(x) > 5))
    logger.info('Source inters: {}, uid: {}, iid: {}.'.format(len(src), len(set(src.uid)), len(set(src.iid))))
    logger.info('Target inters: {}, uid: {}, iid: {}.'.format(len(tgt), len(set(tgt.uid)), len(set(tgt.iid))))
    all_uid = set(src.uid) | set(tgt.uid)
    all_iid = len(set(src.iid)) +len(set(tgt.iid))
    co_uid = set(src.uid) & set(tgt.uid)
    total_user_num = len(all_uid)
    total_item_num = all_iid
    print('All uid: {}, Co uid: {}.'.format(len(all_uid), len(co_uid)))

    # id映射
    uid_map = dict(zip(all_uid, range(len(all_uid))))
    iid_map_src = dict(zip(set(src.iid), range(len(set(src.iid)))))
    iid_map_tgt = dict(zip(set(tgt.iid), range(len(set(src.iid)), len(set(src.iid)) + len(set(tgt.iid)))))
    src.uid = src.uid.map(uid_map)
    src.iid = src.iid.map(iid_map_src)
    tgt.uid = tgt.uid.map(uid_map)
    tgt.iid = tgt.iid.map(iid_map_tgt)

    # 分割测试集和训练集
    src_users = set(src.uid.unique())
    tgt_users = set(tgt.uid.unique())
    co_users = src_users & tgt_users
    train_ov_src = src[src['uid'].isin(co_users)].astype(int).values.tolist()
    train_ov_src_dic = defaultdict(list)
    for data in train_ov_src:
        train_ov_src_dic[data[0]].append(data[1])

    test_users = set(random.sample(co_users, round(test_ratio * len(co_users))))
    logger.info(f"test_users:{len(test_users)}")

    train_src = src
    train_tgt = tgt[tgt['uid'].isin(tgt_users - test_users)]
    train_ov_data_tgt = tgt[tgt['uid'].isin(co_users - test_users)]
    train_ov_data_src = src[src['uid'].isin(co_users - test_users)]
    test_data_tgt = tgt[tgt['uid'].isin(test_users)]
    test_data_src = src[src['uid'].isin(test_users)]

    train_src = train_src.astype(int).values.tolist()
    train_tgt = train_tgt.astype(int).values.tolist()
    train_ov_data_tgt = train_ov_data_tgt.astype(int).values.tolist()
    train_ov_data_src = train_ov_data_src.astype(int).values.tolist()
    test_data_tgt = test_data_tgt.astype(int).values.tolist()
    test_data_src = test_data_src.astype(int).values.tolist()

    return train_src, train_tgt, train_ov_data_src, train_ov_data_tgt, test_data_src, test_data_tgt, total_user_num, total_item_num,

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data_mid', default=0)
    parser.add_argument('--process_data_ready', default=1)
    args = parser.parse_args()
    file_directory = os.path.dirname(os.path.abspath(__file__))
    # os.path.join(file_directory, dataset_config['path'], '2024_06_10_train.txt')
    config_path = 'config.json'
    # config_path = os.path.join(file_directory,'config.json')
    # print(config_path)
    with open(config_path, 'r') as f:
        config = json.load(f)
    if args.process_data_mid:
        for dealing in ['Books', 'CDs_and_Vinyl', 'Movies_and_TV']:
            DataPreprocessingMid(config['root'], dealing).main()
    if args.process_data_ready:
        src_tgt_pairs = config['src_tgt_pairs']
        for test_ratio in [0.2, 0.5, 0.8]:
            for task in ['1', '2', '3']:
                src_name = src_tgt_pairs[task]['src']
                tgt_name = src_tgt_pairs[task]['tgt']
                logger.info(f"task_src:{src_name} task_tgt:{tgt_name} test_ratio:{test_ratio}")
                src = read_mid(config['root'], src_name)
                tgt = read_mid(config['root'], tgt_name)

                train_src,train_tgt,train_ov_src, train_ov_tgt, test_src, test_tgt, total_user_num, total_item_num = get_data(src, tgt, test_ratio)
                save_path = "data/ready/" + str(test_ratio) + "_tgt_" + tgt_name + '_src_' + src_name + '.pkl'
                with open(save_path, 'wb') as f:
                    pickle.dump(train_src, f)
                    pickle.dump(train_tgt, f)
                    pickle.dump(train_ov_src, f)
                    pickle.dump(train_ov_tgt, f)
                    pickle.dump(test_src, f)
                    pickle.dump(test_tgt,f)
                    pickle.dump((total_user_num, total_item_num), f)
