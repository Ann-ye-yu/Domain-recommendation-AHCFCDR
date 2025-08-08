import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time
from logutils import getLogger

class Embedding(nn.Module):
    def __init__(self, n_user, n_item, embedding_size=32):
        super(Embedding, self).__init__()
        self.user_embedding = nn.Parameter(torch.Tensor(n_user, embedding_size))
        self.item_embedding = nn.Parameter(torch.Tensor(n_item, embedding_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding)
        nn.init.xavier_uniform_(self.item_embedding)

    def regularization_loss(self):
        loss_reg = 0.
        loss_reg += torch.sum(torch.sqrt(torch.sum(self.user_embedding ** 2, 1)))
        loss_reg += torch.sum(torch.sqrt(torch.sum(self.item_embedding ** 2, 1)))
        return loss_reg


class GCMCModel(nn.Module):
    def __init__(self, n_user, n_item, n_rating, device, edge_sparse=None, embedding_size=64, hidden_size=64):
        super(GCMCModel, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_rating = n_rating
        self.device = device
        self.edge_sparse = edge_sparse
        self.embedding_model = Embedding(n_user, n_item, embedding_size)
        self.user_embedding = self.embedding_model.user_embedding
        self.item_embedding = self.embedding_model.item_embedding

        # graph法
        # self.GCN_user = nn.Linear(embedding_size, embedding_size)
        # self.GCN_item = nn.Linear(embedding_size, embedding_size)
        # self.fc1_user = nn.Linear(n_rating * embedding_size, embedding_size)
        # self.fc1_item = nn.Linear(n_rating * embedding_size, embedding_size)
        # self.l1 = nn.Linear(embedding_size * 4, hidden_size * 2)
        # self.l2 = nn.Linear(hidden_size * 2, hidden_size)
        # self.l3 = nn.Linear(hidden_size, 1)

        # nn法
        self.l1 = nn.Linear(embedding_size * 3, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.user_bias = nn.Parameter(torch.Tensor(n_user, 1))
        self.item_bias = nn.Parameter(torch.Tensor(n_item, 1))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.user_bias)
        nn.init.zeros_(self.item_bias)

    def regularization_loss(self):
        return self.embedding_model.regularization_loss()

    def forward(self, x, edge_UI=None, edge_IU=None, sample_graph=False):
        user_id = x[:, 0]
        item_id = x[:, 1]

        user_emb = self.user_embedding[user_id]
        item_emb = self.item_embedding[item_id]

        # Feedford Neural Network方法
        # interaction = torch.mul(user_emb, item_emb)
        # ratings = torch.sum(interaction, dim=1)
        #
        # x = torch.cat([user_emb, item_emb, interaction], dim=1)
        # x1 = torch.tanh(self.l1(x))
        # x2 = torch.tanh(self.l2(x1))
        # x3 = self.l3(x2).reshape(-1)
        # user_b = self.user_bias[user_id].reshape(-1)
        # item_b = self.item_bias[item_id].reshape(-1)
        #
        # output = (ratings + x3) / 2. + user_b + item_b
        # graph方法
        # user_h = self.user_embedding
        # item_h = self.item_embedding
        #
        # for n in range(self.n_rating):
        #     if sample_graph:
        #         edge_index_ui_n = self.edge_sparse[self.edge_sparse[:, 2] == n][:, :2].transpose(1, 0)
        #         edge_index_iu_n = edge_index_ui_n[[1, 0], :]
        #         # if self.training: # for dropout sparse edge matrix
        #         #	edge_num_n = edge_index_ui_n.size(1)
        #         #	edge_index_ui_n = edge_index_ui_n[:, torch.randperm(edge_num_n)][:, :int(0.7*edge_num_n)]
        #         #	edge_index_iu_n = edge_index_iu_n[:, torch.randperm(edge_num_n)][:, :int(0.7*edge_num_n)]
        #         edge_ui_n = torch.sparse_coo_tensor(edge_index_ui_n, torch.ones(edge_index_ui_n.size(1)),
        #                                             size=torch.Size([self.n_user, self.n_item])).to(self.device)
        #         edge_iu_n = torch.sparse_coo_tensor(edge_index_iu_n, torch.ones(edge_index_iu_n.size(1)),
        #                                             size=torch.Size([self.n_item, self.n_user])).to(self.device)
        #         gcn_item_h_n = torch.sparse.mm(edge_ui_n, item_h)[user_id]
        #         gcn_user_h_n = torch.sparse.mm(edge_iu_n, user_h)[item_id]
        #         item_din = torch.sparse.mm(edge_ui_n, torch.ones(self.n_item, 1).to(self.device))[user_id] + 1
        #         user_din = torch.sparse.mm(edge_iu_n, torch.ones(self.n_user, 1).to(self.device))[item_id] + 1
        #         gcn_item_h_n = gcn_item_h_n / item_din
        #         gcn_user_h_n = gcn_user_h_n / user_din
        #         gcn_item_h_n = F.dropout(gcn_item_h_n, p=0.3, training=self.training)
        #         gcn_user_h_n = F.dropout(gcn_user_h_n, p=0.3, training=self.training)
        #     # edge_index_ui_n = self.edge_sparse
        #     # edge_index_iu_n = edge_index_ui_n[[1,0], :]
        #     # edge_ui_n = torch.sparse_coo_tensor(edge_index_ui_n, torch.ones(edge_index_ui_n.size(1)), size=torch.Size([self.n_user, self.n_item])).to(self.device)
        #     # edge_iu_n = torch.sparse_coo_tensor(edge_index_iu_n, torch.ones(edge_index_iu_n.size(1)), size=torch.Size([self.n_item, self.n_user])).to(self.device)
        #     # gcn_item_h_n = torch.sparse.mm(edge_ui_n, item_h)[user_id]
        #     # gcn_user_h_n = torch.sparse.mm(edge_iu_n, user_h)[item_id]
        #     # item_din = torch.sparse.mm(edge_ui_n, torch.ones(self.n_item, 1).to(self.device))[user_id] + 1
        #     # user_din = torch.sparse.mm(edge_iu_n, torch.ones(self.n_user, 1).to(self.device))[item_id] + 1
        #     # gcn_item_h_n = gcn_item_h_n / item_din
        #     # gcn_user_h_n = gcn_user_h_n / user_din
        #     # gcn_item_h_n = F.dropout(gcn_item_h_n, p=0.3, training=self.training)
        #     # gcn_user_h_n = F.dropout(gcn_user_h_n, p=0.3, training=self.training)
        #     else:
        #         edge_UI_n = edge_UI[n].float()
        #         edge_IU_n = edge_IU[n].float()
        #         edge_UI_n = F.dropout(edge_UI_n, p=0.3, training=self.training)
        #         edge_IU_n = F.dropout(edge_IU_n, p=0.3, training=self.training)
        #         gcn_user_h_n = torch.matmul(edge_IU_n, user_h)
        #         gcn_item_h_n = torch.matmul(edge_UI_n, item_h)
        #     # gcn_user_output = torch.relu(self.GCN_user(gcn_user_h_n))
        #     # gcn_item_output = torch.relu(self.GCN_item(gcn_item_h_n))
        #     gcn_user_h_n = torch.relu(self.GCN_user(gcn_user_h_n))
        #     gcn_item_h_n = torch.relu(self.GCN_item(gcn_item_h_n))
        #     if n <= 0:
        #         gcn_user_h = gcn_user_h_n
        #         gcn_item_h = gcn_item_h_n
        #     else:
        #         gcn_user_h = torch.cat([gcn_user_h, gcn_user_h_n], dim=-1)
        #         gcn_item_h = torch.cat([gcn_item_h, gcn_item_h_n], dim=-1)
        #
        # gcn_user_output = self.fc1_user(gcn_user_h)
        # gcn_item_output = self.fc1_item(gcn_item_h)
        #
        # interaction1 = torch.mul(user_emb, item_emb)
        # interaction2 = torch.mul(user_emb, gcn_item_output)
        # interaction3 = torch.mul(gcn_user_output, item_emb)
        # interaction4 = torch.mul(gcn_user_output, gcn_item_output)
        # x = torch.cat([interaction1, interaction2, interaction3, interaction4], dim=-1)
        # x1 = torch.tanh(self.l1(x))
        # x2 = torch.tanh(self.l2(x1))
        # x3 = self.l3(x2).reshape(-1)
        #
        # user_b = self.user_bias[user_id].reshape(-1)
        # item_b = self.item_bias[item_id].reshape(-1)
        #
        # output = x3 + user_b + item_b
        output =torch.sum(user_emb*item_emb,1)

        return output

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)

    def load_embedding(self, path):
        pretrained_dict = torch.load(path)
        model_dict = self.embedding_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.embedding_model.load_state_dict(model_dict)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RelationGAT(nn.Module):
    def __init__(self, in_size, out_size):
        super(RelationGAT, self).__init__()
        self.wq = nn.Linear(in_size, out_size, bias=False)  # (dim,dim)
        self.wk = nn.Linear(in_size, out_size, bias=False)
        self.wv = nn.Linear(in_size, out_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)

    def forward(self, x, neighbor):  # x:(batch,dim), neighbor:(batch,500,dim)
        x = self.wq(x).unsqueeze(1)  # (batch,1,dim)
        neighbor = self.wk(neighbor) # (1024,500,64)
        # gat_input = torch.cat([x.repeat(1, neighbor.size(1), 1), neighbor], dim=2) # (1024,500,128)
        gat_input = torch.sum(
            torch.mul(x.repeat(1, neighbor.size(1), 1), neighbor), dim=2
        )  # (1024,500)
        attn = F.softmax(gat_input, dim=1)  # (batch,500)
        neighbor = neighbor.transpose(1, 2).contiguous() # (1024,64,500)
        test1 = attn.unsqueeze(2) #(batch,500,1)
        test2 =torch.matmul(neighbor, test1) # (batch,64,1)
        # test3 = self.wv(torch.matmul(neighbor, attn.unsqueeze(2)))
        gat_output = self.wv(
            torch.matmul(neighbor, attn.unsqueeze(2)).squeeze(2)
        )  # (1024,64)
        return gat_output


class IRMC_GC_Model(nn.Module):
    def __init__(self, n_user, n_item, n_rating, supp_users, device, edge_sparse=None,
                 out_size=None,
                 # head_num=None,
                 sample_num=500,
                 src_pretrain_path=None,
                 tgt_pretrain_path=None,
                 args = None):
        super(IRMC_GC_Model, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_rating = n_rating
        self.device = device
        self.edge_sparse = edge_sparse
        self.supp_users = supp_users.to(self.device)  # key user的索引列表
        self.supp_user_num = supp_users.size(0)
        # self.head_num = head_num
        self.sample_num = sample_num
        self.GAT_unit = nn.ModuleList()
        self.dropout_rate = 0.3
        self.embedding_size = args.emb_size
        self.hidden_size = args.hid_size
        self.head_num = args.head_num
        self.out_size = out_size
        if self.out_size is None:
            self.out_size = self.embedding_size
        for i in range(self.head_num):
            self.GAT_unit.append(RelationGAT(self.embedding_size, self.out_size))
        self.w_out = nn.Linear(self.out_size * self.head_num, self.out_size, bias=False)

        # self.user_embedding = nn.Parameter(torch.Tensor(n_user, embedding_size), requires_grad=False)
        # self.item_embedding = nn.Parameter(torch.Tensor(n_item, embedding_size), requires_grad=False)

        self.src_model = self.load_embedding_nn(src_pretrain_path)
        self.tgt_model = self.load_embedding_nn(tgt_pretrain_path)

        # GCN方法
        # self.GCN_user = nn.Linear(embedding_size, embedding_size)
        # self.GCN_item = nn.Linear(embedding_size, embedding_size)
        #
        # self.fc1_user = nn.Linear(n_rating * embedding_size, embedding_size)
        # self.fc1_item = nn.Linear(n_rating * embedding_size, embedding_size)
        #
        # self.l1 = nn.Linear(embedding_size * 4, hidden_size * 2)
        # self.l2 = nn.Linear(hidden_size * 2, hidden_size)
        # self.l3 = nn.Linear(hidden_size, 1)
        # nn方法
        self.l1 = nn.Linear(self.embedding_size * 2, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.embedding_size)
        self.l3 = nn.Linear(self.embedding_size, 1)
        self.l4 = nn.Linear(self.embedding_size *2, self.embedding_size)

        self.user_bias = nn.Parameter(torch.Tensor(n_user, 1), requires_grad=True)
        self.item_bias = nn.Parameter(torch.Tensor(n_item, 1), requires_grad=False)

        # 域分类器
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.embedding_size, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        # self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

        # 用户表征聚合层
        self.W_att = nn.Sequential(nn.Linear(self.embedding_size, self.embedding_size),
                                   nn.Tanh())
        self.dropout = nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
        self.W_agg = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.user_bias)
        nn.init.zeros_(self.item_bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def user_fea_encode(self,user_id,user_his,user_hl,emb_model):
        mask = torch.arange(user_his.size(1))[None, :].to(self.device)  # (1,100)
        mask = mask < user_hl[:, None]  # (1024,100)
        history_emb = self.src_model.item_embedding[user_his]  # (1024,100,64)
        history_emb[~mask] = torch.zeros(self.src_model.item_embedding.size(1)).to(self.device)  # (1024,100,64), src_user的item_embedding

        u_emb = emb_model.user_embedding[user_id]  # (1024,64),src_user的预训练得到的embedding

        key = self.W_att(history_emb)  # (1024,100,64), b x seq_len x attention_dim
        mask = history_emb.sum(dim=-1) == 0  # (1024,100)
        attention = torch.bmm(key, u_emb.unsqueeze(-1)).squeeze(-1)  # (1024,100),b x seq_len
        attention = self.masked_softmax(attention, mask)  # 用户100个交互的item的注意力权重 (1024,100)
        if self.dropout is not None:
            attention = self.dropout(attention)
        output = torch.bmm(attention.unsqueeze(1), history_emb).squeeze(1)  # (1024,64)
        user_init_emb = self.W_agg(output)  # (1024,64)
        return user_init_emb

    def get_user_emb(self,ui, user_fea_emb,):
        for i in range(self.head_num):
            # key user的选择
            # slice_list = torch.randint(0, self.supp_user_num, (self.sample_num,1)).squeeze().to( self.device)  # (num_users, 500)
            # slice_list = torch.cat((torch.arange(self.sample_num//2),torch.arange(len(self.supp_users)-250, len(self.supp_users))),dim=0).to(int)
            # slice_list = torch.arange(self.sample_num * 2)
            # sample_index = torch.stack([slice_list for _ in range(x.size(0))])
            sample_index = torch.randint(0, self.supp_user_num, (ui.size(0), self.sample_num)).to(self.device)
            sample_users = self.supp_users[sample_index]  # (1024,500)
            sample_user_emb = self.tgt_model.user_embedding[sample_users]  # (1024,500,64)

            gat_output_i = self.GAT_unit[i](user_fea_emb, sample_user_emb)  # (1024,64)
            if i == 0:
                gat_output = gat_output_i
            else:
                gat_output = torch.cat([gat_output, gat_output_i], dim=1)

        user_emb = self.w_out(gat_output)  # (batchsize,dim)
        return user_emb

    def forward(self, x, edge_UI=None, edge_IU=None, src_his=None, src_hl=None, tgt_his=None, tgt_hl=None, alpha=None, train=False):
        user_id = x[:, 0]  # (1024,)
        item_id = x[:, 1]  # (1024,)
        batch = x.size(0)
        sim_pre = None
        src_domain_output = None
        tgt_domain_output = None
        output_t = None
        # mask = torch.arange(src_his.size(1))[None, :].to(self.device) # (1,100)
        # mask = mask < src_hl[:, None] #(1024,100)
        # history_emb = self.src_model.item_embedding[src_his] # (1024,100,64)
        # history_emb[~mask] = torch.zeros(self.src_model.item_embedding.size(1)).to(self.device) # (1024,100,64)
        # user_init_emb = torch.sum(self.src_model.item_embedding[src_his], dim=1) # (1024,64)
        # user_init_emb /= src_hl[:, None].float()  # (1024,64)
        # test = self.src_model.user_embedding[user_id]
        # 改为注意力机制
        user_src_fea = self.user_fea_encode(user_id,src_his,src_hl,self.src_model)
        user_emb = self.get_user_emb(x,user_src_fea)  # (batch,dim)
        item_emb = self.tgt_model.item_embedding[item_id]   # (batch,dim)

        # if train:

        user_tgt_fea = self.user_fea_encode(user_id, tgt_his, tgt_hl, self.tgt_model)
        hybrid_tgt_emb = self.get_user_emb(x, user_tgt_fea)  # (batch,dim)

        if train:
            sim_pre = F.cosine_similarity(user_emb.unsqueeze(1).repeat(1,batch,1),hybrid_tgt_emb.unsqueeze(0).repeat(batch,1,1),dim=2) # (batch.batch)
            sim_pre = sim_pre/0.07  # (batch,batch)

            # 域分类器
            reverse_src_fea = ReverseLayerF.apply(user_emb, alpha)  # (1024,64)
            src_domain_output = self.domain_classifier(reverse_src_fea)  # (batch,2)

            reverse_tgt_fea = ReverseLayerF.apply(hybrid_tgt_emb, alpha)  # (batch,64)
            tgt_domain_output = self.domain_classifier(reverse_tgt_fea)  # (batch,2)

        x_t = torch.cat([hybrid_tgt_emb, item_emb], dim=1)  # (batchsize,128)
        x1_t = torch.tanh(self.l1(x_t))  # (batchsize,dim)
        x2_t = torch.tanh(self.l2(x1_t))  # (batchsize,dim)
        x3_t = self.l3(x2_t).reshape(-1)  # (batchsize,)
        output_t = x3_t

        # nn法
        # x = torch.cat([user_emb, item_emb], dim=1)  # (batchsize,128)
        # x1 = torch.tanh(self.l1(x))  # (batchsize,dim)
        # x2 = torch.tanh(self.l2(x1))  # (batchsize,dim)
        # x3 = self.l3(x2).reshape(-1)   # (batchsize,)
        # output = x3

        # MF法
        output =torch.sum(user_emb*item_emb,1)
        x2 = user_emb*item_emb # (batch,dim)
        # 保存embedding
        # 画图用
        # x = torch.cat([user_emb, item_emb], dim=1)  # (batchsize,128) # (batchsize,64)
        # x1 = torch.tanh(self.l4(x))   # (batchsize,64)
        # output = self.l3(x1).reshape(-1)

        out_emb_s = x2
        out_emb_t = x2_t
        return output, sim_pre, src_domain_output, tgt_domain_output, output_t, out_emb_s,out_emb_t

    def masked_softmax(self, X, mask):
        # use the following softmax to avoid nans when a sequence is entirely masked
        X = X.masked_fill_(mask, 0)
        e_X = torch.exp(X)
        return e_X / (e_X.sum(dim=1, keepdim=True) + 1.e-12)

    def embedding_lookup(self, x):
        return self.user_embedding[x]

    def embedding_lookup_supp(self, user_id):
        return self.user_embedding[user_id]

    def embedding_lookup_que(self, user_id, history, history_len):
        mask = torch.arange(history.size(1))[None, :].to(self.device)
        mask = mask < history_len[:, None]
        history_emb = self.item_embedding[history]
        history_emb[~mask] = torch.zeros(self.item_embedding.size(1)).to(self.device)
        user_init_emb = torch.sum(self.item_embedding[history], dim=1)
        user_init_emb /= history_len[:, None].float()

        for i in range(self.head_num):
            sample_index = torch.arange(0, self.supp_user_num).unsqueeze(0).repeat(user_id.size(0), 1)
            sample_users = self.supp_users[sample_index]
            sample_user_emb = self.user_embedding[sample_users]
            gat_output_i = self.GAT_unit[i](user_init_emb, sample_user_emb)
            if i == 0:
                gat_output = gat_output_i
            else:
                gat_output = torch.cat([gat_output, gat_output_i], dim=1)
        user_emb = self.w_out(gat_output)

        return user_emb

    def get_attns(self, user_id, history, history_len):
        mask = torch.arange(history.size(1))[None, :].to(self.device)
        mask = mask < history_len[:, None]
        history_emb = self.item_embedding[history]
        history_emb[~mask] = torch.zeros(self.item_embedding.size(1)).to(self.device)
        user_init_emb = torch.sum(self.item_embedding[history], dim=1)
        user_init_emb /= history_len[:, None].float()

        for i in range(self.head_num):
            sample_index = torch.arange(0, self.supp_user_num).unsqueeze(0).repeat(user_id.size(0), 1)
            sample_users = self.supp_users[sample_index]
            sample_user_emb = self.user_embedding[sample_users]
            gat_attn_i = self.GAT_unit[i].get_attn(user_init_emb, sample_user_emb)
            gat_attn_i = gat_attn_i.unsqueeze(2)
            if i == 0:
                gat_attns = gat_attn_i
            else:
                gat_attns = torch.cat([gat_attns, gat_attn_i], dim=2)

        return gat_attns

    def load_model(self, path):
        model_dict = torch.load(path)
        self.load_state_dict(model_dict)

    # def load_embedding(self, path):
    #     pretrained_dict = torch.load(path)
    #     model_dict = self.embedding_model.state_dict()
    #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #     model_dict.update(pretrained_dict)
    #     self.embedding_model.load_state_dict(model_dict)

    def load_embedding_nn(self, path):
        # pretrained_dict = torch.load(path)
        # model_dict = self.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.load_state_dict(model_dict, strict=False)
        model = GCMCModel(n_user=self.n_user, n_item=self.n_item, n_rating=self.n_rating, device=self.device).to(
            self.device)
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)
        return model
