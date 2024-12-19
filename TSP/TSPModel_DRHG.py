# 第一次embed前，就把point拼起来：5 - 128 - decoder
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from datetime import datetime


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.mode = model_params['mode']
        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.encoded_nodes = None

    def forward(self, state, selected_node_list, solution, current_step, point_couples=None, endpoint_mask=None):

        batch_size_V = state.data.size(0)
        problem_size = state.data.size(1)
        # state.data.size(): [batch, problem_size, 2]

        second_points_gather_id = point_couples[:, :, 1].unsqueeze(-1).repeat((1,1,2)) 
        second_points_data = torch.gather(state.data, dim=1, index=second_points_gather_id)
        data_both_point = torch.concat((state.data, second_points_data, endpoint_mask.unsqueeze(-1)), dim=2) #[batch, problem_size, 5]
        # print(state.data.size())

        if self.mode == 'train':
            # self, data, selected_node_list, batch_size_V, problem_size
            probs = self.decoder(self.encoder(data_both_point), selected_node_list, batch_size_V, problem_size)

            # shape: [ B, V - current_step]
            selected_student = probs.argmax(dim=1)  # shape: B
            selected_teacher = solution[:, current_step - 1]  # shape: B
            prob = probs[torch.arange(batch_size_V)[:, None], selected_teacher[:, None]].reshape(batch_size_V, 1)  # shape: [B, 1]

        elif self.mode == 'test':
            # if (decode_method == 'greedy' and repair == False):
            if current_step <= 1:
                self.encoded_nodes = self.encoder(data_both_point)

            probs = self.decoder(self.encoded_nodes, selected_node_list, batch_size_V, problem_size)
            # probs shape: [ B * k , V - current_step]
            # selected_node_list shape [ B * k, current_step]

            selected_student = probs.argmax(dim=1)  # shape:  B * k
            selected_teacher = selected_student  # shape:  B * k
            prob = 1

        else: raise NotImplementedError()

        return selected_teacher, prob, 1, selected_student



########################################
# ENCODER
########################################
class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        self.embedding = nn.Linear(5, embedding_dim, bias=True)


    def forward(self, data):
        out = self.embedding(data)
        return out


########################################
# DECODER
########################################
class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        decoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_first_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_2 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_3 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_4 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_5 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_6 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_7 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_8 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_9 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_10 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_11 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_12 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_13 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_14 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_15 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.embedding_last_node_16 = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.layers = nn.ModuleList([DecoderLayer(**model_params) for _ in range(decoder_layer_num)])

        self.Linear_REC = nn.Linear(embedding_dim, 1, bias=True)

    def _get_new_data(self, data, selected_node_list, prob_size, B_V):

        new_list = torch.arange(prob_size)[None, :].repeat(B_V, 1)
        # print('new_list.shape', new_list.shape)
        new_list_len = prob_size - selected_node_list.shape[1]  # shape: [B, V-current_step]
        # print('selected node list ', selected_node_list)
        # print('new_list_len', new_list_len)
        index_2 = selected_node_list.type(torch.long)

        index_1 = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2.shape[1])

        # 把已经选过的node的坐标转化为-2
        new_list[index_1, index_2] = -2
        # print(new_list[torch.gt(new_list, -1)].size())
        unselect_list = new_list[torch.gt(new_list, -1)].view(B_V, new_list_len)

        # ----------------------------------------------------------------------------

        new_data = data

        emb_dim = data.shape[-1]

        new_data_len = new_list_len

        index_2_ = unselect_list.repeat_interleave(repeats=emb_dim, dim=1)

        index_1_ = torch.arange(B_V, dtype=torch.long)[:, None].expand(B_V, index_2_.shape[1])
        # print(prob_size)
        # print(new_data_len)
        index_3_ = torch.arange(emb_dim)[None, :].repeat(repeats=(B_V, new_data_len))

        new_data_ = new_data[index_1_, index_2_, index_3_].view(B_V, new_data_len, emb_dim)

        return new_data_

    def _get_encoding(self,encoded_nodes, node_index_to_pick):
        # encoded_nodes.shape: (batch, problem, embedding)
        # node_index_to_pick.shape: (batch, pomo)

        batch_size = node_index_to_pick.size(0)
        pomo_size = node_index_to_pick.size(1)
        embedding_dim = encoded_nodes.size(2)

        gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
        # shape: (batch, pomo, embedding)

        picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
        # shape: (batch, pomo, embedding)
        return picked_nodes

    def forward(self, data, selected_node_list, batch_size_V, problem_size):
        batch_size_V = data.shape[0]  # B

        problem_size = data.shape[1]
        # print('problem_size',problem_size)
        new_data = data
        # selected_node_list's shape: [B, current_step]

        left_encoded_node = self._get_new_data(new_data, selected_node_list, problem_size, batch_size_V)

        first_and_last_node = self._get_encoding(new_data, selected_node_list[:,[0,-1]])
        embedded_first_node_ = first_and_last_node[:,0]
        embedded_last_node_ = first_and_last_node[:,1]

        embedded_last_no2_node_ = self.embedding_last_node_2(embedded_last_node_)
        embedded_last_no3_node_ = self.embedding_last_node_3(embedded_last_node_)
        embedded_last_no4_node_ = self.embedding_last_node_4(embedded_last_node_)
        embedded_last_no5_node_ = self.embedding_last_node_5(embedded_last_node_)
        embedded_last_no6_node_ = self.embedding_last_node_6(embedded_last_node_)
        embedded_last_no7_node_ = self.embedding_last_node_7(embedded_last_node_)
        embedded_last_no8_node_ = self.embedding_last_node_8(embedded_last_node_)
        embedded_last_no9_node_ = self.embedding_last_node_9(embedded_last_node_)
        embedded_last_no10_node_ = self.embedding_last_node_10(embedded_last_node_)
        embedded_last_no11_node_ = self.embedding_last_node_11(embedded_last_node_)
        embedded_last_no12_node_ = self.embedding_last_node_12(embedded_last_node_)
        embedded_last_no13_node_ = self.embedding_last_node_13(embedded_last_node_)
        embedded_last_no14_node_ = self.embedding_last_node_14(embedded_last_node_)
        embedded_last_no15_node_ = self.embedding_last_node_15(embedded_last_node_)
        embedded_last_no16_node_ = self.embedding_last_node_16(embedded_last_node_)

        embedded_first_node_ = self.embedding_first_node(embedded_first_node_)
        embedded_last_node_ = self.embedding_last_node(embedded_last_node_)



        out = torch.cat((embedded_first_node_.unsqueeze(1),embedded_last_no2_node_.unsqueeze(1),
                         embedded_last_no3_node_.unsqueeze(1),embedded_last_no4_node_.unsqueeze(1),
                         embedded_last_no5_node_.unsqueeze(1),embedded_last_no6_node_.unsqueeze(1),
                         embedded_last_no7_node_.unsqueeze(1), embedded_last_no8_node_.unsqueeze(1),
                         embedded_last_no9_node_.unsqueeze(1), embedded_last_no10_node_.unsqueeze(1),
                         embedded_last_no11_node_.unsqueeze(1), embedded_last_no12_node_.unsqueeze(1),
                         embedded_last_no13_node_.unsqueeze(1), embedded_last_no14_node_.unsqueeze(1),
                         embedded_last_no15_node_.unsqueeze(1), embedded_last_no16_node_.unsqueeze(1),
                         embedded_last_node_.unsqueeze(1)), dim=1)
        # inter = left_encoded_node_2
        inter = torch.cat((embedded_first_node_.unsqueeze(1), embedded_last_no2_node_.unsqueeze(1),
                           embedded_last_no3_node_.unsqueeze(1), embedded_last_no4_node_.unsqueeze(1),
                           embedded_last_no5_node_.unsqueeze(1), embedded_last_no6_node_.unsqueeze(1),
                           embedded_last_no7_node_.unsqueeze(1), embedded_last_no8_node_.unsqueeze(1),
                           embedded_last_no9_node_.unsqueeze(1), embedded_last_no10_node_.unsqueeze(1),
                           embedded_last_no11_node_.unsqueeze(1), embedded_last_no12_node_.unsqueeze(1),
                           embedded_last_no13_node_.unsqueeze(1), embedded_last_no14_node_.unsqueeze(1),
                           embedded_last_no15_node_.unsqueeze(1), embedded_last_no16_node_.unsqueeze(1),
                           embedded_last_node_.unsqueeze(1),left_encoded_node), dim=1)
        
        layer_count = 0
        for layer in self.layers:
            out,inter = layer(out, inter)
            layer_count += 1
        out = inter # [batch, reminding_nodes_number + 2, embedding_dim]
        # print(out.size()) # [batch, 16 + reminding_nodes_number + 2, embedding_dim ]
        # raise(ValueError('stop'))

      
        out = self.Linear_REC(out).squeeze(-1)  # shape: [B*(V-1), reminding_nodes_number + 2, embedding_dim ]
        out[:, 0:17] = out[:, 0:17] + float('-inf')

        props = F.softmax(out, dim=-1)  # shape: [B, remind_nodes_number]
        props = props[:, 17:]


        index_small = torch.le(props, 1e-5)
        props_clone = props.clone()
        props_clone[index_small] = props_clone[index_small] + torch.tensor(1e-7, dtype=props_clone[index_small].dtype)  # 防止概率过小
        props = props_clone

        new_props = torch.zeros(batch_size_V, problem_size)
        # shape: [B*(V-1), problem_size], 作用是把 props的概率填充到new_props里, props的里概率元素与未访问节点的概率一一对应.
        # 构造torch高级索引
        index_1_ = torch.arange(batch_size_V, dtype=torch.long)[:, None].expand(batch_size_V, selected_node_list.shape[1])  # shape: [B*(V-1), n]
        index_2_ = selected_node_list.type(torch.long)  # shape: [B*(V-1), n]
        new_props[index_1_, index_2_] = -2
        index = torch.gt(new_props, -1).view(batch_size_V, -1)
        # 把 props的概率填充到new_props里, props的里概率元素与未访问节点的概率一一对应.

        new_props[index] = props.ravel()

        return new_props

                    

class DecoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.multi_head_combine_2 = nn.Linear(head_num * qkv_dim, embedding_dim)
        # self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)

        self.feedForward = Feed_Forward_Module(**model_params)
        self.feedForward_2 = Feed_Forward_Module(**model_params)
        # self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input2), head_num=head_num)
        v = reshape_by_heads(self.Wv(input2), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v)  # shape: (B, n, head_num*key_dim)

        multi_head_out = self.multi_head_combine(out_concat)  # shape: (B, n, embedding_dim)

        out1 = input1 + multi_head_out
        out2 = self.feedForward(out1)
        out3 = out1 +  out2
        # out3 = multi_head_out
        ################################################################
        ################################################################
        q_2 = reshape_by_heads(self.Wq_2(input2), head_num=head_num)
        k_2 = reshape_by_heads(self.Wk_2(out3), head_num=head_num)
        v_2 = reshape_by_heads(self.Wv_2(out3), head_num=head_num)
        # k = reshape_by_heads(input2, head_num=head_num)
        out_concat_2 = multi_head_attention(q_2, k_2, v_2)  # shape: (B, n, head_num*key_dim)
        multi_head_out_2 = self.multi_head_combine_2(out_concat_2)  # shape: (B, n, embedding_dim)

        out1_2 = input2 + multi_head_out_2
        out2_2 = self.feedForward_2(out1_2)

        out3_2 = out1_2 +  out2_2

        return out3, out3_2
        # shape: (batch, problem, EMBEDDING_DIM)


def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    # q shape: (B, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (B, head_num, n, key_dim)
    # rank2_ninf_mask.shape: (B, problem)
    # rank3_ninf_mask.shape: (B, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))  # shape: (B, head_num, n, n)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    # if rank2_ninf_mask is not None:
    #     score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    # if rank3_ninf_mask is not None:
    #     score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)  # shape: (B, head_num, n, n)

    out = torch.matmul(weights, v)  # shape: (B, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)  # shape: (B, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)  # shape: (B, n, head_num*key_dim)

    return out_concat



class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))
