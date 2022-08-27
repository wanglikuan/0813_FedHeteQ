import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
import numpy as np
import time
import copy
import sys
from torch.utils.data import DataLoader
import torch.nn.functional as F

def flatten(t):
    return t.reshape(t.shape[0], -1)

class Qinv_net(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(Qinv_net, self).__init__()
        self.predict = torch.nn.Linear(n_input, n_output, bias=False)  # 线性输出层
    def forward(self, x):
        x = self.predict(x)
        return x

class clientAVG(Client):
    def __init__(self, device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                 local_steps):
        super().__init__(device, numeric_id, train_slow, send_slow, train_data, test_data, model, batch_size, learning_rate,
                         local_steps)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)


        self.Hetelabel_matrix=[[0,1,2,3,4,5,6,7,8,9],\
            [8,9,5,0,4,3,6,7,1,2],\
                [6,0,2,7,5,9,4,3,8,1],\
                    [1,5,2,0,8,4,7,3,6,9],\
                        [6,0,1,2,3,7,4,5,9,8],\
                            [4,0,2,8,5,3,1,7,6,9],\
                                [3,6,4,0,7,9,5,1,2,8],\
                                    [8,2,6,5,3,7,4,0,9,1],\
                                        [9,1,3,7,4,0,5,6,2,8],\
                                            [2,5,1,3,6,7,8,0,9,4]]

        self.public_data_loader = None
        self.iter_public_data_loader = None

        self.model_layer_list = list(dict(self.model.named_parameters()).keys())
        self.need_frozen_list_f = self.model_layer_list[:-1]
        self.need_frozen_list_Q = self.model_layer_list[-1:]
        self.local_public_logits = []
        self.local_private_logits = []
        self.local_private_label = []
        self.local_origin_yalign_hook = []
        self.Qinv_net = Qinv_net(n_input=10, n_output=10) 
        self.hidden = {}

    def get_publice_data(self, public_data):
        self.public_data_loader = DataLoader(public_data, 495, drop_last=True) # batch size 可更改
        self.iter_public_data_loader = iter(self.public_data_loader)

    def savemodel(self):
        torch.save(self.model.state_dict(), 'fedavg_net_client{}.pt'.format(self.id))

    def label_2_Hete(self, inputy):
        for yid,true_label in enumerate(inputy) :
            inputy[yid] = self.Hetelabel_matrix[self.id][true_label]
        return inputy

    def test_accuracy_Q(self):
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        
        with torch.no_grad():
            for x, y in self.testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = self.label_2_Hete(y) ###########
                y = y.to(self.device)
                output = self.model(x)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

        # self.model.cpu()
        
        return test_acc, test_num

    def train_accuracy_and_loss_Q(self):
        # self.model.to(self.device)
        self.model.eval()

        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in self.trainloaderfull:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = self.label_2_Hete(y) ###########
            y = y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        # self.model.cpu()

        return train_acc, loss, train_num

    def get_next_train_batch_public(self):
        try:
            # Samples a new batch for persionalizing
            (x, y) = next(self.iter_public_data_loader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_public_data_loader = iter(self.public_data_loader)
            (x, y) = next(self.iter_public_data_loader)

        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)

        return x, y

    def get_public_logits(self, public_logits):
        self.local_public_logits = public_logits

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def predict(self):
        self.model.eval()
        self.local_private_logits= []
        self.local_private_label = []
        self.local_origin_yalign_hook = []
############ 准备hook ############
        modules = dict([*self.model.named_modules()])
        layer = modules.get('fc', None)
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
#######################
        #bs = 32
        #dataarray = dataarray.astype(np.float32)
        with torch.no_grad():
            for step in range(1): # local step 可更改
                x, y = self.get_next_train_batch_public()
                y = self.label_2_Hete(y)
#######################################
                self.hidden.clear()
######################################
                logit = self.model(x)
#############################
                hidden = self.hidden[x.device]
                self.hidden.clear()
#############################
                #to do# 加入softmax层
                #Tsoftmax = nn.Softmax(dim=1)
                #加入温度系数T#
                #output_logit = Tsoftmax(logit.float()/1.0)
                #
                #output_logit = Tsoftmax(logit)
                #
                #self.local_private_logits.append(output_logit.cpu().numpy())
                self.local_origin_yalign_hook.append(hidden.cpu().numpy())
                self.local_private_logits.append(logit.cpu().numpy())
        
                self.local_private_label.append(y.cpu().numpy())

        self.local_private_label = np.concatenate(self.local_private_label)    
        
        self.local_private_logits = np.concatenate(self.local_private_logits)
        self.local_origin_yalign_hook = np.concatenate(self.local_origin_yalign_hook)
        #return self.local_private_logits

    def get_truelabel(self):
        self.model.eval()
        local_Q = copy.deepcopy(self.model.Linear_Q)
        local_private_logits = copy.deepcopy(self.local_private_logits)
        local_private_logits = torch.from_numpy(local_private_logits)
        local_private_logits = local_private_logits.to(self.device)
        print('local_private_logits.size: ', local_private_logits.size())
        # with torch.no_grad():
        #     output = torch.matmul(local_private_logits, local_Q.weight.t())
        ######### 使用 Linear_Q的weight 与 logits 求 要对齐的label ###############################
        #local_Qinv = torch.linalg.inv(local_Q.weight) #Q的逆矩阵 
        local_Qinv = torch.inverse(local_Q.weight)
        
        local_Qinv_layer = nn.Linear(10, 10, bias=False)
        local_Qinv_layer.weight.data = local_Qinv
        ### 计算开始 ###
        with torch.no_grad():
            output = local_Qinv_layer(local_private_logits)
        ########################################
        return output #size: [495,10]

    def train(self):

        #print(self.model_layer_list)
        #print(self.need_frozen_list_f)
        #print(self.need_frozen_list_Q)
        ####### 冻结 Q 参数 ##################################
        for param in self.model.named_parameters():
            if param[0] in self.need_frozen_list_Q:
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True
        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        #####################################################

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()
        
        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            if self.train_slow:
                time.sleep(0.1 * np.abs(np.random.rand()))
            x, y = self.get_next_train_batch()
            y = self.label_2_Hete(y)
            self.optimizer.zero_grad()
            output = self.model(x)
            loss = self.loss(output, y)
            #print('cross-entropy loss: ', loss)
            loss.backward()
            self.optimizer.step()
        # self.model.cpu()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_Q(self):
        self.model.train()
######################## 知识蒸馏 #################################################################
        ####### 冻结 F 参数 ##################################
        for param in self.model.named_parameters():
            if param[0] in self.need_frozen_list_Q:
                param[1].requires_grad = True
            else:
                param[1].requires_grad = False

        self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        #####################################################
        total_loss = None
        bs = 45
        for ind in range(0,len(self.local_private_logits),bs):

            origin_yalign = self.local_origin_yalign_hook[ind:(ind+bs)]
            origin_yalign = torch.from_numpy(origin_yalign)
            origin_yalign = origin_yalign.to(self.device)

            labels = self.local_private_label[ind:(ind+bs)]

            data = self.local_private_logits[ind:(ind+bs)]
            data = torch.from_numpy(data)
            data = data.to(self.device)
            y = self.local_public_logits[ind:(ind+bs)]

            self.optimizer.zero_grad()

            ######
            #output = torch.matmul(data, self.model.Linear_Q.weight.t())
            ######### 使用 Linear_Q的weight 与 logits 求 要对齐的label ###############################
            local_Q = copy.deepcopy(self.model.Linear_Q)
            #local_Qinv = torch.linalg.inv(local_Q.weight) #Q的逆矩阵
            #print('client id:', self.id, '    240 Q: ', local_Q.weight) 
            #local_Qinv = torch.pinverse(local_Q.weight)
            #local_Qinv = F.softmax(local_Qinv, dim=-1)

            local_Qinv_layer_opt = torch.optim.SGD(self.Qinv_net.parameters(), lr=(self.learning_rate/10.0)) ## Q 逆矩阵的优化器
            self.Qinv_net.predict.weight.data = local_Q.weight.data
            local_Qinv_layer_opt.zero_grad()

            ### 计算开始 ###
            output = self.Qinv_net(y)
            #################################观察算出的 yalign 与原本的 yalign 的差距#######
            #print('client id: ', self.id,  '    origin yalign: ', origin_yalign[0:9])
            #print('client id: ', self.id, '    output yalign: ', output[0:9])
            ######
            #print('client id: ', self.id,  '    data: ', data[0:4], '    labels: ', labels[0:4])
            #####
            kl_loss = nn.KLDivLoss(reduction="batchmean")
            output = F.log_softmax(output, dim=-1)
            data = F.softmax(data, dim=-1)
            loss = kl_loss(output, data)

            #criterion = nn.MSELoss(reduction='mean')
            #loss = criterion(output, y)
            if total_loss == None:
                total_loss = loss
            else:
                total_loss += loss
            #print('client id: ', self.id, '    kl-loss: ', loss)
            loss_L2 = torch.norm(self.Qinv_net.predict.weight, p=2)
            loss = 0.9 * loss + 0.1 * loss_L2

            loss.backward()
            local_Qinv_layer_opt.step()

            ############## 用更新后的 self.Qinv_net.predict.weight 再算逆矩阵; 更新 self.model.Linear_Q.weight ####################
            #new_local_Q = torch.linalg.inv(self.Qinv_net.predict.weight)
            new_local_Q = self.Qinv_net.predict.weight.data
            #new_local_Q = F.softmax(new_local_Q.float()/0.1, dim=-1)
            with torch.no_grad():
                if self.id == 0:
                    new_local_Q = torch.eye(10, dtype=torch.float32, requires_grad=True).to(self.device)
                self.model.Linear_Q.weight.data = new_local_Q

        #new_local_Q = F.softmax(new_local_Q/0.001, dim=-1)
        #with torch.no_grad():
        #    if self.id == 0:
        #        new_local_Q = torch.eye(10, dtype=torch.float32, requires_grad=True).to(self.device)
        #    self.model.Linear_Q.weight.data = new_local_Q

            #exit(0)
        print('client id: ', self.id, '    Q: ', new_local_Q, '    Q.dtype: ', new_local_Q.dtype)
        print('client id: ', self.id, '    total-kl-loss: ', total_loss)

#########################################################################################
