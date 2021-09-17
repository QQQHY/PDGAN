import _pickle as cPickle
import numpy as np
import utils as ut
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

class KERNEL(nn.Module):
    def __init__(self, itemNum, emb_dim, lamda, param=None, initdelta=0.05):
        super(KERNEL, self).__init__()
        self.itemNum = itemNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param  # embedding parameters: user embedding, item embedding, item bias
        self.initdelta = initdelta  # range of the uniform distribution when initializing params
        self.l_params = []

        if self.param is None:
            print("DPP random initialize embedding") 
            self.item_embeddings = nn.Embedding(self.itemNum, self.emb_dim)
            nn.init.uniform_(self.item_embeddings.weight, -self.initdelta, self.initdelta)
        else:   # otherwise, load pre-calculated values
            print("DPP using pretrain embedding")
            self.item_embeddings = nn.Embedding(self.itemNum, self.emb_dim)            
            self.item_embeddings.weight.data.copy_(torch.from_numpy(self.param[0]))

        self.l_params = [self.item_embeddings]    

    
    def forward(self, i_pos, i_neg):

        # i_pos n items
        self.i_pos_embedding = self.item_embeddings(torch.LongTensor(i_pos))# n*k matrix     
        self.Ln_pos = torch.matmul(self.i_pos_embedding, self.i_pos_embedding.t())

        # i_neg n items
        self.i_neg_embedding = self.item_embeddings(torch.LongTensor(i_neg))  # n*k matrix
        self.Ln_neg = torch.matmul(self.i_neg_embedding, self.i_neg_embedding.t())

        # calculate L
        self.L = torch.matmul(self.item_embeddings.weight, self.item_embeddings.weight.t())

        # Loss
        l_loss = \
            -torch.log(torch.det(self.Ln_pos + torch.mul(torch.exp(torch.Tensor([-6.0])),torch.eye(self.Ln_pos.shape[1]) ))) + \
             torch.log(torch.det(self.Ln_neg + torch.mul(torch.exp(torch.Tensor([-6.0])),torch.eye(self.Ln_neg.shape[1]) )))
        return l_loss
        
    def test(self, i):
        # for test stage, self.i: size of n
        # calculate L[n]
        i_embedding = self.item_embeddings(torch.LongTensor(i))  # n*k matrix
        Ln = torch.matmul(i_embedding, i_embedding.t())
        L = torch.matmul(self.item_embeddings.weight, self.item_embeddings.weight.t())
        p = torch.det(Ln + torch.mul(torch.exp(torch.Tensor([-6.0])) ,torch.eye(Ln.shape[1])/ torch.det (L + torch.eye(self.itemNum)) )) #(1)
        return p

        

    def save_model(self, filename):
        cPickle.dump(self.l_params, open(filename, 'wb'))

    


    def sample(self, scores, K, theta, train_examples):
        # change tensor self.L to numpy L

        alpha = theta/(2*(1-theta)) # alpha theta

        L = np.matmul(self.item_embeddings.weight.detach().cpu().numpy(), self.item_embeddings.weight.detach().cpu().numpy().T) # (6)
        
        if scores is not None:           
            diag = np.diag(np.squeeze(np.exp(alpha * scores)))
            L = np.matmul(np.matmul(diag, L), diag)

        C = defaultdict(list)
        D = np.zeros((self.itemNum,))
        D = D + L.diagonal()
        # is train_examples is given, then sampling should exclude train_examples
        if train_examples is not None:
            D[train_examples] = 0
        j = np.argmax(D)
        Y = [j]
        Z = set(range(self.itemNum))

        for k in range(1, K):
            Z = Z - set(Y)
            for i in Z:
                if len(C[i]) == 0 or len(C[j]) == 0:
                    ei = L.item(j, i) / (D[j] ** 0.5)
                else:
                    ei = (L.item(j, i) - np.sum(np.array(C[i]) * np.array(C[j]))) / (D[j] ** 0.5)
                C[i].append(ei)
                D[i] = D[i] - ei ** 2
            ii = np.array(list(Z))
            jj = np.argmax(D[ii])
            j = ii[jj]
            Y.append(j)
        return Y

def test(DPP_Kernel,workdir,CATE_NUM):
    #########################################################################################
    # Test DPP Kernel
    #########################################################################################

    DPP_Kernel.eval()
    file_cate = workdir + 'ml-100k/u.item'
    cate = ut.get_category(file_cate)

    # testing sampled 10 items from trained kernel
    sampled_items = DPP_Kernel.sample(None, 10, 0.8, None)
    #print("sampled_items",sampled_items)
    p = DPP_Kernel.test(sampled_items)
    print("p",p)
    diversity = ut.diversity_by_category(sampled_items, cate, CATE_NUM)
    print('diversity of sampled 10 items is:' + str(diversity))
    """
                wandb.log({
                    "loss":loss,
                    "p":p,
                    "diversity":diversity,
                })
    """

if __name__ == '__main__':


    
    #########################################################################################
    # Hyper-parameters
    #########################################################################################
    EMB_DIM = 30
    USER_NUM = 943
    ITEM_NUM = 1683
    CATE_NUM = 18
    BATCH_SIZE = 16
    INIT_DELTA = 0.05
    learning_rate=0.01 
    train_epoch=1

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    workdir = 'ml-100k/'
    #########################################################################################
    # Load or Train DPP Kernel
    #######################################################################################

    DPP_Kernel = KERNEL(ITEM_NUM, EMB_DIM, lamda=0.01, param=None, initdelta=INIT_DELTA)

    pos_train_file = workdir + 'pos_diversity_train.txt'
    neg_train_file = workdir + 'neg_diversity_train.txt'
    
    #DPP Training  

    # Load training data
    pos_train_samples = []
    with open(pos_train_file) as fin:
        for line in fin:
            line = line.split()
            items = list(map(int, line[1:]))
            pos_train_samples.append(items)

    neg_train_samples = []
    with open(neg_train_file) as fin:
        for line in fin:
            line = line.split()
            items = list(map(int, line[1:]))
            neg_train_samples.append(items)
    
    for param_tensor in DPP_Kernel.state_dict():
        print(param_tensor,'\t',DPP_Kernel.state_dict()[param_tensor].size())
    # item_embeddings.weight   torch.Size([1683, 30])
    optimizer = torch.optim.SGD(DPP_Kernel.parameters(), lr = learning_rate)

    # training
    for l_epoch in range(train_epoch):
        print('training l_epoch:', l_epoch)

        for i in range(len(pos_train_samples)):
            DPP_Kernel.train()

            pos_input_items = pos_train_samples[i]
            neg_input_items = neg_train_samples[i]
            loss = DPP_Kernel.forward(i_pos=pos_input_items,i_neg=neg_input_items)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()               
    
            if (i%1000==0):
                print("i",i,"loss",loss)
                test(DPP_Kernel,workdir,CATE_NUM)

                


    # Test DPP Kernel
    test(DPP_Kernel,workdir,CATE_NUM)

    DPP_Kernel.save_model(workdir + "dpp_param_pairwise_EMB-DIM={}.pkl".format(EMB_DIM))


