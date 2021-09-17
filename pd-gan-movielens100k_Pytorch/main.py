import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pdb
import pickle
import _pickle as cPickle
import torch.nn.functional as F
from models import *
from dpp_model_pairwise import KERNEL
import utils as ut
import multiprocessing
import argparse
from config import movielens_config


import wandb
cores = multiprocessing.cpu_count()



#########################################################################################
# Hyper-parameters
#########################################################################################

parser = argparse.ArgumentParser(description='PD-GAN')

# training
parser.add_argument('--epochs',       type=int,    default=300,   help='number of epochs to train')
parser.add_argument('--epochs_g',     type=int,    default=1,     help='number of G epochs to train')
parser.add_argument('--epochs_d',     type=int,    default=1,     help='number of D epochs to train')
parser.add_argument('--batch_size',   type=int,    default=16,    help='input batch size for training (default: 16)')

# opt
parser.add_argument('--opt',          type=str,    default='SGD', help='SGD|SGDM|Adam')
parser.add_argument('--eta_G',        type=float,  default=1e-2,  help='learning rate G')
parser.add_argument('--eta_D',        type=float,  default=1e-2,  help='learning rate D')

# PD-GAN
parser.add_argument('--emb_dim',      type=int,    default=30,    help='embedding dimension')
parser.add_argument('--k',            type=int,    default=10,    help='sample top K')
parser.add_argument('--theta',        type=float,  default=0.90,  help='trade-off parm')
parser.add_argument('--INIT_DELTA',   type=float,  default=0.05,  help='the bound of the uniform distribution')
parser.add_argument('--seed',         type=int,    default=1,     help='random seed')
parser.add_argument('--device',       type=str,    default='0')
parser.add_argument('--device_ids',   type=list,   default=[0])
parser.add_argument('--dataset',      type=str,    default="Movielens-100K")
parser.add_argument('--wandb_flag',   type=bool,   default=True)


pdgan_config = parser.parse_args()

EPOCH      = pdgan_config.epochs
epochs_d   = pdgan_config.epochs_d
epochs_g   = pdgan_config.epochs_g
BATCH_SIZE = pdgan_config.batch_size

opt        = pdgan_config.opt
eta_G      = pdgan_config.eta_G 
eta_D      = pdgan_config.eta_D 

EMB_DIM    = pdgan_config.emb_dim
k          = pdgan_config.k
theta      = pdgan_config.theta

INIT_DELTA = pdgan_config.INIT_DELTA

seed       = pdgan_config.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ['CUDA_VISIBLE_DEVICES'] = pdgan_config.device
device     = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device_ids = pdgan_config.device_ids

dataset =  pdgan_config.dataset
if dataset == "Movielens-100K":
    data_config = movielens_config

USER_NUM = data_config.USER_NUM # user size  943 for movielens
ITEM_NUM = data_config.ITEM_NUM # items number  1683 for movielens
CATE_NUM = data_config.CATE_NUM # categories number  18 for movielens
workdir  = data_config.workdir

wandb_flag = pdgan_config.wandb_flag
if wandb_flag:
    wandb.init(project="PD-GAN")
    wandb.config.update(pdgan_config)

torch.set_default_tensor_type(torch.FloatTensor)


all_items = set(range(ITEM_NUM))

filename = "batchsize={}_opt={}_lr={}_emb={}_theta={}_seed={}_dataset={}".\
    format( BATCH_SIZE,  opt   ,eta_G,EMB_DIM,theta,  seed,   dataset,)

#########################################################################################
# Load data
#########################################################################################

user_pos_train = ut.get_train_test_data((workdir + 'movielens-100k-train.txt'))

user_pos_test = ut.get_train_test_data(workdir + 'movielens-100k-test.txt')

item_cate = ut.get_category(workdir + 'ml-100k/u.item')

div_train_data = ut.get_div_train_data(workdir + 'pos_diversity_train.txt')

#########################################################################################
# Initialize generator and discriminator
#########################################################################################
print("load model...")

with open(workdir + "irgan_generator_{}.pkl".format(EMB_DIM), "rb") as f:
    G_param = pickle.load(f, encoding='latin1')

with open(workdir + "dpp_param_pairwise_{}.pkl".format(EMB_DIM), "rb") as f:
    dpp_param = pickle.load(f, encoding='latin1')

G_user_embeddings = torch.autograd.Variable(torch.tensor(G_param[0]).to(device), requires_grad=True)
G_item_embeddings = torch.autograd.Variable(torch.tensor(G_param[1]).to(device), requires_grad=True)
G_item_bias = torch.autograd.Variable(torch.zeros([ITEM_NUM]).to(device), requires_grad=True)   

D_user_embeddings = torch.autograd.Variable((torch.ones(USER_NUM, EMB_DIM)).to(device), requires_grad=True)
D_item_embeddings = torch.autograd.Variable((torch.ones(ITEM_NUM, EMB_DIM)).to(device), requires_grad=True)
D_item_bias = torch.autograd.Variable((torch.zeros(ITEM_NUM)).to(device), requires_grad=True)

DG_init = [D_user_embeddings, D_item_embeddings]
for params in DG_init:
    torch.nn.init.uniform_(params, a=-1*INIT_DELTA, b=INIT_DELTA)

criterion = torch.nn.DataParallel(L2Loss(), device_ids=device_ids)
criterion.to(device)

#model def
generator = torch.nn.DataParallel(Generator(G_user_embeddings, G_item_embeddings, G_item_bias), device_ids=device_ids)
generator.to(device)
discriminator = torch.nn.DataParallel(Discriminator(D_user_embeddings, D_item_embeddings, D_item_bias), device_ids=device_ids)
discriminator.to(device)
dpp = torch.nn.DataParallel(KERNEL(ITEM_NUM, EMB_DIM, lamda=0., param=dpp_param, initdelta=INIT_DELTA), device_ids=device_ids)
dpp.to(device)

#params opt
G_params = list(generator.parameters()) + [G_user_embeddings, G_item_embeddings, G_item_bias]
D_params = list(discriminator.parameters()) + [D_user_embeddings, D_item_embeddings, D_item_bias]
optimizer_G = torch.optim.SGD(G_params, lr = eta_G)
optimizer_D = torch.optim.SGD(D_params, lr = eta_D)
    
lamda = 0.1 / BATCH_SIZE 

dis_log = open(workdir + 'dis_log_' + filename + '.txt', 'w')
gen_log = open(workdir + 'gen_log_' + filename + '.txt', 'w')
    
def test_one_user(user_top_k, u):
    r = []
    for i in user_top_k:
        try:
            if i in user_pos_test[u]:
                r.append(1)
            else:
                r.append(0)
        except KeyError:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ut.ndcg_at_k(r, 3)
    ndcg_5 = ut.ndcg_at_k(r, 5)
    ndcg_10 = ut.ndcg_at_k(r, 10)
    diversity_3 = ut.diversity_by_category(user_top_k[:3], item_cate, CATE_NUM)
    diversity_5 = ut.diversity_by_category(user_top_k[:5], item_cate, CATE_NUM)
    diversity_10 = ut.diversity_by_category(user_top_k[:10], item_cate, CATE_NUM)
    
    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10, diversity_3, diversity_5, diversity_10])

def test(gen_model, dpp_model):
    result = np.array([0.] * 9)

    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    for u in test_users:
        try:
            user_top_k = gen_output(gen_model, dpp_model, u, user_pos_train[u])
            result += test_one_user(user_top_k, u)
        except KeyError:
            pass

    ret = result / test_user_num
    ret = list(ret)    
    return ret

def test_and_save(gen_model, dpp_model):
    result = []
    test_users = list(user_pos_test.keys())
    for u in test_users:        
        if(u%100==0):
            print("u = ",u)
        user_top_k = gen_output(gen_model, dpp_model, u, user_pos_train[u])
        result_of_one_user = test_one_user(user_top_k, u)
        result.append('\t'.join(str(x) for x in result_of_one_user))

    with open(workdir + filename + "_result-for-each-user.txt", 'w')as fout:
        fout.write('\n'.join(result))
    if wandb_flag:
        wandb.save(workdir + filename + "_result-for-each-user.txt")    

def main():


    initial_result = test(generator, dpp)

    # initial_result
    print("   P@03 : {0[0]:%}     P@05 : {0[1]:%}    P@10 : {0[2]:%}\nNDCG@03 : {0[3]:%}  NDCG@05 : {0[4]:%} NDCG@10 : {0[5]:%}\n  CC@03 : {0[6]:%}    CC@05 : {0[7]:%}   CC@10 : {0[8]:%} ".format(initial_result))

    # min-max training
    best = initial_result[1]

    for epoch in range(EPOCH):
        print('----------------- Min-Max Training Epoch = {} of {} -----------------'.format(epoch+1,EPOCH))
        if epoch >= 0:
            for d_epoch in range(epochs_d):
                for u in user_pos_train:
                    sampled_i = gen_output(generator, dpp, u, None) 
                    ground_sets = div_train_data[u]
                    for i in range(len(ground_sets)):
                        D_loss = discriminator(input_user= u , input_item = sampled_i, pred_data_label = torch.tensor(ground_sets[i])) 
                        optimizer_D.zero_grad()
                        D_loss.backward()
                        optimizer_D.step()
                print("\r[D Epoch %d/%d] [loss: %f]" %(d_epoch+1,epochs_d, D_loss.item()))

            for g_epoch in range(epochs_g):
                for u in user_pos_train:
                    sample = gen_output(generator, dpp, u, None) 
                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = discriminator.module.get_reward(u, sample)
                    reward = reward.clone().detach().cpu().numpy()
                    a = list()
                    a.append(reward.tolist())
                    reward = a
                    G_loss = generator(u, torch.tensor(sample), torch.tensor(reward))+ lamda * (criterion(G_user_embeddings) + criterion(G_item_embeddings) + criterion(G_item_bias))
                    
                    optimizer_G.zero_grad()
                    G_loss.backward()
                    optimizer_G.step()
                print("\r[G Epoch %d/%d] [loss: %f] [reward: %f]" %(g_epoch+1, epochs_g, G_loss.item(), reward[0]))

                result = test(generator, dpp)
                print("   P@03 : {0[0]:%}     P@05 : {0[1]:%}    P@10 : {0[2]:%}\nNDCG@03 : {0[3]:%}  NDCG@05 : {0[4]:%} NDCG@10 : {0[5]:%}\n  CC@03 : {0[6]:%}    CC@05 : {0[7]:%}   CC@10 : {0[8]:%} ".format(result))
                if wandb_flag:
                    wandb.log({
                    'p_3': result[0],
                    'p_5': result[1],
                    'p_10': result[2],
                    'ndcg_3': result[3],
                    'ndcg_5': result[4],
                    'ndcg_10': result[5],
                    'diversity_3': result[6],
                    'diversity_5': result[7],
                    'diversity_10': result[8],                    
                    'D_loss':D_loss.item(),
                    'G_loss':G_loss.item(),
                    'reward':reward[0],
                    })
                buf = '\t'.join([str(x) for x in result])
                gen_log.write(str(epoch) + '\t' + buf + '\n')
                gen_log.flush()

                p_5 = result[1]

                if p_5 > best:
                    print('Epoch : ',epoch,'  Best P@05 : ', p_5)
                    best = p_5                 
                    cPickle.dump(G_params, open(workdir + filename + ".pkl", 'wb'))
                    wandb.save("ml-100k/gan_generator_30.pkl") 
                print("\n")

    gen_log.close()

# generate top_k outputs
def gen_output(gen_model, dpp_model, u, train_examples):
    scores = generator.module.all_rating(u)
    scores = scores.detach_().cpu().numpy() 
    sampled_items = dpp_model.module.sample(scores, k, theta, train_examples) # dpp_model sample items according to score 
    return sampled_items

def stat_test():
    print()
    print("load model...")
    print()
    G_param = cPickle.load(open(workdir + filename + ".pkl", 'rb'), encoding='latin1')
    dpp_param = cPickle.load(open(workdir + "dpp_param_pairwise_{}.pkl".format(EMB_DIM), 'rb'), encoding='latin1')

    G_user_embeddings = torch.autograd.Variable(torch.tensor(G_param[0]).to(device), requires_grad=True)
    G_item_embeddings = torch.autograd.Variable(torch.tensor(G_param[1]).to(device), requires_grad=True)
    G_item_bias = torch.autograd.Variable(torch.tensor(G_param[2]).to(device), requires_grad=True)

    generator = torch.nn.DataParallel(Generator(G_user_embeddings, G_item_embeddings, G_item_bias), device_ids=device_ids)
    generator.to(device)
    dpp = torch.nn.DataParallel(KERNEL(ITEM_NUM, EMB_DIM, lamda=0., param=dpp_param, initdelta=INIT_DELTA), device_ids=device_ids)
    dpp.to(device) 

    test_and_save(generator, dpp)

if __name__ == '__main__':
    main()
    stat_test()