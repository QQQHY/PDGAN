"""
Configuration files
"""

class Config():
    def __init__(self):
        self.framework="Pytorch"

        self.epochs = 300 
        self.epochs_g = 1 
        self.epochs_d = 1 

        self.batch_size = 16 

        self.opt = "SGD"  
        self.emb_dim = 30
        self.eta_G = 1e-2
        self.eta_D = 1e-2

        self.INIT_DELTA = 0.05

        self.device = "0"
        self.device_ids = [0]

        self.k = 10
        self.theta =0.90

class BprConfig():
    def __init__(self):
        self.epochs = 200
        self.batch_size = 64
        self.eta = 1e-3   
        self.dir_path = "./data/"
        self.emb_dim = 5
        self.device = "cuda:0"
        self.weight_decay = 1e-5

class MovielensConfig():
    def __init__(self):
        self.USER_NUM = 943
        self.ITEM_NUM = 1683
        self.CATE_NUM = 18  
        self.workdir = 'ml-100k/'

bpr_config = BprConfig()
pdgan_config = Config() 
movielens_config = MovielensConfig()     