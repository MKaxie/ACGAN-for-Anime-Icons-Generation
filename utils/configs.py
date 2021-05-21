import os
import torch

class Configs():# 管理参数

    # parameters
    EPOCHS = 800 
    EPOCH_START = 0
    RESUME = True

    BATCH_SIZE = 4
    NUM_WORKERS = 8

    LRG = 0.02
    LRD = 0.02
    MOMENTUM = 0.9
    WEIGHT_DEACY = 1e-4
    
    GPU_MODE = True
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    DATA_PATH = r'data'
    SAVE_PATH = r'checkpoints/trainedmodels'
    CKECK_DIR = r'checkpoints'
    RESULT_IMGS = r'data/results'
    
    

    