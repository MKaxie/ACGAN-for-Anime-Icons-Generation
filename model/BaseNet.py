
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU, ReLU, Sigmoid
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d



class Generater(nn.Module): # 生成器
    def __init__(self,output=3,noize_dim=102,num_features=26):
        super(Generater,self).__init__()

        self.output_c = output
        self.n_features = num_features
        self.n_dim = noize_dim

        self.fc1 = nn.Sequential( # 
                                nn.Linear(self.n_features+self.n_dim,512),  # [1,128]      ---> [1,512]
                                nn.BatchNorm1d(512),
                                nn.ReLU(),

                                nn.Linear(512,1024),                        # [64,1]      ---> [1024,1]
                                nn.BatchNorm1d(1024),
                                nn.ReLU(),

                                nn.Linear(1024,256*12*12),                  # [1024,1]    ---> [256*12*12,1]
                                nn.BatchNorm1d(256*12*12),
                                nn.ReLU())

        self.deconv1 = nn.Sequential( # 
                                nn.ConvTranspose2d(256,128,4,2,1),          # [256,12,12] ---> [128,24,24]
                                nn.BatchNorm2d(128),
                                nn.ReLU(),

                                nn.ConvTranspose2d(128,64,4,2,1),           # [128,24,24] ---> [64,48,48]
                                nn.BatchNorm2d(64),
                                nn.ConvTranspose2d(64,3,4,2,1),            # [64,48,48]  ---> [32,96,96]
                                nn.Tanh())

        # self.conv = nn.Conv2d(32,self.output_c,3,1,1)                       # [32,96,96]  ---> [3,96,96]

        self.initialize_weights(self)

    def initialize_weights(self,net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self,noize,label):

        # noize = torch.randn([label.shape[0],self.n_dim])
        # print(noize.shape)
        x = torch.cat([noize,label],1)

        x = x.view(len(x),-1)
        x = self.fc1(x)
        x = x.view(-1,256,12,12)
        x = self.deconv1(x)
        # x = self.conv(x)
        # x = F.relu(x)

        return x


class Discriminator(nn.Module): # 判别器
    def __init__(self,input_c=3,scoredim=1,num_features=26) -> None:
        super(Discriminator,self).__init__()

        self.scoredim = scoredim
        self.n_features = num_features
        self.input_c = input_c
        # self.label_embedding = nn.Linear(self.n_features,96*96)

        self.conv = nn.Sequential(
                                nn.Conv2d(self.input_c,64,3,1,1),   # [3,96,96]   ---> [64,96,96]
                                nn.LeakyReLU(0.2,True),
                                nn.MaxPool2d(2,2),                  # [64,96,96]  ---> [64,48,48]

                                nn.Conv2d(64,128,3,1,1),            # [64,48,48]  ---> [128,48,48]
                                nn.BatchNorm2d(128),
                                nn.LeakyReLU(0.2,True),
                                nn.MaxPool2d(2,2),                  # [128,48,48] ---> [128,24,24]

                                nn.Conv2d(128,256,3,1,1),           # [128,24,24] ---> [256,24,24]
                                nn.BatchNorm2d(256),
                                nn.LeakyReLU(0.2,True),
                                nn.MaxPool2d(2,2))                  # [256,24,24] ---> [256,12,12]
       
        self.fc = nn.Sequential(
                                nn.Linear(256*12*12,1024),
                                nn.BatchNorm1d(1024),
                                nn.LeakyReLU(0.2,True),

                                nn.Linear(1024,512),
                                nn.BatchNorm1d(512),
                                nn.LeakyReLU(0.2,True))

        self.score = nn.Sequential(
                                nn.Linear(512,self.scoredim),
                                nn.Sigmoid())
        
        self.classify = nn.Sequential(
                                nn.Linear(512,self.n_features),
                                nn.Sigmoid())

        self.initialize_weights(self)

    def initialize_weights(self,net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    def forward(self,input):
        x = input

        x = self.conv(x)
        x = x.view(-1,256*12*12)
        x = self.fc(x)
        
        score = self.score(x)
        classes = self.classify(x)

        return  classes,score


# class ICONGAN(): # 主类

#     def __init__(self,args) -> None: # 初始化
        
#         # 参数初始化
#         self.Epochs = args.EPOCHS 
#         self.epoch_s = args.EPOCH_START 
#         self.check_s = args.CHECKPOINT_STEP 

#         self.batch_size = args.BATCH_SIZE 
#         self.n_workers = args.NUM_WORKERS 

#         self.lrG =  args.LRG
#         self.lrD = args.LRD 
#         self.momentum = args.MOMENTUM 
#         self.weight_decay = args.WEIGHT_DEACY 
        
#         self.gpu = args.GPU_MODE
#         self.device = args.DEVICE 
        
#         self.data_path = args.DATA_PATH 
#         self.save_path = args.SAVE_PATH 
#         self.result_path = args.RESULT_IMGS
#         self.log_path =  args.LOGS_PATH 

#         # 读取数据
#         self.data = icons(self.data_path)
#         self.trainloader = DataLoader(self.data,
#                                       batch_size=self.batch_size,
#                                       shuffle=True,
#                                       num_workers=self.n_workers,
#                                       drop_last=True)

#         # 模型初始化
#         self.Generater = Generater()
#         self.Discriminater = Discriminator()

#         # 优化器初始化
#         self.G_optimizer = torch.optim.Adam(self.Generater.parameters(),lr=self.lrG,betas=(0.9,0.999),weight_decay=self.weight_decay)
#         self.D_optimizer = torch.optim.Adam(self.Discriminater.parameters(),lr=self.lrD,betas=(0.9,0.999),weight_decay=self.weight_decay)

#         # 损失函数构造
#         self.BCE_loss = nn.BCELoss()
#         self.CE_loss = nn.CrossEntropyLoss()

#         # GPU加速
#         if self.gpu:
#             self.Generater.to(self.device)
#             self.Discriminater.to(self.device)
#             self.BCE_loss = nn.BCELoss().to(self.device)
#             self.CE_loss = nn.CrossEntropyLoss().to(self.device)
        

#     def train(self): # 训练

#         self.train_hist = {}
#         self.train_hist['D_loss'] = []
#         self.train_hist['G_loss'] = []
#         self.train_hist['per_epoch_time'] = []
#         self.train_hist['total_time'] = []

#         self.Discriminater.train()
#         print('----开始训练----')

#         start_time = time.time()
        
#         for epoch in range(self.epoch_s,self.Epochs):
#             self.Generater.train()

#             tq = tqdm(total=len(self.trainloader)*self.batch_size) # 生成进度条
#             tq.set_description(f'Epoch {epoch}') # 进度条描述

#             for idx,(real_img,label) in enumerate(self.trainloader):
#                 if self.gpu:
#                     real_img = real_img.to(self.device)
#                     label = label.to(self.device).long()
#                     real_score = torch.ones([self.batch_size,1]).long().to(self.device) # 真实图片得分
#                     fake_score = torch.zeros([self.batch_size,1]).long().to(self.device) # 生成图片得分

#                 # (1) 更新判别器网络: 最大化 log(D(x)) + log(1 - D(G(z)))
#                 self.Discriminater.zero_grad() # 判别器梯度清零
#                 real_c,real_s = self.Discriminater(real_img)
#                 real_c_loss = self.CE_loss(real_c,label) # 真实图片的分类CEloss
#                 real_s_loss = self.BCE_loss(real_s,real_score) # 真实图片得分BCEloss,得分越高,损失越小

#                 fake_img = self.Generater(label)
#                 fake_c,fake_s = self.Discriminater(fake_img)
#                 fake_c_loss = self.CE_loss(fake_c,label) # 生成图片分类CEloss
#                 fake_s_loss = self.BCE_loss(fake_s,fake_score) # 生成图片得分BCEloss,得分越低，损失越小

#                 Ls_loss = real_s_loss + fake_s_loss # 判别损失
#                 Lc_loss = real_c_loss + fake_c_loss # 分类损失
#                 D_loss = Ls_loss + Lc_loss
#                 self.train_hist['D_loss'].append(D_loss.item()) # 记录判别器损失

#                 D_loss.backward() # 判别器反向传播
#                 self.D_optimizer.step() # 梯度下降

#                 # (2) 更新生成器网络: 最大化log(D(G(z)))
#                 self.Generater.zero_grad() # 生成器梯度清零
#                 fake_img = self.Generater(label)
#                 fake_c,fake_s = self.Discriminater(fake_img)
#                 G_loss = self.BCE_loss(fake_s,real_score) # 生成图片得分BCEloss
#                 G_loss += self.CE_loss(fake_c,label) # 生成图片分类CEloss
                
#                 self.train_hist['G_loss'].append(G_loss.item()) # 记录生成器损失

#                 G_loss.backward() # 生成器反向传播
#                 self.G_optimizer.step() # 梯度下降

#                 tq.set_postfix({'GLoss':G_loss.item(),'DLoss':D_loss.item()}) # 进度条结尾更新参数变化

#             print('判别器平均损失:',mean(self.train_hist['D_loss']))
#             print('生成器平均损失:',mean(self.train_hist['G_loss']))
#             tq.close() # 关闭进度条
    
#     def test(self):
#         pass  

#     def save_checkpoint(self,epoch,model):
#         pass
    
#     def save_model(self): # 保存模型
#         pass

#     def load_model(self): # 读取模型
#         pass

#     def generate(self): # 生成
#         pass

#     def visualize_results(self): # 结果展示
#         pass









