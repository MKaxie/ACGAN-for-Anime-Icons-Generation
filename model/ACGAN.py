import os,sys,time,cv2
sys.path.append(['MyGAN'])
# print(sys.path)
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import LeakyReLU, ReLU, Sigmoid
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.nn.modules.linear import Linear
from torch.nn.modules.pooling import MaxPool2d
import torch.optim 
# from utils.configs import Configs

from torch.utils.data.dataloader import DataLoader
from dataset.ICONS import icons
from BaseNet import Generater,Discriminator



class ICONGAN(): # 主类

    def __init__(self,args) -> None: # 初始化
        
        # 参数初始化
        self.Epochs = args.EPOCHS 
        self.epoch_s = args.EPOCH_START  
        self.resume = args.RESUME # 断点续训

        self.batch_size = args.BATCH_SIZE 
        self.n_workers = args.NUM_WORKERS 

        self.lrG =  args.LRG
        self.lrD = args.LRD 
        self.momentum = args.MOMENTUM 
        self.weight_decay = args.WEIGHT_DEACY 
        
        self.gpu = args.GPU_MODE
        self.device = args.DEVICE 
        
        self.data_path = args.DATA_PATH 
        self.save_path = args.SAVE_PATH 
        self.result_path = args.RESULT_IMGS
        self.check_dir =  args.CKECK_DIR 

        # 读取数据
        self.data = icons(self.data_path)
        self.trainloader = DataLoader(self.data,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.n_workers,
                                      drop_last=True)

        # 模型初始化
        self.Generater = Generater()
        self.Discriminater = Discriminator()

        # 优化器初始化
        self.G_optimizer = torch.optim.Adam(self.Generater.parameters(),lr=self.lrG,betas=(0.9,0.999),weight_decay=self.weight_decay)
        self.D_optimizer = torch.optim.Adam(self.Discriminater.parameters(),lr=self.lrD,betas=(0.9,0.999),weight_decay=self.weight_decay)

        # 损失函数构造
        self.BCE_loss = nn.BCELoss()
        self.CE_loss = nn.CrossEntropyLoss()

        # GPU加速
        if self.gpu:
            self.Generater.to(self.device)
            self.Discriminater.to(self.device)
            self.BCE_loss = nn.BCELoss().to(self.device)
            self.CE_loss = nn.CrossEntropyLoss().to(self.device)
        
    def train(self): # 训练

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.Discriminater.train()

        # 查找断点
        self.loadcheck(self.epoch_s)

        print('----模型训练----')

        start_time = time.time()
        
        for epoch in range(self.epoch_s,self.Epochs):
            self.Generater.train()

            dloss = []
            gloss = []

            tq = tqdm(total=len(self.trainloader)*self.batch_size) # 生成进度条
            tq.set_description(f'Epoch {epoch}') # 进度条描述

            for idx,(real_img,label) in enumerate(self.trainloader):
                if self.gpu:
                    real_img = real_img.to(self.device)
                    noize = torch.randn([self.batch_size,102],dtype=torch.float32)
                    label = label.to(self.device)
                    label_c = label.long()
                    real_score = torch.ones([self.batch_size,1]).to(self.device) # 真实图片得分
                    fake_score = torch.zeros([self.batch_size,1]).to(self.device) # 生成图片得分

                # (1) 更新判别器网络: 最大化 log(D(x)) + log(1 - D(G(z)))
                self.Discriminater.zero_grad() # 判别器梯度清零
                real_c,real_s = self.Discriminater(real_img)
                real_c_loss = self.CE_loss(real_c,label) # 真实图片的分类CEloss
                real_s_loss = self.BCE_loss(real_s,real_score) # 真实图片得分BCEloss,得分越高,损失越小

                fake_img = self.Generater(noize,label)
                fake_c,fake_s = self.Discriminater(fake_img)
                fake_c_loss = self.BCE_loss(fake_c,label) # 生成图片分类CEloss
                fake_s_loss = self.BCE_loss(fake_s,fake_score) # 生成图片得分BCEloss,得分越低，损失越小

                Ls_loss = real_s_loss + fake_s_loss # 判别损失
                Lc_loss = real_c_loss + fake_c_loss # 分类损失
                D_loss = Ls_loss + Lc_loss
                self.train_hist['D_loss'].append(D_loss.item()) # 记录判别器损失
                dloss.append(D_loss.item())

                D_loss.backward() # 判别器反向传播
                self.D_optimizer.step() # 梯度下降

                # (2) 更新生成器网络: 最大化log(D(G(z)))
                self.Generater.zero_grad() # 生成器梯度清零
                fake_img = self.Generater(noize,label)
                fake_c,fake_s = self.Discriminater(fake_img)
                G_loss = self.BCE_loss(fake_s,real_score) # 生成图片得分BCEloss
                G_loss += self.BCE_loss(fake_c,label) # 生成图片分类CEloss
                
                self.train_hist['G_loss'].append(G_loss.item()) # 记录生成器损失
                gloss.append(G_loss.item())

                G_loss.backward() # 生成器反向传播
                self.G_optimizer.step() # 梯度下降

                tq.set_postfix({'GLoss':G_loss.item(),'DLoss':D_loss.item()}) # 进度条结尾更新参数变化

            print('判别器平均损失:',np.mean(dloss))
            print('生成器平均损失:',np.mean(gloss))

            tq.close() # 关闭进度条

            if (epoch+1)%50==0:
                self.save_checkpoint(epoch) # 保存断点

        self.save_checkpoint(self.Epochs-1)
    
    def loadcheck(self,epoch):
        
        if self.resume:
            ckpt = self.check_dir+f'/ckpt_last_{epoch}.pt'
            if os.path.isfile(ckpt):
                checkpoint = torch.load(ckpt)
                self.epoch_s = checkpoint['epoch'] + 1
                self.Discriminater.load_state_dict(checkpoint['Discriminater'])
                self.Generater.load_state_dict(checkpoint['Generater'])
                self.D_optimizer.load_state_dict(checkpoint['D_optimizer'])
                self.G_optimizer.load_state_dict(checkpoint['G_optimizer'])
                print("=> 读取断点 (epoch {})".format(checkpoint['epoch']))
            else:
                print("----未找到断点文件----")

    def save_checkpoint(self,epoch):
        
        time_ = time.strftime("%Y-%m-%d", time.localtime())

        if epoch!=self.Epochs-1:
            checkpoint = {
                'epoch': epoch,
                'Discriminater': self.Discriminater.state_dict(),
                'Generater':self.Generater.state_dict(),
                'D_optimizer': self.D_optimizer.state_dict(),
                'G_optimizer': self.G_optimizer.state_dict() }

            torch.save(checkpoint,self.check_dir+f'/ckpt_last_{epoch}.pt')
        else:
            trainedModel = {
                'Discriminater': self.Discriminater.state_dict(),
                'Generater':self.Generater.state_dict(),
            }
            torch.save(trainedModel,self.save_path+'/last_model.pt')

    def load_model(self): # 读取模型
        model = self.save_path+'/last_model.pt'
        if os.path.isfile(model):
            trainedModel = torch.load(model)
            self.Discriminater.load_state_dict(trainedModel['Discriminater'])
            self.Generater.load_state_dict(trainedModel['Generater'])
        else:
            assert '----未找到已训练模型文件----'

    def generate(self,labels): # 生成
        
        self.Discriminater.eval()
        self.Generater.eval()

        self.load_model()

        imgs = self.Generater(labels)
        scores = self.Discriminater(imgs)

        imgs = imgs.cpu().numpy()
        scores = scores.cpu().numpy()

        for i in range(len(scores)):
            cv2.imwrite(self.result_path+f'/{scores[i]}.jpg',imgs[i])

    def visualize_results(self): # 结果展示
        pass

    def plot_loss(self):

        x = np.array([_ for _ in range(self.Epochs*len(self.trainloader))])
        plt.figure(figsize=(20,15))
        plt.plot(x,self.train_hist['D_loss'],'deepskyblue')
        plt.plot(x,self.train_hist['G_loss'],'orange')
        plt.title('Loss changing waves')
        plt.legend(['D_loss','G_loss'])
        plt.xlabel('EPOCH')
        plt.ylabel('Loss')
        plt.show()


    




    





































