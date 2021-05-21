import os
import cv2
import numpy as np
from glob import glob
import torch
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset,dataloader

# facesPth = D:\PyProfile\11-GAN\MyGAN\data\faces

class icons(Dataset): # 数据集读取，图片与标签
    def __init__(self,dataPath) -> None: # dataset 初始化
        super(icons,self).__init__()

        self.imgList = glob(dataPath+'/faces/*.jpg') # 读取图片列表，生成图片地址列表

        with open(dataPath+'/tag_dict.txt','r') as fp:
            tgdict = eval(fp.read())

        self.tags = embeding(tgdict).encode()

        self.totensor = transforms.ToTensor()
    
    def __getitem__(self,index): # 核心方法，给定索引读取样本与标签

        path = self.imgList[index] # index下的图片位置
        idx = path[37:-4]
        
        img = cv2.imread(path) # 读取图片

        label = self.tags[int(idx)]
        label = label.astype(np.float32)
        label = torch.from_numpy(label)
        # label = label.view(len(label),-1)

        img = self.totensor(img)
        # label = self.totensor(label)

        return img, label

    def __len__(self): # 返回列表长度
        return len(self.imgList)


class embeding(): # 标签encoding
    def __init__(self,tgdict) -> None:
        
        self.tgdict = tgdict
        self.hair_colors = dict()
        self.eye_colors = dict()

    def one_hot(self): # 统计颜色数量，并进行ont-hot处理
        e_color = 0
        h_color = 0

        for item in self.tgdict:
            des = self.tgdict[item].split(' and ')
            # print(des)
            # break
            if len(des)==1: # 只包含一个特征时
                feat = des[0].split(' ')
                if feat[-1]=='hair':
                    self.hair_colors,h_color = self.featcompute(feat,self.hair_colors,h_color)
                    
                elif feat[-1]=='eye':
                    self.eye_colors,e_color = self.featcompute(feat,self.eye_colors,e_color)

            else: # 同时包含眼睛，头发
                hairs = des[0].split(' ')
                eyes = des[1].split(' ')

                self.hair_colors,h_color = self.featcompute(hairs,self.hair_colors,h_color)
                self.eye_colors,e_color = self.featcompute(eyes,self.eye_colors,e_color)

    def featcompute(self,featlist,featdict,ord_co): # 辅助计算

        for co in featlist[:-1]:
            if co not in featdict:
                ord_co += 1
                featdict[co] = ord_co
                
        return featdict,ord_co

    def encode(self,):
        
        self.one_hot()
        hlen = len(self.hair_colors)
        elen = len(self.eye_colors)

        shape = (hlen+elen)
        labeldict = dict()
        for item in self.tgdict:
            des = self.tgdict[item].split(' and ')
            label = np.zeros(shape)
            if len(des)==1:
                feat = des[0].split(' ')
                if feat[-1]=='hair':
                    for co in feat[:-1]:
                        label[self.hair_colors[co]-1] = 1
                elif feat[-1]=='eye':
                    for co in feat[:-1]:
                        label[self.eye_colors[co]-1 + hlen] = 1

            else:
                hairs = des[0].split(' ')
                eyes = des[1].split(' ')

                for co in hairs[:-1]:
                    label[self.hair_colors[co]-1] = 1
                for co in eyes[:-1]:
                    label[self.eye_colors[co]-1 + hlen] = 1

            labeldict[item] = label

        return labeldict
   


if __name__ == '__main__':
    
    path = r'D:\PyProfile\11-GAN\MyGAN\data'

#     with open(path+'/tag_dict.txt','r') as fp:
#         tgdict = eval(fp.read())

#     emb = embeding(tgdict)
#     emb.one_hot()

#     print(emb.hair_colors)
#     print(emb.eye_colors)

    a = icons(path)

    img,label = a[2]


    print(label.shape)
