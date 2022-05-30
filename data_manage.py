import torch
import torch.utils.data as data
import pandas as pd
import os
from skimage import io, transform
import numpy as np
import random
import torchvision.transforms.functional as TF
from torchvision import transforms
import cv2
class CustomDataset:
    def __init__(self,csv_dir,img_dir,transform=None,mode='test'):
        super(CustomDataset, self).__init__()
        if mode == 'test':
            self.csv_dir = pd.read_csv(csv_dir, encoding='euc=kr')
        elif mode == 'train':
            self.csv_dir = pd.read_csv(csv_dir, encoding='euc=kr')
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode


    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        N = random.random()
        angle, translations, scale, shear = transforms.RandomAffine(30).get_params([-15, 15], [0, 0], [1, 1], [0, 0], img_size=(512,380))
        
        num = self.csv_dir.iloc[index, 1]

        img_name = os.path.join(self.img_dir,(self.csv_dir.iloc[index,0]))
        
        folder = os.listdir(img_name+'/Img')

        thermal = os.listdir(img_name+'/Thermal')
        
        if 'Thumbs.db' in folder:
            folder.remove('Thumbs.db')
        if 'Thumbs.db' in thermal:
            thermal.remove('Thumbs.db')            
        folder.sort(key=len)
        if 'ROI1.txt' in thermal:
            thermal.remove('ROI1.txt')
        thermal.sort(key=len)
        #for different file name format
        if (self.csv_dir.iloc[index, 0])[:9] == '22dataset':
            folder = 'img_' + num24(num)+ '.bmp'
            thermal = 'thm_' + num24(num) + '.bmp'
        else:
            folder = folder[0][:-4]
            folder = folder+'_'+num24(num)+'.bmp'

            thermal = thermal[0][:-4]
            thermal = thermal+'_'+num24(num)+'.bmp'

        
        image = io.imread(img_name +'/Img/'+folder)
        therm = io.imread(img_name +'/Thermal/'+thermal)
        if len(therm.shape) == 3:
            therm = cv2.cvtColor(therm, cv2.COLOR_BGR2GRAY)

        landmarks = self.csv_dir.iloc[index,2] -1
        landmarks = np.array(landmarks)
        landmarks = landmarks.astype('float')
        image = torch.from_numpy(image)
        image = image.permute(2,0,1)
        therm = torch.from_numpy(therm)
        therm = therm.unsqueeze(0)
        image = torch.cat((image,therm),dim=0)
        
        
        if self.mode == 'Train':
            image = TF.affine(image,angle,list(translations),scale,list(shear))
            if N>0.5:
                image = TF.hflip(image)
                
  
        #for CoAtNet
        image = TF.resize(image,(224,224))
        
        sample = {'image':image , 'landmarks':torch.from_numpy(landmarks)}
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.csv_dir)

def num24(x):
    x = str(x)
    blank= 4-len(x)
    result =''
    for i in range(blank):
        result+='0'
    return result + x

#PATH = 'C:/Users/y/PycharmProjects/pythonProject/data/1.자유로_도로1/'
#test_dataset = CustomDataset(PATH+'dd.csv',PATH+'img/')

#print(test_dataset[2])
#print(len(test_dataset))
#from torch.utils.data import DataLoader

#dataloader = DataLoader(test_dataset,batch_size=4,shuffle=False,num_workers=0)

#for i,s in enumerate(dataloader,0):
