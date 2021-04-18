# data loader
from __future__ import print_function, division
from PIL import Image

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random

#==========================dataset load==========================

class SalObjDataset(Dataset):
	def __init__(self,dataset_dir,transforms_,rgb=True):
		self.dataset_dir=dataset_dir
		self.file_list = os.listdir(self.dataset_dir)
		self.transform = transforms.Compose(transforms_)
		self.rgb=rgb
	def __len__(self):
		return len(self.file_list)

	def __getitem__(self,idx):
		# sal data loading
		temp_dir=os.listdir(os.path.join(self.dataset_dir,self.file_list[idx]))
		temp_idx=random.randint(0,3)
		chird_dir=temp_dir[temp_idx]
		if self.rgb==True:
			img1=Image.open(self.dataset_dir+'/'+self.file_list[idx]+"/"+chird_dir+"/"+self.file_list[idx]+"_1.jpg")
			img2=Image.open(self.dataset_dir+'/'+self.file_list[idx]+"/"+chird_dir+"/"+self.file_list[idx]+"_2.jpg")
			label=Image.open(self.dataset_dir+'/'+self.file_list[idx]+"/"+"GT"+"/"+self.file_list[idx]+"_ground.jpg")
		else:
			img1=Image.open(self.dataset_dir+'/'+self.file_list[idx]+"/"+chird_dir+"/"+self.file_list[idx]+"_1.jpg").convert('L')
			img2=Image.open(self.dataset_dir+'/'+self.file_list[idx]+"/"+chird_dir+"/"+self.file_list[idx]+"_2.jpg").convert('L')
			label=Image.open(self.dataset_dir+'/'+self.file_list[idx]+"/"+chird_dir+"/"+self.file_list[idx]+"_ground.jpg").convert('L')
		#裁剪为256*256
		img1=img1.resize((256, 256),Image.BICUBIC)
		img2 = img2.resize((256, 256), Image.BICUBIC)
		label =label.resize((256, 256), Image.BICUBIC)
		#水平翻转
		if random.random() <0.5:
			img1=img1.transpose(Image.FLIP_LEFT_RIGHT)
			img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
			label = label.transpose(Image.FLIP_LEFT_RIGHT)
		img1 = self.transform(img1)
		img2 = self.transform(img2)
		label = self.transform(label)
		return img1,img2,label


class DataTest(Dataset):
    def __init__(self,testData_dir,transforms_):
        self.testData_dir=testData_dir
        self.file_list=os.listdir(testData_dir)
        self.transform = transforms.Compose(transforms_)
    def __getitem__(self, idx):
        image1= Image.open(self.testData_dir+"/"+self.file_list[idx]+"/"+self.file_list[idx]+"-A.jpg")
        image2= Image.open(self.testData_dir+"/"+self.file_list[idx]+"/"+self.file_list[idx]+"-B.jpg")
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return image1,image2
    def __len__(self):
        return len(self.file_list)

