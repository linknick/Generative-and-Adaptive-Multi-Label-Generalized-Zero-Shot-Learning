import torch
import numpy as np
import pandas as pd
from PIL import Image
from itertools import zip_longest
from torch.utils.data import Dataset
from torchvision import transforms
    
from os.path import join as PJ

import statistics

class ConceptSet():

    def __init__(self, concept="vec", dataset='voc2007'):

        self.file = PJ('./dataset', dataset, 'list', 'concepts', 'concepts_' + concept + '.txt')
        # iloc: for nuswide 81 concept
        #iloc = -81 if dataset == 'nus_wide' else 0
        iloc = 0

        self.names = pd.read_csv(self.file, header=None, low_memory=False).iloc[0, iloc:].values
        self.vecs = pd.read_csv(self.file, header=None, low_memory=False).iloc[1:, iloc:].values.astype(float)

        self.vecs = torch.from_numpy(self.vecs).type(torch.cuda.FloatTensor)
        


class ClassDataset(Dataset):

    def __init__(self, dataset='nus_wide', mode='train'):
        self.mode = mode
        self.dataset = dataset

        if dataset == 'nus_wide':

            self.img_root = PJ('./dataset', dataset)
            self.csv_file = PJ('./dataset', dataset, 'list', 'train_test', 'ground_truth_'+ self.mode + '.txt')
            self.data = pd.read_csv(self.csv_file, header=None)
            self.data = pd.concat([self.data.iloc[:, 0], self.data.iloc[:, 1:]], axis=1)

            train_num=925
            test_num=81
        elif dataset == 'voc2007':
            #self.img_root = PJ('./dataset', dataset, mode, 'img/')
            #self.csv_file = PJ('./dataset', dataset, mode, mode + '_dataset.txt')
            #self.data = pd.read_csv(self.csv_file)
            
            train_num=10
            test_num=10
            self.img_root = PJ('./dataset', dataset)
            self.csv_file = PJ('./dataset', dataset, 'origin_'+self.mode+'.txt')
            
            self.data = pd.read_csv(self.csv_file)
            
            self.data = pd.concat([self.data.iloc[:, 0], self.data.iloc[:, 1:]], axis=1)
        elif dataset == 'mscoco':
            self.img_root = PJ('./dataset', dataset)
            self.csv_file = PJ('./dataset', dataset, 'coco_'+self.mode+'.txt')
            self.data = pd.read_csv(self.csv_file)
            self.data = pd.concat([self.data.iloc[:, 0], self.data.iloc[:, 1:]], axis=1)

        else:
            self.img_root = PJ('./dataset', dataset)
            self.csv_file = PJ('./dataset', dataset, 'lesa_'+self.mode + '.txt')
            self.data = pd.read_csv(self.csv_file)

            self.data = pd.concat([self.data.iloc[:, 0], self.data.iloc[:, 1:]], axis=1)

        

        # astype and remove empty
        if dataset!='openImage':
            self.data.iloc[:, 1:] = (self.data.iloc[:, 1:] == 1).astype('int64')
            self.data = self.data[(self.data.iloc[:, 1:] == 1).sum(axis=1) > 0]
            self.neg_weight = 1 / ((self.data == 0).sum().sum() / (self.data == 1).sum().sum() + 1)
        else:
            self.data.iloc[:,0] = mode+"/img/"+self.data.iloc[:,0]
            self.data = self.data[(self.data.iloc[:, 2] != '[]')]
            a=self.data.iloc[:,1]
            self.neg_weight = sum(1  for s in a for w in s.split(', ') )/len(a)/7186
            #self.neg_weight = 1 / ((self.data == 0).sum().sum() / (self.data == 1).sum().sum() + 1)
        if mode == 'train':
            #self.data = self.data[(self.data.iloc[:, -test_num:] == 1).sum(axis=1) == 0]
            pass
            #print(self.data)
            
            
        	

        elif mode == 'test':
            #self.data = self.data[(self.data.iloc[:, -test_num:] == 1).sum(axis=1) > 0]
            pass
        	

        #self.neg_weight = 1 / ((self.data == 0).sum().sum() / (self.data == 1).sum().sum() + 1)
        #self.neg_weight = 0.3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            image = Image.open(PJ(self.img_root, self.data.iloc[idx, 0])).convert('RGB')

            image = self.transform(image)
            
        except :
            print(self.img_root, self.data.iloc[idx, 0])
        
        
        if self.dataset == 'openImage':
            
            pos = self.data.iloc[idx, 1]
            neg = self.data.iloc[idx, 2]
            sample = {'image': image, 'pos': pos, 'neg': neg}
        else:
            label = torch.Tensor(self.data.iloc[idx, 1:].tolist())
            sample = {'image': image, 'label': label}
        return sample

    def transform(self, image):
        transform = None
        #if self.mode == 'train':
        #    transform = transforms.Compose([
        #        transforms.RandomResizedCrop(224),
        #        transforms.RandomHorizontalFlip(),
        #        transforms.ToTensor(),
        #        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])])
        """        
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])
        """

        
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])
        #        transforms.RandomHorizontalFlip(),
        #        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])])
        

        elif self.mode == 'test':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()])

        

        #elif self.mode == 'test':
        #    transform = transforms.Compose([
        #        transforms.Resize(256),
        #        transforms.CenterCrop(224),
        #        transforms.ToTensor(),
        #        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225])])
        return transform(image)


if __name__ == '__main__':

    dataset = 'openImage'
    mode = ['train', 'test']
    trainset = ClassDataset(dataset=dataset, mode='test')
    testset = ClassDataset(dataset=dataset, mode='test')
    #validset = ClassDataset(dataset=dataset, mode='test')
    #validset.data=testset.data
    concept_set = ConceptSet(concept='vec', dataset=dataset)
    trainloader= torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False) 
    #testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)
    #validloader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=False)
    
    # show test data
    #print(np.mean(np.mean(concept_set.vecs[i])))
    
    for batch_i, batch_data in enumerate(trainloader, 1):
        img=transforms.ToPILImage()(batch_data['image'][1])
        img.show()
        exit(0)
        #print(batch_i,batch_data['label'])
        label=batch_data['label']
        label = [[int(e)for e in l[1:-1].split(', ')]  for l in label ]
        max_length = max([len(l) for l in label])
        label = [l + l[-1:] * (max_length - len(l)) for l in label]
        #print(label)
        label = torch.zeros(len(label), 7586).scatter_(1, torch.tensor(label), 1).cuda()

        
        
    