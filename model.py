import torch
from torch.autograd import Variable
import torch.nn as nn
from torchvision.models import vgg16_bn as vgg16_bn
from torchvision.models import vgg19_bn as vgg19_bn
from torchvision.models import resnet101 as resnet152
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import gc
from PIL import Image

global DEVICE
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_count=torch.cuda.device_count()
_MODEL = 'vgg19'

class VGG(nn.Module):

    def __init__(self, freeze=True, pretrained=True, **kwarg):
        super(VGG, self).__init__()
        self.k = kwarg['k']
        self.d = kwarg['d']
        self.non_linear_param = 0 if self.k == 1 else self.k + 1
        self.output_num = self.non_linear_param + self.k * (self.d + 1)
        
        
        if _MODEL == 'vgg':

            model = vgg16_bn(pretrained=pretrained)
            self.features = nn.Sequential(*list(model.features.children()))
            if freeze:
                for param in self.features.parameters():
                    param.requires_grad = False
            self.transform = nn.Sequential(*list(model.classifier.children())[:4],nn.Linear(4096, self.output_num))

            self.classifier = nn.Sequential(*list(nn.Linear(4096)))
        elif _MODEL == 'vgg19':

            model = vgg19_bn(pretrained=pretrained)
            self.features = nn.Sequential(*list(model.features.children()))
            if freeze:
                for param in self.features.parameters():
                    param.requires_grad = False
            
            self.transform = nn.Sequential(*list(model.classifier.children())[:2])
            
            
            if freeze:
                for param in self.transform.parameters():
                    param.requires_grad = False
            
            self.classifier = nn.Sequential(nn.Linear(4096,self.output_num))
            
        elif _MODEL == 'resnet':

            model = resnet152(pretrained=pretrained)
            self.features = nn.Sequential(*list(model.children())[:-1])
            if freeze:
                for param in self.features.parameters():
                    param.requires_grad = False
            self.transform = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(1024, self.output_num))


    def forward(self, visual, semantics):
        #split visual to nxn parts
        """
        split_visual_features = torch.tensor([]).cuda()
        size = int(visual.shape[0]/n/n)
        for i in range(n*n):
            split_visual = self.split(visual[size*i:size*(i+1)],n)
          if(split_visual.shape[0]==0):
                break
            split_visual = split_visual.view(-1,3,split_visual.shape[3],split_visual.shape[4]).cuda()
            visual_feature = self.features(split_visual)
            split_visual_features = torch.cat([split_visual_features,visual_feature],0)
        visual_feature = split_visual_features
        """

        #feature extraction
        """ 
        visual_feature = self.features(visual)
        visual_feature = visual_feature.view(visual_feature.size(0), -1)
        visual_matrix = self.transform(visual_feature) # batch_size * 4096d visual feature
        """
        ##visual_matrix = self.feature_extraction(visual)

        #attention
        """
        attention=None
        attention = self.attention(visual_matrix) # (n*n)blocks * batch_size
        attention = attention.view(-1,n*n)
        if(n>1):
            attention = torch.softmax(attention,dim=1)
            attention = attention.view(-1,1)
            visual_matrix = attention*visual_matrix
        """

        ##visual_matrix = visual_matrix.view(visual.shape[0],-1)

        visual_matrix = visual
        visual_matrix = self.classifier(visual_matrix)

        # Classification
        """
        semantics = semantics.expand(visual_matrix.shape[0], self.d, -1)

        if self.non_linear_param == 0:
            
            matrix = visual_matrix[..., :-1].view(-1, self.k, self.d)
            bias = visual_matrix[..., -1].view(-1, self.k)[..., None]            
            semantic_transforms = torch.matmul(matrix, semantics) + bias

        else:
            visual_outer = visual_matrix[..., :self.non_linear_param]
            visual_inner = visual_matrix[..., self.non_linear_param:]

            visual_outer_matrix = visual_outer[..., :-1].view(-1, 1, self.k)
            visual_outer_bias = visual_outer[..., -1].view(-1, 1)[..., None]

            visual_inner_matrix = visual_inner[..., :-self.k].view(-1, self.k, self.d)
            visual_inner_bias = visual_inner[..., -self.k:].view(-1, self.k)[..., None]

            semantic_transforms = torch.matmul(visual_inner_matrix, semantics) + visual_inner_bias
            semantic_transforms = torch.tanh(semantic_transforms)
            semantic_transforms = torch.matmul(visual_outer_matrix, semantic_transforms) + visual_outer_bias
        
        semantic_transforms = semantic_transforms.transpose(1, 2).contiguous()
        semantic_transforms = torch.sigmoid(semantic_transforms)
        """
        semantic_transforms = self.classify(visual_matrix,semantics)
        
        return semantic_transforms
        

    def feature_extraction(self,visual):
        visual_feature = self.features(visual)
        visual_feature = visual_feature.view(visual_feature.size(0), -1)
        visual_matrix = self.transform(visual_feature)
        return visual_matrix

    def classify(self,visual_matrix,semantics):

        semantics = semantics.expand(visual_matrix.shape[0], self.d, -1)

        if self.non_linear_param == 0:
            
            matrix = visual_matrix[..., :-1].view(-1, self.k, self.d)
            bias = visual_matrix[..., -1].view(-1, self.k)[..., None]
            semantic_transforms = torch.matmul(matrix, semantics) + bias
          
        else:
            visual_outer = visual_matrix[..., :self.non_linear_param]
            visual_inner = visual_matrix[..., self.non_linear_param:]

            visual_outer_matrix = visual_outer[..., :-1].view(-1, 1, self.k)
            visual_outer_bias = visual_outer[..., -1].view(-1, 1)[..., None]

            visual_inner_matrix = visual_inner[..., :-self.k].view(-1, self.k, self.d)
            visual_inner_bias = visual_inner[..., -self.k:].view(-1, self.k)[..., None]

            semantic_transforms = torch.matmul(visual_inner_matrix, semantics) + visual_inner_bias
            semantic_transforms = torch.tanh(semantic_transforms)
            semantic_transforms = torch.matmul(visual_outer_matrix, semantic_transforms) + visual_outer_bias
        
        semantic_transforms = semantic_transforms.transpose(1, 2).contiguous()
        semantic_transforms = torch.sigmoid(semantic_transforms)
        return semantic_transforms

    def split(self,visual,n):
        
        #transforms.ToPILImage()(visual[0].cpu()).resize((224,224)).show()
        size=int(visual.shape[3]/n)
        split_visuals=torch.tensor([])

        for i in range(visual.shape[0]):
            img=torch.tensor([])

            for j in range(n*n):
                split_visual = visual[i,:,size*int(j/n):size*(int(j/n)+1),size*(j%n):size*(j%n+1)].cpu()

                #split_visual = transforms.ToTensor()(Image.fromarray(np.float32(split_visual.cpu()), 'RGB').resize((224,224)))
                split_visual = transforms.ToTensor()(transforms.ToPILImage()(split_visual.cpu()).resize((224,224)))
                
                
                split_visual = split_visual.expand(1,1,-1,-1,-1)
                
                img = torch.cat([img,split_visual],1)
                
            split_visuals = torch.cat([split_visuals,img],0)
                    
        #split_visuals = split_visuals.view(visual.shape[0],n*n,3,224,224)

        return split_visuals

class HingeRankLoss():

    def __init__(self, dataset='voc2007', loss_args=None):
        self.alpha = torch.cuda.FloatTensor([loss_args['alpha'], 1 - loss_args['alpha']])
        #self.alpha = torch.cuda.FloatTensor([1-loss_args['alpha'], loss_args['alpha']])
        self.gamma = loss_args['gamma']

    

    def _correct(self, outputs, labels):
        outputs = torch.round(outputs)
        return outputs
        
    
        
            
    def _loss(self, outputs=None, labels=None):
        
        self.outputs = outputs
        self.labels = labels[..., None].type(torch.cuda.FloatTensor)
        
        bce_loss = F.binary_cross_entropy(self.outputs, self.labels)
        #print(bce_loss)
        """
        max_pos = torch.sum(self.labels*((1-self.outputs)))
        max_neg = torch.sum((1-self.labels)*(self.outputs))
        #bce_loss+=(max_pos+max_neg)/2
        print("sum:",max_pos.item(),max_neg.item())
        max_pos = torch.max(self.labels*((1-self.outputs)))
        max_neg = torch.max((1-self.labels)*(self.outputs))
        print("max:",max_pos.item(),max_neg.item())
        max_pos = 1-torch.max(self.labels*((self.outputs)))
        max_neg = 1-torch.max((1-self.labels)*(1-self.outputs))
        print("min:",max_pos.item(),max_neg.item())
        max_pos = torch.sum(self.labels*((1-self.outputs)))/torch.sum(self.labels)
        max_neg = torch.sum((1-self.labels)*(self.outputs))/torch.sum(1-self.labels)
        
        print("mean:",max_pos.item(),max_neg.item())
        """
        
        #bce_loss = F.binary_cross_entropy(self.outputs, self.labels)
        
        #loss = torch.mean(abs(self.labels-self.outputs))
        #print(bce_loss)
        #pt = Variable((-bce_loss).data.exp())
        #print(self.labels.type(torch.cuda.LongTensor).data.reshape(-1))
        #print(self.alpha.data)
        #print(self.alpha.data.gather(0, self.labels.type(torch.cuda.LongTensor).data.reshape(-1)))
        """
        at = self.alpha.data.gather(0, self.labels.type(torch.cuda.LongTensor).data.reshape(-1))/self.alpha.data[0]
        at = at.reshape(self.labels.shape)
        bce_loss = F.binary_cross_entropy(self.outputs, self.labels, weight=at)
        """
        #print(bce_loss)
        #at = at[:-81]

        #self.loss = torch.mean(Variable(at) * (1 - pt) ** self.gamma * bce_loss)
        #self.loss = torch.mean(Variable(at) * bce_loss)
        #self.loss = torch.mean(bce_loss)

        self.loss=bce_loss

        #self.loss+=loss

        return self.loss
    def my_bceloss(self,outputs,labels,weight):
        loss = weight*(labels * torch.log(outputs) + (1-labels)*torch.log(1-outputs))
        return torch.mean(loss)


class ModelCompare():
    def compare(m1,m2):
        print(torch.norm(m1.classifier[0].weight-m2.classifier[0].weight,p=2))
        print(torch.norm(m1.classifier[0].weight-m2.classifier[0].weight,p=1))
        print(torch.mean(m1.classifier[0].weight-m2.classifier[0].weight))
        #print(m1.transform[1].weight)

class Generator(nn.Module):
    def __init__(self,g):
        self.z_dim=300
        self.generate_num=g
        super(Generator, self).__init__()
        #self.fc1 = nn.Sequential(nn.Linear(4096, 2048),nn.LeakyReLU(inplace=True),nn.Linear(2048,1024))
        self.fc1 = nn.Sequential(nn.Linear(4096+300, 4096),nn.LeakyReLU(inplace=True),nn.Linear(4096,4096))
        self.fc21 = nn.Sequential(nn.Linear(4096, self.z_dim))
        self.fc22 = nn.Sequential(nn.Linear(4096, self.z_dim))
        #self.fc3 = nn.Sequential(nn.Linear(self.z_dim+self.z_dim, 2048),nn.LeakyReLU(inplace=True),nn.Linear(2048,4096),nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(self.z_dim+300, 4096),nn.LeakyReLU(inplace=True),nn.Linear(4096,4096),nn.ReLU(inplace=True))

    def encode(self, x, labels, semantics):
        vec = torch.matmul(labels,semantics.T)        
        """
        vec = labels.unsqueeze(-1).cuda()
        vec = vec*semantics.T
        #vec = torch.mean(vec,dim=1)
        
        maxs = torch.argmax(torch.abs(vec),1).cpu()
        onehot=torch.zeros(len(labels), semantics.shape[0], semantics.shape[1]).scatter_(2, maxs.unsqueeze(-1), 1).cuda()
        vec = torch.sum(onehot*semantics.expand(len(labels),semantics.shape[0],-1),dim=2)
        """
        x = torch.cat([x,vec],dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):

        return self.fc3(z)

    def forward(self, x, labels, semantics):
        mu, logvar = self.encode(x,labels, semantics)
        z = self.reparameterize(mu, logvar)
        vec = torch.matmul(labels,semantics.T)
        """
        vec = labels.unsqueeze(-1).cuda()
        vec = vec*semantics.T
        #vec = torch.mean(vec,dim=1)
        
        maxs = torch.argmax(torch.abs(vec),1).cpu()
        onehot=torch.zeros(len(labels), semantics.shape[0], semantics.shape[1]).scatter_(2, maxs.unsqueeze(-1), 1).cuda()
        vec = torch.sum(onehot*semantics.expand(len(labels),semantics.shape[0],-1),dim=2)
        """
        
        z = torch.cat([z,vec],dim=1)
        return self.decode(z), mu, logvar

    def _loss(self,recon_x, x, mu, logvar):
        #BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        #BCE = torch.mean(torch.exp(pow(recon_x-x,2)))
        BCE = torch.mean(pow(recon_x-x,2))
        #BCE += torch.mean(pow(torch.tanh(recon_x)-torch.tanh(x),2))
        
        #BCE = F.binary_cross_entropy_with_logits(recon_x,x)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        #print(KLD,BCE)
        return KLD+BCE

    
    def generate(self,labels , semantics):
        generate_num=int(labels.shape[0])
        noise = torch.randn(generate_num,self.z_dim).cuda()

        
        vec = torch.matmul(labels.cuda(),semantics.T)
        """
        vec = labels.unsqueeze(-1).cuda()
        vec = vec*semantics.T
        #vec = torch.mean(vec,dim=1)
        
        maxs = torch.argmax(torch.abs(vec),1).cpu()
        onehot=torch.zeros(len(labels), semantics.shape[0], semantics.shape[1]).scatter_(2, maxs.unsqueeze(-1), 1).cuda()
        vec = torch.sum(onehot*semantics.expand(len(labels),semantics.shape[0],-1),dim=2)
        """
        vec = torch.cat([noise,vec],dim=1)
        g_feature = self.decode(vec)
        
        return g_feature
    



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Linear(4396,4096),nn.LeakyReLU(),nn.Linear(4096,1))
        
    def forward(self, features,labels,semantics):

        #vec = [vec_i.cuda()*semantics for vec_i in labels]
        #vec = torch.stack(vec).view(features.shape[0],-1)
        
        vec = torch.matmul(labels.cuda(),semantics.T)
        """
        vec = labels.unsqueeze(-1).cuda()
        vec = vec*semantics.T
        #vec = torch.mean(vec,dim=1)
        
        maxs = torch.argmax(torch.abs(vec),1).cpu()
        onehot=torch.zeros(len(labels), semantics.shape[0], semantics.shape[1]).scatter_(2, maxs.unsqueeze(-1), 1).cuda()
        vec = torch.sum(onehot*semantics.expand(len(labels),semantics.shape[0],-1),dim=2)
        """
        
        
        features = torch.cat([features,vec],dim=1)
        
        result = self.discriminator(features)
        return result
    def _loss(self, logits):
                
        
        return torch.mean(logits)




        
