import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
from tensorboardX import SummaryWriter
import pandas as pd
from torch.utils.data import DataLoader
from data_loader import ClassDataset, ConceptSet
from model import VGG, HingeRankLoss, Generator, Discriminator
import utils
import torch.nn.functional as F
from torchvision import transforms
from os.path import join as PJ
import yaml
import math
import random
import numpy as np
import gc
from collections import deque
import os
import psutil
from itertools import zip_longest
import sys
if __name__ == '__main__':
    # setting
    CONFIG = yaml.load(open("config.yaml"))
    EXP_NAME = CONFIG['exp_name']
    DATASET = CONFIG['dataset']
    GAN = True


    LOAD_MODEL = None
    #LOAD_MODEL = PJ('.', 'runs', DATASET, EXP_NAME,'GAN_GT_TEST', 'epoch40.pkl')

    TOP_NUM = [10, 20]
    SAVE_PATH = PJ('.', 'runs', DATASET, EXP_NAME,'GAN_GT_TESTX')

    if(DATASET=='nus_wide'):
        train_num=925
        test_num=81
        torch.cuda.set_device(1)
    elif(DATASET=='voc2007'):
        train_num=10
        test_num=10
        torch.cuda.set_device(0)
    elif(DATASET=='mscoco'):
        train_num=65
        test_num=15
        torch.cuda.set_device(1)
    elif(DATASET=='openImage'):
        train_num=7186
        test_num=400
        torch.cuda.set_device(1)
    # build model
    if LOAD_MODEL is None:
        model = VGG(freeze=CONFIG['freeze'], pretrained=True, k=CONFIG['k'], d=CONFIG['d'])
        model = model.cuda()
    else:
        print("Loading pretrained model")
        model = VGG(freeze=CONFIG['freeze'], pretrained=True, k=CONFIG['k'], d=CONFIG['d'])
        model.load_state_dict(torch.load(LOAD_MODEL))
        model = model.cuda()
    G_model=Generator(CONFIG['train_batch_size']).cuda()
    D_model=Discriminator().cuda()
    D2_model=Discriminator().cuda()

    # load concept set
    concept_set = ConceptSet(concept=CONFIG['concepts'], dataset=CONFIG['dataset'])
    
    concept_set.vecs = F.normalize(concept_set.vecs, p=2, dim=0)
    
    # load dataset
    print("Loading training data")
    trainset = ClassDataset(dataset=CONFIG['dataset'], mode='train')
    print("Loading test data")
    testset = ClassDataset(dataset=CONFIG['dataset'], mode='test')
    trainset.data = trainset.data[(trainset.data.iloc[:, 1] != '[]')]
    testset.data = testset.data[(testset.data.iloc[:, 1] != '[]')]
    
    trainloader = DataLoader(trainset, shuffle=False, batch_size=CONFIG['train_batch_size'])
    testloader = DataLoader(testset, shuffle=False, batch_size=CONFIG['test_batch_size'])
     
    # tensorboard
    writer = SummaryWriter(SAVE_PATH)

    # loss and optim
    criterion = HingeRankLoss(dataset=CONFIG['dataset'],
                              loss_args={"alpha": trainset.neg_weight, "gamma": CONFIG['gamma']})

    L_RATE = np.float64(CONFIG['l_rate'])
    L_RATE *= pow(0.9,CONFIG['start_epoch']-1)

    if CONFIG['freeze']:
        params=[
                {'params': model.classifier.parameters(), 'lr': L_RATE},
                {'params': G_model.parameters(), 'lr': L_RATE},
                {'params': D_model.parameters(), 'lr': L_RATE},
                {'params': D2_model.parameters(), 'lr': L_RATE}
                ]

    else:
        params = [
            {'params': model.parameters(), 'lr': L_RATE},
            {'params': G_model.parameters(), 'lr': L_RATE},
            {'params': D_model.parameters(), 'lr': L_RATE},
            {'params': D2_model.parameters(), 'lr': L_RATE}
        ]

    if CONFIG['optim'] == 'SGD':
        optimizer = optim.SGD(params, L_RATE, momentum=CONFIG['momentum'], weight_decay=CONFIG['weight_decay'])
    elif CONFIG['optim'] == 'Adam':
        optimizer = optim.Adam(params, L_RATE, weight_decay=CONFIG['weight_decay'])
    elif CONFIG['optim'] == 'RMSprop':
        optimizer = optim.RMSprop(params, L_RATE, weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # epoch
    train_vec = concept_set.vecs[:,:train_num]
    test_vec = concept_set.vecs[:,-test_num:]

    for epoch in range(CONFIG['start_epoch'], CONFIG['end_epoch']):
        # train        
        print("train")

        model.train()
        torch.set_grad_enabled(True)
        running_loss = 0.0
        features = torch.FloatTensor([]).cuda()

        for batch_i, batch_data in enumerate(zip_longest(trainloader,testloader), 1):
            
            classify_loss, classify_generated_unseen_loss, D1, G1, VAE, D2, G2, classify_generated_seen_loss, gradient_matching = 0,0,0,0,0,0,0,0,0
            
            optimizer.zero_grad()
            
            if batch_data[0]:
                #classification
                batch_img = batch_data[0]['image'].type(torch.cuda.FloatTensor)  

                if DATASET == 'openImage':
                    pos_list = batch_data[0]['pos']
                    neg_list = batch_data[0]['neg']

                    neg_list = [[int(e) for e in l[1:-1].split(', ')]  for l in neg_list ]
                    max_length = max([len(l) for l in neg_list])
                    neg_list = [l + l[-1:] * (max_length - len(l)) for l in neg_list]
                    label_list = torch.zeros(len(pos_list), train_num).scatter_(1, torch.tensor(neg_list), 1)
                    label_list*=-1

                    pos_list = [[int(e) for e in l[1:-1].split(', ')]  for l in pos_list ]
                    max_length = max([len(l) for l in pos_list])
                    pos_list = [l + l[-1:] * (max_length - len(l)) for l in pos_list]
                    label_list = label_list.scatter_(1, torch.tensor(pos_list), 1)
                    
                    batch_label = label_list
                else:
                    batch_label = batch_data[0]['label'][:,:train_num]

                inputs = torch.autograd.Variable(batch_img)
                inputs = model.feature_extraction(inputs).view(inputs.shape[0],-1)
                inputs.requires_grad=True

                outputs = model(inputs, train_vec)
                penalty=torch.mean(pow(1-torch.sum(outputs,axis=1),2))
                gradients1 = torch_grad(outputs=outputs, inputs=inputs,
                           grad_outputs=torch.ones(outputs.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]
                
                cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

                if DATASET == 'openImage':
                    loss = F.binary_cross_entropy(outputs, torch.floor((batch_label+1)/2).cuda(),weight=torch.abs(batch_label).cuda())
                else:
                    loss = criterion._loss(outputs, batch_label)
                
                classify_loss = loss
                feature = inputs

                #train VAE                    
                if GAN :
                    label_index, label = torch.nonzero(torch.floor((batch_label+1)/2), as_tuple=True)
                    label = torch.zeros(len(label), train_vec.shape[1]).scatter_(1, label.unsqueeze(-1), 1).cuda()
                    
                    refeature, mu, logvar = G_model(feature[label_index], label, train_vec)
                    VAE_loss = G_model._loss(refeature,feature[label_index],mu, logvar)
                    VAE = VAE_loss
                #train D1
                if GAN :
                    label_index, label = torch.nonzero(torch.floor((batch_label+1)/2), as_tuple=True)
                    label = torch.zeros(len(label), train_vec.shape[1]).scatter_(1, label.unsqueeze(-1), 1).cuda()
                    
                    g_features = G_model.generate(label,train_vec)
                    
                    fake_logits = D_model(g_features,label,train_vec)
                    true_logits = D_model(feature[label_index],label,train_vec)
                    
                    f_loss = D_model._loss(fake_logits)
                    t_loss = D_model._loss(true_logits)

                    d_loss = -t_loss+f_loss

                    alpha = torch.rand(label.shape[0], 1)
                    alpha = alpha.expand_as(feature[label_index]).cuda()
                    interpolated = alpha * feature[label_index] + (1 - alpha) * g_features
                    interpolated = Variable(interpolated, requires_grad=True)
                    interpolated = interpolated.cuda()
                    p_interpolated = D_model(interpolated, label.cuda(), train_vec)
                    gradients = torch_grad(outputs=p_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(p_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]
                    gradients = gradients.view(label.shape[0], -1)

                    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                    gp = ((gradients_norm - 1) ** 2).mean()
                    d_loss+= 10*gp
                    D1 = d_loss
                    
                    g_features = G_model.generate(label.cuda(), train_vec)
                    fake_logits = D_model(g_features,label.cuda(),train_vec)
                    g_loss = -D_model._loss(fake_logits)

                    G1 = g_loss
            
            if batch_data[1]:
                #train D2
                batch_img = batch_data[1]['image'].type(torch.cuda.FloatTensor)
                if DATASET == 'openImage':
                    label_list = batch_data[1]['pos']

                    label_list = [[int(e) for e in l[1:-1].split(', ')]  for l in label_list ]
                    max_length = max([len(l) for l in label_list])
                    label_list = [l + l[-1:] * (max_length - len(l)) for l in label_list]
                    label_list = torch.zeros(len(label_list), train_num+test_num).scatter_(1, torch.tensor(label_list), 1)
                    
                    batch_label2 = label_list[:,-test_num:]
                else:
                    batch_label2 = batch_data[1]['label'][:,-test_num:]
                
                inputs = torch.autograd.Variable(batch_img)
                feature = model.feature_extraction(inputs)

                if GAN :                
                    label_index, label = torch.nonzero(batch_label2, as_tuple=True)
                    label = torch.zeros(len(label), test_vec.shape[1]).scatter_(1, label.unsqueeze(-1), 1).cuda()
                    
                    #g means generate
                    g_label = torch.randint(0,test_vec.shape[1],(label.shape[0],1))
                    g_label = [[1 if j in g_label[i] else 0 for j in range(test_vec.shape[1])]for i in range(label.shape[0])]
                    g_label = torch.FloatTensor(g_label)
                    g_features = G_model.generate(g_label,test_vec)
                    
                    fake_logits = D2_model(g_features,torch.zeros(label.shape),test_vec)
                    true_logits = D2_model(feature[label_index],torch.zeros(label.shape),test_vec)

                    f_loss = D2_model._loss(fake_logits)
                    t_loss = D2_model._loss(true_logits)
                    d_loss = -t_loss+f_loss

                    alpha = torch.rand(label.shape[0], 1)
                    alpha = alpha.expand_as(feature[label_index]).cuda()
                    interpolated = alpha * feature[label_index] + (1 - alpha) * g_features
                    interpolated = Variable(interpolated, requires_grad=True)
                    interpolated = interpolated.cuda()
                    p_interpolated = D2_model(interpolated, torch.zeros(label.shape),test_vec)
                    gradients = torch_grad(outputs=p_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(p_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]
                    gradients = gradients.view(label.shape[0], -1)

                    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                    gp = ((gradients_norm - 1) ** 2).mean()
                    d_loss+= 10*gp

                    D2 = d_loss
                    
                    g_features = G_model.generate(label.cuda(), test_vec)
                    fake_logits = D2_model(g_features,torch.zeros(label.shape),test_vec)
                    g_loss = -D2_model._loss(fake_logits)

                    G2 = g_loss
            
            #classification for generate "seen" feature
            if GAN:
                num = batch_label.shape[0]
                #g means generate
                g_label = torch.randint(0,train_vec.shape[1],(num,1))
                g_label = [[1 if j in g_label[i] else 0 for j in range(train_vec.shape[1])]for i in range(num)]
                g_label = torch.FloatTensor(g_label)

                g_features = G_model.generate(g_label, train_vec)
                g_features = Variable(g_features ,requires_grad=True)
                g_outputs = model.classifier(g_features)

                semantic_transforms = model.classify(g_outputs, train_vec)
                penalty=torch.mean(pow(1-torch.sum(semantic_transforms,axis=1),2))
                g_loss = criterion._loss(semantic_transforms, g_label)
                classify_generated_seen_loss = g_loss
            
                gradients2 = torch_grad(outputs=semantic_transforms, inputs=g_features,
                       grad_outputs=torch.ones(semantic_transforms.size()).cuda(),
                       create_graph=True, retain_graph=True)[0]
                if batch_data[0]:
                    gradient_matching = 1-torch.mean(cos(gradients1,gradients2))
            
            #classification for generate "unseen" feature
            if GAN:
                #g means generate
                g_label = torch.randint(0,test_vec.shape[1],(inputs.shape[0],1))
                g_label = [[1 if j in g_label[i] else 0 for j in range(test_vec.shape[1])]for i in range(inputs.shape[0])]
                g_label = torch.FloatTensor(g_label)
                g_features = G_model.generate(g_label, test_vec)
                g_features = Variable(g_features ,requires_grad=True)
                
                g_outputs = model.classifier(g_features)
                semantic_transforms = model.classify(g_outputs, test_vec)
                g_loss = criterion._loss(semantic_transforms, g_label)
                classify_generated_unseen_loss = g_loss
                
     
            #hyper param    
            classify_param =10000# classification trainset 
            generated_param = 1000#classification generated set
            GAN1_param = 1#GAN w/ trainset
            VAE_param = 1000#VAE
            GAN2_param = 1#GAN w/ testset
            loss = classify_param*classify_loss + generated_param*(classify_generated_unseen_loss ) + GAN1_param*(D1+G1) + VAE_param*VAE + GAN2_param*(D2+G2)
            
            running_loss = loss.item()
            writer.add_scalar('train_loss', running_loss, batch_i + (epoch - 1) * len(trainloader))
            
            loss.backward()
            optimizer.step()
            
            print('[%d, %6d] loss: %.3f' % (epoch, batch_i * trainloader.batch_size, running_loss))
            
        torch.save(model.state_dict(), PJ(SAVE_PATH, 'epoch' + str(epoch) + '.pkl'))
        scheduler.step()
        
        #-----------------------------------
        # test        
        print("test")
        model.eval()
        torch.set_grad_enabled(False)

        predicts =[]
        gts = []
        top_record = {t: deque() for t in TOP_NUM}
        GZSL_top_record = {t: deque() for t in TOP_NUM}
        for batch_i, batch_data in enumerate(testloader, 1):
            batch_img = batch_data['image'].type(torch.cuda.FloatTensor)
            if DATASET == 'openImage':
                pos_list = batch_data['pos']
                neg_list = batch_data['neg']

                neg_list = [[int(e) for e in l[1:-1].split(', ')]  for l in neg_list ]
                max_length = max([len(l) for l in neg_list])
                neg_list = [l + l[-1:] * (max_length - len(l)) for l in neg_list]
                label_list = torch.zeros(len(pos_list), train_num+test_num).scatter_(1, torch.tensor(neg_list), 1)
                label_list*=-1

                pos_list = [[int(e) for e in l[1:-1].split(', ')]  for l in pos_list ]
                max_length = max([len(l) for l in pos_list])
                pos_list = [l + l[-1:] * (max_length - len(l)) for l in pos_list]
                label_list = label_list.scatter_(1, torch.tensor(pos_list), 1)
                batch_label = label_list
                label_list=[]
            else:
                batch_label = batch_data['label']
            
            inputs = torch.autograd.Variable(batch_img)
            inputs = model.feature_extraction(inputs).view(batch_img.shape[0],-1)
            outputs = model(inputs, concept_set.vecs[:,:])
            
            # record to calculate map_extraction
            predicts.append(torch.squeeze(outputs).tolist())
            gts.append(torch.squeeze(batch_label).tolist())

            if DATASET == 'openImage':
                pass
            else:
                zsl_label=batch_label[:,-test_num:]
                zsl_idx=torch.nonzero(zsl_label,as_tuple=True)[0]
                zsl_idx=torch.unique(zsl_idx)

                rank_inds = (-torch.squeeze(outputs[zsl_idx,-test_num:])).sort()[1]
                for rank_ind, ground_truth in zip(rank_inds, batch_label[zsl_idx,-test_num:]):
                    for t in TOP_NUM:
                        predict = [1 if i in rank_ind[:t] else 0 for i in range(len(rank_ind))]
                        top_record[t].append([ground_truth.tolist(), predict])
                GZSL_rank_inds = (-torch.squeeze(outputs)).sort()[1]
                for rank_ind, ground_truth in zip(GZSL_rank_inds, batch_label):
                    for t in TOP_NUM:
                        predict = [1 if i in rank_ind[:t] else 0 for i in range(len(rank_ind))]
                        GZSL_top_record[t].append([ground_truth.tolist(), predict])
                        
            print('batch:', batch_i * testloader.batch_size)            
        # evaluation
        # cal map

        predicts = np.concatenate(predicts)
        gts = np.concatenate(gts)
        if DATASET == 'openImage':
            mAPs = utils.compute_AP(predicts[:,-test_num:], gts[:,-test_num:])
        else:
            mAPs = utils.cal_mAP(predicts[:,-test_num:], gts[:,-test_num:])
        text = '| Class |  AP  |\n| :---: | :--: |\n'
        ap_texts = ['| ' + concept_set.names[train_num+col] + ' | ' + '{:.2f}'.format(ap * 100) + ' |' for col, ap in enumerate(mAPs)]
        text += '\n'.join(ap_texts)

        writer.add_text('AP Table', text, epoch)
        writer.add_scalar('mAP', (sum(mAPs) / len(mAPs) * 100), epoch)

        # cal miap
        if DATASET == 'openImage':
            miAP = utils.compute_miAP(predicts[:,-test_num:], gts[:,-test_num:])
        else:
            miAP = utils.cal_miAP(predicts[:,-test_num:], gts[:,-test_num:])
        writer.add_scalar('miAP', (miAP * 100), epoch)

        # cal precision and recall
        text = '| Tables            |  CP  |  CR  |  CF  |  OP  |  OR  |  OF  |\n\
                | :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |\n'
        for t in TOP_NUM:
            if DATASET == 'openImage':
                cf_, cp_, cr_ = utils.evaluate_k(t,predicts[:,-test_num:], gts[:,-test_num:], 'overall')
                of_, op_, or_ = utils.evaluate_k(t,predicts[:,-test_num:], gts[:,-test_num:], 'overall')
            else:
                cp_, cr_, cf_, op_, or_, of_ = utils.cal_rp(top_record[t])
            text += '| top' + str(t) + ' | ' + '{:.2f}'.format(cp_ * 100) + ' | ' + '{:.2f}'.format(cr_ * 100) + \
                    ' | ' + '{:.2f}'.format(cf_ * 100) + ' | ' + '{:.2f}'.format(op_ * 100) + \
                    ' | ' + '{:.2f}'.format(or_ * 100) + ' | ' + '{:.2f}'.format(of_ * 100) + ' |\n'
        writer.add_text('Test Table', text, epoch)
        # GZSL evaluation
        # cal map

        if DATASET == 'openImage':
            mAPs = utils.compute_AP(predicts, gts)
        else:
            mAPs = utils.cal_mAP(predicts, gts)
        text = '| Class |  AP  |\n| :---: | :--: |\n'
        ap_texts = ['| ' + concept_set.names[col] + ' | ' + '{:.2f}'.format(ap * 100) + ' |' for col, ap in enumerate(mAPs)]
        text += '\n'.join(ap_texts)

        writer.add_text('GZSL_AP Table', text, epoch)
        writer.add_scalar('GZSL_mAP', (sum(mAPs) / len(mAPs) * 100), epoch)

        # cal miap
        if DATASET == 'openImage':
            miAP = utils.compute_miAP(predicts, gts)
        else:
            miAP = utils.cal_miAP(predicts, gts)
        writer.add_scalar('GZSL_miAP', (miAP * 100), epoch)

        # cal precision and recall
        text = '| Tables            |  CP  |  CR  |  CF  |  OP  |  OR  |  OF  |\n\
                | :---------------: | :--: | :--: | :--: | :--: | :--: | :--: |\n'
        for t in TOP_NUM:
            if DATASET == 'openImage':
                cf_, cp_, cr_ = utils.evaluate_k(t,predicts, gts, 'overall')
                of_, op_, or_ = utils.evaluate_k(t,predicts, gts, 'overall')
            else:
                cp_, cr_, cf_, op_, or_, of_ = utils.cal_rp(GZSL_top_record[t])
            text += '| top' + str(t) + ' | ' + '{:.2f}'.format(cp_ * 100) + ' | ' + '{:.2f}'.format(cr_ * 100) + \
                    ' | ' + '{:.2f}'.format(cf_ * 100) + ' | ' + '{:.2f}'.format(op_ * 100) + \
                    ' | ' + '{:.2f}'.format(or_ * 100) + ' | ' + '{:.2f}'.format(of_ * 100) + ' |\n'
        writer.add_text('GZSL_Test Table', text, epoch)
        
    writer.close()
