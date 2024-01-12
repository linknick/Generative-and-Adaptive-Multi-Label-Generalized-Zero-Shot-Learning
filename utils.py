import numpy as np
import pandas as pd
import math
from sklearn.metrics import average_precision_score,f1_score,precision_score,recall_score

def mask_unlabeled(predictions,labels):
    const = np.min(predictions)-0.1
    m_predictions = predictions-const
    m_predictions = np.multiply(m_predictions,np.abs(labels))
    return m_predictions
def evaluate_k(k, predictions=None,labels=None,mode_F1='overall'):
    
    predictions = predictions.copy()
    predictions=mask_unlabeled(predictions,labels)
    ## binarize ##
    idx = np.argsort(predictions,axis = 1)
    for i in range(predictions.shape[0]):
        predictions[i][idx[i][-k:]]=1
        predictions[i][idx[i][:-k]]=0
    ## binarize ##
    return compute_F1(predictions,labels,mode_F1)

def compute_F1(predictions,labels,mode_F1):
    if mode_F1 == 'overall':
        mask = predictions == 1
        TP = np.sum(labels[mask]==1)
        p = TP/np.sum(mask)
        r = TP/np.sum(labels==1)
        f1 = 2*p*r/(p+r)
        
#        p_2,r_2,f1_2=compute_F1_fast0tag(predictions,labels)
    else:
        num_class = predictions.shape[1]
        f1 = np.zeros(num_class)
        p = np.zeros(num_class)
        r  = np.zeros(num_class)
        for idx_cls in range(num_class):
            prediction = np.squeeze(predictions[:,idx_cls])
            label = np.squeeze(labels[:,idx_cls]) 
            mask = np.abs(label)==1
            if np.sum(label>0)==0:
                continue
            binary_label=np.clip(label,0,1)
            f1[idx_cls] = f1_score(binary_label,prediction)#AP(prediction,label,names)
           
            p[idx_cls] = precision_score(binary_label,prediction)
            r[idx_cls] = recall_score(binary_label,prediction)
        p=np.sum(p)/num_class
        r=np.sum(r)/num_class
        f1=2*p*r/(p+r)
    return f1,p,r
def compute_AP(predictions,labels):
    num_class = predictions.shape[1]
    ap=np.zeros(num_class)
    for idx_cls in range(num_class):
        prediction = np.squeeze(predictions[:,idx_cls])
        label = np.squeeze(labels[:,idx_cls])
        mask = np.abs(label)==1
        if np.sum(label>0)==0:
            continue
        binary_label=np.clip(label[mask],0,1)
        ap[idx_cls]=average_precision_score(binary_label,prediction[mask])#AP(prediction,label,names)
    return ap
def compute_miAP(predictions,labels):
    iAPs = []
    for predict, gt in zip(predictions, labels):
        mask = np.abs(gt)==1
        predict = predict[mask]
        gt = gt[mask]
        gt = np.where(gt == 1)[0]
        predict_list = np.argsort(-predict)
        if predict_list.sum() == 0:
            iAPs.append(0)
            continue

        idx = np.array(sorted([np.nonzero(predict_list == g)[0][0] + 1 for g in gt]))
        num_hits = np.cumsum([1 for p in np.argsort(-predict) if p in gt])
        scores = num_hits / idx

        iap = sum(scores) / (len(scores) + np.finfo(float).eps)
        iAPs.append(iap)
    miAP = np.mean(iAPs)
    return miAP
def cal_rp(top_record):
    top_record = np.asarray(top_record)

    sucess_per_class = np.logical_and(top_record[:, 0], top_record[:, 1]).sum(axis=0)
    p_per_class = top_record.sum(axis=0)[1]
    g_per_class = top_record.sum(axis=0)[0]
    #print(np.nan_to_num(sucess_per_class / p_per_class).sum(),len(g_per_class))
    cp_ = np.nan_to_num(sucess_per_class / p_per_class).sum() / len(g_per_class)
    cr_ = np.nan_to_num(sucess_per_class / g_per_class).sum() / len(g_per_class)
    cf_ = 2 * cp_ * cr_ / (cp_+ cr_)

    op_ = np.nan_to_num(sucess_per_class.sum() / p_per_class.sum())
    or_ = np.nan_to_num(sucess_per_class.sum() / g_per_class.sum())
    of_ = 2 * op_ * or_ / (op_ + or_)

    return [cp_, cr_, cf_, op_, or_, of_]


def cal_miAP(predicts, gts):
    gts = np.asarray(gts)
    predicts = np.asarray(predicts)
    iAPs = []
    for predict, gt in zip(predicts, gts):
        #print(predict,gt)
        gt = np.where(gt == 1)[0]
        predict_list = np.argsort(-predict)
        if predict_list.sum() == 0:
            iAPs.append(0)
            continue

        idx = np.array(sorted([np.nonzero(predict_list == g)[0][0] + 1 for g in gt]))
        num_hits = np.cumsum([1 for p in np.argsort(-predict) if p in gt])
        scores = num_hits / idx

        iap = sum(scores) / (len(scores) + np.finfo(float).eps)
        iAPs.append(iap)

    miAP = np.mean(iAPs)
    return miAP


def cal_mAP(predicts, gts):
    try:
        gts = np.asarray(gts)
        predicts = np.asarray(predicts)
    except:
        pass
    mAPs = []

    _nan=[]
    _zap=[]
    for col in range(gts.shape[1]):
    #for col in range(len(gts[0])):
        predict = predicts[:, col]
        ground_truth = gts[:, col]
        """
        tp = np.zeros(len(ground_truth))
        fp = np.zeros(len(ground_truth))

        tp[ground_truth[np.argsort(-predict)] == 1] = 1
        fp[ground_truth[np.argsort(-predict)] == 0] = 1

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        recall = tp / (sum(ground_truth == 1) + np.finfo(float).eps)
        precision = tp / (fp + tp + np.finfo(float).eps)
        ap = sum([0 if len(precision[recall >= (t * 0.1)]) == 0 else max(precision[recall >= (t * 0.1)]) for t in range(11)]) / 11
        mAPs.append(ap)
        """
        ap = average_precision_score(ground_truth,predict)
        _zap.append(sum(ground_truth))
        if math.isnan(ap):
            
            ap=0.5
            _nan.append(col)
        elif ap==0:
            _zap.append(col)

        else:
            pass
        mAPs.append(ap)
        #print("NAN:",_nan)
        #print("0AP:",_zap)
    return mAPs
