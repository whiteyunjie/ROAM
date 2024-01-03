from unicodedata import name
import numpy as np
import pandas as pd
import pickle
import json
import os
import sys
from sklearn import metrics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

## class names for each task
cls_name_dict = {
    'int_glioma_tumor_subtyping':['astrocytoma','oligodendroglioma','ependymoma'],
    'ext_glioma_tumor_subtyping3':['astrocytoma','oligodendroglioma'],
    'int_glioma_cls':['normal','gliosis','tumor']
}

def getmetric(cm,num_cls):
    tr= np.trace(cm)
    precs = []
    recs = []
    f1scores = []
    for cid in range(num_cls):
        TP = cm[cid,cid]
        #TN = tr-cm[cid,cid]
        gt = cm[cid] #tp+fn
        pred = cm[:,cid] #tp+fp

        prec = TP/np.sum(pred)
        rec = TP/np.sum(gt)
        f1score = 2 * prec*rec/(prec+rec)

        precs.append(prec)
        recs.append(rec)
        f1scores.append(f1score)
    return np.mean(precs),np.mean(recs),np.mean(f1scores)


def compute_metric_results(args,task):
    metric = {}

    savepath = os.path.join(args.results_dir,'visual_res')
    respath = os.path.join(args.results_dir,'results.json')

    os.makedirs(savepath,exist_ok=True)
    
    with open(respath,'r') as f:
        res = json.load(f)

    test_acc = res['test']['acc']

    clsnames = cls_name_dict[task]


    cls_num = len(clsnames)
    gt = res['test']['trues']
    pred = res['test']['preds']
    cm = confusion_matrix(gt,pred)
    acc_b = balanced_accuracy_score(gt,pred)
    print(cm)

    prec,rec,f1score = getmetric(cm,cls_num)
    print(f'prec:{prec},recall:{rec},f1_score:{f1score}')
    metric['acc'] = test_acc
    metric['precision'] = prec
    metric['recall'] = rec
    metric['f1_score'] = f1score
    metric['balanced_accuracy'] = acc_b

    cm_normal = cm/cm.sum(axis=1)[:,np.newaxis]
    # confusion matrix      
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True,cmap='Blues')
    plt.title(f'confusion matrix mean')

    plt.xlabel('predicted labels')
    plt.ylabel('ground truth labels')
    xlocations = np.array(range(len(clsnames))) + 0.5
    plt.xticks(xlocations,clsnames)
    plt.yticks(xlocations,clsnames,rotation = 90)

    plt.savefig(os.path.join(savepath,'cm_mean.png'))

    # normalized confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_normal,annot=True,cmap='Blues')
    plt.title(f'confusion matrix mean')

    plt.xlabel('predicted labels')
    plt.ylabel('ground truth labels')
    xlocations = np.array(range(len(clsnames))) + 0.5
    plt.xticks(xlocations,clsnames)
    plt.yticks(xlocations,clsnames,rotation = 90)

    plt.savefig(os.path.join(savepath,'normal_cm_mean.png'))


    with open(os.path.join(savepath,'metrics.json'),'w') as f:
        json.dump(metric,f)


# compute metics for specific seed
def compute_metric_results_seed(exp_code,task,seed):
    metric = {}

    results_dir = f'results/{task}/{exp_code}/{seed}'
    savepath = os.path.join(results_dir,'visual_res')
    respath = os.path.join(results_dir,'results.json')

    os.makedirs(savepath,exist_ok=True)
    
    with open(respath,'r') as f:
        res = json.load(f)

    test_acc = res['test']['acc']

    clsnames = cls_name_dict[task]


    cls_num = len(clsnames)
    gt = res['test']['trues']
    pred = res['test']['preds']
    cm = confusion_matrix(gt,pred)
    acc_b = balanced_accuracy_score(gt,pred)
    print(cm)

    prec,rec,f1score = getmetric(cm,cls_num)
    print(f'prec:{prec},recall:{rec},f1_score:{f1score}')
    metric['acc'] = test_acc
    metric['precision'] = prec
    metric['recall'] = rec
    metric['f1_score'] = f1score
    metric['balanced_accuracy'] = acc_b

    cm_normal = cm/cm.sum(axis=1)[:,np.newaxis]
    # confusion matrix      
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True,cmap='Blues')
    plt.title(f'confusion matrix mean')

    plt.xlabel('predicted labels')
    plt.ylabel('ground truth labels')
    xlocations = np.array(range(len(clsnames))) + 0.5
    plt.xticks(xlocations,clsnames)
    plt.yticks(xlocations,clsnames,rotation = 90)

    plt.savefig(os.path.join(savepath,'cm_mean.png'))

    # normalized confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(cm_normal,annot=True,cmap='Blues')
    plt.title(f'confusion matrix mean')

    plt.xlabel('predicted labels')
    plt.ylabel('ground truth labels')
    xlocations = np.array(range(len(clsnames))) + 0.5
    plt.xticks(xlocations,clsnames)
    plt.yticks(xlocations,clsnames,rotation = 90)

    plt.savefig(os.path.join(savepath,'normal_cm_mean.png'))


    with open(os.path.join(savepath,'metrics.json'),'w') as f:
        json.dump(metric,f)


if __name__ == '__main__':
    exp_name = sys.argv[1]
    task = sys.argv[2]
    seed = sys.argv[3]

    compute_metric_results_seed(exp_name,task,seed)
