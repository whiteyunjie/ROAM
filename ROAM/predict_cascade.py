
from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import loss
from torch.utils.data import DataLoader,WeightedRandomSampler
import numpy as np
import h5py
import os
import shutil
import sys
import json
import random

from tqdm import tqdm

from dataset.roidataset import Wsi_Dataset_pred
from models.ROAM import ROAM
from parse_config import parse_args

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

cls_name_dict = {
    'int_glioma_tumor_subtyping':['astrocytoma','oligodendroglioma','ependymoma']
}


def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def read_features(feat_path):
    with h5py.File(feat_path,'r') as hdf5_file:
        features = hdf5_file['features'][:] # num_patches,84,1024
    return torch.from_numpy(features)

def weights_for_balanced_class(train_dataset,weight_cls):
    n = float(len(train_dataset))
    weight = [0]*int(n)
    for idx in range(len(train_dataset)):
        label = train_dataset.get_label(idx)
        weight[idx] = weight_cls[label]
    return weight



def val_epoch(args,model,loader,loss_fn,epoch):
    val_loss = []
    val_acc = []
    preds = []
    trues = []
    probs = []
    total_loss = 0.0
    model.eval()

    with torch.no_grad():
        progressbar = tqdm(loader)
        for i,(feature,_,label) in enumerate(progressbar):
            feature, label = feature.cuda(),label.cuda()
            logits,loss_instance = model(feature,label,inst_level=False) # only slide
            pred = logits.argmax(1).cpu()

            loss_bag = loss_fn(logits,label)

            #print(loss_instance)
            loss = loss_bag

            val_loss.append(loss.item())
            
            if pred == label.cpu():
                val_acc.append(1)
            else:
                val_acc.append(0)
            
            preds.append(float(pred))
            trues.append(int(label.cpu()))
            probs.append(torch.nn.functional.softmax(logits, dim=1).cpu())

            progressbar.set_description(f'epoch: {epoch}, val_acc: {np.mean(val_acc):.4f}, val_loss: {np.mean(val_loss):.4f}, current_inst_loss: {float(loss_instance):.4f}, current_bag_loss: {float(loss_bag):.4f}')

            del loss

        probs = torch.cat(probs)

        return np.mean(val_acc),np.mean(val_loss),preds,trues,probs

def pred_epoch(args,model,loader,epoch):
    preds = []
    probs = []
    model.eval()

    with torch.no_grad():
        progressbar = tqdm(loader)
        for i,(feature,_) in enumerate(progressbar):
            feature = feature.cuda()
            logits,_ = model(feature,inst_level=False) # only slide
            pred = logits.argmax(1).cpu()
            
            preds.append(float(pred))
            probs.append(torch.nn.functional.softmax(logits, dim=1).cpu())

            progressbar.set_description(f'epoch: {epoch}, pred:{float(pred)}')


        probs = torch.cat(probs)

        return preds,probs

if __name__ == "__main__":
    assert len(sys.argv) in [3,4], 'please give configuration file and split seed!'
    ### split seed [s1, s2, s3, s4, s5]

    args, task_info = parse_args(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        args.exp_code = sys.argv[3]
    print(f'exp_code: {args.exp_code}')
    print(f'split seed: {args.split_seed}')

        
    args.n_classes = task_info[args.task]['n_classes']
    if args.task not in task_info:
        raise NotImplementedError
    
    ## ==> save_dir
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    
    args.results_dir = os.path.join(args.results_dir,args.task)
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)
    
    args.results_dir = os.path.join(args.results_dir,f'{args.exp_code}')
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    shutil.copy(sys.argv[1], args.results_dir)
    with open(sys.argv[1], 'r') as f:
        for l in f.readlines():
            print(l.strip())

    args.results_dir = os.path.join(args.results_dir,f'{args.split_seed}')
    if not os.path.exists(args.results_dir):
        os.mkdir(args.results_dir)

    ## pred results save dir
    pred_result_dir = f'prediction_results/{args.split_seed}/{args.task}'
    if not os.path.exists(pred_result_dir):
        os.makedirs(pred_result_dir)
    
    

    if args.embed_type == 'ImageNet':
        patch_dim = 1024
    if args.embed_type == 'RetCCL':
        patch_dim = 2048
    if args.embed_type == 'ctranspath':
        patch_dim = 768
    if args.embed_type == 'simclr-ciga':
        patch_dim = 512



    ### ==> test dataset
    # confirm test slide list
    default_test_tasks = ['int_glioma_cls','int_idh_cls','int_mgmt_cls']

    ## read test slide list file
    test_split_dir = task_info[args.task]['test_split_dir']
    if args.task not in default_test_tasks:
        cascade_test_dir = os.path.join(f'prediction_results/{args.split_seed}',f'cascade_{args.task}_split.npy')
        if os.path.exists(cascade_test_dir):
            test_split_dir = cascade_test_dir
    
    # if args.task in default_test_tasks:
    #     test_split_dir = task_info[args.task]['test_split_dir'] #xiangya
    # else:
    #     # use the results from the previous layer of the cascade system for prediction
    #     test_split_dir = os.path.join(f'prediction_results/{args.split_seed}',f'cascade_{args.task}_split.npy')
    
    ## original test slide list file has label, ignore
    test_info = np.load(test_split_dir)
    test_ids = test_info[0] if len(test_info)==2 else test_info


    data_dir = f'{args.data_root_dir}/feats_{args.embed_type}_norm'
    if args.test_dataset == 'xiangya':
        test_dataset = Wsi_Dataset_pred(slide_ids=test_ids,
                                        csv_path=task_info[args.task]['csv_path'],
                                        data_dir=data_dir)
        test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)
    if args.test_dataset == 'TCGA':
        test_ids,test_labels = np.load(task_info[args.task]['test_split_dir_ext'])
        
        test_dataset = Wsi_Dataset_pred(slide_ids = test_ids,
                                    csv_path = task_info[args.task]['csv_path'],
                                    data_dir = data_dir,
                                    )
        test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)
    
    ### ==> training

    num_folds = 5
    results = {k: {} for k in range(num_folds)}
    probs_all = []

    ## whether setting fixed weights of embedding for each level
    if args.embed_weightx5==None and args.embed_weightx10==None and args.embed_weightx20==None:
        embed_weights = None
        print('use learnabel weights')
    else:
        embed_weights = [args.embed_weightx5,args.embed_weightx10,args.embed_weightx20]
        print('set weights:', embed_weights)
    
    for k in range(num_folds):
        print(f'split: {k}')

        ## set seed
        seed_torch(args.seed)

        # model
        model = ROAM(choose_num = args.topk,
                num_patches = 84,
                patch_dim=patch_dim,
                num_classes=task_info[args.task]['n_classes'],
                roi_level = args.roi_level,
                scale_type = args.scale_type,
                single_level = args.single_level,
                embed_weights=embed_weights,
                dim=args.dim,
                depths=args.depths,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                not_interscale = args.not_interscale,
                dim_head=args.dim_head,
                dropout=args.dropout,
                emb_dropout=args.emb_dropout,
                attn_dropout=args.attn_dropout,
                pool=args.pool,
                ape=args.ape,
                attn_type=args.attn_type,
                shared_pe=args.shared_pe)
        
        model = model.cuda()

        loss_fn = nn.CrossEntropyLoss()
        
        model_path = os.path.join(args.results_dir,f'{args.model_type}_split{str(k)}.pth')
        assert os.path.exists(model_path), 'No trained model checkpoint!'
        model.load_state_dict(torch.load(model_path))
        
        preds,probs = pred_epoch(args,model,test_loader,1)


        results[k]['preds'] = preds

        probs_all.append(probs)

        print(f'end of prediction of split_{k}')

   

    probs_all = torch.stack(probs_all)

    probs_mean = probs_all.mean(0) #b,n_classes
    preds_mean = probs_mean.argmax(1)
    
    results['test'] = {}
    results['test']['preds'] = preds_mean.tolist()
    results['test']['probs'] = probs_mean.tolist()

    final_preds = {}
    cls_names = cls_name_dict[args.task]
    print('predict results:')
    for idx,sid in enumerate(test_ids):
        final_preds[sid] = cls_names[results['test']['preds'][idx]]
        print(f'{sid}:{final_preds[sid]}')

    pred_res_dir = os.path.join(pred_result_dir,'predictions.json')
    print(f'save the predictions to {pred_res_dir}')
    with open(os.path.join(pred_result_dir,'predictions.json'),'w') as f:
        json.dump(final_preds,f)



    with open(os.path.join(pred_result_dir,'results.json'),'w') as f:
        json.dump(results,f)
    
    ## results split: prepare for next level prediction
    preds_res = np.array(results['test']['preds'])
    #print('preds_res:',preds_res)
    if args.task == 'int_glioma_cls':
        slide_ids_c = test_ids[preds_res == 2]
        np.save(os.path.join(f'prediction_results/{args.split_seed}','cascade_int_glioma_tumor_subtyping_split.npy'),slide_ids_c)
    elif args.task == 'int_glioma_tumor_subtyping':
        cat_names = ['int_ast_grade1','int_oli_grade','int_epe_grade']    
        for c in range(task_info[args.task]['n_classes']):
            slide_ids_c = test_ids[preds_res == c]
            if len(slide_ids_c)>0:     
                np.save(os.path.join(f'prediction_results/{args.split_seed}',f'cascade_{cat_names[c]}_split.npy'),slide_ids_c)

        



    

    
