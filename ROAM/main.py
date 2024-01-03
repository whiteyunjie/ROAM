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
import pytorch_warmup as warmup
## internal imports
from dataset.roidataset import Wsi_Dataset_sb, Wsi_Dataset_mb
from models.PTMIL import PTMIL
from compute_metric import compute_metric_results
from parse_config import parse_args

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'

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
    torch.set_num_threads(1) # control number of cpu threads

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

def train_epoch(args,model,loader,optimizer,loss_fn,epoch,warmup_scheduler):
    train_loss = []
    train_bag_loss = []
    train_instance_loss = []

    train_acc = []
    progressbar = tqdm(loader)
    model.train()

    for i,(feat_paths,labels) in enumerate(progressbar):
        
        optimizer.zero_grad()

        total_loss = 0.0
        total_bag_loss = 0.0
        total_instance_loss = 0.0

        preds = []

        for b in range(len(feat_paths)):
            feature = read_features(feat_paths[b])
            
            if args.roi_dropout:
                tmp_num_inst = feature.shape[0]
                if tmp_num_inst >= 16:
                    tmp_idx = random.sample(range(tmp_num_inst), int(0.8*tmp_num_inst))
                    feature = feature[tmp_idx]

            feature = feature.cuda()
            feature = feature.unsqueeze(0)
            label = labels[b:b+1].cuda()

            if args.roi_supervise:
                logits,loss_instance = model(feature,label,inst_level=True)
            else:
                logits,loss_instance = model(feature,label,inst_level=False)

            pred = logits.argmax(1).cpu()
            loss_bag = loss_fn(logits,label)
            loss = loss_bag + args.roi_weight*loss_instance

            total_loss += loss
            total_bag_loss += loss_bag
            total_instance_loss += args.roi_weight*loss_instance

            preds.append(pred)
        total_loss /= args.batch_size
        total_bag_loss /= args.batch_size
        total_instance_loss /= args.batch_size
        total_loss.backward()
        
        optimizer.step()
        if i < len(loader) - 1:
            with warmup_scheduler.dampening():
                pass



        train_loss.append(float(total_loss))
        train_bag_loss.append(float(total_bag_loss))
        train_instance_loss.append(float(total_instance_loss))
        
        preds = torch.tensor(preds)
        acc = torch.sum(preds==labels.cpu())/args.batch_size
        train_acc.append(float(acc))

        progressbar.set_description(f'epoch: {epoch}, train_acc: {np.mean(train_acc):.4f}, train_loss: {np.mean(train_loss):.4f}, train_bag_loss: {np.mean(train_bag_loss):.4f}, train_instance_loss: {np.mean(train_instance_loss):.4f}')

        del loss

    return np.mean(train_acc),np.mean(train_loss)

def val_epoch(args,model,loader,loss_fn,epoch):
    val_loss = []
    val_acc = []
    preds = []
    trues = []
    probs = []
    total_loss = 0.0
    model.eval()
    if args.scale_type != 'ss' and args.embed_weightx5==None:
        print(torch.softmax(model.vit.learned_weights, dim=0))
        print(model.vit.learned_weights)

    with torch.no_grad():
        progressbar = tqdm(loader)
        for i,(feature,_,label) in enumerate(progressbar):
            feature, label = feature.cuda(),label.cuda()
            logits,loss_instance = model(feature,label,inst_level=False)
            pred = logits.argmax(1).cpu()
            loss_bag = loss_fn(logits,label)

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



if __name__ == "__main__":
    ## read configuration file
    assert len(sys.argv) in [3,4], 'please give configuration file and split seed!'
    args, task_info = parse_args(sys.argv[1], sys.argv[2])
    if len(sys.argv) == 4:
        args.exp_code = sys.argv[3]
    print(f'exp_code: {args.exp_code}')
    print(f'split seed: {args.split_seed}')

        
    args.n_classes = task_info[args.task]['n_classes']
    if args.task not in task_info:
        raise NotImplementedError
    
    ## create results directory
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
    
    
    ## select pre-trained model
    if args.embed_type == 'ImageNet':
        patch_dim = 1024
    if args.embed_type == 'RetCCL':
        patch_dim = 2048
    if args.embed_type == 'ctranspath':
        patch_dim = 768
    if args.embed_type == 'simclr-ciga':
        patch_dim = 512


    ## generate test dataset
    if args.test_dataset == 'xiangya':
        test_ids,test_labels = np.load(task_info[args.task]['test_split_dir'])
        test_labels = test_labels.astype(np.int16)
        data_dir = f'{args.data_root_dir}/{args.test_dataset}/feats_{args.roi_level}/feats_{args.embed_type}_norm'
        test_dataset = Wsi_Dataset_sb(slide_ids = test_ids, label_ids= test_labels,
                                    csv_path = task_info[args.task]['csv_path'],
                                    data_dir = data_dir,
                                    label_dict = task_info[args.task]['label_dict'])
        test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)
    if args.test_dataset == 'TCGA':
        test_ids,test_labels = np.load(task_info[args.task]['test_split_dir_ext'])
        data_dir = f'{args.data_root_dir}/{args.test_dataset}/feats_{args.roi_level}/feats_{args.embed_type}_norm'
        test_dataset = Wsi_Dataset_sb(slide_ids = test_ids, label_ids= test_labels,
                                    csv_path = '../data_prepare/data_csv/tcga_data_info_pro.csv',
                                    data_dir = data_dir,
                                    label_dict = task_info[args.task]['label_dict_ext'])
        test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=4)



    ## train
    # weighted sampler for class balancing
    weights_cls = task_info[args.task]['cls_weights']
    weights_cls = np.sum(weights_cls)/weights_cls

    data_split = np.load(task_info[args.task]['split_dir'],allow_pickle=True)
    num_folds = data_split.shape[0]
    results = {k: {} for k in range(num_folds)}
    probs_all = []

    # set weights of embedding for each level
    if args.embed_weightx5==None and args.embed_weightx10==None and args.embed_weightx20==None:
        embed_weights = None
        print('use learnabel weights')
    else:
        embed_weights = [args.embed_weightx5,args.embed_weightx10,args.embed_weightx20]
        print('set weights:', embed_weights)
    
    for k in range(num_folds):
        print(f'split: {k}')
        seed_torch(args.seed)

        model = PTMIL(choose_num = args.topk,
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

        if args.stage == 'train':
            train_x,train_y,val_x,val_y = data_split[k]
            
            data_dir = f'{args.data_root_dir}/xiangya/feats_{args.roi_level}/feats_{args.embed_type}_norm'

            train_dataset = Wsi_Dataset_mb(slide_ids = train_x, label_ids=train_y,
                                        csv_path = task_info[args.task]['csv_path'],
                                        data_dir = data_dir,#'../../../../stor2/yinxiaoxu/glioma/feat/roi',
                                        label_dict = task_info[args.task]['label_dict'])
            if args.weighted_sample:
                weights = weights_for_balanced_class(train_dataset,weights_cls)
                train_loader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=False,shuffle=False,num_workers=4,sampler=WeightedRandomSampler(weights, len(weights)))
            else:
                train_loader = DataLoader(train_dataset,batch_size=args.batch_size,drop_last=False,shuffle=True,num_workers=4)

            val_dataset = Wsi_Dataset_sb(slide_ids = val_x, label_ids=val_y,
                                        csv_path = task_info[args.task]['csv_path'],
                                        data_dir = data_dir,#'../../../../stor2/yinxiaoxu/glioma/feat/roi',
                                        label_dict = task_info[args.task]['label_dict'])
            val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False,num_workers=4)

            if args.optimizer == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'adamw':
                
                '''
                # learnable weights of embeddings for each level
                print('different lr')
                weight_param = list(map(id,[model.vit.learned_weights]))
                print(weight_param)
                base_params = filter(lambda p: id(p) not in weight_param, model.parameters())
                print(list(base_params))
                optimizer = optim.AdamW([{'params':model.vit.learned_weights,'lr':2*args.lr},
                                        {'params':base_params,'lr':args.lr}],weight_decay=args.weight_decay)
                '''
                optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                        weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(model.parameters(),lr=args.lr,
                                      momentum=0.9,
                                      weight_decay=args.weight_decay)
            else:
                raise NotImplementedError
            
            if args.scheduler == 'onplateau': 
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=10, verbose=True)
            elif args.scheduler == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, verbose=True)
            elif args.scheduler == 'none':
                scheduler = None
            else:
                raise NotImplementedError

            warmup_scheduler = warmup.LinearWarmup(optimizer, 5*len(train_loader))



            best_acc = 0
            best_loss = 100
            stop_epochs = args.stop_epochs
            train_acc_all = []
            train_loss_all = []
            val_acc_all = []
            val_loss_all = []

            count = 0
            for epoch in range(args.max_epochs):
                print(optimizer.param_groups[0]['lr'])
                train_acc,train_loss = train_epoch(args,model,train_loader,optimizer,loss_fn,epoch,warmup_scheduler)
                val_acc,val_loss,_,_,_ = val_epoch(args,model,val_loader,loss_fn,epoch)

                
                                
                train_acc_all.append(train_acc)
                train_loss_all.append(train_loss)

                val_acc_all.append(val_acc)
                val_loss_all.append(val_loss)

                if epoch<5:
                    continue
                
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    with warmup_scheduler.dampening():
                        scheduler.step(val_loss)
                elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    with warmup_scheduler.dampening():
                        scheduler.step()
                elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    with warmup_scheduler.dampening():
                        scheduler.step()
                

                if best_loss > val_loss:
                    best_loss = val_loss
                    count = 0
                    # save best pkl
                    model_path = os.path.join(args.results_dir,f'{args.model_type}_split{str(k)}.pth')
                    torch.save(model.state_dict(),model_path)
                    results[k]['best'] = [epoch,val_acc,val_loss]
                else:
                    count += 1
                    print(f'out of best epoch {count}/{stop_epochs}')
                    if count > stop_epochs and epoch>=20:
                        break
            results[k]['train_acc'] = train_acc_all
            results[k]['train_loss'] = train_loss_all
            results[k]['val_acc'] = val_acc_all
            results[k]['val_loss'] = val_loss_all

            best_epoch = results[k]['best'][0]
            best_acc = results[k]['best'][1]
        
        elif args.stage == 'test':
            results[k]['best'] = []

        model_path = os.path.join(args.results_dir,f'{args.model_type}_split{str(k)}.pth')
        assert os.path.exists(model_path), 'No trained model checkpoint!'
        model.load_state_dict(torch.load(model_path))
        
        test_acc,test_loss,preds,trues,probs = val_epoch(args,model,test_loader,loss_fn,1)

        results[k]['best'].append(test_acc)
        results[k]['best'].append(test_loss)
        results[k]['preds'] = preds
        results[k]['trues'] = trues

        probs_all.append(probs)

        

        print(f'best results of split_{k}: acc:{test_acc:.4f},loss:{test_loss:.4f}')
   
    probs_all = torch.stack(probs_all)
    trues = torch.tensor(trues)
    probs_mean = probs_all.mean(0) #b,n_classes
    preds_mean = probs_mean.argmax(1)
    acc_test = torch.sum(preds_mean==trues)/preds_mean.shape[0]
    
    results['test'] = {}
    results['test']['preds'] = preds_mean.tolist()
    results['test']['probs'] = probs_mean.tolist()
    results['test']['acc'] = float(acc_test)
    results['test']['trues'] = trues.tolist()

    print(f'average test accuracy:{acc_test}')

    with open(os.path.join(args.results_dir,'results.json'),'w') as f:
        json.dump(results,f)
    
    with open(os.path.join(args.results_dir,'results_abstract.txt'), 'w') as f:
        for k in range(num_folds):
            f.write('{}\n'.format(str(results[k]['best'])))

    # compute complete metrics
    compute_metric_results(args,args.task)

    
