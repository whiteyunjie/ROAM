import argparse
from secrets import choice
import sys
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
import vahadane
import matplotlib.pyplot as plt

from vis_utils.vit_rollout import VITAttentionRollout
from vis_utils.vit_grad_rollout import VITAttentionGradRollout
from models.ROAM import ROAM_VIS
from models_embed.extractor import resnet50
import models_embed.ResNet as ResNet
from models_embed.ccl import CCL
from models_embed.ctran import ctranspath
from models_embed.simclr_ciga import simclr_ciga_model
from parse_config import parse_args_heatmap_roi

os.environ["CUDA_VISIBLE_DEVICES"]="0"


cls_name_dict = {
    'int_glioma_tumor_subtyping':['astrocytoma','oligodendroglioma','ependymoma'],
    'ext_glioma_tumor_subtyping3':['astrocytoma','oligodendroglioma'],
    'int_glioma_cls':['normal','gliosis','tumor']
}


## for stain normalization
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_patch = transforms.Compose(
                    [# may be other transform
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ]
                )
target_img = np.array(Image.open('visheatmaps/target_roi_6e3.jpg'))
vhd = vahadane.vahadane(LAMBDA1=0.01,LAMBDA2=0.01,fast_mode=0,ITER=100)
Wt,Ht = vhd.stain_separate(target_img)



def preprocess_image(src_img):
    target_roi_size = 2048
    patch_size = 256
    img_batch = []
    img_roi = src_img.resize((target_roi_size,target_roi_size))
    # stain normalization
    Ws,Hs = vhd.stain_separate(np.array(img_roi))
    img = vhd.SPCN(np.array(img_roi),Ws,Hs,Wt,Ht)

    # segment roi to 84 patches with size of 256
    for size in [2048,1024,512]:
        img_cur = Image.fromarray(img).resize((size,size))
        img_array = np.array(img_cur)
        for i in range(0,size,patch_size):
            for j in range(0,size,patch_size):
                img_patch = img_array[i:i+patch_size,j:j+patch_size,:]
                img_patch = transform_patch(img_patch)
                img_batch.append(img_patch)

    img_batch = torch.stack(img_batch) #84,3,256,256
    return img_batch

def masks_preprocess(masks,img,w):
    '''
    Sum the attention masks of three magnification levels, weighted by the model's 'embed_weights'
    parameters, to obtain teh final attention heatmap of the ROI.
    
    Args:
        masks (List of array): attention masks of 3 magnifications (20x:(8,8), 10x:(4,4),5x(2,2))
        img (PIL.Image): ROI image to be visualized.
        w (List of float): attention weights of 3 magnifications, same with parameters 'embed_weights' during training.
    '''
    np_img = np.array(img)[:, :, ::-1] #RGB->BGR
    mask_weighted = 0
    # multi-scale
    for i in range(3):
        mask = masks[i]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask_weighted += w[i]*mask

    masked_img = show_mask_on_image_weighted(np_img, mask_weighted)

    return masked_img,mask_weighted

    



def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    #print(heatmap.shape)
    cam = heatmap + np.float32(img)

    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_mask_on_image_weighted(img, mask, alpha=0.3):
    '''
    draw heatmaps according to attention scores
    '''
    cmap = plt.get_cmap('jet')

    mask_block = (cmap(mask) * 255)[:,:,:3].astype(np.uint8)[:,:,::-1]
    cam = cv2.addWeighted(mask_block,alpha,img,1-alpha,0)

    return cam


if __name__ == '__main__':
    assert len(sys.argv)==3, 'please give configuration file and split seed!'

    
    args, task_info = parse_args_heatmap_roi(sys.argv[1], sys.argv[2])
    if args.level == -1:
        args.level = args.depths[-1]
    ## embed_model
    print('load embedding model')
    if args.embed_type == 'ImageNet':
        embed_model = resnet50(pretrained=True).cuda()
        patch_dim = 1024
    elif args.embed_type == 'RetCCL':
        backbone = ResNet.resnet50
        embed_model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()
        ckpt_path = f'models_embed/RetCCL_ckpt.pth'
        embed_model.load_state_dict(torch.load(ckpt_path),strict=True)
        embed_model.encoder_q.fc = nn.Identity()
        embed_model.encoder_q.instDis = nn.Identity()
        embed_model.encoder_q.groupDis = nn.Identity()
        patch_dim = 2048
    elif args.embed_type == 'ctranspath':
        embed_model = ctranspath().cuda()
        embed_model.head = nn.Identity()
        td = torch.load(r'models_embed/ctranspath.pth')
        embed_model.load_state_dict(td['model'], strict=True)
        patch_dim = 768    
    else:
        embed_model = simclr_ciga_model().cuda()
        patch_dim = 1024

    embed_model.eval()
    
    
    if args.embed_weightx5==None and args.embed_weightx10==None and args.embed_weightx20==None:
        embed_weights = None
        print('use learnabel weights')
    else:
        embed_weights = [args.embed_weightx5,args.embed_weightx10,args.embed_weightx20]
        print('set weights:', embed_weights)
    ## main model ROAM
    print('load main model ROAM')

    model = ROAM_VIS(choose_num = args.topk,
                num_patches = 84,
                patch_dim=patch_dim,
                num_classes=task_info[args.task]['n_classes'],
                roi_level = args.roi_level,
                scale_type = args.scale_type,
                embed_weights=embed_weights,
                dim=args.dim,
                depths=args.depths,
                heads=args.heads,
                mlp_dim=args.mlp_dim,
                dim_head=args.dim_head,
                dropout=args.dropout,
                emb_dropout=args.emb_dropout,
                attn_dropout=args.attn_dropout,
                pool=args.pool,
                ape=args.ape,
                attn_type=args.attn_type,
                shared_pe=args.shared_pe)
    model = model.cuda()


    print('exp_code:',args.exp_code)


    # read topk roi list
    img_root = f'visheatmaps/slide_vis/results/heatmap_production_results/{args.exp_code}/sampled_patches'
    slide_list = pd.read_csv(args.process_list)
    slide_ids = slide_list['slide_id'].values
    labels = slide_list['label'].values
    cls_name = cls_name_dict[args.task]

    for i in range(len(slide_ids)):
        sid = slide_ids[i]
        label = labels[i]
        catname = cls_name[label]
        print(f'===process topk roi of slide: {sid}')

        imagelist  = os.listdir(f'{img_root}/pred_{label}_label_{label}/topk_high_attention')
        
        topidx = 0

        for imgpath in sorted(imagelist):
            img_name = str(topidx) + '_' + sid
            if img_name not in imgpath: continue
            topidx += 1
            print(f'topkidx:{topidx}')
            if topidx > args.topk_num: break
            mask_all = []
            print(f'===process topk roi {topidx}/{args.topk_num}')
            img_save_path = f'visheatmaps/roi_vis/{args.sample}/{args.exp_code}/{args.vis_type}/{catname}/{sid}'
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            image_path = os.path.join(f'{img_root}/pred_{label}_label_{label}/topk_high_attention',imgpath)

            ## read origin image
            img = Image.open(image_path)
            img_batch = preprocess_image(img)

            ## extract features
            input = img_batch.cuda()
            for k in range(5):
                print(f'split{k}')
                model_path = os.path.join(f'results/{args.task}/{args.exp_code}/{args.split_seed}',f'ROAM_split{k}.pth')
                #print('model_path:',model_path)
                model.load_state_dict(torch.load(model_path))
                model.eval()
                print('done!')
              
                features = embed_model(input) #84,2048
                input_tensor = features.unsqueeze(0) #1,84,2048
                #print(input_tensor.device)

                ## get attention grad
                
                if args.category_index is None:
                    print("Doing Attention Rollout")
                    attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
                        discard_ratio=args.discard_ratio)
                    masks = attention_rollout(input_tensor)
                    name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
                else:
                    print("Doing Gradient Attention Rollout")
                    if embed_weights:
                        grad_rollout = VITAttentionGradRollout(model, args.level, discard_ratio=args.discard_ratio,vis_type=args.vis_type,vis_scale=args.vis_scale)
                        w = embed_weights
                    else:
                        grad_rollout,w = VITAttentionGradRollout(model, args.level, discard_ratio=args.discard_ratio,vis_type=args.vis_type,vis_scale=args.vis_scale,learnable_weights=True)
                    masks = grad_rollout(input_tensor, args.category_index)
                    name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                        args.discard_ratio, args.head_fusion)


                print('embed_weights:',w)
                masked_img,mask = masks_preprocess(masks,img,w)

                if args.vis_type == 'grad_rollout':
                    Image.fromarray(masked_img[:,:,::-1]).convert('RGB').resize((1024,1024)).save(os.path.join(img_save_path,f'top{topidx}_seed{args.split_seed}_{args.category_index}_d{args.depths[-1]}_l{args.level}_r{args.discard_ratio}_fold{k}.png'))
                else:
                    Image.fromarray(masked_img[:,:,::-1]).convert('RGB').resize((1024,1024)).save(os.path.join(img_save_path,f'top{topidx}_seed{args.split_seed}_{args.category_index}_d{args.depths[-1]}_fold{k}.png'))
            
                ## exclude mask with nan value
                if not np.isnan(mask).any():
                    mask_all.append(mask)
            
            ## average attention mask for 5 splits in each seed
            mask_avg = np.mean(mask_all,0)

            mask_avg = mask_avg/np.max(mask_avg) #normalization


            np_img = np.array(img)[:, :, ::-1] #RGB->BGR
            masked_img_avg = show_mask_on_image_weighted(np_img, mask_avg)

            if args.vis_type == 'grad_rollout':
                Image.fromarray(masked_img_avg[:,:,::-1]).convert('RGB').resize((1024,1024)).save(os.path.join(img_save_path,f'top{topidx}_seed{args.split_seed}_{args.category_index}_d{args.depths[-1]}_l{args.level}_r{args.discard_ratio}_avg.png'))
            else:
                Image.fromarray(masked_img_avg[:,:,::-1]).convert('RGB').resize((1024,1024)).save(os.path.join(img_save_path,f'top{topidx}_seed{args.split_seed}_{args.category_index}_d{args.depths[-1]}_avg.png'))
            img.resize((1024,1024)).save(os.path.join(img_save_path,f'top{topidx}_seed{args.split_seed}_ori.png'))