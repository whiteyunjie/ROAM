import torch
import torch.nn as nn
import numpy as np
import openslide
import h5py
import os

from torch.utils.data import DataLoader
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from patchdataset import Roi_Seg_Dataset, Patch_Seg_Dataset
from models.extractor import resnet50
import models.ResNet as ResNet
from models.ccl import CCL
from models.ctran import ctranspath
from models.simclr_ciga import simclr_ciga_model

from utils.file_utils import save_hdf5
from utils.utils import collate_features
from PIL import Image
from tqdm import tqdm
import h5py
import openslide
import argparse




def extract_feats(args,h5_file_path,wsi,slide_path,model,output_path,target_roi_size=2048,patch_size=256,levels=[0,1,2],batch_size=1,is_stain_norm=False):
    '''
    extract_feats:
        Extract features of patches within ROIs through pre-trained models.
        For ROI with size of 2048*2048 at 20x magnification, 8*8=64 patches with size of 256 
        will be put into models to extract features.
        args:
            h5_file_path: directory of bag (.h5 file), the coordinates of all patches segmented from the slide(bag)
            wsi: object of WSI ('OpenSlide' object)
            slide_path: directory of WSI
            model: pre-trained model
            output_path: directory to save computed featrues (.h5 file)
            target_roi_size: size of roi at 20x magnification
            patch_size: size of patch used for feature extraction
            levels: at which magnification levels the patch features are extracted (List. 0:20x,1:10x,2:5x)
            batch_size: batch size of patches for feature extraction
            is_stain_norm: whether to perform stain normalization
            resize : whether to resize patches to 224*224
    '''
    # for ctranspath pre-trained model, input size of image should be reset to 224*224 instead 256*256
    if args.pretrained_model == 'ctranspath':
        roi_dataset = Roi_Seg_Dataset(args.pretrained_model,h5_file_path,slide_path,wsi,levels,target_roi_size,patch_size,is_stain_norm,resize=True)
    else:
        roi_dataset = Roi_Seg_Dataset(args.pretrained_model,h5_file_path,slide_path,wsi,levels,target_roi_size,patch_size,is_stain_norm)
    roi_dataloader = DataLoader(roi_dataset,batch_size=batch_size,num_workers=4)
    # first w
    mode = 'w'

    for batch,coords,available in tqdm(roi_dataloader):
        with torch.no_grad():

            for b in range(batch_size):
                if not available[b]:
                    continue

                img_batch = batch[b].cuda()

                features = model(img_batch) # 84,d
                features = features.unsqueeze(0) #1,84,d
                features = features.cpu().numpy()

                coord = coords[b].unsqueeze(0)
                coord = coord.numpy()

                asset_dict = {'features':features,'coords':coord}
                save_hdf5(output_path,asset_dict,attr_dict=None,mode=mode)
                mode = 'a'


def extract_feats_patch(h5_file_path,wsi,slide_path,model,output_path,patch_size=256,batch_size=1,is_stain_norm=False):
    '''
    extract_feats_patch:
        normal function for extracting patch features.
        extract features directly with input patch instead of cutting 
        it into smaller patches for indidual feature extraction.
        args:
            h5_file_path: directory of bag (.h5 file), the coordinates of all patches segmented from the slide(bag)
            wsi: object of WSI ('OpenSlide' object)
            slide_path: directory of WSI
            model: pre-trained model
            output_path: directory to save computed featrues (.h5 file)
            patch_size: size of patch used for feature extraction
            batch_size: batch size of patches for feature extraction
            is_stain_norm: whether to perform stain normalization
    '''
    roi_dataset = Patch_Seg_Dataset(h5_file_path,slide_path,wsi,patch_size,is_stain_norm)
    roi_dataloader = DataLoader(roi_dataset,batch_size=batch_size,num_workers=4,pin_memory=True,collate_fn=collate_features)
    # first w
    mode = 'w'

    for batch,coords,available in tqdm(roi_dataloader):
        with torch.no_grad():
            batch = batch.cuda()
            
            features = model(batch)
            features = features.cpu().numpy()

            if features.shape[0] < 2:
                continue


            features_normal = features[available]
            coords_normal = coords[available]

            if features_normal.shape[0] > 0:
                asset_dict = {'features':features_normal,'coords':coords_normal}
                save_hdf5(output_path,asset_dict,attr_dict=None,mode=mode)
                mode = 'a'


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default='/data/glioma_data/datapro')
parser.add_argument('--data_slide_dir', type=str, default='/data/glioma_data/iapsfile')
parser.add_argument('--csv_path', type=str, default='./xiangya_data_info')
parser.add_argument('--dataset', type=str, default='xiangya')
parser.add_argument('--data_format', type=str, default='roi',choices=['roi','patch'])
parser.add_argument('--feat_dir', type=str, default = '/data/glioma_data/datapro')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=256)
parser.add_argument('--target_roi_size', type=int, default=2048)
parser.add_argument('--level',default=0,type=int,choices=[0,1,2])
parser.add_argument('--is_stain_norm',action='store_true',default=False,help='whether stain normlization')
parser.add_argument('--pretrained_model',type=str,default='ImageNet',choices=['ImageNet','RetCCL','simclr-ciga','ctranspath'],help='model weights for extracting features')
args = parser.parse_args()

if __name__ == '__main__':
    
    # create directory to save generated features
    os.makedirs(args.feat_dir, exist_ok=True)
    if args.is_stain_norm:
        args.feat_dir = os.path.join(args.feat_dir, f'feats_{args.pretrained_model}_norm')
        #os.makedirs(os.path.join(args.feat_dir, f'feats_{args.pretrained_model}_norm'))
    else:
        args.feat_dir = os.path.join(args.feat_dir, f'feats_{args.pretrained_model}')
    os.makedirs(args.feat_dir,exist_ok=True)
    dest_files = os.listdir(args.feat_dir)

    # read slide info csv data
    data_csv = pd.read_csv(args.csv_path)
    slide_id = data_csv['slide_id'].values
    slide_path = data_csv['path'].values

    # calculate magnifications
    roi_size_list = [2048,1024,512]

    levels = [i for i in range(4-args.level)]
    target_roi_size = roi_size_list[args.level]
    
    # select pre-trained model for feature extraction
    if args.pretrained_model == 'ImageNet':
        model = resnet50(pretrained=True).cuda()
    elif args.pretrained_model == 'RetCCL':
        backbone = ResNet.resnet50
        model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()
        ckpt_path = f'models/{args.pretrained_model}_ckpt.pth'
        model.load_state_dict(torch.load(ckpt_path),strict=True)
        model.encoder_q.fc = nn.Identity()
        model.encoder_q.instDis = nn.Identity()
        model.encoder_q.groupDis = nn.Identity()
    elif args.pretrained_model == 'ctranspath':
        model = ctranspath()
        model.head = nn.Identity()
        td = torch.load(r'models/ctranspath.pth')
        model.load_state_dict(td['model'], strict=True)
        model = model.cuda() 
    else:
        model = simclr_ciga_model().cuda()

    #model = nn.DataParallel(model)

    # feature extraction
    model.eval()

    for i in range(len(slide_id)):
        print(f'extract features from {slide_id[i]},{i}/{len(slide_id)}')

        bag_name = slide_id[i]+'.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

        if args.dataset == 'xiangya':
            slide_file_path = args.data_slide_dir + slide_path[i]
        else:
            slide_file_path = slide_path[i]

        if not args.no_auto_skip and slide_id[i]+'.h5' in dest_files:
            print(f'skipped {slide_id[i]}')
            continue
        
        output_path = os.path.join(args.feat_dir, bag_name)
        wsi = openslide.open_slide(slide_file_path)
        '''
        data_format: form of feature extraction
            roi: segment the roi into patches with size of 256*256 and extract features of these patches
            patch: directly extract features of input patches
        '''
        if args.data_format == 'roi':
            extract_feats(args,h5_file_path,wsi,slide_file_path,model,output_path,target_roi_size=target_roi_size,levels = levels,is_stain_norm=args.is_stain_norm)
        else:
            extract_feats_patch(h5_file_path,wsi,slide_file_path,model,output_path,batch_size = args.batch_size,is_stain_norm=args.is_stain_norm)


    