from torchvision import transforms
import pandas as pd
import numpy as np
import time
import pdb
import PIL.Image as Image
import h5py
import openslide
from torch.utils.data import Dataset
import torch
from wsi_core.util_classes import Contour_Checking_fn, isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard
import vahadane

TARGET_IMAGE_DIR = 'visheatmaps/target_image_6e3_256.jpg'
TARGET_IMAGE_DIR2 = 'visheatmaps/target_roi_6e3.jpg'

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_patch = transforms.Compose(
                    [# may be other transform
                    transforms.ToTensor(),
					transforms.Normalize(mean = mean, std = std)
					]
				)

def default_transforms(mean = (0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean = mean, std = std)])
    return t

def get_contour_check_fn(contour_fn='four_pt_hard', cont=None, ref_patch_size=None, center_shift=None):
    if contour_fn == 'four_pt_hard':
        cont_check_fn = isInContourV3_Hard(contour=cont, patch_size=ref_patch_size, center_shift=center_shift)
    elif contour_fn == 'four_pt_easy':
        cont_check_fn = isInContourV3_Easy(contour=cont, patch_size=ref_patch_size, center_shift=0.5)
    elif contour_fn == 'center':
        cont_check_fn = isInContourV2(contour=cont, patch_size=ref_patch_size)
    elif contour_fn == 'basic':
        cont_check_fn = isInContourV1(contour=cont)
    else:
        raise NotImplementedError
    return cont_check_fn



class Wsi_Region(Dataset):
    '''
    args:
        wsi_object: instance of WholeSlideImage wrapper over a WSI
        top_left: tuple of coordinates representing the top left corner of WSI region (Default: None)
        bot_right tuple of coordinates representing the bot right corner of WSI region (Default: None)
        level: downsample level at which to prcess the WSI region
        patch_size: tuple of width, height representing the patch size
        step_size: tuple of w_step, h_step representing the step size
        contour_fn (str): 
            contour checking fn to use
            choice of ['four_pt_hard', 'four_pt_easy', 'center', 'basic'] (Default: 'four_pt_hard')
        t: custom torchvision transformation to apply 
        custom_downsample (int): additional downscale factor to apply 
        use_center_shift: for 'four_pt_hard' contour check, how far out to shift the 4 points
    '''
    def __init__(self, wsi_object, slide_path, top_left=None, bot_right=None, level=0, 
                 patch_size = (4096, 4096), step_size=(512, 512), 
                 target_roi_size = (2048,2048),target_patch_size = (256,256),contour_fn='four_pt_easy',
                 t=None, custom_downsample=1, use_center_shift=False,
                 is_stain_norm = True, target_image_dir = None):
        
        self.custom_downsample = custom_downsample
        self.roi_level = level
        self.slide_path = slide_path
        #print('cont_fn:',contour_fn)

        # downscale factor in reference to level 0
        self.ref_downsample = wsi_object.level_downsamples[level]
        # patch size in reference to level 0
        self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        if self.custom_downsample > 1:
            self.target_patch_size = patch_size
            patch_size = tuple((np.array(patch_size) * np.array(self.ref_downsample) * custom_downsample).astype(int))
            step_size = tuple((np.array(step_size) * custom_downsample).astype(int))
            self.ref_size = patch_size
        else:
            step_size = tuple((np.array(step_size)).astype(int))
            self.ref_size = tuple((np.array(patch_size) * np.array(self.ref_downsample)).astype(int)) 
        
        self.wsi = wsi_object.wsi
        self.level = level
        self.patch_size = target_patch_size
        self.target_roi_size = target_roi_size
        self.roi_size = patch_size
        self.levels = [0,1,2]
        #self.patch_nums = 84
            
        if not use_center_shift:
            center_shift = 0.
        else:
            overlap = 1 - float(step_size[0] / patch_size[0])
            if overlap < 0.25:
                center_shift = 0.375
            elif overlap >= 0.25 and overlap < 0.75:
                center_shift = 0.5
            elif overlap >=0.75 and overlap < 0.95:
                center_shift = 0.625
            else:
                center_shift = 1.0
            #center_shift = 0.375 # 25% overlap
            #center_shift = 0.625 #50%, 75% overlap
            #center_shift = 1.0 #95% overlap
        
        # print(f'=========step_size:{step_size[0]}')
        # print(f'=========patch_size:{patch_size[0]}')
        filtered_coords = []
        #iterate through tissue contours for valid patch coordinates
        for cont_idx, contour in enumerate(wsi_object.contours_tissue): 
            print('processing {}/{} contours'.format(cont_idx, len(wsi_object.contours_tissue)))
            cont_check_fn = get_contour_check_fn(contour_fn, contour, self.ref_size[0], center_shift)
            #print(wsi_object.holes_tissue)
            #print(wsi_object.holes_tissue[cont_idx])
            coord_results, _ = wsi_object.process_contour(contour, wsi_object.holes_tissue[cont_idx], level, '', 
                            patch_size = patch_size[0], step_size = step_size[0], contour_fn=cont_check_fn,
                            use_padding=True, top_left = top_left, bot_right = bot_right)
            if len(coord_results) > 0:
                filtered_coords.append(coord_results['coords'])
        
        #print(filtered_coords)
        #print(len(filtered_coords))
        coords=np.vstack(filtered_coords)

        self.coords = coords
        print('filtered a total of {} coordinates'.format(len(self.coords)))
        
        #target_image_dir = 'visheatmaps/target_roi_6e3.jpg'
        #print(target_image_dir)
        if is_stain_norm:
            self.target_img = np.array(Image.open(target_image_dir))
            #print(self.target_img.shape)
            ## may raise muliti-process problem
            self.vhd = vahadane.vahadane(LAMBDA1=0.01,LAMBDA2=0.01,fast_mode=0,ITER=100)
            self.Wt,self.Ht = self.vhd.stain_separate(self.target_img)
            #self.vhd.fast_mode = 1 #fast separate
        self.is_stain_norm = is_stain_norm

    def __len__(self):
        return len(self.coords)
    
    def stain_norm(self,src_img):
        #print(src_img.shape)
        #Image.fromarray(src_img).save('test.jpg')
        #vhd = vahadane.vahadane(LAMBDA1=0.01,LAMBDA2=0.01,fast_mode=0,ITER=100)
        
        Ws,Hs = self.vhd.stain_separate(src_img)
        #print(src_img.shape)
        #print(Ws,Hs,self.Wt,self.Ht)
        img = self.vhd.SPCN(src_img,Ws,Hs,self.Wt,self.Ht)
        return img
    
    def __getitem__(self, idx):
        coord = self.coords[idx]
        
        try:
            img = self.wsi.read_region(coord, self.roi_level, (self.roi_size[0], self.roi_size[1])).convert('RGB')
        except:
            # or subsequent normal patches will also raise errors
            self.wsi = openslide.open_slide(self.slide_path)
            available = False
            #img = np.zeros((self.roi_size,self.patch_size,3))
            #return img, coord, torch.tensor([False])
        else:
            img = self.wsi.read_region(coord, self.roi_level, (self.roi_size[0], self.roi_size[1])).convert('RGB')
            available = True

        #patch_num_all = np.sum(self.patch_nums)
        patch_num_all = 84
        if not available:
            img_batch = torch.zeros((patch_num_all,3,self.patch_size[0],self.patch_size[0]))
            #print('????????')
        else:
            img_batch = []
            img_roi = img.resize((self.target_roi_size[0],self.target_roi_size[1]))
            #img_roi.save('test_roi.jpg')
            if self.is_stain_norm:
                img_roi = self.stain_norm(np.array(img_roi))
                #Image.fromarray(img_roi).save('test_roi_pro.jpg')
            for level in self.levels:
                roi_size_cur = int(self.target_roi_size[0]/(2**level))
                img_roi = np.array(img_roi)
                img_cur = Image.fromarray(img_roi).resize((roi_size_cur,roi_size_cur))
                
                imgarray = np.array(img_cur)
                for i in range(0,roi_size_cur,self.patch_size[0]):
                    for j in range(0,roi_size_cur,self.patch_size[1]):
                        img_patch = imgarray[i:i+self.patch_size[0],j:j+self.patch_size[1],:]
                        img_patch = transform_patch(img_patch)
                        img_batch.append(img_patch)
                
            #img_batch = torch.stack(img_batch).unsqueeze(0) #(1,84,3,256,256)
            if available:
                img_batch = torch.stack(img_batch)
                #print('img_batch_shape:',img_batch.shape)
        

        return img_batch, coord, torch.tensor([available])
