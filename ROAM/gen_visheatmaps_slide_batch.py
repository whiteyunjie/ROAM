from __future__ import print_function

import numpy as np

import argparse
import gc
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
## models for extracting features
from models_embed.extractor import resnet50
import models_embed.ResNet as ResNet
from models_embed.ccl import CCL
from models_embed.ctran import ctranspath
from models_embed.simclr_ciga import simclr_ciga_model
## models for testing
from models.ROAM import ROAM_VIS
from dataset.vis_dataset import Wsi_Region


from tqdm import tqdm
import h5py
import yaml
import sys
import openslide
import random
from scipy.stats import percentileofscore
from wsi_core.batch_process_utils import initialize_df
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap
from wsi_core.wsi_utils import sample_rois
from utils.file_utils import save_hdf5
from parse_config import read_taskinfo

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

cls_name_dict = {
    'int_glioma_tumor_subtyping':['astrocytoma','oligodendroglioma','ependymoma'],
    'ext_glioma_tumor_subtyping3':['astrocytoma','oligodendroglioma'],
    'int_glioma_cls':['normal','gliosis','tumor']
}

def infer_single_slide(model, feature_h5_path, num_rois=999):
	if num_rois < 1000:
		file = h5py.File(feature_h5_path, "r")
		#coords = file['coords'][:]
		features = file['features'][:]
		file.close()
		features = torch.tensor(features).cuda()
		print(features.shape)
		#print(features.shape)
		with torch.no_grad():

			slide_logits, roi_attns = model(x = features.unsqueeze(0),vis_mode=3)
	# divide into several batches to reduce memory consumption
	else:
		roi_embeddings = []
		for i in range(0,num_rois,1000):
			file = h5py.File(feature_h5_path, "r")			
			features = file['features'][i:min(i+1000,num_rois)]
			file.close()
			features = torch.tensor(features).cuda()
			#print(features.shape)
			with torch.no_grad():
				# generate attention weights for each roi
				roi_embed_batch = model(x=features,vis=True,vis_mode=1) # n_rois,84,2048 --> n_rois,n_classes
				roi_embeddings.append(roi_embed_batch)
		roi_embeddings = torch.concat(roi_embeddings)
		print(roi_embeddings.shape)
		slide_logits,roi_attns = model(x=roi_embeddings,vis=True,vis_mode=2)


	probs = torch.nn.functional.softmax(slide_logits, dim=1).cpu()
	torch.cuda.empty_cache()
	del features
	gc.collect()

	return roi_attns.cpu(),probs

def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key] 
			val = dtype(val)
			if isinstance(val, str):
				if len(val) > 0:
					params[key] = val
			elif not np.isnan(val):
				params[key] = val
			else:
				pdb.set_trace()

	return params

def parse_config_dict(args, config_dict):
	if args.exp_code is not None:
		config_dict['exp_arguments']['exp_code'] = args.exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

def read_features(feat_path):
    with h5py.File(feat_path,'r') as hdf5_file:
        features = hdf5_file['features'][:] # num_patches,84,1024
    return torch.from_numpy(features)


def extract_feats(wsi_object, slide_path, model,output_path,**wsi_kwargs):
	roi_dataset = Wsi_Region(wsi_object, slide_path, **wsi_kwargs)
	roi_dataloader = DataLoader(roi_dataset,batch_size=16,num_workers=4)	
	
	print('total number of patches to process: ', len(roi_dataset))
	num_rois = len(roi_dataset)
	# first w
	mode = 'w'

	for batch,coords,available in tqdm(roi_dataloader):
		with torch.no_grad():
			img_batch = batch.reshape((-1,batch.shape[2],batch.shape[3],batch.shape[4])).cuda()
			features_batch = model(img_batch) # b*84,3,w,h --> b*84,2048
			for b in range(available.shape[0]):
				if not available[b]:
					continue
				features = features_batch[b*84:(b+1)*84]
				features = features.unsqueeze(0)
				features = features.cpu().numpy()

				coord = coords[b].unsqueeze(0)
				coord = coord.numpy()

				#print(features.shape)
				#print(coord.shape)

				asset_dict = {'features':features,'coords':coord}
				save_hdf5(output_path,asset_dict,attr_dict=None,mode=mode)
				mode = 'a'
	return num_rois



if __name__ == '__main__':
	assert len(sys.argv) == 3, 'please give configuration file and split seed!'

    ### split seed [s1, s2, s3, s4, s5]

	config_file_dir = sys.argv[1]
	split_seed = sys.argv[2]
	print(f'split seed: {split_seed}')
	task_info = read_taskinfo(split_seed)

	##====>1. load config file
	config_path = os.path.join('visheatmaps/slide_vis/configs', config_file_dir)
	config_dict = yaml.safe_load(open(config_path, 'r'))

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))
			
	
	## load or set parameters for segmentation and patching
	config_args = config_dict
	patch_args = argparse.Namespace(**config_args['patching_arguments'])
	data_args = argparse.Namespace(**config_args['data_arguments'])

	exp_args = argparse.Namespace(**config_args['exp_arguments'])
	heatmap_args = argparse.Namespace(**config_args['heatmap_arguments'])
	sample_args = argparse.Namespace(**config_args['sample_arguments'])
	model_args = argparse.Namespace(**config_args['model_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	
	def_seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 
					  'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':25, 'a_h': 16, 'max_n_holes':8}
	def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
	
	##====>2. load slide list for visulation
	df = pd.read_csv(os.path.join('visheatmaps/slide_vis/process_list', data_args.process_list))
	df = initialize_df(df, data_args.process_list, def_seg_params, def_filter_params, def_vis_params, def_patch_params, use_heatmap_args=False)

	mask = df['process'] == 1 ## default: all slide are 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	##====>3. prepare state dict of models
	## task information for heatmaps
	model_args.exp_code = '_'.join(map(str, [model_args.task, model_args.depths,
					  				   model_args.embed_type,
                                       model_args.batch_size, model_args.roi_dropout, 
                                       model_args.roi_supervise,
                                       model_args.roi_weight, model_args.topk,
                                       model_args.roi_level,
                                       model_args.scale_type, model_args.single_level,
                                       model_args.not_interscale]))
	model_args.results_dir = os.path.join(model_args.results_dir,model_args.task,f'{model_args.exp_code}',f'{split_seed}')

	# pretrained models for extracting features of slides
	if model_args.embed_type == 'ImageNet':
		model = resnet50(pretrained=True).cuda()
		patch_dim = 1024
	elif model_args.embed_type == 'RetCCL':
		backbone = ResNet.resnet50
		model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()
		ckpt_path = f'models_embed/RetCCL_ckpt.pth'
		model.load_state_dict(torch.load(ckpt_path),strict=True)
		model.encoder_q.fc = nn.Identity()
		model.encoder_q.instDis = nn.Identity()
		model.encoder_q.groupDis = nn.Identity()
		patch_dim = 2048
	elif model_args.embed_type == 'ctranspath':
		model = ctranspath().cuda()
		model.head = nn.Identity()
		td = torch.load(r'models_embed/ctranspath.pth')
		model.load_state_dict(td['model'], strict=True)
		patch_dim = 768    
	else:
		model = simclr_ciga_model().cuda()
		patch_dim = 1024
	
	feature_extractor = model
	feature_extractor.eval()
	
	
	## whether setting fixed weights of embedding for each level
	if model_args.embed_weightx5==None and model_args.embed_weightx10==None and model_args.embed_weightx20==None:
		embed_weights = None
		print('use learnabel weights')
	else:
		embed_weights = [model_args.embed_weightx5,model_args.embed_weightx10,model_args.embed_weightx20]
		print('set weights:', embed_weights)
	# models for generate attention weights
	exp_args.n_classes = task_info[model_args.task]['n_classes']
	print(model_args.embed_weightx5)
	seed_torch(model_args.seed)

	model = ROAM_VIS(choose_num=model_args.topk,
                num_patches=84,
                patch_dim=patch_dim,
                num_classes=task_info[model_args.task]['n_classes'],
				roi_level = model_args.roi_level,
				scale_type = model_args.scale_type,
				single_level= model_args.single_level,
				embed_weights=embed_weights,
                dim=model_args.dim,
                depths=model_args.depths,
                heads=model_args.heads,   #8
                mlp_dim=model_args.mlp_dim,
				not_interscale = model_args.not_interscale,
                dim_head=model_args.dim_head,
                dropout=model_args.dropout,
                emb_dropout=model_args.emb_dropout,
				attn_dropout=model_args.attn_dropout,
				pool=model_args.pool,
                ape=model_args.ape,
                attn_type=model_args.attn_type,
                shared_pe=model_args.shared_pe)
    
	model = model.cuda()
	#print(model)
	print('Done!')

	cls_name =  cls_name_dict[model_args.task]
	reverse_label_dict = {i: cls_name[i] for i in range(len(cls_name))} 

	## saving dir of heatmap and asset files
	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)

	##====>4.process each slide
	for i in range(len(process_stack)):
		slide_id = process_stack.loc[i, 'slide_id']
		print('\nprocessing: ', slide_id)	

		## WSI path
		slide_cur_path = process_stack.loc[i, 'path']
		# if model_args.test_dataset == 'xiangya':
		# 	slide_path = '/data/glioma_data/iapsfile' + slide_cur_path
		# else:
		# 	slide_path = slide_cur_path
		slide_path = slide_cur_path
		label = process_stack.loc[i, 'label']


		grouping = reverse_label_dict[label]

		## saving dir of heatmap
		p_slide_save_dir = os.path.join(exp_args.production_save_dir, model_args.exp_code, str(grouping))
		os.makedirs(p_slide_save_dir, exist_ok=True)
		## saving dir of assert file
		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, model_args.exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		
		if heatmap_args.use_roi:
			x1, x2 = process_stack.loc[i, 'x1'], process_stack.loc[i, 'x2']
			y1, y2 = process_stack.loc[i, 'y1'], process_stack.loc[i, 'y2']
			top_left = (int(x1), int(y1))
			bot_right = (int(x2), int(y2))
		else:
			top_left = None
			bot_right = None
		
		print('slide id: ', slide_id)
		print('top left: ', top_left, ' bot right: ', bot_right)

		## check if the slide already has features
		if 'feat_dir' in process_stack.columns and not process_stack['feat_dir'].isnull().values[i]:
			feature_h5_path = process_stack.loc[i,'feat_dir']

		else:
			mask_file = os.path.join(r_slide_save_dir, slide_id+'_mask.pkl')
			
			# Load segmentation and filter parameters
			seg_params = def_seg_params.copy()
			filter_params = def_filter_params.copy()
			vis_params = def_vis_params.copy()

			seg_params = load_params(process_stack.loc[i], seg_params)
			filter_params = load_params(process_stack.loc[i], filter_params)
			vis_params = load_params(process_stack.loc[i], vis_params)
			
			keep_ids = str(seg_params['keep_ids'])
			if len(keep_ids) > 0 and keep_ids != 'none':
				seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
			else:
				seg_params['keep_ids'] = []

			exclude_ids = str(seg_params['exclude_ids'])
			if len(exclude_ids) > 0 and exclude_ids != 'none':
				seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
			else:
				seg_params['exclude_ids'] = []
			
			## load vis_level,seg_level for each slide(different in each slide)
			if model_args.test_dataset == 'xiangya':
				# check whether the slide can be read correctly
				checkwsi = openslide.open_slide(slide_path)
				try:
					img_test = checkwsi.read_region((0,0),process_stack.loc[i,'preset_vis_level'],checkwsi.level_dimensions[process_stack.loc[i,'preset_vis_level']])
				except:
					checkwsi = openslide.open_slide(slide_path)
					checkwsi.read_region((0,0),process_stack.loc[i,'preset_vis_level']+1,checkwsi.level_dimensions[process_stack.loc[i,'preset_vis_level']+1])
					process_stack.loc[i,'vis_level'] = process_stack.loc[i,'preset_vis_level'] + 1
					process_stack.loc[i,'seg_level'] = process_stack.loc[i,'preset_vis_level'] + 1
					sl = process_stack.loc[i,'preset_vis_level']
					print(f'reset seg level {sl} to {sl+1}')
					vis_params['vis_level'] = process_stack.loc[i,'vis_level']
					seg_params['seg_level'] = process_stack.loc[i,'vis_level']

				else:
					process_stack.loc[i,'vis_level'] = process_stack.loc[i,'preset_vis_level']
					process_stack.loc[i,'seg_level'] = process_stack.loc[i,'preset_vis_level']
					vis_params['vis_level'] = process_stack.loc[i,'vis_level']
					seg_params['seg_level'] = process_stack.loc[i,'vis_level']

			for key, val in seg_params.items():
				print('{}: {}'.format(key, val))

			for key, val in filter_params.items():
				print('{}: {}'.format(key, val))

			for key, val in vis_params.items():
				print('{}: {}'.format(key, val))
			
			##====>5.intialize wsi object
			print('Initializing WSI object')

			wsi_object = initialize_wsi(slide_id, slide_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
			print('Done!')

			wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

			# the actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
			vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

			block_map_save_path = os.path.join(r_slide_save_dir, '{}_blockmap.h5'.format(slide_id))
			mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
			if vis_params['vis_level'] < 0:

				vis_params['vis_level'] = wsi_object.wsi.level_count-1
			mask = wsi_object.visWSI(**vis_params, number_contours=True)
			mask.save(mask_path)
		
			## extract features of slides
			## 2 times: 1. non-overlap 2.overlapped
			
			### non-overlap
			print('extract featrues for non-overlapped patches')
			blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size, 
			'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift,
			'is_stain_norm':True,'target_image_dir':patch_args.target_image_dir}
			feature_h5_path = os.path.join(r_slide_save_dir, slide_id+'.h5')
			if not os.path.isfile(feature_h5_path):
				extract_feats(wsi_object = wsi_object, slide_path = slide_path, model = feature_extractor,
							output_path=feature_h5_path,**blocky_wsi_kwargs)
			else:
				file = h5py.File(feature_h5_path, "r")
				coords = file['coords'][:]
				#num_rois = coords.shape[0]
			
			### overlap
			print('extract featrues for overlapped patches')
			blocky_wsi_kwargs_overlap = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': step_size, 
			'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift,
			'is_stain_norm':True,'target_image_dir':patch_args.target_image_dir}
			feature_h5_path_overlap = os.path.join(r_slide_save_dir, f'{slide_id}_{patch_args.overlap}.h5')
			if not os.path.isfile(feature_h5_path_overlap):
				num_rois = extract_feats(wsi_object = wsi_object, slide_path = slide_path, model = feature_extractor,
							output_path=feature_h5_path_overlap,**blocky_wsi_kwargs_overlap)
			else:
				file = h5py.File(feature_h5_path_overlap, "r")
				coords = file['coords'][:]
				num_rois = coords.shape[0]


						
			process_stack.loc[i,'feat_dir'] = feature_h5_path_overlap
			#process_stack.loc[i, 'bag_size'] = len(features)
			wsi_object.saveSegmentation(mask_file)

		## deal with features per 1000 rois if num_rois > 1000
		
		##====>6. generate attention weights: 5 best model dict
		print('generate reference attenion scores of non-overlaped rois')
		attentions_all = []
		probs_all = []
		for k in range(5):
			print(f'load {k}-th model weights')
			model_path = os.path.join(model_args.results_dir,f'ROAM_split{str(k)}.pth')
			model.load_state_dict(torch.load(model_path))
			model.eval()
			attentions,probs = infer_single_slide(model, feature_h5_path)
			attentions_all.append(attentions)
			probs_all.append(probs)
			torch.cuda.empty_cache()
		
		probs_all = torch.stack(probs_all)
		probs_mean = probs_all.mean(0) #b,n_classes
		preds_mean = probs_mean.argmax(1)
		print(f'prediction:{preds_mean}')

		attentions_all = torch.stack(attentions_all)
		attentions_mean = attentions_all.mean(0)
		attentions_mean = F.softmax(attentions_mean,dim=1)
		ref_scores = attentions_mean[:,preds_mean]
		
		print('generate attenion weights')

		attentions_all = []
		probs_all = []
		for k in range(5):
			print(f'load {k}-th model weights')
			model_path = os.path.join(model_args.results_dir,f'ROAM_split{str(k)}.pth')
			model.load_state_dict(torch.load(model_path))
			model.eval()
			attentions,probs = infer_single_slide(model, feature_h5_path_overlap, num_rois)
			attentions_all.append(attentions)
			probs_all.append(probs)
			torch.cuda.empty_cache()
		

		attentions_all = torch.stack(attentions_all)
		attentions_mean = attentions_all.mean(0)
		attentions_mean = F.softmax(attentions_mean,dim=1)
		attentions_score = attentions_mean[:,preds_mean]


		if not os.path.isfile(block_map_save_path): 
			file = h5py.File(feature_h5_path, "r")
			coords = file['coords'][:]
			file.close()
			asset_dict = {'attention_scores': np.array(ref_scores), 'coords': coords}
			block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		
		# save top 3 predictions
		for c in range(exp_args.n_classes):
			process_stack.loc[i, 'p_{}'.format(c)] = probs_mean.tolist()[0][c]

		## save pred label
		process_stack.loc[i,'pred'] = np.array(preds_mean)[0]
		
		os.makedirs('visheatmaps/slide_vis/results/', exist_ok=True)
		if data_args.process_list is not None:
			process_stack.to_csv('visheatmaps/slide_vis/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
		else:
			process_stack.to_csv('visheatmaps/slide_vis/results/{}.csv'.format(model_args.exp_code), index=False)
		
		##====>6. draw heatmaps
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		## save topk rois
		samples = sample_args.samples
		for sample in samples:
			if sample['sample']:
				tag = "pred_{}_label_{}".format(np.array(preds_mean)[0],label)
				sample_save_dir =  os.path.join(exp_args.production_save_dir, model_args.exp_code, 'sampled_patches', str(tag), sample['name'])
				os.makedirs(sample_save_dir, exist_ok=True)
				print('sampling {}'.format(sample['name']))
				sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
					score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
				for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
					print('coord: {} score: {:.3f}'.format(s_coord, s_score))
					patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
					patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))


		heatmap_save_name = '{}_blockmap.tiff'.format(slide_id)
		if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
			pass
		else:
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
							thresh=-1, patch_size = vis_patch_size, convert_to_percentiles=True, use_roi = heatmap_args.use_roi)
		
			heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
			del heatmap

		save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		if heatmap_args.use_ref_scores:
			ref_scores = scores
		else:
			ref_scores = None
		
		## use ref scores
		attentions_score = attentions_score.detach().numpy()

		if ref_scores is not None:
			for score_idx in range(len(attentions_score)):
				attentions_score[score_idx] = score2percentile(attentions_score[score_idx],ref_scores)
		
		# rewrite h5 file
		file = h5py.File(feature_h5_path_overlap, "r")
		coords = file['coords'][:]
		file.close()
		asset_dict = {'attention_scores': np.array(attentions_score), 'coords': coords}
		save_path = save_hdf5(save_path, asset_dict, mode='w')


		file = h5py.File(save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()

		heatmap_args.vis_level = vis_params['vis_level']
		## max_size: max size of generated slide attention heatmaps
		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur,
		      				 'custom_downsample': heatmap_args.custom_downsample,'max_size':2000}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False

		heatmap_save_name = '{}_{}_roi_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(slide_id, float(patch_args.overlap), int(heatmap_args.use_roi),
																						int(heatmap_args.blur), 
																						int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																						float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																						int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


		if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			pass
		
		else:
			print(os.path.join(p_slide_save_dir, heatmap_save_name))
			heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object,  
						          cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, **heatmap_vis_args, 
						          binarize=heatmap_args.binarize, 
						  		  blank_canvas=heatmap_args.blank_canvas,
						  		  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size,
						  		  overlap=patch_args.overlap, 
						  		  top_left=top_left, bot_right = bot_right, use_roi = heatmap_args.use_roi)
			if heatmap_args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				# for assissted diagnosis
				#heatmap.save(os.path.join(d_slide_save_dir, 'mask.jpg'), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		
		if heatmap_args.save_orig:
			vis_level = vis_params['vis_level']
			heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
			if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				pass
			else:
				heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample, max_size=2000)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))

	with open(os.path.join(exp_args.raw_save_dir, model_args.exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)


