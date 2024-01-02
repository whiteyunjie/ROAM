import torch
import openslide
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
import vahadane


mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform_patch = transforms.Compose(
                    [# may be other transform
                    transforms.ToTensor(),
					transforms.Normalize(mean = mean, std = std)
					]
				)


class Roi_Seg_Dataset(Dataset):
	def __init__(self,
			  embed_type='ImageNet',
			  file_path='',
			  slide_path='',
			  wsi=None,
			  levels=[0,1,2],
			  target_roi_size = 2048,
			  patch_size=256,
			  is_stain_norm=False,
			  resize=False):
		'''
		args:
			embed_type (string): type of pre-trained model. select from ImageNet, RetCCL, ctranspath, simclr-ciga
			file_path (string): directory of bag (.h5 file), the coordinates of all patches segmented from the slide(bag)
			slide_path (string): directory of WSI
			wsi ('OpenSlide' object): object of WSI, for reading regions from WSI according to coordinates.
			levels (Lsit): at which magnification levels the patch features are extracted (0:20x,1:10x,2:5x)
			target_roi_size (int): size of roi at 20x magnification
            patch_size (int): size of patch used for feature extraction
			is_stain_norm (bool): whether to perform stain normalization
			resize (bool): whether to resize patches to 224*224
		'''
		self.file_path = file_path
		self.wsi = wsi
		self.levels = levels #[0,1] or [0,1,2]
		self.patch_size = patch_size
		self.slide_path = slide_path
		self.target_roi_size = target_roi_size
		self.downscale = 0
		self.resize = resize

		'''
		mean and std for normalization of simclr-ciga are different
		'''
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)
		if embed_type == 'simclr-ciga':
			mean = (0.5, 0.5, 0.5)
			std = (0.5, 0.5, 0.5)

		self.transform_patch = transforms.Compose(
                    [# may be other transform
                    transforms.ToTensor(),
					transforms.Normalize(mean = mean, std = std)
					]
				)

		with h5py.File(self.file_path,'r') as f:
			dset = f['coords']
			self.roi_level = f['coords'].attrs['patch_level']
			self.roi_size = f['coords'].attrs['patch_size']
			self.downscale = int(self.roi_size/self.target_roi_size)
			self.length = len(dset)
			patch_num_0 = (self.target_roi_size/self.patch_size)**2
			self.patch_nums = [int(patch_num_0/(2**level)) for level in self.levels]
		

		# select target image for stain normalization
		if target_roi_size == 512:
			target_image_dir = 'target_images/target_image_6e3_512.jpg'
		if target_roi_size == 1024:
			target_image_dir = 'target_images/target_image_6e3_1024.jpg'
		if target_roi_size == 2048:
			target_image_dir = 'target_images/target_roi_6e3.jpg'

		if is_stain_norm:
			self.target_img = np.array(Image.open(target_image_dir))
			self.vhd = vahadane.vahadane(LAMBDA1=0.01,LAMBDA2=0.01,fast_mode=0,ITER=100)
			self.Wt,self.Ht = self.vhd.stain_separate(self.target_img)
		self.is_stain_norm = is_stain_norm

                   
        
	def __len__(self):
		return self.length
	
	def stain_norm(self,src_img):
		'''
		perform stain normalization for source img
		input (numpy array, shape: (3,h,w)): source image
		output (numpy array, shape: (3,h,w)): normalized image
		'''
		std = np.std(src_img[:,:,0].reshape(-1))
		# exclude images with large backgrounds
		if std < 5:
			return src_img,False
		else:
			Ws,Hs = self.vhd.stain_separate(src_img)
			img = self.vhd.SPCN(src_img,Ws,Hs,self.Wt,self.Ht)
		return img,True

	def __getitem__(self,idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]

		try:
			img = self.wsi.read_region(coord, self.roi_level, (self.roi_size, self.roi_size)).convert('RGB')
		except:
			# or subsequent normal patches will also raise errors
			self.wsi = openslide.open_slide(self.slide_path)
			available = False

		else:
			img = self.wsi.read_region(coord, self.roi_level, (self.roi_size, self.roi_size)).convert('RGB')
			available = True

		patch_num_all = np.sum(self.patch_nums)
		if not available:
			if self.resize:
				img_batch = torch.zeros((patch_num_all,3,224,224))
			else:
				img_batch = torch.zeros((patch_num_all,3,self.patch_size,self.patch_size))
				print(f'not available: {img_batch.shape}')
		else:
			img_batch = []
			img_roi = img.resize((self.target_roi_size,self.target_roi_size))

			if self.is_stain_norm:
				img_roi,flag = self.stain_norm(np.array(img_roi))

			if not flag:
				img_roi = torch.zeros((patch_num_all,3,self.patch_size,self.patch_size))
				available = False
			else:
				for level in self.levels:
					roi_size_cur = int(self.target_roi_size/(2**level))
					img_roi = np.array(img_roi)
					img_cur = Image.fromarray(img_roi).resize((roi_size_cur,roi_size_cur))
					
					imgarray = np.array(img_cur)
					for i in range(0,roi_size_cur,self.patch_size):
						for j in range(0,roi_size_cur,self.patch_size):
							img_patch = imgarray[i:i+self.patch_size,j:j+self.patch_size,:]
							if self.resize:
								img_patch = Image.fromarray(img_patch).resize((224,224))
								img_patch = np.array(img_patch)
							img_patch = self.transform_patch(img_patch)
							img_batch.append(img_patch)
					

				if available:
					img_batch = torch.stack(img_batch)

		return img_batch, coord, torch.tensor([available])


### for features extraction of single patch
class Patch_Seg_Dataset(Dataset):
	def __init__(self,
			  file_path,
			  slide_path,
			  wsi,
			  patch_size=256,
			  is_stain_norm=False):

		'''
		args:
			file_path (string): directory of bag (.h5 file), the coordinates of all patches segmented from the slide(bag)
			slide_path (string): directory of WSI
			wsi ('OpenSlide' object): object of WSI, for reading regions from WSI according to coordinates.
            patch_size (int): size of patch used for feature extraction
			is_stain_norm (bool): whether to perform stain normalization
		'''

		self.file_path = file_path
		self.wsi = wsi
		self.target_patch_size = patch_size
		self.slide_path = slide_path
		self.downscale = 0

		with h5py.File(self.file_path,'r') as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.downscale = int(self.patch_size/self.target_patch_size)
			self.length = len(dset)
		
		print(self.patch_size)

		if self.patch_size == 256:
			target_image_dir = 'target_images/target_image_6e3_256.jpg'
		if self.patch_size == 512:
			target_image_dir = 'target_images/target_image_6e3_512.jpg'
		if self.patch_size == 1024:
			target_image_dir = 'target_images/target_image_6e3_1024.jpg'
		if is_stain_norm:
			self.target_img = np.array(Image.open(target_image_dir))

			self.vhd = vahadane.vahadane(LAMBDA1=0.01,LAMBDA2=0.01,fast_mode=0,ITER=100)
			self.Wt,self.Ht = self.vhd.stain_separate(self.target_img)
			#self.vhd.fast_mode = 1 #fast separate
		self.is_stain_norm = is_stain_norm

                   
        
	def __len__(self):
		return self.length
	
	def stain_norm(self,src_img):

		std = np.std(src_img[:,:,0].reshape(-1))
		if std < 10:
			return src_img,False
		else:
			Ws,Hs = self.vhd.stain_separate(src_img)
			img = self.vhd.SPCN(src_img,Ws,Hs,self.Wt,self.Ht)
			return img,True

	def __getitem__(self,idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		#print(coord)
		try:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		except:
			# or subsequent normal patches will also raise errors
			self.wsi = openslide.open_slide(self.slide_path)
			available = False

		else:
			img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
			available = True

		if not available:
			img_patch = torch.ones((1,3,self.target_patch_size,self.target_patch_size))

		else:
			img_patch = img.resize((self.target_patch_size,self.target_patch_size))

			flag = True
			if self.is_stain_norm:
				img_patch,flag = self.stain_norm(np.array(img_patch))

			if not flag:
				img_patch = torch.ones((1,3,self.target_patch_size,self.target_patch_size))
				available = False
			else:

				img_patch = transform_patch(img_patch)
				img_patch = img_patch.unsqueeze(0)

		return img_patch, coord, torch.tensor([available])

