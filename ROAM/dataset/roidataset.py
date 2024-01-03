import os
import torch
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


# generate dataset with batchsize of 1
class Wsi_Dataset_sb(Dataset):
    def __init__(self,slide_ids,label_ids,csv_path,data_dir,label_dict):
        '''
        Args:
                slide_ids (list): Ids of all WSIs in the dataset
                label_ids (list): Labels of all WSIs in the dataset
                csv_path (string): Path to the csv file with complete data information of all available WSIs
                data_dir (string): Root directory of all WSI data
                label_dict (dict): Dictionary with key, value pairs for converting label to int that can be used fot the current task
        '''
        super(Wsi_Dataset_sb,self).__init__()
        self.data_csv = pd.read_csv(csv_path)
        self.slide_ids_avl = self.data_csv['slide_id'].values
        self.slide_cls_ids = [[] for i in range(len(label_dict))]

        self.data_dir = data_dir
        self.label_dict = label_dict

        self.slide_data = []
        self.slide_label = []

        for i in range(len(label_ids)):
            if slide_ids[i] in self.slide_ids_avl:
                self.slide_data.append(slide_ids[i])
                self.slide_label.append(self.label_dict[label_ids[i]])
                self.slide_cls_ids[self.label_dict[label_ids[i]]].append(slide_ids[i])
        assert len(self.slide_data)==len(self.slide_label)
    
    def __len__(self):
        return len(self.slide_data)
    
    def get_label(self,idx):
        label = self.slide_label[idx]
        return label


    def __getitem__(self, idx):
        slide_id = self.slide_data[idx]
        label = self.slide_label[idx]

        feat_path = os.path.join(self.data_dir,f'{slide_id}.h5')
        with h5py.File(feat_path,'r') as hdf5_file:
            features = hdf5_file['features'][:] # num_patches,84,1024
            coords = hdf5_file['coords'][:] # num_patches,2
        
        features = torch.from_numpy(features)


        return features,coords,label

# generate dataset with batchsize exceeding 1
class Wsi_Dataset_mb(Dataset):
    def __init__(self,slide_ids,label_ids,csv_path,data_dir,label_dict):
        super(Wsi_Dataset_mb,self).__init__()
        self.data_csv = pd.read_csv(csv_path)
        self.slide_ids_avl = self.data_csv['slide_id'].values
        self.slide_cls_ids = [[] for i in range(len(label_dict))]

        self.data_dir = data_dir
        self.label_dict = label_dict

        self.slide_data = []
        self.slide_label = []

        for i in range(len(label_ids)):
            if slide_ids[i] in self.slide_ids_avl:
                self.slide_data.append(slide_ids[i])
                self.slide_label.append(self.label_dict[label_ids[i]])
                self.slide_cls_ids[self.label_dict[label_ids[i]]].append(slide_ids[i])
        assert len(self.slide_data)==len(self.slide_label)

    
    def __len__(self):
        return len(self.slide_data)

    def get_label(self,idx):
        label = self.slide_label[idx]
        return label

    def __getitem__(self, idx):
        slide_id = self.slide_data[idx]
        label = self.slide_label[idx]

        feat_path = os.path.join(self.data_dir,f'{slide_id}.h5')

        return feat_path,label # return path instead of specific data


## for cascade predict, no labels
class Wsi_Dataset_pred(Dataset):
    def __init__(self,slide_ids,csv_path,data_dir):
        super(Wsi_Dataset_pred,self).__init__()
        self.data_csv = pd.read_csv(csv_path)
        self.slide_ids_avl = self.data_csv['slide_id'].values

        self.data_dir = data_dir

        #self.slide_data = self.slide_data[self.slide_data['slide_id'].isin(sample_ids)].reset_index(drop=True)
        self.slide_data = []

        for i in range(len(slide_ids)):
            if slide_ids[i] in self.slide_ids_avl:
                self.slide_data.append(slide_ids[i])            
    
    def __len__(self):
        return len(self.slide_data)
    

    def __getitem__(self, idx):
        slide_id = self.slide_data[idx]

        feat_path = os.path.join(self.data_dir,f'{slide_id}.h5')
        with h5py.File(feat_path,'r') as hdf5_file:
            features = hdf5_file['features'][:] # num_patches,84,1024
            coords = hdf5_file['coords'][:] # num_patches,2
        
        features = torch.from_numpy(features)

        return features,coords