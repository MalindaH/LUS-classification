import pandas as pd
# import numpy as np
# import os
import cv2
import torch
import math
from torch import nn
# import torch.nn.functional as F
# from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
# import matplotlib.pyplot as plt
# from tqdm import tqdm

from resnet_uscl import ResNetUSCL


torch.cuda.empty_cache()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
device = 'cpu'

model_path_number = 1
batch_size = 4
feature_embedding_dim = 128
epochs = 5


pocus_folder = '../covid19_ultrasound/data/pocus_videos/convex/'
pocus_metadata = '../covid19_ultrasound/data/dataset_metadata.csv'

covidus_folder_uncropped = '../COVID-US/data/video/'
covidus_folder_cropped = '../COVID-US/data/video/cropped/' 
# covidus_metadata = '../COVID-US/utils/video_metadata.csv'
covidus_metadata = '../COVID-US/utils/video_metadata_cropped.txt'

toSave = True
metadata_df = pd.read_csv('metadata_df1.csv')
video_frames_path = '../data/video_frames/'
video_frames_aug_path = '../data/video_frames_augmented1/'
video_features_path = '../data/video_features1/'
video_features_aug_path = '../data/video_features_augmented1/'
max_video_len = 250 # set max len to 250 frames: cut off or pad to make them all the same length

classes = ["COVID-19", "pneumonia", "regular"]
classes_idx = {"COVID-19": 0, "Bacterial pneumonia": 1, "Viral pneumonia": 1, "regular": 2}


## extract and store image features
# video_lens = torch.empty(metadata_df.shape[0])
class ResnetExtractor(nn.Module):
    def __init__(self):
        super(ResnetExtractor, self).__init__()
        self.rn_model = resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.rn_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x
# nn_resnet = ResnetExtractor().to(device)
# print(nn_resnet)

state_dict_path = '../USCL/checkpoint/best_model.pth'
def load_uscl():
    net = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=True)
    # import pretrained model weights
    state_dict = torch.load(state_dict_path)
    new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                if not (k.startswith('l')
                        | k.startswith('fc'))}  # # discard MLP and fc
    model_dict = net.state_dict()

    model_dict.update(new_dict)
    net.load_state_dict(model_dict)
    return net

class USCLExtractor(nn.Module):
    def __init__(self):
        super(USCLExtractor, self).__init__()
        self.rn_model = load_uscl()
        self.features = nn.Sequential(*list(self.rn_model.children())[:-2]) # remove projection MLP and classifier layers
            
    def forward(self, x):
        x = self.features(x)
        return x
# nn_uscl = USCLExtractor().to(device)
# print(nn_uscl)

class FeatureExtractor():
    def __init__(self, network="USCL"):
        self.extractor = USCLExtractor().to(device)
        if network!="USCL": # network=="resnet"
            self.extractor = ResnetExtractor().to(device)
        self.extractor.eval()

    def extract(self, img):
        return self.extractor(img)

uscl_extractor = FeatureExtractor("USCL")



class ImageDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df
        self.transform = transform
        self.extractor = uscl_extractor

    def __len__(self):
        return metadata_df.shape[0]*6

    def __getitem__(self, idx):
        images = torch.empty(0).to(device) # 250, 512
        filename = ""
        idx2 = idx
        if idx < metadata_df.shape[0]:
            filename = video_frames_path+self.metadata_df.iloc[idx2]['frames_filename']
        else:
            idx2 = idx % metadata_df.shape[0]
            i = math.floor(idx/metadata_df.shape[0])-1
            # print(idx, idx2, i)
            filename = video_frames_aug_path+self.metadata_df.iloc[idx2]['frames_filename']+"_batch"+str(i)
        j=0
        while j < min(self.metadata_df.iloc[idx2]['frame_count']-1, max_video_len-1):
        # for j in range(self.metadata_df.iloc[idx]['frame_count']):
            # img = cv2.imread(video_frames_path+self.metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg")
            img = cv2.imread(filename+"_"+str(j)+".jpg")
            if self.transform is not None:
                img = self.transform(img)
            features = self.extractor.extract(img[None, :].to(device)) # 1,512,1,1
            images = torch.cat((images, features[:,:,0,0]), 0)
            j += 1
        # video_len = j+1
        while j < max_video_len:
            images = torch.cat((images, torch.zeros((1,512)).to(device)), 0) # 250, 512
            j += 1
        if toSave:
            # torch.save(images, video_features_aug_path+self.metadata_df.iloc[idx]['frames_filename']+'.pt')
            if idx < metadata_df.shape[0]:
                torch.save(images, video_features_path+self.metadata_df.iloc[idx2]['frames_filename']+'.pt')
            else:
                idx2 = idx % metadata_df.shape[0]
                i = math.floor(idx/metadata_df.shape[0])-1
                torch.save(images, video_features_aug_path+self.metadata_df.iloc[idx2]['frames_filename']+"_batch"+str(i)+'.pt')
        # video_lens[idx] = video_len
        return images

image_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
])

all_image_data = ImageDataset(metadata_df, image_transform)
all_image_dataloader = DataLoader(all_image_data, batch_size=batch_size, drop_last=False)

## extract and store image features: only run once
for idx, data in enumerate(all_image_dataloader):
    # images, label_idx, video_len = data
    images = data
# print(video_lens)
# if toSave:
#     torch.save(video_lens, video_features_aug_path+'video_lens.pt')
exit()

