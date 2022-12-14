import pandas as pd
import numpy as np
# import os
import cv2
import random
import math
import torch
from torch import nn
import torch.nn.functional as F
# from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50
# import matplotlib.pyplot as plt
from tqdm import tqdm

from resnet_uscl import ResNetUSCL
# from load_data import FeatureExtractor


torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'

model_path_number = 1
batch_size = 16
epochs = 5


pocus_folder = '../covid19_ultrasound/data/pocus_videos/convex/'
pocus_metadata = '../covid19_ultrasound/data/dataset_metadata.csv'

covidus_folder_uncropped = '../COVID-US/data/video/'
covidus_folder_cropped = '../COVID-US/data/video/cropped/' 
# covidus_metadata = '../COVID-US/utils/video_metadata.csv'
covidus_metadata = '../COVID-US/utils/video_metadata_cropped.txt'

metadata_df = pd.read_csv('metadata_df1.csv')
video_frames_path = '../data/video_frames/'
video_features_path = '../data/video_features/'
max_video_len = 250 # set max len to 250 frames: cut off or pad to make them all the same length

classes = ["COVID-19", "pneumonia", "regular"]
classes_idx = {"COVID-19": 0, "Bacterial pneumonia": 1, "Viral pneumonia": 1, "regular": 2}



## extract and store image features
video_lens = torch.empty(metadata_df.shape[0])
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
        return metadata_df.shape[0]

    def __getitem__(self, idx):
        # images = torch.empty(0).to(device)
        j = math.floor(random.random()*self.metadata_df.iloc[idx]['frame_count'])
        # j=0
        # while j < min(self.metadata_df.iloc[idx]['frame_count']-1, max_video_len-1):
        # for j in range(self.metadata_df.iloc[idx]['frame_count']):
        img = cv2.imread(video_frames_path+self.metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg") # 224, 224, 3
        if self.transform is not None:
            img = self.transform(img)
        feature = self.extractor.extract(img[None, :].to(device)) # 1,512,1,1
        feature = torch.squeeze(feature)
        label = self.metadata_df.iloc[idx]['class']
        label_idx = classes_idx[label]
        # images = torch.cat((images, features[:,:,0,0]), 0)
        # j += 1
        # video_len = j+1
        # while j < max_video_len:
        #     images = torch.cat((images, torch.zeros((1,512)).to(device)), 0) # 250, 512
        #     j += 1
        # label = self.metadata_df.iloc[idx]['class']
        # label_idx = classes_idx[label]
        # torch.save(images, video_features_path+self.metadata_df.iloc[idx]['frames_filename']+'.pt')
        # video_lens[idx] = video_len
        # return images, label_idx, video_len
        return feature, label_idx

image_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
])

all_image_data = ImageDataset(metadata_df, image_transform)
train_size = int(all_image_data.__len__()*0.7)
test_size = all_image_data.__len__() - train_size
train_data, test_data = torch.utils.data.random_split(all_image_data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)


class BaseModel(nn.Module):
    def __init__(self):
      super(BaseModel, self).__init__()
      self.fc1 = nn.Linear(512, 3)

    def forward(self, image_feature): # batch_size, 512
        output = self.fc1(image_feature) # batch_size, 2048
        return output

baseline_net = BaseModel().to(device)





def eval_test(model):
    model.eval()
    
    gtAcc = []
    for i, data in enumerate(test_dataloader):
        image, label_idx = data
        label_idx = label_idx.to(device)
        
        outputs = baseline_net(image)
        
        for j in range(outputs.shape[0]):
            ans = torch.argmax(outputs[j]).cpu().numpy()
            correct_ans = label_idx[j].item()
            # print("ans:",ans, correct_ans)
            if ans == correct_ans:
                gtAcc.append(1)
            else:
                gtAcc.append(0)
        avgGTAcc = float(sum(gtAcc))/len(gtAcc)
    model.train()

    return avgGTAcc


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(baseline_net.parameters(), lr=0.001)

eval_every = len(train_dataloader)//2

baseline_net.train()

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    num_correct = 0
    for i, data in enumerate(train_dataloader):
        image, label_idx = data
        label_idx = label_idx.to(device)
        # print(image, image.shape) # batch_size, 512
        
        optimizer.zero_grad()
        outputs = baseline_net(image)
        loss = criterion(outputs, label_idx)
        
        loss.backward()
        optimizer.step()
        
        for j in range(label_idx.shape[0]):
            if torch.argmax(outputs[j]) == label_idx[j]:
                num_correct += 1

        # print statistics
        running_loss += loss.item()
        if i % eval_every == eval_every-1:    # print every (eval_every) mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / eval_every:.3f}')
            print(f'[{epoch + 1}, {i + 1:5d}] accuracy: {num_correct / eval_every / batch_size}')
            running_loss = 0.0
            num_correct = 0

    print("validation accuracy: ", eval_test(baseline_net))


torch.save(baseline_net.state_dict(), f'torch_models/baseline_net{model_path_number}')  


