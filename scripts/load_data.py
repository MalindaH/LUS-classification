import pandas as pd
# import numpy as np
# import os
import cv2
import torch
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

metadata_df = pd.read_csv('metadata_df1.csv')
video_frames_path = '../data/video_frames/'
video_features_path = '../data/video_features2/'
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
        images = torch.empty(0).to(device)
        j=0
        while j < min(self.metadata_df.iloc[idx]['frame_count']-1, max_video_len-1):
        # for j in range(self.metadata_df.iloc[idx]['frame_count']):
            img = cv2.imread(video_frames_path+self.metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg")
            if self.transform is not None:
                img = self.transform(img)
            features = self.extractor.extract(img[None, :].to(device)) # 1,512,1,1
            images = torch.cat((images, features[:,:,0,0]), 0)
            j += 1
        video_len = j+1
        while j < max_video_len:
            images = torch.cat((images, torch.zeros((1,512)).to(device)), 0) # 250, 512
            j += 1
        # label = self.metadata_df.iloc[idx]['class']
        # label_idx = classes_idx[label]
        torch.save(images, video_features_path+self.metadata_df.iloc[idx]['frames_filename']+'.pt')
        video_lens[idx] = video_len
        # return images, label_idx, video_len
        return images

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25])
])

all_image_data = ImageDataset(metadata_df, image_transform)
all_image_dataloader = DataLoader(all_image_data, batch_size=batch_size, drop_last=False)

## extract and store image features: only run once
for idx, data in enumerate(all_image_dataloader):
    # images, label_idx, video_len = data
    images = data
print(video_lens) # [ 41.,  21.,  83.,  40.,  40.,  59.,  60.,  60.,  59.,  60.,  39.,  33.,
        #  40.,  60.,  40., 100., 223., 131., 250., 104., 206., 250., 250., 246.,
        # 250., 200., 218., 183., 250., 250., 236., 250., 208., 215., 217., 207.,
        # 247., 250., 209., 250., 250., 250., 215., 250., 218., 250.,  27.,  24.,
        #  26.,  41.,  25.,  27.,  42.,  42.,  36.,  32., 113., 191., 174., 137.,
        # 182.]
torch.save(video_lens, video_features_path+'video_lens.pt')
exit()

# video_lens = torch.load(video_features_path+'video_lens.pt')
# class ImageFeatureDataset(Dataset):
#     def __init__(self, metadata_df):
#         self.metadata_df = metadata_df

#     def __len__(self):
#         return metadata_df.shape[0]

#     def __getitem__(self, idx):
#         images = torch.load(video_features_path+self.metadata_df.iloc[idx]['frames_filename']+'.pt')
#         label = self.metadata_df.iloc[idx]['class']
#         label_idx = classes_idx[label]
#         # video_len = video_lens[idx]
#         video_len = self.metadata_df.iloc[idx]['frame_count']
#         return images, label_idx, video_len

# training_data = ImageFeatureDataset(metadata_df)
# train_size = int(training_data.__len__()*0.75)
# test_size = training_data.__len__() - train_size
# train_data, test_data = torch.utils.data.random_split(training_data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
# train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
# test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)




# ## text feature extractor LSTM: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
# class LSTMModel(nn.Module):
#     def __init__(self):
#       super(LSTMModel, self).__init__()
#       self.lstm_size = 512
#     #   self.embedding_dim = 512
#       self.num_layers = 2

#     #   self.embedding = nn.Embedding(
#     #       num_embeddings=n_vocab,
#     #       embedding_dim=self.embedding_dim,
#     #       padding_idx=0
#     #   )
#       self.lstm = nn.LSTM(
#           input_size=512,
#           hidden_size=self.lstm_size,
#           num_layers=self.num_layers,
#           batch_first=True,
#           bidirectional=True
#       )
#       self.tanh = nn.Tanh()
#       self.fc1 = nn.Linear(2*self.num_layers*self.lstm_size, 3)

#     def forward(self, images, video_len): # batch_size, 250,512
#         # text_emb = self.embedding(q) # batch_size,23,512
#         # images = self.tanh(images)
#         # packed_input = pack_padded_sequence(images, video_len, batch_first=True, enforce_sorted=False)
#         _, (ht, ct) = self.lstm(images) # ct: 4, batch_size, 512

#         ct = ct.transpose(0, 1) # batch_size, 4, 512
#         ct = ct.reshape(ct.size()[0], -1) # batch_size, 2048
#         output = self.fc1(ct) # batch_size, 2048
#         return output

# lstm_net = LSTMModel().to(device)



# class AttnDecoderRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_video_len):
#         super(AttnDecoderRNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.dropout_p = dropout_p
#         self.max_length = max_length

#         self.embedding = nn.Embedding(self.output_size, self.hidden_size)
#         self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
#         self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.gru = nn.GRU(self.hidden_size, self.hidden_size)
#         self.out = nn.Linear(self.hidden_size, self.output_size)

#     def forward(self, input, hidden, encoder_outputs):
#         embedded = self.embedding(input).view(1, 1, -1)
#         embedded = self.dropout(embedded)

#         attn_weights = F.softmax(
#             self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
#         attn_applied = torch.bmm(attn_weights.unsqueeze(0),
#                                  encoder_outputs.unsqueeze(0))

#         output = torch.cat((embedded[0], attn_applied[0]), 1)
#         output = self.attn_combine(output).unsqueeze(0)

#         output = F.relu(output)
#         output, hidden = self.gru(output, hidden)

#         output = F.log_softmax(self.out(output[0]), dim=1)
#         return output, hidden, attn_weights

#     def initHidden(self):
#         return torch.zeros(1, 1, self.hidden_size, device=device)



# def eval_test(model):
#     model.eval()
    
#     gtAcc = []
#     for i, data in enumerate(test_dataloader):
#         images, label_idx, video_len = data
#         label_idx = label_idx.to(device)
        
#         outputs = lstm_net(images, video_len)
        
#         for j in range(outputs.shape[0]):
#             ans = torch.argmax(outputs[j]).cpu().numpy()
#             correct_ans = label_idx[j]
#             print("ans:",ans, correct_ans)
#             if ans == correct_ans:
#                 gtAcc.append(1)
#             else:
#                 gtAcc.append(0)
#         avgGTAcc = float(sum(gtAcc))/len(gtAcc)
#         model.train()

#     return avgGTAcc


# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(lstm_net.parameters(), lr=0.001)

# eval_every = len(train_dataloader)//3

# lstm_net.train()

# for epoch in tqdm(range(epochs)):
#     running_loss = 0.0
#     num_correct = 0
#     for i, data in enumerate(train_dataloader):
#         images, label_idx, video_len = data
#         label_idx = label_idx.to(device)
#         # print(images) # batch_size, 250, 512
#         # print(video_len) # batch_size
#         # print(lstm_net(images, video_len))
        
#         optimizer.zero_grad()
#         outputs = lstm_net(images, video_len) # batch_size, 3
#         loss = criterion(outputs, label_idx)
        
#         loss.backward()
#         optimizer.step()
        
#         for j in range(label_idx.shape[0]):
#             if torch.argmax(outputs[j]) == label_idx[j]:
#                 num_correct += 1

#         # print statistics
#         running_loss += loss.item()
#         if i % eval_every == eval_every-1:    # print every (eval_every) mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / eval_every:.3f}')
#             print(f'[{epoch + 1}, {i + 1:5d}] accuracy: {num_correct / eval_every / batch_size}')
#             running_loss = 0.0
#             num_correct = 0

#     print("validation accuracy: ", eval_test(lstm_net))


# torch.save(lstm_net.state_dict(), f'torch_models/lstm_net{model_path_number}')  


