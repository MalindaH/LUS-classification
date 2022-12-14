import pandas as pd
import numpy as np
# import os
import cv2
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


torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# device = 'cpu'

model_path_number = 2
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



# video_lens = torch.load(video_features_path+'video_lens.pt')
class ImageFeatureDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata_df = metadata_df

    def __len__(self):
        return metadata_df.shape[0]

    def __getitem__(self, idx):
        images = torch.load(video_features_path+self.metadata_df.iloc[idx]['frames_filename']+'.pt').to(device)
        label = self.metadata_df.iloc[idx]['class']
        label_idx = classes_idx[label]
        # video_len = video_lens[idx]
        video_len = self.metadata_df.iloc[idx]['frame_count']
        return images, label_idx, video_len

training_data = ImageFeatureDataset(metadata_df)
train_size = int(training_data.__len__()*0.7)
test_size = training_data.__len__() - train_size
train_data, test_data = torch.utils.data.random_split(training_data, [train_size, test_size], generator=torch.Generator().manual_seed(42))
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)




## text feature extractor LSTM: https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
class LSTMModel(nn.Module):
    def __init__(self):
      super(LSTMModel, self).__init__()
      self.lstm_size = 512
    #   self.embedding_dim = 512
      self.num_layers = 2

    #   self.embedding = nn.Embedding(
    #       num_embeddings=n_vocab,
    #       embedding_dim=self.embedding_dim,
    #       padding_idx=0
    #   )
      self.lstm = nn.LSTM(
          input_size=512,
          hidden_size=self.lstm_size,
          num_layers=self.num_layers,
          batch_first=True,
          bidirectional=True
      )
      self.tanh = nn.Tanh()
      self.fc1 = nn.Linear(2*self.num_layers*self.lstm_size, 3)

    def forward(self, images, video_len): # batch_size, 250,512
        # text_emb = self.embedding(q) # batch_size,23,512
        # images = self.tanh(images)
        # packed_input = pack_padded_sequence(images, video_len, batch_first=True, enforce_sorted=False)
        _, (ht, ct) = self.lstm(images) # ct: 4, batch_size, 512

        ct = ct.transpose(0, 1) # batch_size, 4, 512
        ct = ct.reshape(ct.size()[0], -1) # batch_size, 2048
        output = self.fc1(ct) # batch_size, 2048
        return output

lstm_net = LSTMModel().to(device)



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=max_video_len):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



def eval_test(model):
    model.eval()
    
    gtAcc = []
    for i, data in enumerate(test_dataloader):
        images, label_idx, video_len = data
        label_idx = label_idx.to(device)
        
        outputs = lstm_net(images, video_len)
        
        for j in range(outputs.shape[0]):
            ans = torch.argmax(outputs[j]).cpu().numpy()
            correct_ans = label_idx[j].item()
            print("ans:",ans, correct_ans)
            if ans == correct_ans:
                gtAcc.append(1)
            else:
                gtAcc.append(0)
        avgGTAcc = float(sum(gtAcc))/len(gtAcc)
        model.train()

    return avgGTAcc


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm_net.parameters(), lr=0.001)

eval_every = len(train_dataloader)//2

lstm_net.train()

for epoch in tqdm(range(epochs)):
    running_loss = 0.0
    num_correct = 0
    for i, data in enumerate(train_dataloader):
        images, label_idx, video_len = data
        label_idx = label_idx.to(device)
        # print(images) # batch_size, 250, 512
        # print(video_len) # batch_size
        # print(lstm_net(images, video_len))
        
        optimizer.zero_grad()
        outputs = lstm_net(images, video_len) # batch_size, 3
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

    print("validation accuracy: ", eval_test(lstm_net))


torch.save(lstm_net.state_dict(), f'torch_models/lstm_net{model_path_number}')  


