import pandas as pd
import numpy as np
# import os
import cv2
import math
import random
# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import torchvision.transforms as transforms
# from torchvision.models import resnet50
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm


# torch.cuda.empty_cache()
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(device)
# device = 'cpu'

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
video_frames_aug_path = '../data/video_frames_augmented1/'
video_features_path = '../data/video_features/'
max_video_len = 250 # set max len to 250 frames: cut off or pad to make them all the same length
plots_folder = './plots/'

classes = ["COVID-19", "pneumonia", "regular"]
classes_idx = {"COVID-19": 0, "Bacterial pneumonia": 1, "Viral pneumonia": 1, "regular": 2}


datagen = ImageDataGenerator() # rotation_range=30, horizontal_flip=True, shear_range=0.2, zoom_range=0.2

num_new_image = 5
for idx in range(metadata_df.shape[0]):
    img = []
    for j in range(metadata_df.iloc[idx]['frame_count']):
        # im = tf.keras.utils.load_img(video_frames_path+metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg")
        im = cv2.imread(video_frames_path+metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg")
        img.append(im)
    # img = np.array(img) # 41, 224, 224, 3
    # batchs = len(img)
    
    for j in range(num_new_image):
        subplotx = math.ceil(math.sqrt(len(img)))
        theta_parameters = {'theta': random.random()*30, 'tx': random.random()*0.4-0.2, 'ty': random.random()*0.4-0.2, \
            'shear': random.random()*0.2, 'flip_horizontal': round(random.random()), 'zx': random.random()*0.4+0.8, \
            'zy': random.random()*0.4+0.8, 'brightness': random.random()*0.4+0.8}
        print(theta_parameters)
        
        for i in range(len(img)):
            transformed_im = datagen.apply_transform(img[i], theta_parameters)
            plt.subplot(subplotx, subplotx, 1 + i)
            plt.imshow(transformed_im.astype('uint8'))
            cv2.imwrite(video_frames_aug_path+metadata_df.iloc[idx]['frames_filename']+"_batch"+str(j)+"_"+str(i)+".jpg", transformed_im)
        plt.savefig(plots_folder+"aug_plot_"+str(idx)+"_"+str(j)+".png")
    
    # datagen.fit(img)
    # i=0
    # for img_batch in datagen.flow(img):
    #     for img in img_batch:
    #         plt.subplot(subplotx, subplotx, 1 + i)
    #         plt.imshow(img.astype('uint8'))
    #         cv2.imwrite(video_frames_aug_path+metadata_df.iloc[idx]['frames_filename']+"_batch"+str(i)+".jpg", img)
    #         i+=1    
    #     if i >= batchs:   
    #         break
    break
        