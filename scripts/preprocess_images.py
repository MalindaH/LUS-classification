import pandas as pd
import numpy as np
# import os
import cv2
import math
import random
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# import keras_cv

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
IMG_SIZE = 224


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


# ## some random classical image augmentation
# datagen = ImageDataGenerator() # rotation_range=30, horizontal_flip=True, shear_range=0.2, zoom_range=0.2
# num_new_image = 5
# for idx in range(metadata_df.shape[0]):
#     img = [] # 41, 224, 224, 3
#     for j in range(metadata_df.iloc[idx]['frame_count']):
#         # im = tf.keras.utils.load_img(video_frames_path+metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg")
#         im = cv2.imread(video_frames_path+metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg")
#         img.append(im)
    
#     for j in range(num_new_image):
#         subplotx = math.ceil(math.sqrt(len(img)))
#         theta_parameters = {'theta': random.random()*30, 'tx': random.random()*0.3-0.15, 'ty': random.random()*0.3-0.15, \
#             'shear': random.random()*0.15, 'flip_horizontal': round(random.random()), 'zx': random.random()*0.3+0.85, \
#             'zy': random.random()*0.3+0.85, 'brightness': random.random()*0.4+0.8}
#         # print(theta_parameters)
        
#         for i in range(len(img)):
#             transformed_im = datagen.apply_transform(img[i], theta_parameters)
#             plt.subplot(subplotx, subplotx, 1 + i)
#             plt.imshow(transformed_im.astype('uint8'))
#             cv2.imwrite(video_frames_aug_path+metadata_df.iloc[idx]['frames_filename']+"_batch"+str(j)+"_"+str(i)+".jpg", transformed_im)
#         plt.savefig(plots_folder+"aug_plot_"+metadata_df.iloc[idx]['frames_filename']+"_batch"+str(j)+".png")


## CutMix: https://keras.io/examples/vision/cutmix/

def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

@tf.function
def get_box(lambda_value):
    cut_rat = tf.math.sqrt(1.0 - lambda_value)

    cut_w = IMG_SIZE * cut_rat  # rw
    cut_w = tf.cast(cut_w, tf.int32)

    cut_h = IMG_SIZE * cut_rat  # rh
    cut_h = tf.cast(cut_h, tf.int32)

    cut_x = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # rx
    cut_y = tf.random.uniform((1,), minval=0, maxval=IMG_SIZE, dtype=tf.int32)  # ry

    boundaryx1 = tf.clip_by_value(cut_x[0] - cut_w // 2, 0, IMG_SIZE)
    boundaryy1 = tf.clip_by_value(cut_y[0] - cut_h // 2, 0, IMG_SIZE)
    bbx2 = tf.clip_by_value(cut_x[0] + cut_w // 2, 0, IMG_SIZE)
    bby2 = tf.clip_by_value(cut_y[0] + cut_h // 2, 0, IMG_SIZE)

    target_h = bby2 - boundaryy1
    if target_h == 0:
        target_h += 1

    target_w = bbx2 - boundaryx1
    if target_w == 0:
        target_w += 1

    return boundaryx1, boundaryy1, target_h, target_w

@tf.function
def cutmix(train_ds_one, train_ds_two):
    (image1, label1), (image2, label2) = train_ds_one, train_ds_two

    alpha = [0.25]
    beta = [0.25]

    # Get a sample from the Beta distribution
    lambda_value = sample_beta_distribution(1, alpha, beta)

    # Define Lambda
    lambda_value = lambda_value[0][0]

    # Get the bounding box offsets, heights and widths
    boundaryx1, boundaryy1, target_h, target_w = get_box(lambda_value)

    # Get a patch from the second image (`image2`)
    crop2 = tf.image.crop_to_bounding_box(
        image2, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image2` patch (`crop2`) with the same offset
    image2 = tf.image.pad_to_bounding_box(
        crop2, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
    )
    # Get a patch from the first image (`image1`)
    crop1 = tf.image.crop_to_bounding_box(
        image1, boundaryy1, boundaryx1, target_h, target_w
    )
    # Pad the `image1` patch (`crop1`) with the same offset
    img1 = tf.image.pad_to_bounding_box(
        crop1, boundaryy1, boundaryx1, IMG_SIZE, IMG_SIZE
    )

    # Modify the first image by subtracting the patch from `image1`
    # (before applying the `image2` patch)
    image1 = image1 - img1
    # Add the modified `image1` and `image2`  together to get the CutMix image
    image = image1 + image2

    # Adjust Lambda in accordance to the pixel ration
    lambda_value = 1 - (target_w * target_h) / (IMG_SIZE * IMG_SIZE)
    lambda_value = tf.cast(lambda_value, tf.float32)

    # Combine the labels of both images
    label = lambda_value * label1 + (1 - lambda_value) * label2
    return image, label

for idx in range(metadata_df.shape[0]):
    img = [] # num_frames, 224, 224, 3
    for j in range(metadata_df.iloc[idx]['frame_count']):
        im = cv2.imread(video_frames_path+metadata_df.iloc[idx]['frames_filename']+"_"+str(j)+".jpg") # 224,224,3
        # im2 = cv2.resize(im, [IMG_SIZE, IMG_SIZE]) # 224,224,3
        img.append(im)
    

## MixUp
        