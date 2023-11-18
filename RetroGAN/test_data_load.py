import glob
import random
import os
import torchvision.transforms as transforms
import torch
import random
import pickle
from torch.utils.data import Dataset
from PIL import Image

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def video_load(video_path,nt=5):
     video_info = []
     video_folder = os.listdir(video_path)
     for video_name in video_folder:
         print('video_name',video_name)
         num_img = len(os.listdir(os.path.join(video_path,video_name)))
         print('num image in video_name',video_name,num_img)
         for j in range(num_img-nt+1):
             print('creating index set')
             index_set = []
             for k in range(j, j + nt):
                 print('append', k,'png','in index_set')
                 index_set.append(os.path.join(video_path,os.path.join(video_name,str(k)+".png")))
             print('adding', j, 'th','index_set')
             video_info.append(index_set)
     print('length of video_info:',len(video_info))
     return video_info

testvideo = video_load('processed_video\\test\\traj_0_to_1',nt = 5)
trainvideo = video_load('processed_video\\train\\traj_0_to_2',nt = 5)
valvideo = video_load('processed_video\\val\\traj_0_to_0',nt = 5)
len(testvideo)
len(trainvideo)
len(valvideo)
import random
import pickle
test_output = open('./processed_video/test_data.pkl', 'wb')
train_output = open('./processed_video/train_data.pkl', 'wb')
val_output = open('./processed_video/val_data.pkl', 'wb')
pickle.dump(testvideo, test_output)
pickle.dump(trainvideo, train_output)
pickle.dump(valvideo, val_output)