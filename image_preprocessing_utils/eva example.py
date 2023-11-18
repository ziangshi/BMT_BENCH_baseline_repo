import os
import glob
from numpy import append
import torch
import torch
import os
import math
import os.path as osp
import math
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from typing import Tuple
from scipy.linalg import sqrtm
#from video_diffusion_pytorch.evaluate import calc_metric, load_i3d_pretrained, get_fvd_feats, frechet_distance
import lpips
import numpy as np
from PIL import Image
import torch
# https://github.com/universome/fvd-comparison
i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"

'''
# Load the image
image1_path = 'results_ivrnn/generated_video_frames/frame_011.png'
image2_path = 'results_ivrnn/real_video_frames/frame_011.png'
image1 = np.array(Image.open(image1_path))
image2 = np.array(Image.open(image2_path))

# Convert to PyTorch tensor
#image1_tensor = torch.tensor(image1, dtype=torch.float64)
#image2_tensor = torch.tensor(image2, dtype=torch.float64)
print(image2.shape)#(64, 64, 3)
print(image1.shape)


print(image1.min())#5
print(image2.min())
print(image1.max())#251
print(image2.max())
val_range_1 = image1.max() - image1.min()
val_range_2 = image2.max() - image2.min()
print(val_range_1)#246
print(val_range_2)

print(image1.dtype)
print(compare_psnr(image1, image2,data_range=None))
'''






#correct number 34.1847 #008
#correct number 33.2064 #011
#correct number inf #001
#correct number inf #000
#correct number 35.42 #002
#correct number 36.469 #003
#correct number 36.068 #004

#image1_path = './results_ivrnn/generated_video_frames\\frame_000.png'
#image1 = np.array(Image.open(image1_path))
#print(image1.shape)

'''
# multiple tensors usage
psnr_list = []
ssim_list = []
lpips_list = []
ground_all_0 = torch.load('svg_tensor_result/real/svg_video_real_0.pt')
ground_all_1 = torch.load('svg_tensor_result/real/svg_video_real_1.pt')

fake_all_0 = torch.load('svg_tensor_result/generated/svg_video_generated_0.pt')
fake_all_1 = torch.load('svg_tensor_result/generated/svg_video_generated_1.pt')

stacked_tensor_ground = torch.vstack((ground_all_0.unsqueeze(0), ground_all_1.unsqueeze(0)))
stacked_tensor_fake = torch.vstack((fake_all_0.unsqueeze(0), fake_all_1.unsqueeze(0)))
#fake = fake_all.unsqueeze(0)
#ground = ground_all.unsqueeze(0)
# https://github.com/universome/fvd-comparison
  # best forward scores
for i in range(len(stacked_tensor_ground)):
    print('num:', i)
    ground_all=stacked_tensor_ground[i].unsqueeze(0)
    print(ground_all.shape)
    fake_all=stacked_tensor_fake[i].unsqueeze(0)
    num_frame = fake_all.shape[1]
    for j in range(2, num_frame):
        img1 = ground_all[:, :, j, :, :].clone()  # img1 = video1[:, :, i, :, :].clone()
        img2 = fake_all[:, :, j, :, :].clone()  # img2 = video2[:, :, i, :, :].clone()
        loss_fn_alex = lpips.LPIPS(net='alex').to(fake_all.device)
        lpips_list.append(loss_fn_alex(img1, img2).item())
        print('lpips', loss_fn_alex(img1, img2).item())
        img1 = img1.squeeze().numpy()
        img2 = img2.squeeze().numpy()
        # print shape max min dtype
        print(img1.shape)
        print(img2.shape)

        print(img1.max())
        print(img2.max())
        print(img1.min())
        print(img2.min())

        print(img1.dtype)
        psnr_list.append(compare_psnr(img1, img2, data_range=None))
        ssim_list.append(compare_ssim(img1, img2, data_range=None, channel_axis=0))
        print('psnr', compare_psnr(img1, img2, data_range=None))
        print('ssim', compare_ssim(img1, img2, data_range=None, channel_axis=0))
psnr = sum(psnr_list) / len(psnr_list)
ssim = sum(ssim_list) / len(ssim_list)
avg_lpips = sum(lpips_list) / len(lpips_list)
print("psnr ", psnr)
print("ssim ", ssim)
print("avg_lpips ", avg_lpips)

f=open('ivrnn_result.txt','a')
f.write( 'sample' + ': ' + str(1) + '  psnr:' + str(psnr) + '\t')
f.write('  ssim:' + str(ssim) + '\t')
f.write('  lpips:' + str(lpips) + '\n')
f.close()
'''

psnr_list = []
ssim_list = []
lpips_list = []

fake_all = torch.load('ablation_tensor/ivrnn2_10_0.pt').unsqueeze(0)
ground_all = torch.load('ablation_tensor/ground_0.pt').unsqueeze(0)

num_frame= fake_all.shape[1]
for i in range(2,num_frame):
    img1 = ground_all[:,:, i, :, :].clone() # img1 = video1[:, :, i, :, :].clone()
    img2 = fake_all[:,:, i, :, :].clone() # img2 = video2[:, :, i, :, :].clone()
    loss_fn_alex = lpips.LPIPS(net='alex').to(fake_all.device)
    lpips_list.append(loss_fn_alex(img1, img2).item())
    print('lpips',loss_fn_alex(img1, img2).item())
    img1 = img1.squeeze().numpy()
    img2 = img2.squeeze().numpy()
    #print shape max min dtype
    print(img1.shape)
    print(img2.shape)

    print(img1.max())
    print(img2.max())
    print(img1.min())
    print(img2.min())

    print(img1.dtype)
    psnr_list.append(compare_psnr(img1, img2, data_range=None))
    ssim_list.append(compare_ssim(img1, img2, data_range=None, channel_axis=0))
    print('psnr',compare_psnr(img1, img2,data_range=None))
    print('ssim',compare_ssim(img1, img2, data_range=None, channel_axis=0))

psnr = sum(psnr_list) / len(psnr_list)
ssim = sum(ssim_list) / len(ssim_list)
avg_lpips = sum(lpips_list) / len(lpips_list)

print("psnr ", psnr)
print("ssim ", ssim)
print("avg_lpips ", avg_lpips)

f=open('rvd_result.txt','a')
f.write( 'sample' + ': ' + str(1) + '  psnr:' + str(psnr) + '\t')
f.write('  ssim:' + str(ssim) + '\t')
f.write('  lpips:' + str(lpips) + '\n')
f.close()

# normalize the original to range [0.0, 1.0]
#original_normed = (ground_all - val_min) / val_range
# apply identical normalization to the denoised image (important!)
#im_bayes_normed = (fake_all - val_min) / val_range



