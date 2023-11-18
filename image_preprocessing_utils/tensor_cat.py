import torch

ground_all_0 = torch.load('ivrnn_tensor_result/real/ivrnn_video_real_0.pt')
ground_all_1 = torch.load('ivrnn_tensor_result/real/ivrnn_video_real_1.pt')

fake_all_0 = torch.load('ivrnn_tensor_result/generated/ivrnn_video_generated_0.pt')
fake_all_1 = torch.load('ivrnn_tensor_result/generated/ivrnn_video_generated_1.pt')

stacked_tensor = torch.vstack((ground_all_0.unsqueeze(0), ground_all_1.unsqueeze(0)))

print(stacked_tensor.shape)