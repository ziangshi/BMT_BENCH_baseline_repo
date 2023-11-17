# training config
N_CONTEXT = 4
N_BATCH = 5
BATCH_SIZE = 1
# n_step = 1000000
# scheduler_checkpoint_step = 100000
# log_checkpoint_step = 4000
# gradient_accumulate_every = 1
# lr = 5e-5
# decay = 0.8
# minf = 0.2
# ema_decay = 0.99
optimizer = "adam"  # adamw or adam
# ema_step = 10
# ema_start_step = 2000

# diffusion config
loss_type = "l1"
iteration_step = 1600

context_dim_factor = 1
transform_dim_factor = 1
init_num_of_frame = 4
pred_modes = "noise"
clip_noise = True
transform_modes = "residual"
val_num_of_batch = 1
backbone = "resnet"
aux_loss = False
additional_note = ""


# data config
'''
data_configs = [
{
    "dataset_name": "simu",
    "data_path": "a8",
    "sequence_length": 20,
    "img_size": 128,
    "img_channel": 1,
    "add_noise": False,
    "img_hz_flip": False,
},
{
    "dataset_name": "city",
    "data_path": "*",
    "sequence_length": 20,
    "img_size": 128,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
},
{
    "dataset_name": "bair_robot_pushing",
    "data_path": "*",
    "sequence_length": 20,
    "img_size": 64,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
},
{
    "dataset_name": "kth_actions",
    "data_path": "*",
    "sequence_length": 16,
    "img_size": 64,
    "img_channel": 1,
    "add_noise": False,
    "img_hz_flip": False,
},
 {
     "dataset_name": "city",
     "data_path": "/extra/ucibdl0/shared/data",
     "sequence_length": 20,
     "img_size": 256,
     "img_channel": 3,
     "add_noise": False,
     "img_hz_flip": False,
 },
]
'''

data_configs = {
    "dataset_name": "bat",
    "data_path": "processed_video/video_train",
    "sequence_length": 8,
    "img_size": 64,
    "img_channel": 3,
    "add_noise": False,
    "img_hz_flip": False,
}


result_root = "model_results"
tensorboard_root = "tensorboard_results"