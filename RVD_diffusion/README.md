# RVDtestbatminton
experiment using framework from the paper "Diffusion Probabilistic Modeling for Video Generation" by Ruihan Yang et al. 
code base:https://github.com/buggyyang/RVD/tree/main
DDPM codebase: https://github.com/lucidrains/denoising-diffusion-pytorch

# Parse Arg use --ndevice updated May 19th, 2023
--ndevice is the argument of the number of cuda devices using for multiprocess
example terminal command will be like:
python batminton_train_PARSEARG.py --ndevice 1(I used this for Pycharm community edition)

# How to run
Run batmintontrain.py to get generated model
Run batmintontest.py to get generated video
# MODEL Parameters
obtained original code in https://github.com/buggyyang/RVD/blob/main/config.py
### training config
n_step = 1000000
scheduler_checkpoint_step = 100000
log_checkpoint_step = 4000
gradient_accumulate_every = 1
lr = 5e-5
decay = 0.8
minf = 0.2
ema_decay = 0.99
optimizer = "adam"  # adamw or adam
ema_step = 5
ema_start_step = 2000

### load
load_model = True
load_step = False

### diffusion config
loss_type = "l1"
iteration_step = 1600

context_dim_factor = 1
transform_dim_factor = 1
init_num_of_frame = 4
pred_mode = "noise"
clip_noise = True
transform_mode = "residual"
val_num_of_batch = 1
backbone = "resnet"
aux_loss = False
additional_note = ""

user can change the training "n_step" and diffusion "iteration_step", "init_num_of_frames" depends on the machines

# Data Location
more data can acquire by accessing the google drive: https://drive.google.com/drive/folders/189VTY_XFp6DnciozmrbVJ1HtdkXWdLrW?usp=share_link

# some troubleshooting
1. use this line : rm -rf `find -type d -name .ipynb_checkpoints` to remove any possible ipynb_checkpoints if using ipynb platform
2. rank and world size may alter depends on the use of machines(could also use the parse arg option)
3. some error may come from imports and module references, please send emails at shiziang200033@gmail.com if there is any other issues

# Reference
1.@misc{yang2022diffusion,
      title={Diffusion Probabilistic Modeling for Video Generation}, 
      author={Ruihan Yang and Prakhar Srivastava and Stephan Mandt},
      year={2022},
      eprint={2203.09481},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
2. @misc{ho2020denoising,
      title={Denoising Diffusion Probabilistic Models}, 
      author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
      year={2020},
      eprint={2006.11239},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
