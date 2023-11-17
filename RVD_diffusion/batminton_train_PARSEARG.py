'''
data loading required import
'''
import os
import tarfile
import numpy as np
import torch
import numpy as np
import torchvision.transforms as torch_transforms
import torchvision.transforms.functional as VF
from PIL import Image, ImageChops
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
import torch
'''
trainer required import
'''
import config
import argparse
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from modules.denoising_diffusion import GaussianDiffusion
from modules.unet import Unet
from modules.trainer import Trainer
from modules.temporal_models import HistoryNet, CondNet
from torch.nn.parallel import DistributedDataParallel as DDP

'''
required a modules folders with necessary py files of Diffusion and unet under root folder
'''


'''
transform
'''
class Resize(object):
    """
    Resizes a PIL image or sequence of PIL images.
    img_size can be an int, list or tuple (width, height)
    """

    def __init__(self, img_size):
        if type(img_size) != tuple and type(img_size) != list:
            img_size = (img_size, img_size)
        self.img_size = img_size

    def __call__(self, input):
        if type(input) == list:
            return [im.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR) for im in input]
        return input.resize((self.img_size[0], self.img_size[1]), Image.BILINEAR)

class RandomSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """

    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, input):
        if type(input) == list:
            input_seq_len = len(input)
        elif "shape" in dir(input):
            input_seq_len = input.shape[0]
        max_start_ind = input_seq_len - self.seq_len + 1
        assert max_start_ind > 0, (
            "Sequence length longer than input sequence length: " + str(input_seq_len) + "."
        )
        start_ind = np.random.choice(range(max_start_ind))
        #start_ind = torch.randint(0, max_start_ind, (1,)).item()
        return input[start_ind : start_ind + self.seq_len]

class ImageToTensor(object):
    """
    Converts a PIL image or sequence of PIL images into (a) PyTorch tensor(s).
    """

    def __init__(self):
        self.to_tensor = torch_transforms.ToTensor()

    def __call__(self, input):
        if type(input) == list:
            return [self.to_tensor(i) for i in input]
        return self.to_tensor(input)

class ConcatSequence(object):
    """
    Concatenates a sequence (list of tensors) along a new axis.
    """

    def __init__(self):
        pass

    def __call__(self, input):
        return torch.stack(input)

class FixedSequenceCrop(object):
    """
    Randomly crops a sequence (list or tensor) to a specified length.
    """

    def __init__(self, seq_len, start_index=0):
        self.seq_len = seq_len
        self.start_index = start_index

    def __call__(self, input):
        return input[self.start_index : self.start_index + self.seq_len]
Compose = torch_transforms.Compose

'''
our directory structure is similar to big dataset
'''

class BIG(Dataset):
    """
    Dataset object for BAIR robot pushing dataset. The dataset must be stored
    with each video in a separate directory:
        /path
            /0
                /0.png
                /1.png
                /...
            /1
                /...
    """

    """
    We use the same format of the video directory for our dataset
    """

    def __init__(self, path, transform=None, add_noise=False, img_mode=False):
        assert os.path.exists(
            path), 'Invalid path to UCF+HMDB data set: ' + path
        self.path = path
        self.transform = transform
        self.video_list = os.listdir(self.path)
        self.img_mode = img_mode
        self.add_noise = add_noise

    def __getitem__(self, ind):
        # load the images from the ind directory to get list of PIL images
        img_names = os.listdir(os.path.join(
            self.path, self.video_list[ind]))
        img_names = [img_name.split('.')[0] for img_name in img_names]
        img_names.sort(key=float)
        if not self.img_mode:
            imgs = [Image.open(os.path.join(
                self.path, self.video_list[ind], i + '.png')) for i in img_names]
        else:
            select = torch.randint(0, len(img_names), (1,))
            imgs = [Image.open(os.path.join(
                self.path, self.video_list[ind], img_names[select] + '.png'))]
        if self.transform is not None:
            # apply the image/video transforms
            imgs = self.transform(imgs)

        # imgs = imgs.unsqueeze(1)

        if self.add_noise:
            imgs = imgs + (torch.rand_like(imgs)-0.5) / 256.

        return imgs

    def __len__(self):
        # total number of videos
        return len(self.video_list)
'''
load dataset 
'''
def load_dataset(data_config):
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case
    if dataset_name == 'bat':
        #from .datasets import BIG
        train_transforms = []

        transforms = [
            Resize(data_config["img_size"]),
            RandomSequenceCrop(data_config["sequence_length"]),
            ImageToTensor(),
            ConcatSequence(),
        ]
        val_transforms = [
            Resize(data_config["img_size"]),
            FixedSequenceCrop(data_config["sequence_length"]),
            ImageToTensor(),
            ConcatSequence(),
        ]
        train_trans = Compose(transforms)
        test_trans = Compose(val_transforms)
        train_path = "./processed_video/video_train"
        val_path = "./processed_video/video_val"
        train = BIG(
            os.path.join(train_path),
            train_trans,
            data_config["add_noise"],
        )
        val = BIG(
            os.path.join(val_path),
            test_trans,
            data_config["add_noise"],
        )
    else:
        raise Exception("Dataset name not found.")
    return train, val
'''
load data
'''
'''
requires transposed_collate
'''
from torch.utils.data.dataloader import default_collate
def train_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    # assert transposed_batch.shape[0] >= 4
    # idx = torch.randint(4, transposed_batch.shape[0] + 1, size=(1,)).item()
    # return transposed_batch[:idx]
    return transposed_batch


def test_transposed_collate(batch):
    """
    Wrapper around the default collate function to return sequences of PyTorch
    tensors with sequence step as the first dimension and batch index as the
    second dimension.

    Args:
        batch (list): data examples
    """
    batch = filter(lambda img: img is not None, batch)
    collated_batch = default_collate(list(batch))
    transposed_batch = collated_batch.transpose_(0, 1)
    return transposed_batch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def load_data(data_config, batch_size, num_workers=4, pin_memory=True, sequence=True, distributed=False):
    """
    Wrapper around load_dataset. Gets the dataset, then places it in a DataLoader.

    Args:
        data_config (dict): data configuration dictionary
        batch_size (dict): run configuration dictionary
        num_workers (int): number of threads of multi-processed data Loading
        pin_memory (bool): whether or not to pin memory in cpu
        sequence (bool): whether data examples are sequences, in which case the
                         data loader returns transposed batches with the sequence
                         step as the first dimension and batch index as the
                         second dimension
    """
    '''
    if data_config["img_size"] == 64:
    embed_dim = 48
    transform_dim_mults = (1, 2, 2, 4)
    dim_mults = (1, 2, 4, 8)
    batch_size = 2
elif data_config["img_size"] in [128, 256]:
    embed_dim = 64
    transform_dim_mults = (1, 2, 3, 4)
    dim_mults = (1, 1, 2, 2, 4, 4)
    batch_size = 1
else:
    raise NotImplementedErrorc
    '''
    train, val = load_dataset(data_config)
    train_spl = DistributedSampler(train) if distributed else None
    val_spl = DistributedSampler(val, shuffle=False) if distributed else None

    if train is not None:
        train = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=False if distributed else True,
            collate_fn=train_transposed_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=train_spl
        )

    if val is not None:
        val = DataLoader(
            val,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_transposed_collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=val_spl
        )
    return train, val







parser = argparse.ArgumentParser(description="values from bash script")
parser.add_argument("--ndevice", type=int, required=True, help="number of cuda device")
args = parser.parse_args()
#set addr and port
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "63000"

def schedule_func(ep):
    return max(config.decay ** ep, config.minf)
'''
def ddp_setup(rank:int, world_size:int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "63000"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
'''


def main(rank, world_size):

    #ddp_setup( rank=rank, world_size=world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    train_data, val_data = load_data(
        config.data_config,
        config.batch_size,
        pin_memory=True,
        num_workers=4,
    )

    denoise_model = Unet(
        dim=config.embed_dim,
        context_dim_factor=config.context_dim_factor,
        channels=config.data_config["img_channel"],
        dim_mults=config.dim_mults,
    )

    context_model = CondNet(
        dim=int(config.context_dim_factor * config.embed_dim),
        channels=config.data_config["img_channel"],
        backbone=config.backbone,
        dim_mults=config.dim_mults,
    )

    transform_model = (
        HistoryNet(
            dim=int(config.transform_dim_factor * config.embed_dim),
            channels=config.data_config["img_channel"],
            #residual
            context_mode=config.transform_mode,
            dim_mults=config.transform_dim_mults,
            backbone=config.backbone
        )
        if config.transform_mode
        in ["residual"]
        else None
    )


    diffusion = GaussianDiffusion(
        denoise_fn=denoise_model,
        history_fn=context_model,
        transform_fn=transform_model,
        pred_mode=config.pred_mode,
        clip_noise=config.clip_noise,
        timesteps=config.iteration_step,
        loss_type=config.loss_type,
        aux_loss=False,
    ).to(rank)

    diffusion = DDP(diffusion, device_ids=[rank])

    trainer = Trainer(
        rank=rank,
        diffusion_model=diffusion,
        train_dl=train_data,
        val_dl=val_data,
        sample_num_of_frame=config.init_num_of_frame + 1,
        init_num_of_frame=config.init_num_of_frame,
        scheduler_function=schedule_func,
        scheduler_checkpoint_step=config.scheduler_checkpoint_step,
        ema_decay=config.ema_decay,
        train_lr=config.lr,
        train_num_steps=config.n_step,
        step_start_ema=config.ema_start_step,
        update_ema_every=config.ema_step,
        save_and_sample_every=config.log_checkpoint_step,
        results_folder=os.path.join(config.result_root, f"{config.model_name}/"),
        tensorboard_dir=os.path.join(config.tensorboard_root, f"{config.model_name}/"),
        model_name=config.model_name,
        val_num_of_batch=config.val_num_of_batch,
        optimizer=config.optimizer
    )
    '''
        if config.load_model:
        trainer.load(load_step=config.load_step)
    '''


    trainer.train()


if __name__ == "__main__":
    mp.spawn(main, args=(args.ndevice,), nprocs=args.ndevice, join=True)
    dist.barrier()
    dist.destroy_process_group()