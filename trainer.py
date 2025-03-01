import torch
# from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from datasets.tode_dataset import get_dataset
import yaml
import os
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = True
# model = TODE_HEAD(channels_in =1).cuda()

model = Unet(
        dim = 24,
        init_dim = None,
        out_dim = 1,
        dim_mults = (1, 2, 4, 8),
        channels = 5,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 16,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 1000,    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    objective = 'pred_x0',
    )

cfg_filename = './configs/train_cgsyn+ood_val_cgreal.yaml'
with open(cfg_filename, 'r') as cfg_file:
    cfg_params = yaml.load(cfg_file, Loader = yaml.FullLoader)
dataset_params = cfg_params.get('dataset', {'data_dir': 'data'})
train_dataset = get_dataset(dataset_params,split='train')
val_dataset = get_dataset(dataset_params={"test": {"type": "cleargrasp-syn", "data_dir": cfg_params['dataset']['test']['data_dir'], "image_size": (320, 240),\
    "use_augmentation": False, "depth_min": 0.0, "depth_max": 10.0,  "depth_norm": 1.0}}, split = 'test')
test_dataset = get_dataset(dataset_params,split='test') # cg real

test_dataset.image_paths = test_dataset.image_paths[:16]
# tode_encoder = diffusion.tode_encoder
tode_encoder = None

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 8e-5,
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 4,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = False,              # whether to calculate fid during training
    train_set= train_dataset,
    val_set= test_dataset,
    test_set= test_dataset,
    save_and_sample_every = 2000,
    num_samples = 16,
    num_fid_samples = 1,
    results_folder = './train_cg_spade_simple',
    tode_encoder = tode_encoder,
    log_dir = '../experiments/train_cg_spade_simple'
    
)
trainer.load(1)
trainer.train()


sampled_seq = diffusion.sample(batch_size = 4)
sampled_seq.shape # (4, 32, 128)