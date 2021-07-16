from typing import NamedTuple
from types import SimpleNamespace
import torch
import datetime
import os
import argparse
from logger import logger
from typing import Optional
from dataclasses import dataclass, field

# Default device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--DEBUG', action='store_true')
parser.add_argument('--DP', action='store_true')
parser.add_argument('--load-model', action='store_true')
parser.add_argument('--fitzpatrick17k', action='store_true')
parser.add_argument('--image-dir', type=str, default='./data/images/')
parser.add_argument('--csv-dir', type=str, default='./data/csv')
parser.add_argument('--run-folder', type=str, default='outputs')
parser.add_argument('--test-no', type=int, default=0)
parser.add_argument('--image-size', type=int, help='size of images', default=256)
parser.add_argument('--batch-size', type=int, help='size of batch', default=64)
parser.add_argument('--num-workers', type=int, help='number of workers', default=16)
parser.add_argument('--epochs', type=int, help='max number of epochs')
# parser.add_argument('--save-epoch', type=int, help='epoch to save on', default=100)
parser.add_argument('--z-dim', type=int, help='dimensionality of latent space')
parser.add_argument('--alpha', type=float, help='importance of debiasing')
parser.add_argument('--num-bins', type=int, help='importance of debiasing')
parser.add_argument('--max-images', type=int, help='total size of database')
parser.add_argument('--eval-freq', type=int, help='total size of database')
parser.add_argument('--debias-type', type=str, help='type of debiasing used', default='none')
parser.add_argument("--path-to-model", type=str, help='Path to stored model')
parser.add_argument("--debug-mode", type=bool, help='Debug mode')
parser.add_argument("--use-h5", type=bool, help='Use h5')
# parser.add_argument("--folder_name", type=str, help='folder_name_to_save in')
parser.add_argument("--eval-name", type=str, help='eval name')
parser.add_argument('--stride', type=float, help='importance of debiasing')
parser.add_argument('--eval-dataset', type=str, help='Name of eval dataset [ppb/h5_imagenet/h5]')
parser.add_argument('--save-sub-images', type=bool, help='Save images')
parser.add_argument('--model-name', type=str, help='name of the model to evaluate')
parser.add_argument('--hist-size', type=bool, help='Number of histogram')
parser.add_argument('--run-mode', type=str, help='Type of main.py run')
parser.add_argument('--perturbation-range', type=float, nargs='+', help='list of 7 values to perturb by', default=[])
parser.add_argument('--interp1', type=int, help='first image to interpolate', default=0)
parser.add_argument('--interp2', type=int, help='second image to interpolate', default=0)
parser.add_argument('--var-to-perturb', type=int, help='latent variable to perturb', default=5)
parser.add_argument('-f', type=str, help='Path to kernel json')


class EmptyObject():
    def __getattribute__(self, idx):
        return None

ARGS, unknown = parser.parse_known_args()
if len(unknown) > 0:
    logger.warning(f'There are some unknown args: {unknown}')

num_workers = 16 if ARGS.num_workers is None else ARGS.num_workers

def create_folder_name(foldername):
    if foldername == "":
        return foldername

    suffix = ''
    count = 0
    while True:
        if not os.path.isdir(f"results/{foldername}{suffix}"):
            foldername = f'{foldername}{suffix}'
            return foldername
        else:
            count += 1
            suffix = f'_{count}'

@dataclass
class Config:
    # Running main for train, eval or both
    run_mode: str = 'both' if ARGS.run_mode is None else ARGS.run_mode
    # # Folder name of the run
    # run_folder: str = '' if ARGS.test_no is None else str(ARGS.test_no)
    # # Path to CelebA images
    # path_to_fp17k_images: str = '../data/images/fitzpatrick17k_128/'
    # # Path to ISIC 202 images
    # path_to_isic20_images: str = './data/images/isic_20_train_128'
    # # Path to evaluation images (Faces)
    # path_to_eval_face_images: str = '/data/images/fitzpatrick17k_128'

    load_model: bool = ARGS.load_model

    # # Path to stored model
    path_to_model: Optional[str] = ARGS.path_to_model or ARGS.test_no
    # Type of debiasing used
    debias_type: str = ARGS.debias_type or 'none'
    # name of the model to evaluate
    model_name: str = ARGS.model_name or 'model.pth'
    # Random seed for reproducability
    random_seed: int = 0
    # Device to use
    device: torch.device = DEVICE
    test_no = ARGS.test_no
    # eval file name
    eval_name: str = ARGS.eval_name or "evaluation_results.txt"
    # Batch size
    batch_size: int = ARGS.batch_size or 64
    # Number of bins
    num_bins: int = ARGS.num_bins or 10
    # Epochs
    epochs: int = ARGS.epochs or 50
    # Z dimension
    z_dim: int = ARGS.z_dim or 200
    # Alpha value
    alpha: float = ARGS.alpha or 0.01
    #image indexes to interpolate between
    interp1 = ARGS.interp1
    interp2 = ARGS.interp2
    # stride used for evaluation windows
    stride: float = ARGS.stride or 0.2
    # Dataset size
    max_images: int = ARGS.max_images or -1
    # Eval frequence
    eval_freq: int = ARGS.eval_freq or 5
    # Number workers
    num_workers: int = 16 if ARGS.num_workers is None else ARGS.num_workers
    # Image size
    image_size: int = ARGS.image_size
    # Number windows evaluation
    sub_images_nr_windows: int = 15
    # Evaluation window minimum
    eval_min_size: int = 30
    # Evaluation window maximum
    eval_max_size: int = ARGS.image_size
    # Uses h5 instead of the imagenet files
    use_h5: bool = True if ARGS.use_h5 is None else ARGS.use_h5
    # Debug mode prints several statistics
    debug_mode: bool = False if ARGS.debug_mode is None else ARGS.debug_mode
    # Dataset for evaluation
    eval_dataset: str = ARGS.eval_dataset or 'ppb'
    # Images to save
    save_sub_images: bool = False if ARGS.save_sub_images is None else ARGS.save_sub_images
    # Hist size
    hist_size: int = 1000 if ARGS.hist_size is None else ARGS.hist_size
    # Batch size for how many sub images to batch
    sub_images_batch_size: int = 10
    # Minimum size for sub images
    sub_images_min_size: int = 30
    # Maximum size for sub images
    sub_images_max_size: int = 256
    # Stride of sub images
    sub_images_stride: float = 0.2
    perturbation_range = ARGS.perturbation_range
    var_to_perturb = ARGS.var_to_perturb

    def __post_init__(self, printing=False):
        # self.run_folder = create_run_folder(self.run_folder)
        if printing:
            logger.save(f"Saving new run files to {ARGS.test_no}")


def init_trainining_results(config: Config):
    # Write run-folder name
    if not os.path.exists("results"):
        os.makedirs("results")

    config.__post_init__(printing=True)
    os.makedirs(f'results/plots/{config.test_no}/best_and_worst', exist_ok=True)
    os.makedirs(f'results/plots/{config.test_no}/bias_probs', exist_ok=True)
    os.makedirs(f'results/plots/{config.test_no}/reconstructions/perturbations', exist_ok=True)
    os.makedirs(f'results/logs/{config.test_no}', exist_ok=True)
    os.makedirs(f'results/weights/{config.test_no}', exist_ok=True)

    with open(f"results/logs/{config.test_no}/flags.txt", "w") as write_file:
      write_file.write(f"z_dim = {config.z_dim}\n")
      write_file.write(f"alpha = {config.alpha}\n")
      write_file.write(f"epochs = {config.epochs}\n")
      write_file.write(f"batch size = {config.batch_size}\n")
      write_file.write(f"eval frequency = {config.eval_freq}\n")
      write_file.write(f"max images = {config.max_images}\n")
      write_file.write(f"debiasing type = {config.debias_type}\n")


    if config.debug_mode:
        os.makedirs(f"results/{config.test_no}/debug")

    with open(f"results/logs/{config.test_no}/training_results.csv", "a+") as write_file:
        write_file.write("epoch,train_loss,valid_loss,train_acc,valid_acc\n")

    with open(f"results/logs/{config.test_no}/flags.txt", "w") as wf:
        wf.write(f"debias_type: {config.debias_type}\n")
        wf.write(f"alpha: {config.alpha}\n")
        wf.write(f"z_dim: {config.z_dim}\n")
        wf.write(f"batch_size: {config.batch_size}\n")
        wf.write(f"max_images: {config.max_images}\n")
        wf.write(f"use_h5: {config.use_h5}\n")

default_config = Config()
