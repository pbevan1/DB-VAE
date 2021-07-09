from enum import Enum
# from datasets.h5celeba import H5CelebA
import os
# from datasets.h5imagenet import H5Imagenet
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.sampler import RandomSampler
# import h5py
import numpy as np
from typing import Optional, List, NamedTuple, Union
from logger import logger
import cv2
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split

from datasets.data_utils import CountryEnum, DataLoaderTuple, GenderEnum, SkinColorEnum
from datasets.celeb_a import CelebDataset
from datasets.imagenet import ImagenetDataset
from datasets.ppb import PPBDataset
from setup import *
from datasets.generic import *


# class FitzDataset(GenericImageDataset):
#     """Dataset for CelebA"""
#
#     def __init__(self, csv, **kwargs):
#         super().__init__(**kwargs)
#
#         self.csv = csv.reset_index(drop=True)
#         self.store: pd.DataFrame = self.init_store(csv.filepath)
#         self.classification_label = 1
#         self.transform = transform
#
#     def read_image(self, idx: int):
#         img: Image = Image.open(os.path.join(
#             self.path_to_images,
#             self.store.iloc[idx].image_id)
#         )
#
#         return img
#
#     def __len__(self):
#         return self.csv.shape[0]
#
#     def init_store(self, path_to_celeba_bbox_file):
#         """Sets self.store to be a dataframe of the bbox file."""
#         if not os.path.exists(csv.filepath):
#             logger.error(f"Path to bbox does not exist at {csv.filepath}!")
#             raise Exception
#
#         try:
#             store = pd.read_table(csv.filepath, delim_whitespace=True)
#             return store
#         except:
#             logger.error(
#                 f"Unable to read the bbox file located at {csv.filepath}"
#             )


# class FitzDataset(Dataset):
#     def __init__(self, csv, split, mode, transform=None):
#
#         self.csv = csv.reset_index(drop=True)
#         self.split = split
#         self.mode = mode
#         self.transform = transform
#         self.ARGS = ARGS
#
#     def __len__(self):
#         return self.csv.shape[0]
#
#     def __getitem__(self, index):
#         row = self.csv.iloc[index]
#
#         image = cv2.imread(row.filepath)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         if self.transform is not None:
#             res = self.transform(image=image)
#             image = res['image'].astype(np.float32)
#         else:
#             image = image.astype(np.float32)
#
#         image = image.transpose(2, 0, 1)
#
#         data = torch.tensor(image).float()
#
#         if self.mode == 'test':
#             return data, torch.tensor(self.csv.iloc[index].target).long()
#
#         # returning different data based on what test is being run
#         else:
#             return data, torch.tensor(self.csv.iloc[index].target).long(), torch.tensor(index).long(), torch.tensor(
#                 self.csv.iloc[index].fitzpatrick).long()

def get_df():
    df_fitz_34 = []
    df_fitz_56 = []
    df_train = pd.read_csv(f'{ARGS.csv_dir}/fitzpatrick17k.csv')
    # df_train['fitzpatrick'] = df_train['fitzpatrick'].astype(np.float32)
    df_train = df_train.loc[
              (df_train.three_partition_label != 'non-neoplastic') & (df_train.qc != '3 Wrongly labelled'), :]
    df_train = df_train.loc[df_train['url'].str.contains('http', na=False), :]
    df_train = df_train.loc[df_train['fitzpatrick'] != -1, :]
    df_train = pd.get_dummies(df_train, columns=['three_partition_label'], drop_first=True)
    df_train.rename(columns={'three_partition_label_malignant': 'target'}, inplace=True)
    df_train['image_name'] = 0
    for i, url in enumerate(df_train.url):
        if 'atlasderm' in url:
            df_train.loc[df_train['url'] == url, 'image_name'] = f'atlas{i}.jpg'
        else:
            df_train.loc[df_train['url'] == url, 'image_name'] = url.split('/', -1)[-1]
    # adding column with path to file
    df_train['filepath'] = df_train['image_name'].apply(lambda x: f'{ARGS.image_dir}/fitzpatrick17k_128/{x}')
    # mapping fitzpatrick image to class index
    fp2idx = {d: idx for idx, d in enumerate(sorted(df_train['fitzpatrick'].unique()))}
    df_train['fitzpatrick'] = df_train['fitzpatrick'].map(fp2idx)
    #splitting to test and train
    # df_fitz_34 = df_train.loc[(df_train['fitzpatrick'] == 2) | (df_train['fitzpatrick'] == 3), :]
    # df_fitz_56 = df_train.loc[(df_train['fitzpatrick'] == 4) | (df_train['fitzpatrick'] == 5), :]
    # df_train = df_train.loc[(df_train['fitzpatrick'] == 0) | (df_train['fitzpatrick'] == 1), :]
    # df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42, shuffle=True)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, shuffle=True)
    # df_train['fold'] = 0
    # df_train.iloc[781:1562,-1] = 1
    # df_train.iloc[1562:2343, -1] = 2
    # df_train.iloc[2343:3134, -1] = 3
    # df_train.iloc[3134:3915, -1] = 4
    print(len(df_train))
    print(len(df_val))
    print(len(df_test))

    mel_idx = 1

    return df_train, df_val, df_test, mel_idx

# def get_transforms():
#     #augmentations for all training data
#     transforms = A.Compose([
#         A.Resize(128, 128),
#         A.Normalize()#mean, std)
#     ])
#     return transforms

def split_dataset(dataset, train_size: float, random_seed, max_images: Optional[int] = None):
    """Splits a dataset into a certain (maximum) size."""
    # Shuffle indices of the dataset
    idxs: np.array = np.arange(len(dataset))
    np.random.seed(random_seed)
    np.random.shuffle(idxs)

    # Sample sub-selection
    sampled_idxs: np.array = idxs if not max_images else idxs[:min(max_images, len(idxs))]

    # Split dataset
    split: int = int(np.floor(train_size * len(sampled_idxs)))
    train_idxs = sampled_idxs[:split]
    valid_idxs = sampled_idxs[split:]

    # Subsample dataset with given validation indices
    train_data = Subset(dataset, train_idxs)
    valid_data = Subset(dataset, valid_idxs)

    return train_data, valid_data

def concat_datasets(dataset_a, dataset_b, proportion_a: Optional[float] = None):
    """Concatenates two different datasets."""
    if proportion_a:
        proportion_b = 1 - proportion_a
        # Calculate amount of dataset
        nr_dataset_a: int = int(np.floor(proportion_a * len(dataset_a)))
        nr_dataset_b: int = int(np.floor(proportion_b * len(dataset_b)))

    else:
        nr_dataset_a = len(dataset_a)
        nr_dataset_b = len(dataset_b)

    # Subsample the datasets
    sampled_dataset_a = Subset(dataset_a, np.arange(nr_dataset_a))
    sampled_dataset_b = Subset(dataset_b, np.arange(nr_dataset_b))

    return ConcatDataset([sampled_dataset_a, sampled_dataset_b])

def make_h5_datasets(path_to_h5_train: str, **kwargs):
    """Create the H5 dataset based on an existing H5 dataset on the machine."""
    if not os.path.exists(path_to_h5_train):
        logger.error(
            f"Unable to find h5 file for training file at {path_to_h5_train}",
            next_step="Will stop training / evaluation",
            tip="Double check your path_to_h5_train in your config, and check if you downloaded the appropriate .h5 file"
        )
        raise Exception

    # Read h5
    with h5py.File(path_to_h5_train, mode='r') as h5_file:
        labels = h5_file['labels'][()].flatten()
        files = h5_file['images']

        idxs_faces = np.where(labels > 0)[0].flatten()
        idxs_nonfaces = np.where(labels == 0)[0].flatten()

        files_faces: h5py.Dataset = files[idxs_faces.tolist()]
        files_nonfaces: h5py.Dataset = files[idxs_nonfaces.tolist()]

        dataset_nonfaces: H5Imagenet = H5Imagenet(files_nonfaces, path_to_images='', **kwargs)
        dataset_faces: H5CelebA = H5CelebA(files_faces, path_to_images='', **kwargs)

        return dataset_faces, dataset_nonfaces


def make_train_and_valid_loaders(
    batch_size: int,
    max_images: int,
    path_to_imagenet_images: str,
    path_to_celeba_images: str,
    num_workers: int,
    shuffle: bool = True,
    train_size: float = 0.8,
    proportion_faces: float = 0.5,
    enable_debias: bool = True,
    sample_bias_with_replacement: bool = True,
    use_h5: bool = True,
    random_seed = '',
    **kwargs
):
    """Create two dataloaders, one for training data and one for validation data."""
    nr_images: Optional[int] = max_images if max_images >= 0 else None

    # Create the datasets
    if use_h5:
        logger.info("Creating the celeb and imagenet dataset from the h5 file!")
        celeb_dataset, imagenet_dataset = make_h5_datasets(**kwargs)
    else:
        logger.info("Creating the Imagenet and CelebDataset")
        imagenet_dataset = ImagenetDataset(path_to_images=path_to_imagenet_images, **kwargs)
        celeb_dataset = CelebDataset(path_to_images=path_to_celeba_images, **kwargs)

    # Split both datasets into training and validation
    celeb_train, celeb_valid = split_dataset(celeb_dataset, train_size, random_seed, nr_images)
    imagenet_train, imagenet_valid = split_dataset(imagenet_dataset, train_size, random_seed, nr_images)
    logger.info(f"Sizes of dataset are:\n"
                f"Celeb-train: {len(celeb_train)}\n"
                f"Celeb-valid: {len(celeb_valid)}\n"
                f"Imagenet-train: {len(imagenet_train)}\n"
                f"Imagenet-valid: {len(imagenet_valid)}\n")

    # Nonfaces loaders
    train_nonfaces_loader: DataLoader = DataLoader(imagenet_train, batch_size=int(batch_size / 2), shuffle=shuffle, num_workers=num_workers)
    valid_nonfaces_loader: DataLoader = DataLoader(imagenet_valid, batch_size=int(batch_size / 2), shuffle=False, num_workers=num_workers)

    # Init some weights
    init_weights = torch.rand(len(celeb_train)).tolist()

    # Define samplers: random for non-debias, weighed for debiasing
    random_train_sampler = RandomSampler(celeb_train)
    weights_sampler_train = WeightedRandomSampler(init_weights, len(celeb_train), replacement=sample_bias_with_replacement)

    train_sampler = weights_sampler_train if enable_debias else random_train_sampler

    # Define the face loaders
    train_faces_loader: DataLoader = DataLoader(celeb_train, sampler=train_sampler, batch_size=int(batch_size / 2), num_workers=num_workers)
    valid_faces_loader: DataLoader = DataLoader(celeb_valid, batch_size=int(batch_size / 2), shuffle=shuffle, num_workers=num_workers)

    # Make tuple of the data-loaders
    train_loaders: DataLoaderTuple = DataLoaderTuple(train_faces_loader, train_nonfaces_loader)
    valid_loaders: DataLoaderTuple = DataLoaderTuple(valid_faces_loader, valid_nonfaces_loader)

    return train_loaders, valid_loaders

class EvalDatasetType(Enum):
    """Defines a enumerator the makes it possible to double check dataset types."""
    PBB_ONLY = 'ppb'
    IMAGENET_ONLY = 'imagenet'
    H5_IMAGENET_ONLY = 'h5_imagenet'

def make_eval_loader(
    num_workers: int,
    path_to_eval_face_images: str,
    csv,
    # path_to_eval_nonface_images: str,
    # filter_exclude_gender: List[str] = [],
    # filter_exclude_country: List[str] = [],
    filter_skin_color = 5,
    max_images: int = -1,
    proportion_faces: float = 0.5,
    dataset_type: str = EvalDatasetType.PBB_ONLY.value,
    **kwargs
):
    """Creates an evaluaion data loader."""

    dataset = GenericImageDataset(csv=csv, filter_skin_color=filter_skin_color)
    # if dataset_type == EvalDatasetType.PBB_ONLY.value:
    #     logger.info('Evaluating on PPB')
    #
    #     dataset = PPBDataset(
    #         path_to_images=path_to_eval_face_images,
    #         path_to_metadata=csv,
    #         filter_excl_country=filter_exclude_country,
    # #         filter_excl_gender=filter_exclude_gender,
    # filter_excl_skin_color=filter_skin_color,
    #         get_sub_images=True,
    #         **kwargs
    #     )
    # elif dataset_type == EvalDatasetType.IMAGENET_ONLY.value:
    #     logger.info('Evaluating on Imagenet')
    #
    #     dataset = ImagenetDataset(
    #         path_to_images=path_to_eval_nonface_images,
    #         get_sub_images=True,
    #         **kwargs
    #     )
    # else:
    #     logger.info('Evaluating on Imagenet H5')
    #
    #     _, h5_nonfaces = make_h5_datasets(**kwargs)
    #
    #     dataset = H5Imagenet(
    #         path_to_images='',
    #         h5_dataset=h5_nonfaces.store,
    #         get_sub_images=True,
    #         **kwargs
    #     )

    # If max images, sample only a smaller amount
    nr_images: Optional[int] = max_images if max_images >= 0 else None
    if nr_images:
        logger.info(f"Evaluation on a sub-set of {nr_images} nr images")
        dataset = subsample_dataset(dataset, nr_images, random=True)

    # Concat and wrap with loader
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    return data_loader

def subsample_dataset(dataset: Dataset, nr_subsamples: int, random=False):
    """Create a specified number of subsamples from a dataset."""
    idxs = np.arange(nr_subsamples)

    if random:
        idxs = np.random.choice(np.arange(len(dataset)), nr_subsamples)

    return Subset(dataset, idxs)


def sample_dataset(dataset: Dataset, nr_samples: int):
    """Create a tensor stack of a specified number from a given dataset."""
    max_nr_items: int = min(nr_samples, len(dataset))
    idxs = np.random.permutation(np.arange(len(dataset)))[:max_nr_items]

    return torch.stack([dataset[idx][0] for idx in idxs])

# def sample_idxs_from_loaders(idxs, data_loaders, label):
#     """Returns data id's from a DataLoaderTupler."""
#     if label == 1:
#         dataset = data_loaders.faces.dataset
#     else:
#         dataset = data_loaders.nonfaces.dataset
#
#     return torch.stack([dataset[idx.item()][0] for idx in idxs])

def sample_idxs_from_loader(idxs, data_loader, label):
    """Returns data id's from a dataloader."""
    if label == 1:
        dataset = data_loader.dataset
    else:
        dataset = data_loader.dataset

    return torch.stack([dataset[idx.item()][0] for idx in idxs])

def make_hist_loader(dataset, batch_size):
    """Retrun a data loader that return histograms from the data."""
    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    return DataLoader(dataset, batch_sampler=batch_sampler)
