from enum import Enum
import os
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler, BatchSampler, SequentialSampler
from torch.utils.data.dataset import Subset
import numpy as np
from typing import Optional
from logger import logger
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from setup import *
from datasets.generic import *

def get_df():

    if args.fitzpatrick17k:
        df_train = pd.read_csv(f'{args.csv_dir}/fitzpatrick17k.csv')
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
        df_train['filepath'] = df_train['image_name'].apply(lambda x: f'{args.image_dir}/fitzpatrick17k_128/{x}')
        # mapping fitzpatrick image to class index
        fp2idx = {d: idx for idx, d in enumerate(sorted(df_train['fitzpatrick'].unique()))}
        df_train['fitzpatrick'] = df_train['fitzpatrick'].map(fp2idx)

    else:
        # loading train csv
        df_train = pd.read_csv(os.path.join(args.csv_dir, 'isic_train_20-19-18-17.csv'))
        #removing Mclass images from training data to prevent leakage
        df_train = df_train.loc[df_train.mclassd !=1, :]

        # removing 2018 & 2019 from training data
        df_train = df_train.loc[(df_train.year != 2018) & (df_train.year != 2019), :]
        df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)

        # setting cv folds for 2017 data
        df_train.loc[(df_train.year != 2020), 'fold'] = df_train['tfrecord'] % 5
        tfrecord2fold = {
            2: 0, 4: 0, 5: 0,
            1: 1, 10: 1, 13: 1,
            0: 2, 9: 2, 12: 2,
            3: 3, 8: 3, 11: 3,
            6: 4, 7: 4, 14: 4,
        }
        # setting cv folds for 2020 data
        df_train.loc[(df_train.year == 2020), 'fold'] = df_train['tfrecord'].map(tfrecord2fold)
        #putting image filepath into column
        df_train.loc[(df_train.year == 2020), 'filepath'] = df_train['image_name'].apply(
            lambda x: os.path.join(f'{args.image_dir}/isic_20_train_{args.image_size}/{x}.jpg'))
        df_train.loc[(df_train.year != 2020), 'filepath'] = df_train['image_name'].apply(
            lambda x: os.path.join(f'{args.image_dir}/isic_19_train_{args.image_size}', f'{x}.jpg'))

        # # mapping image size to index as proxy for instrument
        # size2idx = {d: idx for idx, d in enumerate(sorted(df_train['size'].unique()))}
        # df_train['instrument'] = df_train['size'].map(size2idx)

    df_train, df_test = train_test_split(df_train, test_size=0.2, random_state=42, shuffle=True)
    df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=42, shuffle=True)
    # df_train['fold'] = 0
    # df_train.iloc[781:1562,-1] = 1
    # df_train.iloc[1562:2343, -1] = 2
    # df_train.iloc[2343:3134, -1] = 3
    # df_train.iloc[3134:3915, -1] = 4
    print(f'training set size: {len(df_train)}')
    print(f'validation set size: {len(df_val)}')
    print(f'test set size: {len(df_test)}')

    mel_idx = 1

    return df_train, df_val, df_test, mel_idx

class EvalDatasetType(Enum):
    """Defines a enumerator the makes it possible to double check dataset types."""
    PBB_ONLY = 'ppb'
    IMAGENET_ONLY = 'imagenet'
    H5_IMAGENET_ONLY = 'h5_imagenet'

def make_eval_loader(
    num_workers: int,
    csv,
    filter_skin_color = 5,
    **kwargs
):
    """Creates an evaluaion data loader."""
    dataset = GenericImageDataset(csv=csv, filter_skin_color=filter_skin_color)
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
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

    return DataLoader(dataset, batch_sampler=batch_sampler)
