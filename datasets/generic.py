
from torchvision.transforms import ToPILImage
from datasets.data_utils import DatasetOutput, default_transform
from typing import Callable
from PIL import Image
from .data_utils import slide_windows_over_img, DatasetOutput
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from setup import *
import cv2
import numpy as np

class GenericImageDataset(Dataset):
    """Generic dataset which defines all basic operations for the images."""
    def __init__(
        self,
        csv,
        filter_skin_color = None,
        path_to_images: str = ARGS.image_dir,
        get_sub_images: bool = False,
        sub_images_nr_windows: int = 10,
        sub_images_batch_size: int = 10,
        sub_images_min_size: int = 30,
        sub_images_max_size: int = 128,
        sub_images_stride: float = 0.2,
        transform: Callable = default_transform, ###
        **kwargs
    ):
        self.filter_skin_color = filter_skin_color
        self.csv = csv.reset_index(drop=True)
        if filter_skin_color!=None:
            self.csv = self.csv.loc[self.csv.fitzpatrick == self.filter_skin_color, :]

        # self.store: pd.DataFrame = self.init_store(self.csv.filepath)
        self.path_to_images = path_to_images
        self.transform = transform

        # Sub images properties
        self.get_sub_images = get_sub_images
        self.sub_images_min_size = sub_images_min_size
        self.sub_images_max_size = sub_images_max_size
        self.sub_images_nr_windows = sub_images_nr_windows
        self.sub_images_batch_size = sub_images_batch_size
        self.sub_images_stride = sub_images_stride

        self.pil_transformer = ToPILImage()

        # Create store for data
        self.store = None

    def __getitem__(self, idx: int):
        row = self.csv.iloc[idx]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        data = torch.tensor(image).float()

        sub_images: torch.Tensor = torch.tensor(0)

        # Extract sub images if applicable
        if self.get_sub_images:
            sub_images = slide_windows_over_img(
                data,
                min_win_size=self.sub_images_min_size,
                max_win_size=self.sub_images_max_size,
                nr_windows=self.sub_images_nr_windows,
                stride=self.sub_images_stride
            )

        return DatasetOutput(
            image=data,
            label=torch.tensor(self.csv.iloc[idx].target).long(),
            idx=torch.tensor(idx).long(),
            sub_images=sub_images
        )


    def read_image(self, idx: int):
        """Interface, returns an PIL Image using the index."""
        pass

    def _apply_filters_to_metadata(self):
        """Allows filters to filter out countries, skin-colors and genders."""
        result = self.csv

        # if len(self.filter_excl_country):
        #     result = result.query('country not in @self.filter_excl_country')
        #
        # if len(self.filter_excl_gender):
        #     result = result.query('gender not in @self.filter_excl_gender')

        # if len(self.filter_excl_skin_color):
        #     try:
        result = result.loc[result.fitzpatrick!=self.filter_skin_color, :]
                # result = result.query('bi_fitz not in @self.filter_excl_skin_color')
            # except:
            #     logger.error("bi_fitz can't be found in the metadata datadframe",
            #                  next_step="The skin color wont be applied",
            #                  tip="Rename the bi.fitz column to be bi_fitz in the metadata csv")

        return result

    def __len__(self):
        return self.csv.shape[0]

# class FitzDataset(GenericImageDataset):
#     """Dataset for CelebA"""
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#         self.store: pd.DataFrame = self.init_store(path_to_celeba_bbox_file)
#         self.classification_label = 1
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
#         return len(self.store)

    # def init_store(self, path_to_celeba_bbox_file):
    #     """Sets self.store to be a dataframe of the bbox file."""
    #     if not os.path.exists(path_to_celeba_bbox_file):
    #         logger.error(f"Path to bbox does not exist at {path_to_celeba_bbox_file}!")
    #         raise Exception
    #
    #     try:
    #         store = pd.read_table(path_to_celeba_bbox_file, delim_whitespace=True)
    #         return store
    #     except:
    #         logger.error(
    #             f"Unable to read the bbox file located at {path_to_celeba_bbox_file}"
    #         )