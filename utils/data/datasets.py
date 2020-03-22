import os
import pandas as pd
import numpy as np
import pathlib
import nibabel as nib

from sklearn.model_selection import train_test_split
from .load import load_itk


class MRIDataset:

    __DATA_MODES__ = ['train', 'val', 'test']
    __IMAGE_TYPES__ = ['flair', 't1']
    __FILE_FORMATS__ = ['.mgh', '.mhd']

    def __init__(self, dataset_root, mode, image_type, with_mask, metadata, transform=None):
        super().__init__()

        if mode not in self.__DATA_MODES__:
            raise NameError(f"{mode} is not correct; correct modes: {self.__DATA_MODES__}")
        elif image_type not in self.__IMAGE_TYPES__:
            raise NameError(f"{image_type} is not correct; correct image types: {self.__IMAGE_TYPES__}")

        self.dataset_root = dataset_root
        self.transform = transform
        self.mode = mode
        self.with_mask = with_mask
        self.metadata = metadata
        self.image_type = image_type

    def __getitem__(self, index):
        row = self.metadata.iloc[index]

        if self.image_type == 'flair':
            # if
            data = load_itk(os.path.join(self.dataset_root, row['path_flair_image']))
            if self.with_mask:
                data *= np.load(os.path.join(self.dataset_root, row['path_mask_flair_brain']))
        else:
            data = load_itk(os.path.join(self.dataset_root, row['path_t1_image']))
            if self.with_mask:
                data *= np.load(os.path.join(self.dataset_root, row['path_mask_t1_brain']))

        if self.transform:
            data = self.transform(data)

        return data, row['class']

    def __len__(self):
        return len(self.metadata)

    @staticmethod
    def get_metadata(mode, dataset_root, val_size=0.15, random_seed=None):
        metadata = pd.read_csv(os.path.join(dataset_root, 'metadata.csv'))

        if mode not in ['train', 'val', 'test']:
            raise NameError(f"{mode} is not correct; correct modes: {['train', 'val', 'test']}")

        if mode != 'test':
            train_df, val_df = train_test_split(metadata, test_size=val_size, random_state=random_seed,
                                                stratify=metadata['class'])
            val_path = os.path.join(dataset_root, f'val.csv')
            train_path = os.path.join(dataset_root, f'train.csv')

            val_df.to_csv(val_path, index=False)
            train_df.to_csv(train_path, index=False)

            return {'train': train_df.reset_index(drop=True), 'val': val_df.reset_index(drop=True)}
        else:
            return {'test': metadata.reset_index(drop=True)}


# metadata = MRIDataset.get_metadata('train', './train_val_data', random_seed=1)['train']
# dataset = MRIDataset(mode='train', dataset_root='./train_val_data', image_type='flair', with_mask=False, metadata=metadata)
#
# for batch, label in dataset:
#     print(label)
#     break
# print(pathlib.Path(os.path.join('train_val_data', 'norm 1 rs/AX FLAIR norma 1 rs.mgh.gz')))

# brain = nib.load(os.path.join('train_val_data', 'norm 1 rs/AX FLAIR norma 1 rs.mgh'))
# print(brain)