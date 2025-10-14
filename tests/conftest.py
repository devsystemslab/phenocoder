from spatialdata.models import Image2DModel, Image3DModel
import pytest
from pathlib import Path
import pandas as pd
import os
from skimage import io
import numpy as np


@pytest.fixture
def get_metadata(dir_images: str, regex: str = None) -> pd.DataFrame:
    """
    Get metadata from image filenames
    :param dir_images:
    :param regex: regex pattern to match metadata in filename, defaults to Yokogawa naming convention.
    :return:
    """
    images = os.listdir(dir_images)
    images = [image for image in images if '.tif' in image]
    if regex is None:
        regex = r'_(?P<well_id>[A-Z]\d{2})_T(?P<time_point>\d{4})F(?P<field_id>\d{3})L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})Z(?P<z_stack_id>\d{2})C(?P<channel_id>\d{2})\.tif$'
    df = pd.DataFrame({'file': images, 'dir_images': str(dir_images)})
    df = df.join(df['file'].str.extractall(regex).groupby(level=0).last())
    # remove rows that have nan in any column
    df = df[~df.isna().any(axis=1)]
    return df


@pytest.fixture(scope='module')
def example_imgs():
    """Factory that creates image arrays based on dimensions"""

    def _load_images(dims: str = '2d'):
        if dims == '2d':
            dir_images = Path(__file__).parent / 'data' / '2d' / 'imgs'
        elif dims == '3d':
            dir_images = Path(__file__).parent / 'data' / '3d' / 'imgs'
        else:
            raise ValueError('dims must be "2d" or "3d".')

        imgs = np.asarray(
            [
                io.imread(Path(dir_images, file))
                for file in sorted(os.listdir(dir_images))
            ]
        )
        return imgs

    return _load_images


@pytest.fixture
def sample_data_2d(example_imgs):
    """
    Generate example 2d data from tiff files.
    """
    imgs = Image2DModel()
    imgs = imgs.parse(example_imgs('2d'), dims=['c', 'y', 'x'])
    return imgs


@pytest.fixture
def sample_data_3d(example_imgs):
    """
    Generate example 3d data from tiff files.
    """
    imgs = Image3DModel()
    imgs = imgs.parse(example_imgs('3d'), dims=['c', 'z', 'y', 'x'])
    return imgs
