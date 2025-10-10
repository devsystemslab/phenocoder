from spatialdata.models import Image2DModel, Image3DModel
import pytest
from pathlib import Path
import pandas as pd
import os
from skimage import io
import numpy as np

# TODO:
# - setup example data for 2d and 3d imaging data
# - generate individual spatial data objects for each sample and generate merged spatial data from it


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


@pytest.fixture
def data_dir(dims: int):
    """Reusable path to test data"""
    if dims == 2:
        return Path(__file__).parent / 'data' / '2d'
    elif dims == 3:
        return Path(__file__).parent / 'data' / '3d'
    else:
        raise ValueError('dims must be 2 or 3.')


@pytest.fixture
def sample_data(dims: int):
    """
    Generate sample data from tiff files.
    """
    if dims in (2, 3):
        df_images = get_metadata(data_dir(dims))
    else:
        raise ValueError('dims must be 2 or 3.')
    # read images
    df_images['image'] = df_images.apply(
        lambda row: io.imread(row['dir_images'] + '/' + row['file']), axis=1
    )
    imgs = np.asarray(df_images['image'])
    if dims == 2:
        imgs_sd = Image2DModel()
        imgs_sd = imgs_sd.parse(imgs)
    if dims == 3:
        imgs_sd = Image3DModel()
        imgs_sd = imgs_sd.parse(imgs)
    return imgs_sd
