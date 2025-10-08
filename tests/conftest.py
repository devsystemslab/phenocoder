import spatialdata as sd
import pytest
from pathlib import Path


# tests for converting tiff to spatial data zarr and loading spatial data objects from phenocoder class
@pytest.fixture
def data_dir():
    """Reusable path to test data"""
    return Path(__file__).parent / 'data'


@pytest.fixture
def sample_data():
    return None
