from typing import Any
import spatialdata as sd
from phenocoder.model import CVAE, CondCVAE, VAE
from phenocoder.generator import DatasetGenerator, NucleiPatchGenerator, DatasetMerger
from phenocoder.train import train_model
from phenocoder.phenocode import encode_nuclei_patches
from phenocoder.spatial import run_spatial_feature_processing
from phenocoder.cluster import run_clustering


class Phenocoder:
    def __init__(self, *kwargs: Any) -> None:
        self.sdata: sd.SpatialData = None
        self.model: str = None
        self.sample_key: str = None

    def add_sdata(self, sdata) -> None:
        """
        Add a SpatialData object to the Phenocoder instance.

        Parameters:
        sdata (SpatialData): The SpatialData object to be added.

        Returns:
        None
        """
        self.sdata = sdata

    def validate_sdata(self):
        """
        Validate the SpatialData object.

        Returns:
        None
        """
        if self.sample_key is None:
            raise ValueError('Sample key is not set.')
        if self.sample_key not in self.sdata.obs.columns:
            raise ValueError(f'Sample key "{self.sample_key}" not found in sdata.obs')
        else:
            pass

    def generate_dataset(self):
        pass

    def train_model(self):
        pass

    def load_model(self):
        pass

    def encode(self):
        pass

    def cluster(self):
        pass

    def generate_spatialgraph_stats(self):
        pass

    def spatialgraph_embedding(self):
        pass
