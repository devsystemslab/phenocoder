from typing import Any
import spatialdata as sd
from phenocoder.model import CVAE, CondCVAE
from phenocoder.generator import DatasetGenerator, NucleiPatchGenerator, DatasetMerger
from phenocoder.train import train_model
from phenocoder.phenocode import encode_nuclei_patches
from phenocoder.spatial import run_spatial_feature_processing
from phenocoder.cluster import run_clustering


class Phenocoder:
    """
    A class for performing unsupervised morphometric phenotyping on spatial data.

    Attributes:
        sdata (SpatialData): The SpatialData object.
        model (str): The model to be used for phenotyping.
        sample_key (str): The key for the sample in the SpatialData object.

    Methods:
        add_sdata(sdata): Add a SpatialData object to the Phenocoder instance.
        validate_sdata(): Validate the SpatialData object.
        generate_dataset(): Generate a dataset for phenotyping.
        train_model(): Train a model for phenotyping.
        load_model(): Load a model for phenotyping.
        encode(): Encode the data using the trained model.
        cluster(): Cluster the encoded data.
        generate_spatialgraph_stats(): Generate statistics for spatial graphs.
        spatialgraph_embedding(): Generate a spatial graph embedding.
    """

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
        """
        Generate a image patch dataset for phenotyping.

        Returns:
        None
        """
        pass

    def train_model(self):
        """
        Train a Phenocoder model.

        Returns:
        None
        """
        pass

    def load_model(self):
        """
        Load a Phenocoder model.

        Returns:
        None
        """
        pass

    def encode(self):
        """
        Encode dataset.

        Returns:
        None
        """
        pass

    def cluster(self):
        """
        Cluster latent encoded nuclei.

        Returns:
        None
        """
        pass

    def generate_spatialgraph_stats(self):
        """
        Generate spatial graph statistics for each sample.

        Returns:
        None
        """
        pass

    def spatialgraph_embedding(self):
        """
        Generate spatial graph embedding from all samples.

        Returns:
        None
        """
        pass
