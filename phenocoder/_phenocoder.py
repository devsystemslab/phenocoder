from typing import Any
import anndata as ad
import spatialdata as sd
from pathlib import Path

from phenocoder.model import CVAE, CondCVAE
from phenocoder.generator import DatasetGenerator, NucleiPatchGenerator, DatasetMerger
from phenocoder.train import train_model
from phenocoder.phenocode import encode_nuclei_patches
from phenocoder.spatial import run_spatial_feature_processing
from phenocoder._cluster import run_clustering


class Phenocoder:
    """
    A class for performing unsupervised morphometric spatial phenotyping on microscopy image data.

    Attributes:
        sdata (SpatialData): The SpatialData object.
        model (str): The type of model to be used for phenotyping.
        model_dir (str|Path): Path to model directory.
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

    def __init__(self, **kwargs: Any) -> None:
        self.sdata: sd.SpatialData = None
        self.adata: ad.AnnData = None
        self.model: CVAE | CondCVAE = None
        self.model_dir: str | Path = None
        self.sample_key: str = None
        self.data_dir: str | Path = None

    def add_sdata(self, sdata: sd.SpatialData) -> None:
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

    def load_model(self):
        """
        Load a Phenocoder model.

        Returns:
        None
        """
        pass

    def initialize_model(self):
        """
        Initialize a Phenocoder model.

        Returns:
        None
        """
        pass

    def train(self, **kwargs):
        """
        Train a Phenocoder model.

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

    def spatialgraph_stats(self):
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
