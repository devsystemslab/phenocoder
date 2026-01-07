from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import keras
import pandas as pd
import spatialdata as sd
import yaml

from phenocoder.generator import DatasetLoader, PatchGenerator
from phenocoder.model import CVAE, CondCVAE
from phenocoder.spatial import SpatialGraphAnalyzer
from phenocoder.utils import write_training_plots_to_tensorboard


class Phenocoder:
    """
    A class for performing unsupervised morphometric spatial phenotyping on microscopy image data.

    The Phenocoder class provides a complete workflow for spatial phenotyping using variational
    autoencoders. It supports both conditional and non-conditional models, and integrates with
    SpatialData objects for handling spatial omics data.

    Attributes:
        sdata (SpatialData | None): The SpatialData object containing spatial omics data.
        adata (AnnData | None): AnnData object (deprecated, data should be in sdata.tables).
        model (CVAE | CondCVAE | None): The variational autoencoder model for phenotyping.
        model_dir (str | Path | None): Path to model directory for saving/loading models.
        model_oh_enc: One-hot encoder for conditional model inputs.
        model_config (dict | None): Configuration parameters for the model.
        sample_key (str | None): The key for identifying samples in the SpatialData object.
        data_dir (str | Path | None): Directory path for dataset storage.
        data_generator_train: Training data generator for model training.
        data_generator_val: Validation data generator for model training.
        df_conditions (DataFrame | None): DataFrame containing condition information for conditional models.

    Example:
        >>> phenocoder = Phenocoder()
        >>> phenocoder.add_sdata(sdata)
        >>> phenocoder.data_dir = "path/to/data"
        >>> phenocoder.sample_key = "sample_id"
        >>> phenocoder.generate_dataset(dir_input="input", dir_segmented="segmented")
        >>> phenocoder.initialize_model(n_latent_dim=64, n_dense_dim=256, conditional=False)
        >>> phenocoder.train(n_epochs=100)
        >>> encoded_data = phenocoder.encode()
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize a new Phenocoder instance.

        Args:
            **kwargs: Optional keyword arguments to set initial attributes. Supported keys:
                - sdata: SpatialData instance
                - adata: AnnData instance
                - model: preconstructed model (CVAE or CondCVAE)
                - model_dir: path to model directory
                - model_oh_enc: one-hot encoder for conditional models
                - model_config: model configuration dict or path to config
                - sample_key: key used to identify samples in sdata tables
                - spatial_key: spatial key name/index
                - table_key: table key in sdata.tables
                - data_dir: base directory for datasets
                - datasets: list of dataset identifiers
                - data_generator_train: training data generator
                - data_generator_val: validation data generator
                - df_conditions: DataFrame of condition labels
                - image_key: key of images in sdata.images

        Any attributes not provided in kwargs will be initialized to None.
        """
        # Core data containers
        self.sdata: sd.SpatialData | None = kwargs.get('sdata', None)
        self.adata: ad.AnnData | None = kwargs.get('adata', None)

        # Model-related
        self.model: CVAE | CondCVAE | None = kwargs.get('model', None)
        self.model_dir: str | Path | None = kwargs.get('model_dir', None)
        self.model_oh_enc = kwargs.get('model_oh_enc', None)

        self.model_config = kwargs.get('model_config', None)

        # Keys for sdata access
        self.sample_key: str | None = kwargs.get('sample_key', None)
        self.spatial_key: str | None = kwargs.get('spatial_key', None)
        self.table_key: str | None = kwargs.get('table_key', None)
        self.image_key: str | None = kwargs.get('image_key', None)

        # Dataset and generator related
        self.data_dir: str | Path | None = kwargs.get('data_dir', None)
        self.datasets: list[str] | None = kwargs.get('datasets', None)
        self.data_generator_train = kwargs.get('data_generator_train', None)
        self.data_generator_val = kwargs.get('data_generator_val', None)
        self.df_conditions = kwargs.get('df_conditions', None)

        # Other optional attributes that may be set later
        self.model_name: str | None = kwargs.get('model_name', None)
        self.dir_tensorboard: Path | None = kwargs.get('dir_tensorboard', None)
        self.data_loader = kwargs.get('data_loader', None)

    def add_sdata(self, sdata: sd.SpatialData) -> None:
        """
        Add a SpatialData object to the Phenocoder instance.

        Args:
            sdata (SpatialData): The SpatialData object containing spatial omics data
                and microscopy images to be processed.

        Returns:
            None

        Example:
            >>> phenocoder = Phenocoder()
            >>> phenocoder.add_sdata(sdata)
        """
        self.sdata = sdata

    def generate_dataset(
        self, dataset, dir_dataset, spatial_key_index=None, scale=True
    ) -> None:
        """
        Generate an image patch dataset for phenotyping from input microscopy images.

        Creates image patches from input images and segmentation masks, with options for
        sampling strategies and multi-channel processing. The generated dataset is used
        for training the variational autoencoder model.

        Args:
            dataset (str): Name/identifier for the dataset being generated.
            dir_dataset (str | Path): Directory path for storing the generated dataset.
            spatial_key_index (str | None, optional): Spatial key index to use, integer relating to z-index in image array. If None, uses
                the instance's spatial_key attribute. Defaults to None.
            scale (bool, optional): Whether to scale the image patches. Defaults to True.

        Returns:
            None

        Example:
            >>> phenocoder.generate_dataset(
            ...     dataset="experiment_001",
            ...     dir_dataset="/path/to/datasets"
            ... )
        """
        if spatial_key_index is None:
            spatial_key_index = self.spatial_key
        self.data_dir = dir_dataset
        if self.datasets is None:
            self.datasets = [dataset]
        else:
            self.datasets = self.datasets.append(dataset)
        self.patch_generator = PatchGenerator(
            sdata=self.sdata,
            sample_key=self.sample_key,
            table_key=self.table_key,
            image_key=self.image_key,
            spatial_key=spatial_key_index,
            scale=scale,
        )
        self.patch_generator.generate_dataset(dataset, dir_output=self.data_dir)

    def initialize_model(
        self,
        n_latent_dim: int,
        n_dense_dim: int,
        conditions: list[str],
        dropout: float = 0.25,
        batch_size: int = 64,
        n_workers: int = 1,
        input_shape: tuple[int, ...] = (128, 128, 4),
        conv_layers: tuple[int, ...] = (8, 16, 32, 64, 128),
        beta: float = 1,
    ) -> None:
        """
        Initialize a CVAE or conditional CVAE model with specified parameters.

        Sets up the model architecture, data generators, and saves configuration files.
        Creates model directory structure and prepares the model for training.

        Args:
            n_latent_dim (int): Dimensionality of the latent space.
            n_dense_dim (int): Dimensionality of dense layers in the model.
            conditional (bool): Whether to use conditional VAE (requires condition labels).
            dropout (float, optional): Dropout rate for regularization. Defaults to 0.25.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            n_workers (int, optional): Number of workers for data loading. Defaults to 1.
            input_shape (tuple[int, int, int], optional): Input image shape (height, width, channels).
                Defaults to (128, 128, 4).
            conv_layers (tuple[int, ...], optional): Number of filters in each convolutional layer.
                Defaults to (8, 16, 32, 64, 128).
            beta (float, optional): Beta parameter for beta-VAE (controls KL divergence weight).
                Defaults to 1.

        Returns:
            None

        Raises:
            ValueError: If data_dir is not specified.

        Example:
            >>> phenocoder.initialize_model(
            ...     n_latent_dim=64,
            ...     n_dense_dim=256,
            ...     conditional=False,
            ...     dropout=0.25,
            ...     beta=0.01
            ... )
        """
        self.model_name = f'latent_{n_latent_dim}_dense_{n_dense_dim}_dropout_{dropout}_beta_{beta}_{pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")}'
        if conditions:
            self.model_name = f'cond_{self.model_name}'
        if self.data_dir is None:
            raise ValueError('.data_dir must be specified')
        if self.datasets is None:
            raise ValueError('.datasets must be specified')
        self.model_dir = Path(self.data_dir, 'models', self.model_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.data_loader = DatasetLoader(
            datasets=self.datasets,
            dir_datasets=self.data_dir,
            sample_key=self.sample_key,
        )
        self.data_loader.load_datasets()
        self.data_loader.set_train_val_split()

        self.model_config = {
            'n_latent_dim': n_latent_dim,
            'n_dense_dim': n_dense_dim,
            'input_shape': list(input_shape),
            'conv_layers': list(conv_layers),
            'dropout': dropout,
            'dir_dataset': self.data_dir,
            'batch_size': batch_size,
            'n_workers': n_workers,
            'beta': beta,
        }

        if conditions:
            (
                self.data_generator_train,
                self.data_generator_val,
                self.model_oh_enc,
            ) = self.data_loader.get_generators(conditions=conditions)
            self.model_config.update(
                {
                    'conditional': True,
                    'conditions_dim': self.data_generator_train.conditions.shape[-1],
                }
            )
        else:
            self.data_generator_train, self.data_generator_val = (
                self.data_loader.get_generators()
            )
            self.model_config.update({'conditional': False})

        with open(Path(self.model_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.model_config, file)

        # set up model
        if self.model_config['conditional']:
            self.model = CondCVAE(
                n_classes=self.data_generator_train.conditions.shape[-1],
                input_shape=input_shape,
                latent_dim=n_latent_dim,
                dense_dim=n_dense_dim,
                conv_layers=conv_layers,
                dropout=dropout,
                beta=beta,
            )
            joblib.dump(self.model_oh_enc, Path(self.model_dir, 'oh_encoder.joblib'))
        else:
            self.model = CVAE(
                input_shape=input_shape,
                latent_dim=n_latent_dim,
                dense_dim=n_dense_dim,
                conv_layers=conv_layers,
                dropout=dropout,
                beta=beta,
            )

    def load_model(self) -> None:
        """
        Load a pre-trained phenocoder model from disk.

        Reconstructs the model architecture from saved configuration and loads
        the trained weights. Also loads the one-hot encoder for conditional models.

        Returns:
            None

        Note:
            Requires model_config to be set to the path of the configuration file.

        Example:
            >>> phenocoder.model_config = "path/to/config.yaml"
            >>> phenocoder.load_model()
        """
        with open(self.model_config, 'r') as file:
            self.model_config = yaml.load(file, Loader=yaml.FullLoader)

        if self.model_config['conditional']:
            self.model = CondCVAE(
                input_shape=tuple(self.model_config['input_shape']),
                latent_dim=self.model_config['n_latent_dim'],
                dense_dim=self.model_config['n_dense_dim'],
                conv_layers=tuple(self.model_config['conv_layers']),
                n_classes=self.model_config['conditions_dim'],
            )
            self.oh_enc = joblib.load(Path(self.model_directory, 'oh_encoder.joblib'))
        else:
            self.model = CVAE(
                input_shape=tuple(self.model_config['input_shape']),
                latent_dim=self.model_config['n_latent_dim'],
                dense_dim=self.model_config['n_dense_dim'],
                conv_layers=tuple(self.model_config['conv_layers']),
            )
        self.model.compile()
        self.model.load_weights(Path(self.model_directory, 'model.weights.h5'))

    def summarize_model(self) -> None:
        """
        Display model architecture and configuration summary.

        Prints the model configuration parameters and architecture summaries
        for both encoder and decoder components.

        Returns:
            None

        Example:
            >>> phenocoder.summarize_model()
        """
        print('Model summary:')
        for key, value in self.model_config.items():
            print(f'{key}: {value}')
        self.model.encoder.summary()
        self.model.decoder.summary()

    def train(
        self,
        n_epochs: int = 100,
        learning_rate: float = 0.001,
        min_learning_rate: float = 0.0001,
        factor_learning_rate: float = 0.2,
        learning_rate_patience: int = 3,
        early_stopping_patience: int = 5,
        plot: bool = True,
        n_preview: int = 300,
        plot_frac: float = 0.001,
    ) -> None:
        """
        Train the initialized model with specified hyperparameters and callbacks.

        Performs model training with early stopping, learning rate reduction, and
        TensorBoard logging. Optionally generates visualization plots.

        Args:
            n_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            learning_rate (float, optional): Initial learning rate for optimizer. Defaults to 0.001.
            min_learning_rate (float, optional): Minimum learning rate for learning rate scheduler.
                Defaults to 0.0001.
            learning_rate_patience (int, optional): Number of epochs without improvement before
                reducing learning rate. Defaults to 3.
            early_stopping_patience (int, optional): Number of epochs without improvement before
                stopping training. Defaults to 5.
            plot (bool, optional): Whether to generate visualization plots after training.
                Defaults to True.
            n_preview (int, optional): Number of samples to use for reconstruction plots.
                Defaults to 300.
            plot_frac (float, optional): Fraction of data to use for latent space visualization.
                Defaults to 0.001.

        Returns:
            None

        Example:
            >>> phenocoder.train(
            ...     n_epochs=200,
            ...     learning_rate=0.0005,
            ...     early_stopping_patience=10,
            ...     plot=True
            ... )
        """
        self.dir_tensorboard = Path(self.data_dir, 'tensorboard_logs', self.model_name)

        if not self.dir_tensorboard.exists():
            self.dir_tensorboard.mkdir(parents=True, exist_ok=True)

        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        self.model.fit(
            self.data_generator_train,
            validation_data=self.data_generator_val,
            epochs=n_epochs,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_patience,
                    restore_best_weights=True,
                ),
                keras.callbacks.TensorBoard(log_dir=self.dir_tensorboard),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=factor_learning_rate,
                    patience=learning_rate_patience,
                    min_lr=min_learning_rate,
                ),
            ],
        )

        self.model.build(self.model_config['input_shape'])
        self.model.save_weights(Path(self.model_dir, 'model.weights.h5'))

        if plot:
            write_training_plots_to_tensorboard(
                self.model,
                self.data_generator_train,
                self.model_oh_enc,
                self.dir_tensorboard,
                n_preview=n_preview,
                plot_frac=plot_frac,
            )

    def encode(
        self, batch_size: int = 64, scale=True, spatial_key_index=None
    ) -> ad.AnnData:
        """
        Encode nuclei patches into latent space representations using the trained model.

        Processes all samples in the SpatialData object, extracts nuclei patches,
        and encodes them into the learned latent space. Results are aggregated
        by nucleus label and returned as an AnnData object.

        Args:
            batch_size (int, optional): Batch size for encoding predictions. Defaults to 64.
            filter_encodable_conditions (bool, optional): Whether to filter out conditions
                that cannot be encoded by the model (for conditional models). Defaults to False.

        Returns:
            ad.AnnData: AnnData object containing encoded latent representations with
                nuclei metadata in .obs and latent dimensions in .X.

        Note:
            This method contains dataset-specific code that should be generalized.

        Example:
            >>> encoded_data = phenocoder.encode(batch_size=128)
            >>> print(encoded_data.shape)  # (n_nuclei, n_latent_dim)
        """
        adata = []
        samples = self.sdata.tables[self.table_key].obs[self.sample_key].unique()
        if self.patch_generator is None:
            if spatial_key_index is None:
                spatial_key_index = self.spatial_key
            self.patch_generator = PatchGenerator(
                sdata=self.sdata,
                sample_key=self.sample_key,
                table_key=self.table_key,
                image_key=self.image_key,
                spatial_key=spatial_key_index,
                scale=scale,
            )
        for sample in samples:
            patches, df_patches = self.patch_generator.get_patches(sample)
            if df_patches.empty:
                continue
            if self.model_config['conditional']:
                conditions = self.model_oh_enc.transform(
                    df_patches[self.model_oh_enc.feature_names_in_]
                )
                _, _, z = self.model.encoder.predict(
                    [patches, conditions], batch_size=batch_size
                )
            else:
                _, _, z = self.model.encoder.predict(patches, batch_size=batch_size)

            #  TODO: Should store results in sdata.tables instead of creating standalone AnnData -> sdata.parse table or add as obsm to existing table
            adata.append(
                ad.AnnData(
                    X=z,
                    obs=self.sdata.tables[self.table_key][df_patches.index].obs,
                    var=pd.DataFrame(
                        index=[f'phc_latent_{i + 1}' for i in range(z.shape[-1])]
                    ),
                )
            )
        self.sdata.tables['phenocoder'] = ad.concat(adata)

    def spatialgraph_stats(self) -> None:
        """
        Generate statistics for spatial neighborhood graphs of each sample.

        Computes spatial graph-based statistics such as neighborhood composition,
        spatial clustering coefficients, and other graph-based metrics for each
        sample in the dataset.

        Returns:
            None

        Note:
            This method is not yet implemented. Implementation should compute
            spatial graph statistics and store results in sdata.tables.

        Todo:
            - Implement spatial graph construction
            - Add graph-based statistical measures
            - Store results in sdata.tables with appropriate keys
        """
        pass
        # TODO: add analyze methods than runs SpatialGraphAnalyzer over all samples

    def spatialgraph_embedding(self) -> None:
        """
        Generate spatial graph embeddings from all samples.

        Creates low-dimensional embeddings that capture spatial relationships
        between nuclei across all samples in the dataset. This can be used
        for sample-level comparisons and spatial pattern analysis.

        Returns:
            None

        Note:
            This method is not yet implemented. Implementation should use
            the run_spatial_feature_processing function and store results
            in sdata.tables.

        Todo:
            - Implement spatial graph embedding algorithm
            - Add parameters for embedding hyperparameters
            - Store embedding results in sdata.tables
        """
        pass
