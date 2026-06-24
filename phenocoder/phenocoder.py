from pathlib import Path
from typing import Any

import anndata as ad
import bbknn
import joblib
import keras
import numpy as np
import pandas as pd
import scanpy as sc
import spatialdata as sd
import yaml

from phenocoder.generator import DatasetLoader, PatchGenerator
from phenocoder.model import CVAE, CondCVAE
from phenocoder.spatial import SpatialGraphAnalyzer, spatial_message_passing
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
        self.patch_generator = kwargs.get('patch_generator', None)

        # Other optional attributes that may be set later
        self.model_name: str | None = kwargs.get('model_name', None)
        self.dir_tensorboard: Path | None = kwargs.get('dir_tensorboard', None)
        self.data_loader = kwargs.get('data_loader', None)

    def __repr__(self) -> str:
        """
        Return a formatted string representation of the Phenocoder instance.

        Returns:
            str: A summary of the Phenocoder object's structure and configuration.
        """
        lines = ['Phenocoder object with:']

        # SpatialData info
        if self.sdata is not None:
            lines.append(f'sdata: {self.sdata}')
        else:
            lines.append('sdata: None')
        # Adata info
        if self.adata is not None:
            lines.append(f'adata: {self.adata}')
        else:
            lines.append('adata: None')
        # Model info
        if self.model is not None:
            model_type = type(self.model).__name__
            lines.append(f'model: {model_type}')
        else:
            lines.append('model: None')

        # Configuration keys
        config_info = []
        if self.sample_key is not None:
            config_info.append(f'sample_key={repr(self.sample_key)}')
        if self.table_key is not None:
            config_info.append(f'table_key={repr(self.table_key)}')
        if self.image_key is not None:
            config_info.append(f'image_key={repr(self.image_key)}')
        if self.spatial_key is not None:
            config_info.append(f'spatial_key={repr(self.spatial_key)}')

        if config_info:
            lines.append(f'config: {", ".join(config_info)}')

        # Dataset info
        if self.datasets is not None and len(self.datasets) > 0:
            lines.append(f'datasets: {len(self.datasets)} dataset(s)')

        return '\n'.join(lines)

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
        self,
        dataset,
        dir_dataset,
        patch_size=(128, 128),
        spatial_key_index=None,
        scale=True,
        metadata_keys=None,
        scale_percentile=1,
        scale_per_sample=True,
    ) -> None:
        """
        Generate an image patch dataset for phenotyping from input microscopy images.

        Creates image patches from input images and segmentation masks, with options for
        sampling strategies and multi-channel processing. The generated dataset is used
        for training the variational autoencoder model.

        Args:
            dataset (str): Name/identifier for the dataset being generated.
            dir_dataset (str | Path): Directory path for storing the generated dataset.
            spatial_key_index (str | None, optional): Spatial key index to use, integer relating to z-index in image array.
            If None, uses the instance's spatial_key attribute. Defaults to None.
            scale (bool, optional): Whether to scale the image patches. Defaults to True.
            metadata_keys (list[str] | None, optional): Additional columns from the table's
                ``.obs`` to carry into ``patches.csv`` so they can be used as conditioning
                variables (e.g. a ``sample``/donor column). Defaults to None.
            scale_percentile (float, optional): Percentile (0-100) for the per-slice low/high
                used in normalization; the high uses ``100 - scale_percentile``. Defaults to 1
                (1/99 stretch).
            scale_per_sample (bool, optional): If True (default), normalize each sample to its
                own intensity range (per sample+channel). If False, use one global range per
                channel across all samples (original behaviour). NOTE: pass the SAME value to
                ``encode`` so training and inference scale identically.

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
            patch_size=patch_size,
            metadata_keys=metadata_keys,
            scale_percentile=scale_percentile,
            scale_per_sample=scale_per_sample,
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
        beta: float = 0.01,
    ) -> None:
        """
        Initialize a CVAE or conditional CVAE model with specified parameters.

        Sets up the model architecture, data generators, and saves configuration files.
        Creates model directory structure and prepares the model for training.

        Args:
            n_latent_dim (int): Dimensionality of the latent space.
            n_dense_dim (int): Dimensionality of dense layers in the model.
            conditions (list[str]): List of column names in the data to use as conditions
                for conditional VAE. If empty list or None, uses non-conditional CVAE.
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
            ValueError: If datasets is not specified.

        Example:
            >>> # Non-conditional model
            >>> phenocoder.initialize_model(
            ...     n_latent_dim=64,
            ...     n_dense_dim=256,
            ...     conditions=[],
            ...     dropout=0.25,
            ...     beta=0.01
            ... )
            >>> # Conditional model
            >>> phenocoder.initialize_model(
            ...     n_latent_dim=64,
            ...     n_dense_dim=256,
            ...     conditions=['dataset', 'z'],
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
            ) = self.data_loader.get_generators(
                conditions=conditions,
                dim=input_shape[:2],
                n_channels=input_shape[-1],
            )
            self.model_config.update(
                {
                    'conditional': True,
                    'conditions_dim': self.data_generator_train.conditions.shape[-1],
                }
            )
        else:
            self.data_generator_train, self.data_generator_val = (
                self.data_loader.get_generators(
                    dim=input_shape[:2], n_channels=input_shape[-1]
                )
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
        config_path = Path(self.model_config)
        self.model_dir = config_path.parent

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
            self.model_oh_enc = joblib.load(Path(self.model_dir, 'oh_encoder.joblib'))
        else:
            self.model = CVAE(
                input_shape=tuple(self.model_config['input_shape']),
                latent_dim=self.model_config['n_latent_dim'],
                dense_dim=self.model_config['n_dense_dim'],
                conv_layers=tuple(self.model_config['conv_layers']),
            )
        self.model.compile()
        self.model.build(tuple(self.model_config['input_shape']))
        self.model.load_weights(Path(self.model_dir, 'model.weights.h5'))

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
        self,
        batch_size: int = 64,
        scale: bool = True,
        spatial_key_index: str = None,
        scale_percentile: int = 1,
        scale_per_sample: bool = True,
        spatial_message_passing_radius: int = None,
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
            scale_percentile (float, optional): Percentile (0-100) for per-slice low/high, used
                only when the patch_generator is (re)built here. MUST match the value used in
                ``generate_dataset`` for this model's dataset. Defaults to 1.
            scale_per_sample (bool, optional): Per-sample vs global normalization, used only when
                the patch_generator is (re)built here. MUST match ``generate_dataset`` for this
                model's dataset, else inference scales differently than training. Defaults to True.

        Returns:
            ad.AnnData: AnnData object containing encoded latent representations with
                nuclei metadata in .obs and latent dimensions in .X.

        Note:
            This method contains dataset-specific code that should be generalized.

        Example:
            >>> encoded_data = phenocoder.encode(batch_size=128)
            >>> print(encoded_data.shape)  # (n_nuclei, n_latent_dim)
        """
        adatas = []
        samples = self.sdata.tables[self.table_key].obs[self.sample_key].unique()
        if self.patch_generator is None:
            if spatial_key_index is None:
                spatial_key_index = self.spatial_key
            # condition columns the encoder was trained on that must be carried
            # from obs into the patches dataframe; 'z' comes from spatial coords and
            # 'dataset' is reconstructed from the saved patches.csv below.
            metadata_keys = None
            if self.model_config['conditional']:
                metadata_keys = [
                    c
                    for c in self.model_oh_enc.feature_names_in_
                    if c not in ('z', 'dataset')
                ]
            self.patch_generator = PatchGenerator(
                sdata=self.sdata,
                sample_key=self.sample_key,
                table_key=self.table_key,
                image_key=self.image_key,
                spatial_key=spatial_key_index,
                scale=scale,
                patch_size=tuple(self.model_config['input_shape'][:2]),
                metadata_keys=metadata_keys,
                scale_percentile=scale_percentile,
                scale_per_sample=scale_per_sample,
            )
            self.patch_generator.init_patches()
            patches_dfs = [
                pd.read_csv(Path(self.model_config['dir_dataset'], ds, 'patches.csv'))
                for ds in self.datasets
            ]
            patches_meta = pd.concat(patches_dfs, ignore_index=True)
            sample_to_dataset = (
                patches_meta.groupby(self.sample_key)['dataset'].first().to_dict()
            )
            self.patch_generator.patches['dataset'] = self.patch_generator.patches[
                self.sample_key
            ].map(sample_to_dataset)
            if scale:
                stats_dfs = [
                    pd.read_csv(Path(self.model_config['dir_dataset'], ds, 'stats.csv'))
                    for ds in self.datasets
                ]
                self.patch_generator.df_stats = pd.concat(stats_dfs, ignore_index=True)
                self.patch_generator.get_scaling_percentiles()
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
            adata = ad.AnnData(
                X=z,
                obs=self.sdata.tables[self.table_key][df_patches.index].obs,
                obsm=self.sdata.tables[self.table_key][df_patches.index].obsm,
                uns=self.sdata.tables[self.table_key][df_patches.index].uns,
                var=pd.DataFrame(
                    index=[f'phc_latent_{i + 1}' for i in range(z.shape[-1])]
                ),
            )
            if spatial_message_passing_radius is not None:
                adata = spatial_message_passing(
                    adata, radius=spatial_message_passing_radius
                )
            adatas.append(adata)
        self.sdata.tables['phenocoder'] = ad.concat(adatas)

    def spatialgraph_stats(
        self,
        cluster_key: str = 'leiden',
        spatial_key: str = 'spatial',
        radii: tuple[int, ...] = (25, 50),
        table_key: str | None = None,
        use_subunits: bool = False,
        dim_subunit: tuple[int, int, int] = (500, 500, 100),
        min_obs_per_subunit: int = 100,
        max_obs_per_subunit: int | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Generate statistics for spatial neighborhood graphs of each sample or subunit.

        Computes spatial graph-based statistics such as neighborhood composition,
        spatial clustering coefficients, and other graph-based metrics for each
        sample (or spatial subunit within samples) using the SpatialGraphAnalyzer.

        Args:
            cluster_key (str, optional): Key in adata.obs containing cluster labels.
                Defaults to 'leiden'.
            spatial_key (str, optional): Key in adata.obsm containing spatial coordinates.
                Defaults to 'spatial'.
            radii (tuple[int, ...], optional): Tuple of radii to use for spatial neighbor
                calculations. Defaults to (25, 50).
            table_key (str | None, optional): Key in sdata.tables to analyze. If None,
                uses self.table_key. Defaults to None.
            use_subunits (bool, optional): Whether to partition samples into spatial
                subunits and compute statistics per subunit instead of per sample.
                Defaults to False.
            dim_subunit (tuple[int, int, int], optional): Dimensions (x, y, z) of each
                spatial subunit in micrometers. Only used if use_subunits=True.
                Defaults to (500, 500, 100).
            min_obs_per_subunit (int, optional): Minimum number of observations required
                per subunit. Subunits with fewer observations are filtered out.
                Only used if use_subunits=True. Defaults to 100.
            max_obs_per_subunit (int | None, optional): Maximum number of observations
                per subunit. Subunits with more observations are randomly subsampled.
                If None, no subsampling is performed. Only used if use_subunits=True.
                Defaults to None.
            verbose (bool, optional): Whether to print progress information during
                subunit partitioning. Defaults to False.

        Returns:
            None

        Raises:
            ValueError: If table_key is not specified and self.table_key is None.
            ValueError: If cluster_key is not found in the table's obs.

        Note:
            Results are stored in self.adata as an AnnData object with one row per
            sample (if use_subunits=False) or per subunit (if use_subunits=True)
            containing all computed spatial statistics.

        Example:
            >>> # Sample-level analysis
            >>> phenocoder.spatialgraph_stats(
            ...     cluster_key='leiden',
            ...     radii=(25, 50, 100)
            ... )
            >>> # Subunit-level analysis
            >>> phenocoder.spatialgraph_stats(
            ...     cluster_key='leiden',
            ...     radii=(25, 50),
            ...     use_subunits=True,
            ...     dim_subunit=(500, 500, 100),
            ...     min_obs_per_subunit=100
            ... )
        """
        if table_key is None:
            table_key = self.table_key

        if table_key is None:
            raise ValueError('table_key must be specified or set in self.table_key')

        if table_key not in self.sdata.tables:
            raise ValueError(f'Table {table_key} not found in sdata.tables')

        adata = self.sdata.tables[table_key]

        if cluster_key not in adata.obs.columns:
            raise ValueError(
                f'cluster_key "{cluster_key}" not found in adata.obs. '
                f'Available columns: {list(adata.obs.columns)}'
            )

        if spatial_key not in adata.obsm.keys():
            raise ValueError(
                f'spatial_key "{spatial_key}" not found in adata.obsm. '
                f'Available keys: {list(adata.obsm.keys())}'
            )

        if self.sample_key is None:
            raise ValueError('sample_key must be specified')

        if self.sample_key not in adata.obs.columns:
            raise ValueError(
                f'sample_key "{self.sample_key}" not found in adata.obs. '
                f'Available columns: {list(adata.obs.columns)}'
            )

        # Get unique samples
        samples = adata.obs[self.sample_key].unique()

        results = []

        if use_subunits:
            # Import SpatialSubunitSampler
            from phenocoder.sampling import SpatialSubunitSampler

            for sample in samples:
                # Subset data for this sample
                adata_sample = adata[adata.obs[self.sample_key] == sample].copy()

                # Check if sample has enough cells
                if len(adata_sample) == 0:
                    print(f'Warning: Sample {sample} has no cells, skipping.')
                    continue

                # Partition into subunits
                sampler = SpatialSubunitSampler(
                    adata=adata_sample,
                    dim_subunit=dim_subunit,
                    min_obs=min_obs_per_subunit,
                    spatial_key=spatial_key,
                    verbose=verbose,
                )
                sampler.partition()
                sampler.filter()

                if max_obs_per_subunit is not None:
                    sampler.sample(max_obs=max_obs_per_subunit)

                if len(sampler.subunits) == 0:
                    print(
                        f'Warning: Sample {sample} has no valid subunits after filtering, skipping.'
                    )
                    continue

                if verbose:
                    print(
                        f'Sample {sample}: {len(sampler.subunits)} subunits after filtering'
                    )

                # Process each subunit
                for subunit_key, subunit_data in sampler.subunits.items():
                    # Get observations for this subunit
                    subunit_obs_indices = subunit_data['obs_indices']
                    adata_subunit = adata_sample[subunit_obs_indices].copy()

                    # Check if subunit has enough cells and clusters
                    if len(adata_subunit) == 0:
                        continue

                    n_clusters = len(adata_subunit.obs[cluster_key].unique())
                    if n_clusters <= 1:
                        if verbose:
                            print(
                                f'Warning: Sample {sample}, subunit {subunit_key} has '
                                f'{n_clusters} cluster(s), skipping (need >1 for spatial stats).'
                            )
                        continue

                    # Run spatial graph analysis for this subunit
                    try:
                        subunit_index = f'{sample}_subunit_{subunit_data["id"]}'
                        sga = SpatialGraphAnalyzer(
                            adata=adata_subunit,
                            cluster_key=cluster_key,
                            spatial_key=spatial_key,
                            radii=radii,
                            index=subunit_index,
                        )
                        sga.run()
                        df_subunit = sga.to_df()

                        # Add metadata columns
                        df_subunit[self.sample_key] = sample
                        df_subunit['subunit_id'] = subunit_data['id']
                        df_subunit['subunit_key'] = str(subunit_key)
                        df_subunit['subunit_n_obs'] = len(subunit_obs_indices)

                        results.append(df_subunit)
                    except Exception as e:
                        if verbose:
                            print(
                                f'Warning: Failed to compute stats for sample {sample}, '
                                f'subunit {subunit_key}: {str(e)}'
                            )
                        continue

            if len(results) == 0:
                print('Warning: No spatial statistics computed for any subunits.')
                return

            df_stats = pd.concat(results, axis=0, join='outer')
            df_stats = df_stats.fillna(0)

            # Set index but keep sample_key and subunit metadata as columns
            df_stats = df_stats.reset_index(drop=True)

            # Separate metadata from stats
            metadata_cols = [
                self.sample_key,
                'subunit_id',
                'subunit_key',
                'subunit_n_obs',
            ]
            stat_cols = [col for col in df_stats.columns if col not in metadata_cols]

            self.adata = ad.AnnData(
                X=df_stats[stat_cols].values,
                obs=df_stats[metadata_cols].reset_index(drop=True),
                var=pd.DataFrame(index=stat_cols),
            )

        else:
            # sample-level analysis
            for sample in samples:
                # Subset data for this sample
                adata_sample = adata[adata.obs[self.sample_key] == sample].copy()

                # Check if sample has enough cells and clusters
                if len(adata_sample) == 0:
                    print(f'Warning: Sample {sample} has no cells, skipping.')
                    continue

                n_clusters = len(adata_sample.obs[cluster_key].unique())
                if n_clusters <= 1:
                    print(
                        f'Warning: Sample {sample} has {n_clusters} cluster(s), '
                        f'skipping (need >1 for spatial stats).'
                    )
                    continue

                # Run spatial graph analysis
                try:
                    sga = SpatialGraphAnalyzer(
                        adata=adata_sample,
                        cluster_key=cluster_key,
                        spatial_key=spatial_key,
                        radii=radii,
                        index=str(sample),
                    )
                    sga.run()
                    df_sample = sga.to_df()
                    results.append(df_sample)
                except Exception as e:
                    print(
                        f'Warning: Failed to compute stats for sample {sample}: {str(e)}'
                    )
                    continue

            if len(results) == 0:
                raise ValueError(
                    'Warning: No spatial statistics computed for any samples.'
                )
                return

            df_stats = pd.concat(results, axis=0, join='outer')
            df_stats = df_stats.fillna(0)
            df_stats.index.name = self.sample_key
            df_stats = df_stats.loc[:, ~df_stats.columns.duplicated()]

            self.adata = ad.AnnData(
                X=df_stats.values,
                obs=pd.DataFrame(index=df_stats.index),
                var=pd.DataFrame(index=df_stats.columns),
            )

    def spatialgraph_embedding(
        self,
        n_dim: int,
        scale: bool = True,
        variable_features: bool = False,
        batch_correction: bool = False,
        batch_key: str = None,
        confounder_key: str = None,
        n_neighbors: int = 15,
        umap: bool = True,
        obs_keys: str | list[str] | None = None,
    ) -> None:
        """
        Generate spatial graph embeddings from all samples.

        Creates low-dimensional embeddings that capture spatial relationships
        between nuclei across all samples in the dataset. This can be used
        for sample-level comparisons and spatial pattern analysis.

        Args:
            n_dim (int): Number of principal components to compute.
            scale (bool, optional): Whether to scale the data. Defaults to True.
            variable_features (bool, optional): Whether to select highly variable features.
                Defaults to False.
            batch_correction (bool, optional): Whether to apply batch correction using BBKNN.
                Defaults to False.
            batch_key (str | None, optional): Column name in adata.obs or sdata.tables to use
                for batch correction. Required if batch_correction=True. Defaults to None.
            confounder_key (str | list[str] | None, optional): Column name(s) to use as
                confounders in batch correction. Defaults to None.
            n_neighbors (int, optional): Number of neighbors for neighbor graph construction.
                Used in both bbknn.bbknn and sc.pp.neighbors. Defaults to 15.
            umap (bool, optional): Whether to compute UMAP embedding. Defaults to True.
            obs_keys (str | list[str] | None, optional): Column name(s) in
                ``sdata.tables[table_key].obs`` to carry into ``self.adata.obs`` as
                per-sample metadata (e.g. condition/treatment groups), so the UMAP
                can be colored by them. Each value is taken per sample via
                ``groupby(sample_key).first()`` and must be constant within a sample.
                Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If self.adata is None or not set.
            ValueError: If batch_correction=True but batch_key is None.
            ValueError: If batch_key is not found in adata.obs or sdata.tables.

        Note:
            Results are stored in self.adata with:
            - .layers['raw']: Raw data before scaling
            - .obsm['X_pca']: PCA coordinates
            - .obsm['X_umap']: UMAP coordinates (if umap=True)
            - .obs['leiden']: Leiden cluster assignments

        Example:
            >>> phenocoder.spatialgraph_embedding(
            ...     n_dim=50,
            ...     batch_correction=True,
            ...     batch_key='plate_id',
            ...     n_neighbors=20
            ... )
        """

        if self.adata is None:
            raise ValueError(
                'self.adata is None. Run spatialgraph_stats() first to compute '
                'spatial statistics before generating embeddings.'
            )

        # Carry per-sample metadata (e.g. condition groups) into adata.obs so the
        # embedding can be inspected/colored by them. spatialgraph_stats builds the
        # sample-level adata with an empty obs, so these columns are looked up from
        # the source table and mapped on by sample.
        if obs_keys is not None:
            if isinstance(obs_keys, str):
                obs_keys = [obs_keys]
            if self.sdata is None or self.table_key is None:
                raise ValueError(
                    'obs_keys requires self.sdata and self.table_key to look up '
                    'per-sample metadata.'
                )
            table = self.sdata.tables[self.table_key]
            if self.sample_key not in table.obs.columns:
                raise ValueError(
                    f'sample_key "{self.sample_key}" not found in '
                    f'sdata.tables["{self.table_key}"].obs'
                )
            # per-row sample identifier: the sample_key column if present
            # (subunit-level adata), otherwise the index (sample-level adata)
            if self.sample_key in self.adata.obs.columns:
                sample_ids = self.adata.obs[self.sample_key].astype(str)
            else:
                sample_ids = self.adata.obs.index.to_series().astype(str)
            for key in obs_keys:
                if key not in table.obs.columns:
                    raise ValueError(
                        f'obs_key "{key}" not found in '
                        f'sdata.tables["{self.table_key}"].obs. '
                        f'Available columns: {list(table.obs.columns)}'
                    )
                mapping = table.obs.groupby(self.sample_key)[key].first()
                mapping.index = mapping.index.astype(str)
                self.adata.obs[key] = sample_ids.map(mapping.to_dict()).values

        # Store raw data
        self.adata.layers['raw'] = self.adata.X.copy()

        # Scale data
        if scale:
            sc.pp.scale(self.adata)
            self.adata.X[np.isnan(self.adata.X)] = 0

        # Handle batch correction metadata
        if batch_correction:
            if confounder_key is None:
                confounder_key = []
            elif isinstance(confounder_key, str):
                confounder_key = [confounder_key]

            if batch_key is None:
                raise ValueError('batch_key must be specified for batch correction')

            # Check if batch_key is in adata.obs
            if batch_key not in self.adata.obs.columns:
                # Try to get it from sdata.tables
                if self.sdata is None or self.table_key is None:
                    raise ValueError(
                        f'batch_key "{batch_key}" not found in adata.obs and '
                        f'sdata/table_key not available for lookup.'
                    )

                if self.table_key not in self.sdata.tables:
                    raise ValueError(
                        f'table_key "{self.table_key}" not found in sdata.tables'
                    )

                table = self.sdata.tables[self.table_key]

                # Check if batch_key exists in the table
                if batch_key not in table.obs.columns:
                    raise ValueError(
                        f'batch_key "{batch_key}" not found in adata.obs or '
                        f'sdata.tables["{self.table_key}"].obs'
                    )

                # Get unique batch values per sample_key
                if self.sample_key not in table.obs.columns:
                    raise ValueError(
                        f'sample_key "{self.sample_key}" not found in '
                        f'sdata.tables["{self.table_key}"].obs'
                    )

                # Create mapping from sample_key to batch_key
                batch_mapping = (
                    table.obs.groupby(self.sample_key)[batch_key].first().to_dict()
                )

                # Map batch_key to adata.obs using the index
                self.adata.obs[batch_key] = self.adata.obs.index.map(batch_mapping)

                # Also add confounder_key if specified
                for conf_key in confounder_key:
                    if conf_key not in self.adata.obs.columns:
                        if conf_key in table.obs.columns:
                            conf_mapping = (
                                table.obs.groupby(self.sample_key)[conf_key]
                                .first()
                                .to_dict()
                            )
                            self.adata.obs[conf_key] = self.adata.obs.index.map(
                                conf_mapping
                            )

            # Apply ridge regression for batch correction
            bbknn.ridge_regression(
                self.adata,
                batch_key=[batch_key],
                confounder_key=confounder_key if confounder_key else [],
            )

        # Select highly variable features if requested
        if variable_features:
            sc.pp.highly_variable_genes(self.adata)

        # Check and adjust n_dim if necessary
        max_dim = self.adata.obs.shape[0] - 1
        if n_dim > max_dim:
            print(
                f'Warning: n_dim ({n_dim}) exceeds maximum possible ({max_dim}). '
                f'Setting n_dim to {max_dim}.'
            )
            n_dim = max_dim

        # Compute PCA
        sc.pp.pca(
            self.adata,
            n_comps=n_dim,
            use_highly_variable=variable_features,
        )

        if batch_correction:
            bbknn.bbknn(
                self.adata, batch_key=batch_key, neighbors_within_batch=n_neighbors
            )
        else:
            sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, use_rep='X_pca')

        if umap:
            sc.tl.umap(self.adata, n_components=2)
