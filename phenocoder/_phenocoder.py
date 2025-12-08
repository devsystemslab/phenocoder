from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import keras
import pandas as pd
import spatialdata as sd
import tensorflow as tf
import yaml

from phenocoder.generator import DatasetLoader, PatchGenerator
from phenocoder.model import CVAE, CondCVAE
from phenocoder.spatial import SpatialGraphAnalyzer
from phenocoder.utils import plot_latent_space, plot_reconstructions, plot_to_image


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
            **kwargs: Additional keyword arguments (currently unused).
        """
        self.sdata: sd.SpatialData = None
        self.adata: ad.AnnData = None
        self.model: CVAE | CondCVAE = None
        self.model_dir: str | Path = None
        self.model_oh_enc = None
        self.model_config = None
        self.sample_key: str = None
        self.spatial_key: str = None
        self.table_key: str = None
        self.data_dir: str | Path = None
        self.datasets: list[str] = None
        self.data_generator_train = None
        self.data_generator_val = None
        self.df_conditions = None

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

    def validate_sdata(self) -> None:
        """
        Validate the SpatialData object for required structure and keys.

        Checks that the sample_key is set and exists in the SpatialData object structure.
        This method should be called before other operations that depend on the SpatialData object.

        Raises:
            ValueError: If sample_key is not set or not found in the SpatialData object.

        Returns:
            None
        """
        if self.sample_key is None:
            raise ValueError('Sample key is not set.')
        if self.sample_key not in self.sdata.table[self.table_key].obs.columns:
            raise ValueError(
                f'Sample key "{self.sample_key}" not found in sdata.table["{self.table_key}"]'
            )
        else:
            pass

    # TODO: update method signature, etc and dataset generator to work with sdata
    # TODO: CRITICAL - Method signature is file-based (dir_input, dir_segmented) instead of using self.sdata
    # TODO: DatasetGenerator should be initialized with sdata, not file paths
    def generate_dataset(self, dataset, dir_dataset, spatial_key_index=None) -> None:
        """
        Generate an image patch dataset for phenotyping from input microscopy images.

        Creates image patches from input images and segmentation masks, with options for
        sampling strategies and multi-channel processing. The generated dataset is used
        for training the variational autoencoder model.

        Args:
            dir_dataset (str): Directory path for storing the generated dataset.
        Returns:
            None

        Raises:
            ValueError: If data_dir is not set.

        Example:
            >>> phenocoder.generate_dataset()
        """
        if spatial_key_index is None:
            spatial_key_index = self.spatial_key
        self.data_dir = dir_dataset
        if self.datasets is None:
            self.datasets = [dataset]
        else:
            self.datasets = self.datasets.append(dataset)
        dataset_generator = PatchGenerator(
            dataset=dataset,
            sdata=self.sdata,
            dir_output=self.data_dir,
            sample_key=self.sample_key,
            table_key=self.table_key,
            image_key=self.image_key,
            spatial_key=spatial_key_index,
        )
        dataset_generator.generate_dataset()
        dataset_generator.save_stats()

    def initialize_model(
        self,
        n_latent_dim: int,
        n_dense_dim: int,
        conditional: bool,
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
        if conditional:
            self.model_name = f'cond_{self.model_name}'
        if self.data_dir is None:
            raise ValueError('.data_dir must be specified')
        if self.datasets is None:
            raise ValueError('.datasets must be specified')
        self.model_dir = Path(self.data_dir, 'models', self.model_name)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        (
            self.data_generator_train,
            self.data_generator_val,
            self.df_conditions,
            self.model_oh_enc,
        ) = setup_generators(
            self.datasets,
            self.data_dir,
            conditional,
            batch_size=batch_size,
            n_workers=n_workers,
            dim=input_shape[:2],
            n_channels=input_shape[2],
            shuffle=True,
        )

        self.model_config = {
            'n_latent_dim': n_latent_dim,
            'n_dense_dim': n_dense_dim,
            'input_shape': list(input_shape),
            'conv_layers': list(conv_layers),
            'conditional': conditional,
            'dropout': dropout,
            'dir_dataset': self.data_dir,
            'batch_size': batch_size,
            'n_workers': n_workers,
            'beta': beta,
            'quantiles_low': self.data_generator_train.quantiles_low.tolist(),
            'quantiles_high': self.data_generator_train.quantiles_high.tolist(),
            'conditions_dim': self.data_generator_train.conditions.shape[-1],
        }

        with open(Path(self.model_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.model_config, file)
        joblib.dump(self.model_oh_enc, Path(self.model_dir, 'oh_encoder.joblib'))

        # self.df_conditions.to_csv(Path(self.model_dir, 'df_files.csv'), index=False)

        # set up model
        if conditional:
            self.model = CondCVAE(
                n_classes=self.data_generator_train.conditions.shape[-1],
                input_shape=input_shape,
                latent_dim=n_latent_dim,
                dense_dim=n_dense_dim,
                conv_layers=conv_layers,
                dropout=dropout,
                beta=beta,
            )
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
        else:
            self.model = CVAE(
                input_shape=tuple(self.model_config['input_shape']),
                latent_dim=self.model_config['n_latent_dim'],
                dense_dim=self.model_config['n_dense_dim'],
                conv_layers=tuple(self.model_config['conv_layers']),
            )
        self.model.compile()
        self.model.load_weights(f'{self.model_directory}/model.weights.h5')
        self.oh_enc = joblib.load(f'{self.model_directory}/oh_encoder.joblib')

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
                    factor=0.2,
                    patience=learning_rate_patience,
                    min_lr=min_learning_rate,
                ),
            ],
        )

        self.model.build(self.model_config['input_shape'])
        self.model.save_weights(Path(self.model_dir, 'model.weights.h5'))

        if plot:
            file_writer = tf.summary.create_file_writer(self.dir_tensorboard)
            figure_reconstructions = plot_reconstructions(
                self.model,
                self.data_generator_train,
                n_preview=n_preview,
                return_fig=True,
                show=False,
            )
            with file_writer.as_default():
                tf.summary.image(
                    'input vs reconstruction',
                    plot_to_image(figure_reconstructions),
                    step=0,
                )
            figure_latent_space = plot_latent_space(
                self.model,
                self.data_generator_train,
                self.model_oh_enc,
                sample_frac=plot_frac,
                return_fig=True,
                show=False,
            )
            with file_writer.as_default():
                tf.summary.image(
                    'latent space', plot_to_image(figure_latent_space), step=0
                )

    def encode(
        self, batch_size: int = 64, filter_encodable_conditions: bool = False
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
        results = []
        samples = self.sdata.tables[self.table_key].obs[self.sample_key].unique()
        nuclei_patch_generator = PatchGenerator()
        for sample in samples:
            patches, df = nuclei_patch_generator.get_patches(
                sample,
                self.sdata.tables[self.table_key],
                scale=True,
                quantiles_low=self.model_config['quantiles_low'],
                quantiles_high=self.model_config['quantiles_high'],
                channels=self.sdata.images[self.image_key].coords.values,
            )
            df = df.reset_index()
            if df.empty:
                continue
            if self.model_config['conditional']:
                df_cond = df[['plate_id', 'z']]  # fix: dataset specific
                # rename plate_id to dataset
                df_cond = df_cond.rename(
                    columns={'plate_id': 'dataset'}
                )  # fix: dataset specific
                # reset index
                if filter_encodable_conditions:
                    # filter out conditions which cannot be encoded
                    idx = df_cond.index[
                        (df_cond['z'].isin(self.oh_enc.categories_[1]))
                        & (df_cond['dataset'].isin(self.oh_enc.categories_[0]))
                    ]
                    conditions = self.oh_enc.transform(
                        df_cond.iloc[idx][self.oh_enc.feature_names_in_]
                    )
                    patches = patches[idx]
                    df = df.iloc[idx]
                else:
                    conditions = self.oh_enc.transform(
                        df_cond[self.oh_enc.feature_names_in_]
                    )
                _, _, z = self.model.encoder.predict(
                    [patches, conditions], batch_size=batch_size
                )
            else:
                _, _, z = self.model.encoder.predict(patches, batch_size=batch_size)
            df_z = pd.DataFrame(z, columns=[f'z_{i}' for i in range(z.shape[-1])])
            # reset z to pixel coordinates
            df['z'] = df['z'] / 0.322 * 10  # fix: dataset specific

            # drop non numeric cols
            df = df.drop(columns=['well_id', 'plate_id'])  # fix: dataset specific
            # add df_z to df
            df = pd.concat([df, df_z], axis=1)
            df = df.groupby('label').mean()
            df = df.reset_index()
            df_z = df[[f'z_{i}' for i in range(z.shape[-1])]]
            df = df.drop(columns=[f'z_{i}' for i in range(z.shape[-1])])

            df = df.assign(well_id=sample)

            df.index = (
                df['label'].astype(str) + '_' + df['well_id'] + '_' + df['plate_id']
            )
            # TODO: Should store results in sdata.tables instead of creating standalone AnnData
            adata = ad.AnnData(
                X=df_z.values,
                obs=df,
                var=pd.DataFrame(index=[f'z_{i + 1}' for i in range(df_z.shape[-1])]),
            )
            results.append(adata)
        adata = ad.concat(results)
        adata.obs['label'] = adata.obs.index.copy()
        return adata

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
