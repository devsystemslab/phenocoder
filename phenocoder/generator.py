from pathlib import Path

import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
from spatialdata import SpatialData
from tqdm import tqdm


class PatchGenerator:
    """
    Generator for image patches and image patch datasets from spatial data.

    This class handles the extraction of image patches and statistics
    from spatial data objects, primarily for use in deep learning workflows.
    """

    def __init__(
        self,
        sdata: SpatialData,
        image_key: str,
        spatial_key: str,
        table_key: str,
        sample_key: str,
        scale: bool,
    ):
        """
        Initialize DatasetGenerator.

        Parameters
        ----------
        sdata : SpatialData
            Spatial data object containing images and tables
        image_key : str
            Key for accessing images in sdata.images
        spatial_key : str
            Key for accessing spatial coordinates in sdata.tables
        table_key : str
            Key for accessing tables in sdata.tables
        sample_key : str
            Key for sample identification in observations
        dir_output : str or Path
            Output directory for generated datasets
        max_workers : int, default 12
            Maximum number of worker threads for parallel processing
        """
        self.sdata = sdata
        self.image_key = image_key
        self.spatial_key = spatial_key
        self.table_key = table_key
        self.sample_key = sample_key
        self.scale = scale
        image_key_init = '_'.join(
            [
                self.image_key,
                self.sdata.tables[self.table_key].obs[self.sample_key].unique()[0],
            ]
        )
        self.channels = self.sdata.images[image_key_init].coords['c'].values.tolist()
        self.image_size = self.sdata.images[image_key_init].shape[-2:]
        self.patch_size = (128, 128)
        self.df_stats = pd.DataFrame()
        self.patches = None
        self.percentiles_low = None
        self.percentiles_high = None

    def init_patches(self):
        """
        Initialize patch positions from spatial coordinates.

        Extracts spatial coordinates from the data, filters positions that
        would result in patches extending beyond image boundaries, and
        assigns batch IDs.
        """
        self.patches = pd.DataFrame(
            self.sdata.tables[self.table_key].obsm[self.spatial_key],
            columns=['y', 'x', 'z'],
            index=self.sdata.tables[self.table_key].obs.index,
        )
        self.patches[self.sample_key] = self.sdata.tables[self.table_key].obs[
            self.sample_key
        ]
        # round to integer
        self.patches['x'] = self.patches['x'].astype(int)
        self.patches['y'] = self.patches['y'].astype(int)

        # filter x and y that are within image boundaries when patch size is added
        x_min, x_max = (
            self.patch_size[0] // 2,
            self.image_size[1] - self.patch_size[0] // 2,
        )
        y_min, y_max = (
            self.patch_size[0] // 2,
            self.image_size[0] - self.patch_size[0] // 2,
        )
        self.patches = self.patches[
            (self.patches['x'] >= x_min)
            & (self.patches['x'] <= x_max)
            & (self.patches['y'] >= y_min)
            & (self.patches['y'] <= y_max)
        ]

        self.patches['id'] = np.arange(0, len(self.patches))

    def extract_patch(
        self,
        img: np.ndarray,
        id: int,
    ):
        """
        Extract a patch from an image centered on specified coordinates.

        Parameters
        ----------
        img : ndarray
            Input image array
        id : int
            Batch ID corresponding to patch position

        Returns
        -------
        ndarray
            Extracted image patch
        """
        # extract patch centered on x and y
        x = int(self.patches[self.patches['id'] == id]['x'].iloc[0])
        y = int(self.patches[self.patches['id'] == id]['y'].iloc[0])

        if img.ndim == 4:
            z = int(self.patches[self.patches['id'] == id]['z'].iloc[0])
            img = img[:, z, ...]
        x_max = x + self.patch_size[0] // 2
        y_max = y + self.patch_size[1] // 2
        x_min = x - self.patch_size[0] // 2
        y_min = y - self.patch_size[1] // 2
        img = img[:, y_min:y_max, x_min:x_max]
        assert img.shape[-2:] == self.patch_size
        if self.scale:
            if self.percentiles_high is None or self.percentiles_low is None:
                raise ValueError('Percentiles need to be provided for scaling')
            img = img.astype(np.float32)
            for i in range(img.shape[0]):
                if self.percentiles_high[i] == self.percentiles_low[i]:
                    self.percentiles_high[i] += 1
                img[i] = np.clip(
                    (img[i] - self.percentiles_low[i])
                    / (self.percentiles_high[i] - self.percentiles_low[i]),
                    0,
                    1,
                )
            img = img.astype(np.float32)

        return img

    def __get_image_stats__(
        self, imgs: np.ndarray, id: str, id_name: str, percentile: int = 1
    ):
        """
        Calculate comprehensive statistics for image data.

        Parameters
        ----------
        imgs : ndarray
            Input image array
        id : str
            Identifier for the image/patch
        id_name : str
            Name of the ID column

        Returns
        -------
        DataFrame
            Statistics for each channel including mean, std, quantiles, etc.
        """
        if imgs.ndim == 3:
            imgs = imgs[:, np.newaxis, :, :]

        mean = np.mean(imgs, axis=(-2, -1))
        std = np.std(imgs, axis=(-2, -1))
        median = np.median(imgs, axis=(-2, -1))
        mad = np.median(np.abs(imgs - np.median(imgs)), axis=(-2, -1))
        max = np.max(imgs, axis=(-2, -1))
        min = np.min(imgs, axis=(-2, -1))
        percentile_low = np.percentile(imgs, percentile, axis=(-2, -1))
        percentile_high = np.percentile(imgs, 100 - percentile, axis=(-2, -1))

        df = pd.concat(
            [
                pd.DataFrame(
                    {
                        id_name: id,
                        'channel': self.channels[i],
                        'z': np.arange(imgs.shape[1]),
                        'mean': mean[i],
                        'std': std[i],
                        'percentile_high': percentile_high[i],
                        'percentile_low': percentile_low[i],
                        'median': median[i],
                        'mad': mad[i],
                        'max': max[i],
                        'min': min[i],
                    }
                )
                for i in range(len(self.channels))
            ]
        )

        return df

    def generate_image_stats(self, sample_id: str):
        """
        Generate statistics for all patches in a sample.

        Parameters
        ----------
        sample_id : str or int
            Sample identifier for which to generate statistics
        """

        df_patch_positions = self.patches[self.patches[self.sample_key] == sample_id]
        if len(df_patch_positions) > 0:
            # load images
            imgs = np.asarray(self.sdata.images['_'.join([self.image_key, sample_id])])
            df_stat = self.__get_image_stats__(imgs, sample_id, 'sample_id')
            self.df_stats = pd.concat([self.df_stats, df_stat])

    def select_patches(self, sample_id: str):
        """
        Select all patches of a given sample.

        Parameters
        ----------
        sample_id : str or int
            Sample identifier for which to select patches

        Returns
        -------
        df_patches_sample : pd.DataFrame
            DataFrame containing patch information
        img : np.ndarray
            Image array
        """
        # get all files that need to be written
        df_patches_sample = self.patches[self.patches[self.sample_key] == sample_id]
        img_key_sample = '_'.join([self.image_key, sample_id])
        img = np.asarray(self.sdata.images[img_key_sample])
        return df_patches_sample, img

    def write_patches(self, sample_id: str):
        """
        Write all patches of a given samples to disk as numpy arrays.

        Parameters
        ----------
        sample_id : str or int
            Sample identifier for which to write patches
        """
        df_patches_sample, img = self.select_patches(sample_id)
        for id, file in zip(df_patches_sample['id'], df_patches_sample['file']):
            np.save(Path(self.dir_dataset, file), self.extract_patch(img, id))

    def get_patches(self, sample_id: str):
        """
        Return all patches of a given sample.

        Parameters
        ----------
        sample_id : str or int
            Sample identifier for which to retrieve patches

        Returns
        -------
        list of np.ndarray
            List of patches as numpy arrays
        pd.DataFrame
            DataFrame containing patch information
        """
        df_patches_sample, img = self.select_patches(sample_id)
        patches = np.asarray(
            [self.extract_patch(img, id) for id in df_patches_sample['id']]
        )
        return np.moveaxis(patches, 1, -1), df_patches_sample

    def get_scaling_percentiles(self):
        """
        Extract and set scaling percentiles from computed statistics.

        Computes conservative percentiles across all samples and z-stacks for each channel
        to use for normalization during patch extraction. Uses the minimum of percentile_low
        values and maximum of percentile_high values to avoid clipping any samples.

        Sets the percentiles_low and percentiles_high attributes based on the
        percentile_low and percentile_high values in df_stats.

        Raises
        ------
        ValueError
            If statistics have not been computed yet (df_stats is None or empty)
        """
        if self.df_stats is None or self.df_stats.empty:
            raise ValueError('Statistics not computed yet')

        # Group by channel and compute conservative percentiles to avoid clipping
        # Use min for low percentile (darkest values) and max for high percentile (brightest values)
        self.percentiles_low = self.df_stats.groupby('channel')['percentile_low'].min()
        self.percentiles_high = self.df_stats.groupby('channel')[
            'percentile_high'
        ].max()

        # Set percentiles in the same order as self.channels
        self.percentiles_low = self.percentiles_low.loc[self.channels].values
        self.percentiles_high = self.percentiles_high.loc[self.channels].values

    def generate_dataset(
        self,
        dataset: str,
        dir_output: str,
        n_samples: int = None,
        n_patches: int = None,
    ):
        """
        Generate complete dataset with patches and statistics.

        Parameters
        ----------
        sampling_frac : float, optional
            Fraction of samples to process
        n_patches : int, optional
            Number of patches to sample from all available patches
        """
        self.dir_output = Path(dir_output)
        self.dir_dataset = Path(dir_output, dataset)
        self.dir_output.mkdir(exist_ok=True, parents=True)
        self.dir_dataset.mkdir(exist_ok=True, parents=True)
        self.samples = self.sdata.tables[self.table_key].obs[self.sample_key].unique()
        if n_samples is not None:
            self.samples = np.random.choice(self.samples, n_samples, replace=False)
        self.init_patches()
        [
            self.generate_image_stats(sample)
            for sample in tqdm(self.samples, desc='Generating image statistics')
        ]
        self.df_stats.to_csv(Path(self.dir_dataset, 'stats.csv'), index=False)
        if self.scale:
            self.get_scaling_percentiles()
        if n_patches is not None:
            self.patches = self.patches.sample(n_patches, replace=False)
            self.samples = self.patches[self.sample_key].unique()
        self.patches['file'] = self.patches.apply(
            lambda row: f'{row[self.sample_key]}_{row["id"]}.npy',
            axis=1,
        )
        self.patches['dataset'] = dataset
        self.patches.to_csv(Path(self.dir_dataset, 'patches.csv'))
        [self.write_patches(id) for id in tqdm(self.samples, desc='Writing patches')]


class SequenceGenerator(Sequence):
    """
    Keras Sequence generator for loading image patches from disk during training.

    This generator loads patches from disk and applies optional data
    augmentation and normalization for training deep learning models.
    """

    def __init__(
        self,
        ids: list,
        batch_size=32,
        dim=(128, 128),
        n_channels=4,
        shuffle=True,
        flip=False,
        conditions=None,
        return_conditions=False,
        **kwargs,
    ):
        """
        Initialize SequenceGenerator.

        Parameters
        ----------
        ids : list
            List of file paths for patches to load
        batch_size : int, default 32
            Number of patches per batch
        dim : tuple, default (128, 128)
            Spatial dimensions of patches
        n_channels : int, default 4
            Number of channels in patches
        shuffle : bool, default True
            Whether to shuffle patch order each epoch
        scale : bool, default False
            Whether to apply intensity scaling
        flip : bool, default False
            Whether to apply random flipping augmentation
        percentiles_low : array-like, optional
            Low quantiles for intensity normalization
        percentiles_high : array-like, optional
            High quantiles for intensity normalization
        conditions : array-like, optional
            Condition labels for conditional generation
        return_conditions : bool, default False
            Whether to return conditions along with patches
        **kwargs
            Additional arguments passed to parent Sequence class
        """
        super().__init__(**kwargs)
        self.indexes = None
        self.dim = dim
        self.batch_size = batch_size
        self.ids = ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.flip = flip
        self.conditions = conditions
        self.on_epoch_end()

    def __len__(self):
        """
        Get number of batches per epoch.

        Returns
        -------
        int
            Number of batches that fit in the dataset
        """
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Parameters
        ----------
        index : int
            Batch index

        Returns
        -------
        ndarray or tuple
            Batch of patches, optionally with conditions
        """
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        ids_temp = [self.ids[k] for k in indexes]
        X = self.__data_generation(ids_temp)

        if self.flip:
            for i in range(X.shape[0]):
                # add random horizontal flip
                if np.random.rand() < 0.5:
                    X[i,] = np.fliplr(X[i,])
                # add random vertical flip
                if np.random.rand() < 0.5:
                    X[i,] = np.flipud(X[i,])

        if self.conditions is not None:
            cond = self.conditions[indexes]
            return X, cond
        else:
            return X

    def on_epoch_end(self):
        """
        Update indexes after each epoch.

        Shuffles the order of patches if shuffle is enabled.
        """
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids_temp):
        """
        Generate batch data by loading patches from disk.

        Parameters
        ----------
        list_ids_temp : list
            List of file paths for the current batch

        Returns
        -------
        ndarray
            Batch of loaded image patches
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, idx in enumerate(ids_temp):
            # Store sample
            X[i,] = np.moveaxis(np.load(idx), 0, -1)

        return X


class DatasetLoader:
    """
    Utility class for merging multiple datasets and their statistics.

    This class combines statistics from multiple dataset directories and
    provides unified access to files and scaling parameters.
    """

    def __init__(self, datasets: list, dir_datasets: str, sample_key: str):
        """
        Initialize DatasetMerger.

        Parameters
        ----------
        datasets : list
            List of dataset names to merge
        dir_datasets : str
            Base directory containing dataset subdirectories
        """
        self.dir_datasets = dir_datasets
        self.datasets = datasets
        self.sample_key = sample_key
        self.stats_imgs = None
        self.patches = None

    def load_datasets(self):
        """
        Loads and merge statistics from all specified datasets.

        Combines stats.csv files from each
        dataset directory and creates unified dataframes with file paths.
        """
        self.stats = []
        self.patches = []
        for dataset in self.datasets:
            self.stats.append(
                pd.read_csv(Path(self.dir_datasets, dataset, 'stats.csv'))
            )
            self.patches.append(
                pd.read_csv(Path(self.dir_datasets, dataset, 'patches.csv'))
            )
        self.stats = pd.concat(self.stats)
        self.patches = pd.concat(self.patches)

    def set_train_val_split(self, batch_size=64, split: float = 0.8):
        """
        Setup generators for training and validation.

        Parameters
        ----------
        dir_datasets : str
            Directory containing datasets
        conditional : bool, default False
            Whether to return conditions with data
        batch_size : int, default 64
            Batch size for generators
        dim : tuple, default (128, 128)
            Dimensions of patches
        n_channels : int, default 4
            Number of channels
        shuffle : bool, default True
            Whether to shuffle data
        n_workers : int, default 1
            Number of workers for data generation

        Returns
        -------
        tuple
            Training generator, validation generator, dataframe, and encoder
        """
        self.load_datasets()
        self.patches = self.patches.sample(frac=1, random_state=42, replace=False)
        df_samples = self.patches.groupby([self.sample_key, 'dataset']).count()
        df_samples = df_samples.reset_index().sample(
            frac=1, random_state=42, replace=False
        )
        n_train = int(df_samples.shape[0] * split)
        df_samples['split'] = [
            'train' if i < n_train else 'val' for i in range(df_samples.shape[0])
        ]
        self.patches = pd.merge(
            self.patches,
            df_samples[[self.sample_key, 'dataset', 'split']],
            on=[self.sample_key, 'dataset'],
            how='left',
        )
        # drop remainders of splits regarding batch_size
        self.patches = (
            self.patches.groupby('split')
            .apply(lambda x: x.iloc[: -(x.shape[0] % batch_size)])
            .reset_index(drop=True)
        )
        # expand files to complete paths
        self.patches['file_path'] = self.patches.apply(
            lambda x: Path(self.dir_datasets, x['dataset'], x['file']), axis=1
        )

    def get_generators(
        self,
        conditions: list[str],
        batch_size: int = 64,
        dim=(128, 128),
        n_channels=4,
        shuffle=True,
        n_workers=1,
    ):
        if conditions:
            enc = OneHotEncoder()
            cond = enc.fit_transform(self.patches[conditions]).toarray()
            generator_train = SequenceGenerator(
                self.patches[self.patches['split'] == 'train']['file_path'].values,
                conditions=cond[self.patches['split'] == 'train'],
                batch_size=batch_size,
                dim=dim,
                n_channels=n_channels,
                shuffle=shuffle,
                return_conditions=True,
                workers=n_workers,
            )
            generator_val = SequenceGenerator(
                self.patches[self.patches['split'] == 'val']['file_path'].values,
                conditions=cond[self.patches['split'] == 'val'],
                batch_size=batch_size,
                dim=dim,
                n_channels=n_channels,
                shuffle=shuffle,
                return_conditions=True,
                workers=n_workers,
            )
            return generator_train, generator_val, enc
        else:
            generator_train = SequenceGenerator(
                self.patches[self.patches['split'] == 'train']['file_path'].values,
                batch_size=batch_size,
                dim=dim,
                n_channels=n_channels,
                shuffle=shuffle,
                return_conditions=False,
                workers=n_workers,
            )
            generator_val = SequenceGenerator(
                self.patches[self.patches['split'] == 'val']['file_path'].values,
                batch_size=batch_size,
                dim=dim,
                n_channels=n_channels,
                shuffle=shuffle,
                return_conditions=False,
                workers=n_workers,
            )
            return generator_train, generator_val
