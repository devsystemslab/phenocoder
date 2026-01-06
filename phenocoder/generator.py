import os
from pathlib import Path

import numpy as np
import pandas as pd
from keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


class PatchGenerator:
    """
    Generator for image patches and image patch datasets from spatial data.

    This class handles the extraction of image patches and statistics
    from spatial data objects, primarily for use in deep learning workflows.
    """

    def __init__(
        self,
        dataset: str,
        sdata,
        image_key: str,
        spatial_key: str,
        table_key: str,
        sample_key: str,
        dir_output: str | Path,
        max_workers: int = 12,
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
        image_key_init = '_'.join(
            [
                self.image_key,
                self.sdata.tables[self.table_key].obs[self.sample_key].unique()[0],
            ]
        )
        self.channels = self.sdata.images[image_key_init].coords['c'].values.tolist()
        self.image_size = self.sdata.images[image_key_init].shape[-2:]
        self.patch_size = (128, 128)
        self.max_workers = max_workers
        self.dir_output = Path(dir_output)
        self.dir_dataset = Path(dir_output, dataset)
        self.df_images = None
        self.df_stats = pd.DataFrame()
        self.df_stats_patches = pd.DataFrame()
        self.df_stats_patches_sampled = None
        self.patches = None
        self.ids = None

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
        img,
        id,
        scale=False,
        percentiles_high=None,
        percentiles_low=None,
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
        if scale:
            if percentiles_high is None or percentiles_low is None:
                raise ValueError('Percentiles need to be provided for scaling')
            img = img.astype(np.float32)
            for i in range(img.shape[-1]):
                img[..., i] = np.clip(
                    (img[..., i] - percentiles_low[i])
                    / (percentiles_high[i] - percentiles_low[i]),
                    0,
                    1,
                )
            img = img.astype(np.float32)

        return img

    def __get_image_stats__(self, imgs, id, id_name, percentile=1):
        """
        Calculate comprehensive statistics for image data.

        Parameters
        ----------
        imgs : ndarray
            Input image array
        id : str or int
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

    def generate_image_stats(self, sample_id):
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

    def write_patches(self, sample_id):
        """
        Write all patches of a given samples to disk as numpy arrays.

        Parameters
        ----------
        sample_id : str or int
            Sample identifier for which to write patches
        """

        # get all files that need to be written
        df_patches_sample = self.patches[self.patches[self.sample_key] == sample_id]
        img_key_sample = '_'.join([self.image_key, sample_id])
        img = np.asarray(self.sdata.images[img_key_sample])
        for id, file in zip(df_patches_sample['id'], df_patches_sample['file']):
            np.save(Path(self.dir_dataset, file), self.extract_patch(img, id))

    def generate_dataset(
        self,
        n_samples=None,
        n_patches=None,
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
        if n_patches is not None:
            self.patches = self.patches.sample(n_patches, replace=False)
            self.samples = self.patches[self.sample_key].unique()
        self.patches['file'] = self.patches.apply(
            lambda row: f'{row[self.sample_key]}_{row["id"]}.npy',
            axis=1,
        )
        self.patches.to_csv(Path(self.dir_dataset, 'patches.csv'))
        [self.write_patches(id) for id in tqdm(self.samples, desc='Writing patches')]


class SequenceGenerator(Sequence):
    """
    Keras Sequence generator for loading image patches during training.

    This generator loads patches from disk and applies optional data
    augmentation and normalization for training deep learning models.
    """

    def __init__(
        self,
        list_ids: list,
        batch_size=32,
        dim=(128, 128),
        n_channels=4,
        shuffle=True,
        scale=False,
        flip=False,
        percentiles_low=None,
        percentiles_high=None,
        conditions=None,
        return_conditions=False,
        **kwargs,
    ):
        """
        Initialize PatchGenerator.

        Parameters
        ----------
        list_ids : list
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
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.flip = flip
        self.scale = scale
        self.percentiles_low = percentiles_low
        self.percentiles_high = percentiles_high
        self.conditions = conditions
        self.return_conditions = return_conditions
        self.on_epoch_end()

    def __len__(self):
        """
        Get number of batches per epoch.

        Returns
        -------
        int
            Number of batches that fit in the dataset
        """
        return int(np.floor(len(self.list_ids) / self.batch_size))

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
        list_ids_temp = [self.list_ids[k] for k in indexes]
        X = self.__data_generation(list_ids_temp)

        if self.scale:
            for i in range(X.shape[-1]):
                if self.percentiles_high[i] == self.percentiles_low[i]:
                    self.percentiles_high[i] += 1
                X[..., i] = np.clip(
                    (X[..., i] - self.percentiles_low[i])
                    / (self.percentiles_high[i] - self.percentiles_low[i]),
                    0,
                    1,
                )
            X = X.astype(np.float32)

        if self.flip:
            for i in range(X.shape[0]):
                # add random horizontal flip
                if np.random.rand() < 0.5:
                    X[i,] = np.fliplr(X[i,])
                # add random vertical flip
                if np.random.rand() < 0.5:
                    X[i,] = np.flipud(X[i,])

        if self.return_conditions and self.conditions is not None:
            cond = self.conditions[indexes]
            return X, cond
        else:
            return X

    def on_epoch_end(self):
        """
        Update indexes after each epoch.

        Shuffles the order of patches if shuffle is enabled.
        """
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
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
        for i, idx in enumerate(list_ids_temp):
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
                pd.read_csv(Path(self.dir_datasets, dataset, 'stats.csv')).assign(
                    dataset=dataset
                )
            )
            self.patches.append(
                pd.read_csv(Path(self.dir_datasets, dataset, 'patches.csv')).assign(
                    dataset=dataset
                )
            )
        self.stats = pd.concat(self.stats)
        self.patches = pd.concat(self.patches)

    def get_scaling_stats(self):
        """
        Get scaling statistics across all datasets.

        Returns
        -------
        tuple
            Tuple of (percentiles_low, percentiles_high) for each channel,
            representing the global min/max quantiles for normalization
        """
        percentiles_low = self.stats.groupby('channel')['percentile_low'].min()
        percentiles_high = self.stats.groupby('channel')['percentile_high'].max()
        return percentiles_low, percentiles_high

    def setup_generators(
        self,
        datasets,
        dir_datasets,
        conditional=False,
        batch_size=64,
        dim=(128, 128),
        n_channels=4,
        shuffle=True,
        scale=True,
        n_workers=1,
    ):
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
        scale : bool, default True
            Whether to scale data using quantiles
        n_workers : int, default 1
            Number of workers for data generation

        Returns
        -------
        tuple
            Training generator, validation generator, dataframe, and encoder
        """
        self.load_datasets()
        percentiles_low, percentiles_high = self.get_scaling_stats()
        # randomize
        self.patches = self.patches.sample(frac=1, random_state=42, replace=False)
        # get well dataset combinations and split into train and val
        df_samples = self.patches.groupby([self.sample_key, 'dataset']).count()
        df_samples = df_samples.reset_index().sample(
            frac=1, random_state=42, replace=False
        )
        # train val split
        n_train = int(df_samples.shape[0] * 0.8)
        df_samples['split'] = [
            'train' if i < n_train else 'val' for i in range(df_samples.shape[0])
        ]
        # TODO: add stratify by sample key!
        # merge df with df_well_train
        self.patches = pd.merge(
            self.patches,
            df_samples[[self.sample_key, 'dataset', 'split']],
            on=[self.sample_key, 'dataset'],
            how='left',
        )
        # drop remainders of split columns regarding batch_size grouped by split
        self.patches = (
            self.patches.groupby('split')
            .apply(lambda x: x.iloc[: -(x.shape[0] % batch_size)])
            .reset_index(drop=True)
        )
        # expand files to complete paths and add to path in self.patches -> Path(self.dir_datasets, self.patches['dataset'], file)
        self.patches['file_path'] = self.patches.apply(
            lambda x: Path(self.dir_datasets, x['dataset'], x['file']), axis=1
        )

        if conditional:
            # one hot encode conditions with sklearn
            enc = OneHotEncoder()
            cond = enc.fit_transform(self.patches[['dataset', 'z']]).toarray()
            # convert conditions to numpy array
            cond_train = cond[self.patches['split'] == 'train']
            cond_val = cond[self.patches['split'] == 'val']
        # TODO: fix that the SequenceGenerator just works for the CondVAE...
        generator_train = SequenceGenerator(
            self.patches[self.patches['split'] == 'train']['file_path'].values,
            conditions=cond_train,
            batch_size=batch_size,
            dim=dim,
            n_channels=n_channels,
            shuffle=shuffle,
            scale=scale,
            percentiles_low=percentiles_low.values,
            percentiles_high=percentiles_high.values,
            return_conditions=conditional,
            workers=n_workers,
        )

        generator_val = SequenceGenerator(
            self.patches[self.patches['split'] == 'val']['file_path'].values,
            conditions=cond_val,
            batch_size=batch_size,
            dim=dim,
            n_channels=n_channels,
            shuffle=shuffle,
            scale=scale,
            percentiles_low=percentiles_low.values,
            percentiles_high=percentiles_high.values,
            return_conditions=conditional,
            workers=n_workers,
        )

        return generator_train, generator_val, self.patches, enc
