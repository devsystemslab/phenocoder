from __future__ import annotations

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
        patch_size: tuple[int, int] = (128, 128),
        metadata_keys: list[str] | None = None,
        scale_percentile: float = 1,
        scale_per_sample: bool = True,
    ):
        """
        Initialize PatchGenerator.

        Args:
            sdata (SpatialData): Spatial data object containing images and tables
            image_key (str): Key prefix for accessing images in sdata.images (images are read per sample
                as ``f"{image_key}_{sample}"``)
            spatial_key (str): Key in ``sdata.tables[table_key].obsm`` holding the (y, x, z) coordinates used
                to center patches
            table_key (str): Key for accessing the object table in sdata.tables
            sample_key (str): obs column used for sample identification
            scale (bool): Whether patches are intensity-scaled using the computed percentiles
            patch_size (tuple of int): Patch (height, width) extracted around each object. Defaults to (128, 128).
            metadata_keys (list of str, optional): Additional columns from ``sdata.tables[table_key].obs`` to copy into the
                patches dataframe (and ``patches.csv``)
            scale_percentile (float): Percentile (in 0-100) used when computing each slice's low/high in the
                image statistics. Defaults to 1.
            scale_per_sample (bool): If True, aggregate the per-slice percentiles per (sample, channel) so each
                sample is normalized to its own range; if False, aggregate globally per channel. Defaults to True.
        """
        self.sdata = sdata
        self.image_key = image_key
        self.spatial_key = spatial_key
        self.table_key = table_key
        self.sample_key = sample_key
        self.scale = scale
        self.scale_percentile = scale_percentile
        self.scale_per_sample = scale_per_sample
        self.metadata_keys = metadata_keys or []
        image_key_init = '_'.join(
            [
                self.image_key,
                self.sdata.tables[self.table_key].obs[self.sample_key].unique()[0],
            ]
        )
        self.channels = self.sdata.images[image_key_init].coords['c'].values.tolist()
        self.image_size = self.sdata.images[image_key_init].shape[-2:]
        self.patch_size = patch_size
        self.df_stats = pd.DataFrame()
        self.patches = None
        self.percentiles_low = None
        self.percentiles_high = None
        # per-sample lookups {sample_id: array over channels}, built by
        # get_scaling_percentiles when scale_per_sample is True
        self.sample_percentiles_low = None
        self.sample_percentiles_high = None

    def init_patches(self) -> None:
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
        # carry user-requested obs columns so they can be used as conditions
        for key in self.metadata_keys:
            if key in (self.sample_key, 'x', 'y', 'z'):
                continue
            self.patches[key] = self.sdata.tables[self.table_key].obs[key]
        # round to integer
        self.patches['x'] = self.patches['x'].astype(int)
        self.patches['y'] = self.patches['y'].astype(int)

        # filter x and y that are within image boundaries when patch size is added
        # (per-sample, since images may differ in size)
        filtered = []
        for sample_id, grp in self.patches.groupby(self.sample_key):
            image_size = self.sdata.images[
                '_'.join([self.image_key, str(sample_id)])
            ].shape[-2:]
            x_min, x_max = (
                self.patch_size[0] // 2,
                image_size[1] - (self.patch_size[0] - self.patch_size[0] // 2),
            )
            y_min, y_max = (
                self.patch_size[1] // 2,
                image_size[0] - (self.patch_size[1] - self.patch_size[1] // 2),
            )
            grp = grp[
                (grp['x'] >= x_min)
                & (grp['x'] < x_max)
                & (grp['y'] >= y_min)
                & (grp['y'] < y_max)
            ]
            filtered.append(grp)
        self.patches = pd.concat(filtered)

        self.patches['id'] = np.arange(0, len(self.patches))

    def extract_patch(
        self,
        img: np.ndarray,
        id: int,
    ) -> np.ndarray:
        """
        Extract a patch from an image centered on specified coordinates.

        Args:
            img (ndarray): Input image array
            id (int): Batch ID corresponding to patch position

        Returns:
            ndarray: Extracted image patch
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
        self, imgs: np.ndarray, id: str, id_name: str, percentile: float | None = None
    ):
        """
        Calculate comprehensive statistics for image data.

        Args:
            imgs (ndarray): Input image array
            id (str): Identifier for the image/patch
            id_name (str): Name of the ID column
            percentile (float, optional): Percentile (0-100) for the low/high columns; the high uses
                ``100 - percentile``. Defaults to ``self.scale_percentile``.

        Returns:
            DataFrame: Statistics for each channel including mean, std, quantiles, etc.
        """
        if percentile is None:
            percentile = self.scale_percentile
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

    def generate_image_stats(self, sample_id: str) -> None:
        """
        Generate statistics for all patches in a sample.

        Args:
            sample_id (str or int): Sample identifier for which to generate statistics
        """

        df_patch_positions = self.patches[self.patches[self.sample_key] == sample_id]
        if len(df_patch_positions) > 0:
            # load images
            imgs = np.asarray(self.sdata.images['_'.join([self.image_key, sample_id])])
            df_stat = self.__get_image_stats__(imgs, sample_id, 'sample_id')
            self.df_stats = pd.concat([self.df_stats, df_stat])

    def select_patches(self, sample_id: str) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Select all patches of a given sample.

        Args:
            sample_id (str or int): Sample identifier for which to select patches

        Returns:
            df_patches_sample (pd.DataFrame): DataFrame containing patch information
            img (np.ndarray): Image array
        """
        # get all files that need to be written
        df_patches_sample = self.patches[self.patches[self.sample_key] == sample_id]
        img_key_sample = '_'.join([self.image_key, sample_id])
        img = np.asarray(self.sdata.images[img_key_sample])
        if self.scale and self.scale_per_sample:
            # activate this sample's own scaling range for extract_patch.
            # copy: extract_patch may bump percentiles_high in place when low==high
            if self.sample_percentiles_low is None:
                raise ValueError(
                    'Per-sample scaling requested but percentiles not computed; '
                    'call get_scaling_percentiles() first'
                )
            self.percentiles_low = self.sample_percentiles_low[sample_id].copy()
            self.percentiles_high = self.sample_percentiles_high[sample_id].copy()
        return df_patches_sample, img

    def write_patches(self, sample_id: str) -> None:
        """
        Write all patches of a given samples to disk as numpy arrays.

        Args:
            sample_id (str or int): Sample identifier for which to write patches
        """
        df_patches_sample, img = self.select_patches(sample_id)
        for id, file in zip(df_patches_sample['id'], df_patches_sample['file']):
            np.save(Path(self.dir_dataset, file), self.extract_patch(img, id))

    def get_patches(self, sample_id: str) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Return all patches of a given sample.

        Args:
            sample_id (str or int): Sample identifier for which to retrieve patches

        Returns:
            list of np.ndarray: List of patches as numpy arrays
            pd.DataFrame: DataFrame containing patch information
        """
        df_patches_sample, img = self.select_patches(sample_id)
        patches = np.asarray(
            [self.extract_patch(img, id) for id in df_patches_sample['id']]
        )
        return np.moveaxis(patches, 1, -1), df_patches_sample

    def get_scaling_percentiles(self) -> None:
        """
        Extract and set scaling percentiles from computed statistics.

        Aggregates the per-slice ``percentile_low`` / ``percentile_high`` values in
        ``df_stats`` into a conservative range -- minimum of lows (darkest) and
        maximum of highs (brightest) -- used to normalize patches in
        ``extract_patch``. The grouping depends on ``scale_per_sample``:

        - ``scale_per_sample=True`` (default): aggregate per (sample, channel), so
          each sample is scaled to its own intensity range. Stored in
          ``sample_percentiles_low`` / ``sample_percentiles_high`` keyed by sample;
          ``select_patches`` activates the right one per sample.
        - ``scale_per_sample=False``: aggregate per channel across all samples/slices
          (the original global behaviour). Stored directly in ``percentiles_low`` /
          ``percentiles_high``.

        Raises:
            ValueError: If statistics have not been computed yet (df_stats is None or empty)
        """
        if self.df_stats is None or self.df_stats.empty:
            raise ValueError('Statistics not computed yet')

        if self.scale_per_sample:
            # Per (sample, channel): conservative range over the sample's own slices.
            low = self.df_stats.groupby(['sample_id', 'channel'])[
                'percentile_low'
            ].min()
            high = self.df_stats.groupby(['sample_id', 'channel'])[
                'percentile_high'
            ].max()
            # -> {sample_id: array ordered like self.channels} for per-channel indexing
            low = low.unstack('channel')[self.channels]
            high = high.unstack('channel')[self.channels]
            self.sample_percentiles_low = {s: r.values for s, r in low.iterrows()}
            self.sample_percentiles_high = {s: r.values for s, r in high.iterrows()}
        else:
            # Global per channel across all samples/slices (original behaviour).
            low = self.df_stats.groupby('channel')['percentile_low'].min()
            high = self.df_stats.groupby('channel')['percentile_high'].max()
            self.percentiles_low = low.loc[self.channels].values
            self.percentiles_high = high.loc[self.channels].values

    def generate_dataset(
        self,
        dataset: str,
        dir_output: str,
        n_samples: int = None,
        n_patches: int = None,
    ) -> None:
        """
        Generate complete dataset with patches and statistics.

        Args:
            dataset (str): Name/identifier for the dataset being generated
            dir_output (str): Directory path for storing the generated dataset
            n_samples (int, optional): Number of samples to randomly select for processing. If None, processes all samples.
            n_patches (int, optional): Number of patches to randomly sample from all available patches. If None, uses all patches.
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
        batch_size: int = 32,
        dim: tuple = (128, 128),
        n_channels: int = 4,
        shuffle: bool = True,
        flip: bool = False,
        conditions: np.ndarray | None = None,
        return_conditions: bool = False,
        **kwargs,
    ):
        """
        Initialize SequenceGenerator.

        Args:
            ids (list): List of file paths for patches to load
            batch_size (int): Number of patches per batch. Defaults to 32.
            dim (tuple): Spatial dimensions of patches. Defaults to (128, 128).
            n_channels (int): Number of channels in patches. Defaults to 4.
            shuffle (bool): Whether to shuffle patch order each epoch. Defaults to True.
            flip (bool): Whether to apply random horizontal/vertical flipping augmentation. Defaults to False.
            conditions (array-like, optional): One-hot encoded condition labels for conditional generation. If provided, each
                batch is returned as ``(patches, conditions)``
            return_conditions (bool): Accepted for API symmetry; conditions are returned whenever ``conditions`` is set. Defaults to False.
            **kwargs: Additional arguments passed to parent Sequence class
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

        Returns:
            int: Number of batches that fit in the dataset
        """
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.

        Args:
            index (int): Batch index

        Returns:
            ndarray or tuple: Batch of patches, optionally with conditions
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

        Args:
            ids_temp (list): List of file paths for the current batch

        Returns:
            ndarray: Batch of loaded image patches
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
        Initialize DatasetLoader.

        Args:
            datasets (list): List of dataset names to merge
            dir_datasets (str): Base directory containing dataset subdirectories
            sample_key (str): obs column used to group patches into samples for the train/val split
        """
        self.dir_datasets = dir_datasets
        self.datasets = datasets
        self.sample_key = sample_key
        self.stats_imgs = None
        self.patches = None

    def load_datasets(self) -> None:
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

    def set_train_val_split(self, batch_size: int = 64, split: float = 0.8) -> None:
        """
        Assign each patch to a train or validation split.

        Splits are made at the sample level (grouped by ``sample_key`` and ``dataset``) so all
        patches of a sample land in the same split, then each split is truncated to a whole
        number of batches. Adds ``split`` and ``file_path`` columns to ``self.patches``.

        Args:
            batch_size (int): Batch size used to drop the remainder so each split is batch-aligned. Defaults to 64.
            split (float): Fraction of samples assigned to the training split. Defaults to 0.8.
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
        dim: tuple[int, int] = (128, 128),
        n_channels: int = 4,
        shuffle: bool = True,
        flip: bool = False,
        n_workers: int = 1,
    ) -> tuple:
        """
        Build the training and validation Keras Sequence generators.

        Requires ``set_train_val_split`` to have been called (patches must have ``split`` and
        ``file_path`` columns).

        Args:
            conditions (list of str): obs/patch columns to one-hot encode and feed as conditions. If empty, plain
                (non-conditional) generators are returned
            batch_size (int): Number of patches per batch. Defaults to 64.
            dim (tuple): Spatial (height, width) of patches. Defaults to (128, 128).
            n_channels (int): Number of image channels. Defaults to 4.
            shuffle (bool): Whether to shuffle patch order each epoch. Defaults to True.
            n_workers (int): Number of worker processes for the Keras Sequence. Defaults to 1.

        Returns:
            tuple: ``(train_generator, val_generator, one_hot_encoder)`` if ``conditions`` is non-empty,
                otherwise ``(train_generator, val_generator)``
        """
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
                flip=flip,
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
                flip=flip,
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
