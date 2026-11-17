from __future__ import annotations

import functools
import multiprocessing
import pickle
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from keras.utils import Sequence
from sklearn.preprocessing import OneHotEncoder
from spatialdata import SpatialData
from tqdm import tqdm

#: ``patch_transform(patch, obs_row) -> np.ndarray``: maps a cut (c, h, w) patch and the
#: object's row of the table's ``.obs`` to the (c', h, w) array that is saved / encoded.
PatchTransform = Callable[[np.ndarray, pd.Series], np.ndarray]


def patch_transform_name(patch_transform: PatchTransform | None) -> str | None:
    """
    Return the import path of a patch transform.

    The name is written to a dataset's ``dataset.yaml`` and a model's ``config.yaml``, so
    ``encode`` can tell whether it is given the transform the model was trained with.

    Args:
        patch_transform (callable | None): A function, a ``functools.partial`` of one, or a
            callable object.

    Returns:
        str | None: ``"module.qualname"``; for a partial,
            ``"functools.partial(module.qualname, *args, **kwargs)"``; None if no transform.
    """
    if patch_transform is None:
        return None
    if isinstance(patch_transform, functools.partial):
        parts = [patch_transform_name(patch_transform.func)]
        parts += [repr(a) for a in patch_transform.args]
        parts += [f'{k}={v!r}' for k, v in patch_transform.keywords.items()]
        return f'functools.partial({", ".join(parts)})'
    qualname = getattr(patch_transform, '__qualname__', None)
    if qualname is None:  # a callable instance: name its class
        qualname = type(patch_transform).__qualname__
    module = getattr(patch_transform, '__module__', None)
    if module is None:  # a C-implemented callable: name it alone
        return qualname
    return f'{module}.{qualname}'


def cut_patch(
    img: np.ndarray,
    y: int,
    x: int,
    z: int,
    patch_size: tuple[int, int],
    scale: bool = False,
    percentiles_low: np.ndarray | None = None,
    percentiles_high: np.ndarray | None = None,
    patch_transform: PatchTransform | None = None,
    obs_row: pd.Series | None = None,
) -> np.ndarray:
    """
    Cut one patch centred on (y, x) and optionally scale and transform it.

    Shared by ``PatchGenerator.extract_patch`` and the parallel writer, so both produce
    identical patches.

    Args:
        img (ndarray): (c, y, x) or (c, z, y, x) image.
        y (int): Patch centre row.
        x (int): Patch centre column.
        z (int): Slice to cut from, used only for a 4D image.
        patch_size (tuple of int): Patch size, as in ``PatchGenerator``.
        scale (bool): Whether to scale each channel to [0, 1] with the percentiles.
        percentiles_low (ndarray, optional): Per-channel low, required if ``scale``.
        percentiles_high (ndarray, optional): Per-channel high, required if ``scale``.
        patch_transform (callable, optional): Applied last, as
            ``patch_transform(patch, obs_row)``. Its output is returned.
        obs_row (pd.Series, optional): The object's row of the table's ``.obs``, passed to
            ``patch_transform``.

    Returns:
        ndarray: The (c, h, w) patch, or the transform's output.
    """
    if img.ndim == 4:
        img = img[:, z, ...]
    # patch_size is (height, width). Start half a patch before the centre and take
    # exactly height x width pixels, so odd sizes are not one pixel short (the centre
    # pixel is at index size // 2).
    height, width = patch_size
    y_min = y - height // 2
    x_min = x - width // 2
    img = img[:, y_min : y_min + height, x_min : x_min + width]
    assert img.shape[-2:] == tuple(patch_size)
    if scale:
        if percentiles_high is None or percentiles_low is None:
            raise ValueError('Percentiles need to be provided for scaling')
        img = img.astype(np.float32)
        for i in range(img.shape[0]):
            span = percentiles_high[i] - percentiles_low[i]
            if span == 0:
                # a constant channel: avoid dividing by zero, without changing the
                # stored percentiles (which the next patch would then see)
                span = 1
            img[i] = np.clip((img[i] - percentiles_low[i]) / span, 0, 1)
        img = img.astype(np.float32)
    if patch_transform is not None:
        # copy: unscaled, `img` is a view into the whole image, which an in-place
        # transform would otherwise corrupt for every later patch
        img = np.asarray(patch_transform(img.copy(), obs_row))
        if img.ndim != 3:
            raise ValueError(
                f'patch_transform must return a (c, h, w) array, got shape {img.shape}'
            )

    return img


def _write_sample_patches(
    image,
    files: list[str],
    yxz: np.ndarray,
    obs_rows: pd.DataFrame | None,
    dir_dataset: Path,
    cut_kwargs: dict,
) -> None:
    """
    Write one sample's patches; the task run by each worker of the parallel writer.

    Args:
        image: The sample's image (a lazy ``xarray.DataArray`` from ``sdata.images``,
            loaded here, in the worker).
        files (list of str): Output file name per patch.
        yxz (ndarray): (n, 3) patch centres, in the order of ``files``.
        obs_rows (pd.DataFrame | None): The patches' ``.obs`` rows in the same order, if
            there is a transform.
        dir_dataset (Path): Output directory.
        cut_kwargs (dict): ``patch_size``, ``scale``, ``percentiles_low/high`` and
            ``patch_transform`` for ``cut_patch``.
    """
    if hasattr(image, 'compute'):
        # one worker per sample already: keep dask from starting its own thread pool
        image = image.compute(scheduler='synchronous')
    img = np.asarray(image)
    for i, file in enumerate(files):
        y, x, z = (int(v) for v in yxz[i])
        obs_row = obs_rows.iloc[i] if obs_rows is not None else None
        np.save(
            Path(dir_dataset, file),
            cut_patch(img, y, x, z, obs_row=obs_row, **cut_kwargs),
        )


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
        patch_transform: PatchTransform | None = None,
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
            patch_transform (callable, optional): ``patch_transform(patch, obs_row) -> ndarray``,
                applied to every patch after cutting and (if ``scale``) scaling. ``patch`` is the
                (c, h, w) array and ``obs_row`` the object's row of ``sdata.tables[table_key].obs``.
                Its (c', h, w) output is what is written and encoded, so it may change the
                number of channels and the dtype. For ``n_workers > 1`` it must be picklable: a
                module-level function or a ``functools.partial`` of one. Defaults to None.
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
        self.patch_transform = patch_transform
        image_key_init = '_'.join(
            [
                self.image_key,
                self.sdata.tables[self.table_key].obs[self.sample_key].unique()[0],
            ]
        )
        self.channels = self.sdata.images[image_key_init].coords['c'].values.tolist()
        self.image_size = self.sdata.images[image_key_init].shape[-2:]
        # tuple: extract_patch compares it with an array shape
        self.patch_size = tuple(patch_size)
        self.df_stats = pd.DataFrame()
        self.patches = None
        self.percentiles_low = None
        self.percentiles_high = None
        # per-sample lookups {sample_id: array over channels}, built by
        # get_scaling_percentiles when scale_per_sample is True
        self.sample_percentiles_low = None
        self.sample_percentiles_high = None
        # lookups built by init_patches: patch id -> (y, x, z) and -> row position in
        # the table's .obs; ids are 0..n-1, so both are plain arrays indexed by id
        self._yxz = None
        self._obs_pos = None
        # (patches the index was built for, {sample: row positions in patches})
        self._sample_rows = None

    def init_patches(self) -> None:
        """
        Initialize patch positions from spatial coordinates.

        Extracts spatial coordinates from the data, filters positions that
        would result in patches extending beyond image boundaries, and
        assigns batch IDs. Also builds the lookups ``extract_patch`` uses, so finding a
        patch's position does not scan the whole table.
        """
        obs = self.sdata.tables[self.table_key].obs
        self.patches = pd.DataFrame(
            self.sdata.tables[self.table_key].obsm[self.spatial_key],
            columns=['y', 'x', 'z'],
            index=obs.index,
        )
        self.patches[self.sample_key] = obs[self.sample_key]
        # positional, so the obs row is found even if obs names repeat; removed below
        self.patches['_obs_pos'] = np.arange(len(obs))
        # carry user-requested obs columns so they can be used as conditions
        for key in self.metadata_keys:
            if key in (self.sample_key, 'x', 'y', 'z'):
                continue
            self.patches[key] = self.sdata.tables[self.table_key].obs[key]
        # round to integer
        self.patches['x'] = self.patches['x'].astype(int)
        self.patches['y'] = self.patches['y'].astype(int)

        # filter x and y that are within image boundaries when patch size is added
        # (per-sample, since images may differ in size). patch_size is (height, width):
        # the height bounds y, the width bounds x.
        height, width = self.patch_size
        filtered = []
        for sample_id, grp in self.patches.groupby(self.sample_key):
            image_size = self.sdata.images[
                '_'.join([self.image_key, str(sample_id)])
            ].shape[-2:]
            x_min, x_max = (
                width // 2,
                image_size[1] - (width - width // 2),
            )
            y_min, y_max = (
                height // 2,
                image_size[0] - (height - height // 2),
            )
            grp = grp[
                (grp['x'] >= x_min)
                & (grp['x'] < x_max)
                & (grp['y'] >= y_min)
                & (grp['y'] < y_max)
            ]
            filtered.append(grp)
        self.patches = pd.concat(filtered)
        self._obs_pos = self.patches.pop('_obs_pos').to_numpy()

        self.patches['id'] = np.arange(0, len(self.patches))
        self._yxz = self.patches[['y', 'x', 'z']].to_numpy()
        self._sample_rows = None

    def _rows_of_sample(self, sample_id: str) -> np.ndarray:
        """
        Return the row positions of a sample's patches in ``self.patches``.

        Grouping once and caching, instead of comparing the whole sample column for every
        sample, keeps per-sample access linear overall. The cache is rebuilt whenever
        ``self.patches`` is replaced (e.g. subsampled by ``n_patches``).

        Args:
            sample_id (str or int): Sample identifier

        Returns:
            ndarray: Row positions, in table order (empty if the sample has no patches).
        """
        if self._sample_rows is None or self._sample_rows[0] is not self.patches:
            groups = self.patches.groupby(
                self.sample_key, sort=False, observed=True
            ).indices
            self._sample_rows = (self.patches, groups)
        return self._sample_rows[1].get(sample_id, np.array([], dtype=int))

    def _sample_scaling(self, sample_id: str) -> tuple:
        """
        Return the (low, high) percentiles that scale a sample's patches.

        Args:
            sample_id (str or int): Sample identifier

        Returns:
            tuple: (percentiles_low, percentiles_high). Per-sample arrays are copies, so
                the stored per-sample lookups cannot be changed through them.
        """
        if self.scale and self.scale_per_sample:
            if self.sample_percentiles_low is None:
                raise ValueError(
                    'Per-sample scaling requested but percentiles not computed; '
                    'call get_scaling_percentiles() first'
                )
            return (
                self.sample_percentiles_low[sample_id].copy(),
                self.sample_percentiles_high[sample_id].copy(),
            )
        return self.percentiles_low, self.percentiles_high

    def extract_patch(
        self,
        img: np.ndarray,
        id: int,
    ) -> np.ndarray:
        """
        Extract a patch from an image centered on specified coordinates.

        The patch is cut, scaled if ``self.scale``, and passed through
        ``self.patch_transform`` if one is set (see ``cut_patch``).

        Args:
            img (ndarray): Input image array
            id (int): Batch ID corresponding to patch position

        Returns:
            ndarray: Extracted image patch
        """
        if self._yxz is None:
            raise ValueError('Patch positions not set; call init_patches() first')
        y, x, z = (int(v) for v in self._yxz[id])
        obs_row = None
        if self.patch_transform is not None:
            obs_row = self.sdata.tables[self.table_key].obs.iloc[self._obs_pos[id]]
        return cut_patch(
            img,
            y,
            x,
            z,
            self.patch_size,
            scale=self.scale,
            percentiles_low=self.percentiles_low,
            percentiles_high=self.percentiles_high,
            patch_transform=self.patch_transform,
            obs_row=obs_row,
        )

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
        # Deviation from the channel's own median, taken over all of its slices. Centring
        # on one median across every channel measures the gap between channels rather than
        # the spread within one; centring per slice hides drift from slice to slice.
        channel_median = np.median(imgs, axis=(1, -2, -1))
        mad = np.median(
            np.abs(imgs - channel_median[:, np.newaxis, np.newaxis, np.newaxis]),
            axis=(-2, -1),
        )
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

        if len(self._rows_of_sample(sample_id)) > 0:
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
        df_patches_sample = self.patches.iloc[self._rows_of_sample(sample_id)]
        img_key_sample = '_'.join([self.image_key, sample_id])
        img = np.asarray(self.sdata.images[img_key_sample])
        if self.scale and self.scale_per_sample:
            # activate this sample's own scaling range for extract_patch
            self.percentiles_low, self.percentiles_high = self._sample_scaling(
                sample_id
            )
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

    def _sample_task(self, sample_id: str) -> tuple:
        """
        Build the arguments of ``_write_sample_patches`` for one sample.

        Only this sample's rows, scaling range and (still lazy) image go to the worker,
        never the whole SpatialData or patch table.

        Args:
            sample_id (str or int): Sample identifier

        Returns:
            tuple: Positional arguments for ``_write_sample_patches``.
        """
        df_patches_sample = self.patches.iloc[self._rows_of_sample(sample_id)]
        ids = df_patches_sample['id'].to_numpy()
        obs_rows = None
        if self.patch_transform is not None:
            obs_rows = self.sdata.tables[self.table_key].obs.iloc[self._obs_pos[ids]]
        low, high = self._sample_scaling(sample_id)
        cut_kwargs = {
            'patch_size': self.patch_size,
            'scale': self.scale,
            'percentiles_low': low,
            'percentiles_high': high,
            'patch_transform': self.patch_transform,
        }
        return (
            self.sdata.images[f'{self.image_key}_{sample_id}'],
            df_patches_sample['file'].tolist(),
            self._yxz[ids],
            obs_rows,
            self.dir_dataset,
            cut_kwargs,
        )

    def _write_patches_parallel(self, n_workers: int) -> None:
        """
        Write every sample's patches with a pool of ``n_workers`` processes.

        One task per sample, so memory is about ``n_workers`` times the largest image.
        Workers are spawned, not forked, which behaves the same on Linux and macOS and is
        safe after TensorFlow has started its threads. The output is identical to
        ``n_workers=1``: both go through ``cut_patch`` with the same inputs.

        Args:
            n_workers (int): Number of worker processes.

        Raises:
            ValueError: If ``patch_transform`` cannot be pickled.
        """
        if self.patch_transform is not None:
            try:
                pickle.dumps(self.patch_transform)
            except Exception as err:
                raise ValueError(
                    f'patch_transform {patch_transform_name(self.patch_transform)!r} '
                    'cannot be sent to worker processes. With n_workers > 1 it must be '
                    'a module-level function or a functools.partial of one (not a '
                    'lambda or a nested function).'
                ) from err
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=min(n_workers, len(self.samples)), mp_context=context
        ) as executor:
            futures = [
                executor.submit(_write_sample_patches, *self._sample_task(sample))
                for sample in self.samples
            ]
            for future in tqdm(
                as_completed(futures), total=len(futures), desc='Writing patches'
            ):
                future.result()

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
        dir_dataset: str | Path,
        n_samples: int = None,
        n_patches: int = None,
        n_workers: int = 1,
    ) -> None:
        """
        Generate complete dataset with patches and statistics.

        Writes ``patches.csv``, one ``.npy`` per patch, ``dataset.yaml`` (see
        ``write_dataset_config``) and, only if ``self.scale``, ``stats.csv``. Without
        scaling the image statistics are not needed, and computing them reads every image
        in full, so they are skipped.

        Args:
            dataset (str): Name/identifier for the dataset being generated
            dir_dataset (str | Path): Directory to write this dataset into. Already fully
                resolved by the caller (``Phenocoder.dir_dataset``) -- the dataset name is
                *not* appended here.
            n_samples (int, optional): Number of samples to randomly select for processing. If None, processes all samples.
            n_patches (int, optional): Number of patches to randomly sample from all available patches. If None, uses all patches.
            n_workers (int): Number of processes writing patches, one sample at a time each.
                The files are identical to ``n_workers=1``; ``patches.csv`` is written once,
                by this process. Defaults to 1 (no worker processes).
        """
        self.dir_dataset = Path(dir_dataset)
        self.dir_dataset.mkdir(exist_ok=True, parents=True)
        self.samples = self.sdata.tables[self.table_key].obs[self.sample_key].unique()
        if n_samples is not None:
            self.samples = np.random.choice(self.samples, n_samples, replace=False)
        self.init_patches()
        if self.scale:
            [
                self.generate_image_stats(sample)
                for sample in tqdm(self.samples, desc='Generating image statistics')
            ]
            self.df_stats.to_csv(Path(self.dir_dataset, 'stats.csv'), index=False)
            self.get_scaling_percentiles()
        else:
            # Regenerating an unscaled dataset over a scaled one: the old stats.csv
            # describes patches that no longer exist, and encode(scale=True) would
            # read it and scale with it instead of reporting that it is missing.
            Path(self.dir_dataset, 'stats.csv').unlink(missing_ok=True)
        if n_patches is not None:
            self.patches = self.patches.sample(n_patches, replace=False)
            self.samples = self.patches[self.sample_key].unique()
        self.patches['file'] = (
            self.patches[self.sample_key].astype(str)
            + '_'
            + self.patches['id'].astype(str)
            + '.npy'
        )
        self.patches['dataset'] = dataset
        self.patches.to_csv(Path(self.dir_dataset, 'patches.csv'))
        self.write_dataset_config(dataset)
        if n_workers > 1 and len(self.samples) > 1:
            self._write_patches_parallel(n_workers)
        else:
            [
                self.write_patches(id)
                for id in tqdm(self.samples, desc='Writing patches')
            ]

    def write_dataset_config(self, dataset: str) -> None:
        """
        Write ``dataset.yaml``: the settings the dataset's patches were made with.

        ``DatasetLoader`` reads it back, so ``initialize_model`` can record the
        ``patch_transform`` name in the model's ``config.yaml``.

        Args:
            dataset (str): Name of the dataset, written into the file.
        """
        config = {
            'dataset': dataset,
            # plain Python types: safe_dump refuses numpy scalars
            'patch_size': [int(p) for p in self.patch_size],
            'scale': bool(self.scale),
            'scale_percentile': float(self.scale_percentile),
            'scale_per_sample': bool(self.scale_per_sample),
            'patch_transform': patch_transform_name(self.patch_transform),
        }
        Path(self.dir_dataset, 'dataset.yaml').write_text(yaml.safe_dump(config))


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
        # float32: the model computes in float32, and numpy's float64 default would double
        # the batch memory for nothing
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
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

    def __init__(self, datasets: list, dir_datasets: dict, sample_key: str):
        """
        Initialize DatasetLoader.

        Args:
            datasets (list): List of dataset names to merge
            dir_datasets (dict): Mapping of dataset name -> directory holding that dataset's
                ``patches.csv`` / ``stats.csv`` / patch ``.npy`` files. Resolved by the caller
                (``Phenocoder.dir_dataset``) so datasets may live outside the project root.
            sample_key (str): obs column used to group patches into samples for the train/val split
        """
        self.dir_datasets = {name: Path(path) for name, path in dir_datasets.items()}
        self.datasets = datasets
        self.sample_key = sample_key
        self.stats_imgs = None
        self.patches = None
        # dataset name -> contents of its dataset.yaml (absent for datasets generated
        # before dataset.yaml existed)
        self.dataset_configs: dict[str, dict] = {}

    def load_datasets(self) -> None:
        """
        Loads and merge statistics from all specified datasets.

        Combines the ``patches.csv`` files of every dataset, and their ``stats.csv`` where
        present: datasets generated with ``scale=False`` have none. Also reads each
        dataset's ``dataset.yaml``, if it has one.
        """
        stats = []
        self.patches = []
        self.dataset_configs = {}
        for dataset in self.datasets:
            dir_dataset = self.dir_datasets[dataset]
            if Path(dir_dataset, 'stats.csv').exists():
                stats.append(pd.read_csv(Path(dir_dataset, 'stats.csv')))
            self.patches.append(pd.read_csv(Path(dir_dataset, 'patches.csv')))
            path_config = Path(dir_dataset, 'dataset.yaml')
            if path_config.exists():
                self.dataset_configs[dataset] = yaml.safe_load(path_config.read_text())
        self.stats = pd.concat(stats) if stats else None
        self.patches = pd.concat(self.patches)

    def get_patch_transform(self) -> str | None:
        """
        Return the name of the ``patch_transform`` the loaded datasets were made with.

        Requires ``load_datasets`` to have been called. Datasets without a
        ``dataset.yaml`` count as made without a transform.

        Returns:
            str | None: The transform's name (see ``patch_transform_name``), or None.

        Raises:
            ValueError: If the datasets were made with different transforms; their patches
                would not mean the same thing to one model.
        """
        names = {
            dataset: self.dataset_configs.get(dataset, {}).get('patch_transform')
            for dataset in self.datasets
        }
        if len(set(names.values())) > 1:
            raise ValueError(
                f'The datasets were made with different patch transforms: {names}. '
                'Train one model per transform, or regenerate them with the same one.'
            )
        return next(iter(names.values()), None)

    def set_train_val_split(self, batch_size: int = 64, split: float = 0.8) -> None:
        """
        Assign each patch to a train or validation split.

        Splits are made at the sample level (grouped by ``sample_key`` and ``dataset``) so all
        patches of a sample land in the same split, then each split is truncated to a whole
        number of batches. Adds ``split`` and ``file_path`` columns to ``self.patches``.

        Args:
            batch_size (int): Batch size used to drop the remainder so each split is batch-aligned. Defaults to 64.
            split (float): Fraction of samples assigned to the training split. Defaults to 0.8.

        Raises:
            ValueError: If a split holds fewer than ``batch_size`` patches, leaving it with no
                complete batch.
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
        # Drop each split's remainder so it is a whole number of batches. SequenceGenerator
        # floors len(ids)/batch_size and so never yields a partial batch anyway; this keeps
        # `patches` consistent with what actually gets trained on.
        #
        # NB `iloc[:n - n % batch_size]`, not `iloc[:-(n % batch_size)]`: when the remainder
        # is 0 the latter is `iloc[:0]`, which silently discards the entire split -- the
        # exactly-batch-aligned case became the worst case.
        before = self.patches.groupby('split').size()
        counts = self.patches.groupby('split')['split'].transform('size')
        keep = self.patches.groupby('split').cumcount() < (counts - counts % batch_size)
        self.patches = self.patches[keep].reset_index(drop=True)

        # A split with fewer patches than batch_size truncates to nothing. Left alone this
        # surfaces much later as an opaque pandas or Keras error on an empty generator, so
        # say what happened here.
        after = self.patches.groupby('split').size()
        starved = {
            name: int(before.get(name, 0))
            for name in ('train', 'val')
            if int(after.get(name, 0)) == 0
        }
        if starved:
            raise ValueError(
                f'batch_size={batch_size} leaves no complete batch for split(s) '
                f'{starved} (name: patches available). Training would receive no data. '
                f'Use a smaller batch_size, or generate more patches.'
            )
        # expand files to complete paths
        self.patches['file_path'] = self.patches.apply(
            lambda x: Path(self.dir_datasets[x['dataset']], x['file']), axis=1
        )

    def get_generators(
        self,
        conditions: list[str] | None = None,
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
            conditions (list of str, optional): obs/patch columns to one-hot encode and feed as
                conditions. If empty or None, plain (non-conditional) generators are returned
            batch_size (int): Number of patches per batch. Defaults to 64.
            dim (tuple): Spatial (height, width) of patches. Defaults to (128, 128).
            n_channels (int): Number of image channels. Defaults to 4.
            shuffle (bool): Whether to shuffle patch order each epoch. Defaults to True.
            flip (bool): Whether to randomly flip training patches. Validation patches are
                never flipped, so the validation loss is measured on the data as it is.
                Defaults to False.
            n_workers (int): Number of worker processes for the Keras Sequence. Defaults to 1.

        Returns:
            tuple: ``(train_generator, val_generator, one_hot_encoder)`` if ``conditions`` is non-empty,
                otherwise ``(train_generator, val_generator)``
        """
        enc = None
        cond = None
        if conditions:
            enc = OneHotEncoder()
            cond = enc.fit_transform(self.patches[conditions]).toarray()

        def make_generator(split: str) -> SequenceGenerator:
            in_split = (self.patches['split'] == split).to_numpy()
            return SequenceGenerator(
                self.patches[in_split]['file_path'].values,
                conditions=cond[in_split] if cond is not None else None,
                batch_size=batch_size,
                dim=dim,
                n_channels=n_channels,
                shuffle=shuffle,
                flip=flip and split == 'train',
                return_conditions=cond is not None,
                workers=n_workers,
            )

        generator_train = make_generator('train')
        generator_val = make_generator('val')
        if conditions:
            return generator_train, generator_val, enc
        return generator_train, generator_val
