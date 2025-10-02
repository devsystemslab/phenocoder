import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import Sequence
from skimage import io
from skimage.util import view_as_windows
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


class DatasetGenerator:
    """
    DatasetGenerator
    """

    def __init__(
        self,
        dir_input,
        dir_output,
        max_workers=12,
        mode='grid',
        dir_segmented=None,
        channels=['01', '02', '03', '04'],
    ):
        """
        Initialize DatasetGenerator
        :param dir_input:
        :param dir_output:
        :param max_workers:
        :param dir_segmented:
        """
        self.max_workers = max_workers
        self.mode = mode
        self.dir_input = dir_input
        self.dir_output = dir_output
        self.dir_segmented = dir_segmented
        self.dir_dataset = os.path.join(dir_output, 'dataset')
        self.df_images = None
        self.df_stats = pd.DataFrame()
        self.df_stats_patches = pd.DataFrame()
        self.df_stats_patches_sampled = None
        self.image_size = (3814, 3814)
        self.patch_size = (128, 128)
        self.patch_positions = None
        self.channels = channels
        self.ids = None

    def get_metadata(self, path) -> pd.DataFrame:
        """
        Get metadata
        :param path:
        :return:
        """
        images = os.listdir(path)
        images = [image for image in images if '.tif' in image]
        regex = r'_(?P<well_id>[A-Z]\d{2})_T(?P<time_point>\d{4})F(?P<field_id>\d{3})L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})Z(?P<z_stack_id>\d{2})C(?P<channel_id>\d{2})\.tif$'
        df = pd.DataFrame({'file': images, 'dir_images': str(path)})
        df = df.join(df['file'].str.extractall(regex).groupby(level=0).last())
        df['id'] = df['well_id'] + '_' + df['z_stack_id']
        # remove rows that have nan in any column
        df = df[~df.isna().any(axis=1)]
        return df

    def init_metadata(self):
        """
        Initialize metadata
        :return:
        """
        if self.dir_input is not None:
            self.df_images = self.get_metadata(self.dir_input)

    def init_output(self):
        """
        Initialize output directories
        :return:
        """
        os.makedirs(self.dir_output, exist_ok=True)
        os.makedirs(self.dir_dataset, exist_ok=True)

    def init_ids(self, sampling_frac=None, qc_path=None):
        """
        Initialize ids
        :param sampling_frac:
        :return:
        """
        # filter ids from excluded wells
        if qc_path is not None:
            df = pd.read_csv(qc_path)
            bad_wells = df[df['decision'] == 'bad']['well_id'].tolist()
            print(bad_wells)
            self.df_images = self.df_images[~self.df_images['well_id'].isin(bad_wells)]

        self.ids = self.df_images['id'].unique()
        if sampling_frac is not None:
            self.ids = np.random.choice(
                self.ids, int(sampling_frac * len(self.ids)), replace=False
            )

    def init_patch_positions(self):
        """
        Initialize patch positions from image size and patch size
        :return:
        """
        patch_positions = []
        self.patch_positions = []
        if self.mode == 'grid':
            for i in range(
                0, self.image_size[0] - self.patch_size[0], self.patch_size[0]
            ):
                for j in range(
                    0, self.image_size[1] - self.patch_size[1], self.patch_size[1]
                ):
                    patch_positions.append((i, j))
            for i in self.ids:
                df = pd.DataFrame(patch_positions, columns=['x', 'y']).assign(id=i)
                self.patch_positions.append(df)
            self.patch_positions = pd.concat(self.patch_positions).reset_index(
                drop=True
            )

        if self.mode == 'segmented':
            for i in tqdm(self.ids, desc='Reading positions'):
                file = os.path.join(self.dir_segmented, f'{i}.csv')
                if os.path.exists(file):
                    df = pd.read_csv(file).assign(id=i)
                    self.patch_positions.append(df)
            self.patch_positions = pd.concat(
                [df for df in self.patch_positions if not df.empty]
            )
            self.patch_positions = self.patch_positions.rename(
                columns={'centroid-0': 'y', 'centroid-1': 'x'}
            )
            # round to integer
            self.patch_positions['x'] = self.patch_positions['x'].astype(int)
            self.patch_positions['y'] = self.patch_positions['y'].astype(int)
            # drop other columns than id, x and y
            self.patch_positions = self.patch_positions[['id', 'x', 'y']]
            # filter x and y that are within image boundaries when patch size is added
            x_min, x_max = (
                self.patch_size[0] // 2,
                self.image_size[1] - self.patch_size[0] // 2,
            )
            y_min, y_max = (
                self.patch_size[0] // 2,
                self.image_size[0] - self.patch_size[0] // 2,
            )
            self.patch_positions = self.patch_positions[
                (self.patch_positions['x'] >= x_min)
                & (self.patch_positions['x'] <= x_max)
                & (self.patch_positions['y'] >= y_min)
                & (self.patch_positions['y'] <= y_max)
            ]

        self.patch_positions['batch_id'] = np.arange(0, len(self.patch_positions))

    def load_image(self, file):
        """
        Load image
        :param file:
        :return:
        """
        image_path = os.path.join(self.dir_input, file)
        image = io.imread(image_path)
        return image

    def extract_patch(self, img, batch_id):
        """
        Extract patch from image
        :param img:
        :param batch_id:
        :return:
        """
        if self.mode == 'grid':
            x = int(
                self.patch_positions[self.patch_positions['batch_id'] == batch_id][
                    'x'
                ].iloc[0]
            )
            x_max = x + self.patch_size[0]
            y = int(
                self.patch_positions[self.patch_positions['batch_id'] == batch_id][
                    'y'
                ].iloc[0]
            )
            y_max = int(y) + self.patch_size[1]
            img = img[y:y_max, x:x_max, :]

        if self.mode == 'segmented':
            # extract patch centered on x and y
            x = int(
                self.patch_positions[self.patch_positions['batch_id'] == batch_id][
                    'x'
                ].iloc[0]
            )
            y = int(
                self.patch_positions[self.patch_positions['batch_id'] == batch_id][
                    'y'
                ].iloc[0]
            )
            x_max = x + self.patch_size[0] // 2
            y_max = y + self.patch_size[1] // 2
            x_min = x - self.patch_size[0] // 2
            y_min = y - self.patch_size[1] // 2
            img = img[y_min:y_max, x_min:x_max, :]

        assert img.shape[:-1] == self.patch_size
        return img

    def __get_image_stats__(self, imgs, id, id_name):
        """
        Get image statistics
        :param imgs:
        :param id:
        :param id_name:
        :return:
        """
        mean = np.mean(imgs, axis=(0, 1))
        std = np.std(imgs, axis=(0, 1))
        median = np.median(imgs, axis=(0, 1))
        mad = np.median(np.abs(imgs - np.median(imgs)), axis=(0, 1))
        max = np.max(imgs, axis=(0, 1))
        min = np.min(imgs, axis=(0, 1))
        quantile_low = np.percentile(imgs, 1, axis=(0, 1))
        quantile_high = np.percentile(imgs, 99, axis=(0, 1))
        df = pd.DataFrame(
            {
                id_name: id,
                'channel': self.channels,
                'mean': mean,
                'std': std,
                'quantile_high': quantile_high,
                'quantile_low': quantile_low,
                'median': median,
                'mad': mad,
                'max': max,
                'min': min,
            }
        )
        return df

    def generate_patch_stats(self, sample_id):
        """
        Generate patch statistics
        :param sample_id:
        :return:
        """
        df_batch_positions = self.patch_positions[
            self.patch_positions['id'] == sample_id
        ]
        if len(df_batch_positions) > 0:
            df_batch_images = self.df_images[self.df_images['id'] == sample_id]
            # filter for channels
            df_batch_images = df_batch_images[
                df_batch_images['channel_id'].isin(self.channels)
            ]
            # sort by channel
            df_batch_images = df_batch_images.sort_values(
                by=['channel_id'], ascending=True
            )
            # load images
            imgs = np.asarray(
                [self.load_image(file) for file in df_batch_images['file']]
            )
            imgs = np.moveaxis(imgs, 0, -1)
            df_stat = self.__get_image_stats__(imgs, sample_id, 'id')
            self.df_stats = pd.concat([self.df_stats, df_stat])
            patches = [
                self.extract_patch(imgs, batch_id)
                for batch_id in df_batch_positions['batch_id']
            ]
            # get patch stats
            df_stat_patches = pd.concat(
                [
                    self.__get_image_stats__(patch, batch_id, 'batch_id')
                    for patch, batch_id in zip(patches, df_batch_positions['batch_id'])
                ]
            )
            df_stat_patches['id'] = sample_id
            # left merge with df_batch_positions
            df_stat_patches = df_stat_patches.merge(
                df_batch_positions, on=['batch_id', 'id'], how='left'
            )
            df_stat_patches['file'] = (
                df_stat_patches['id'].astype(str)
                + '_'
                + df_stat_patches['batch_id'].astype(str)
                + '.npy'
            )
            # add to self.df_stats_patches
            self.df_stats_patches = pd.concat([self.df_stats_patches, df_stat_patches])

    def write_patches(self, sample_id):
        """
        Write patches to disk
        :param sample_id:
        :return:
        """
        # get all files that need to be written
        df_patches_sample = self.df_stats_patches_sampled[
            self.df_stats_patches_sampled['id'] == sample_id
        ]
        df_images_sample = self.df_images[self.df_images['id'] == sample_id]
        # filter for channels
        df_batch_images = df_images_sample[
            df_images_sample['channel_id'].isin(self.channels)
        ]
        # sort by channel
        df_batch_images = df_batch_images.sort_values(by=['channel_id'], ascending=True)
        # load images
        imgs = np.asarray([self.load_image(file) for file in df_batch_images['file']])
        imgs = np.moveaxis(imgs, 0, -1)
        patches = [
            self.extract_patch(imgs, batch_id)
            for batch_id in df_patches_sample['batch_id']
        ]
        [
            np.save(
                os.path.join(self.dir_dataset, f'{sample_id}_{batch_id}.npy'), patch
            )
            for batch_id, patch in zip(df_patches_sample['batch_id'], patches)
        ]

    def generate_dataset(
        self,
        sampling_frac=None,
        n_patches=None,
        n_bins=20,
        per_channel=False,
        qc_path=None,
    ):
        """
        Generate dataset
        :param sampling_frac:
        :param n_patches:
        :param n_bins:
        :param per_channel:
        :return:
        """
        self.init_metadata()
        self.init_output()
        self.init_ids(sampling_frac=sampling_frac, qc_path=qc_path)
        self.init_patch_positions()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(
                tqdm(
                    executor.map(self.generate_patch_stats, self.ids),
                    desc='Generating patch statistics',
                    total=len(self.ids),
                )
            )
        self.sample_patches(n_patches=n_patches, n_bins=n_bins, per_channel=per_channel)
        # set ids to sampled ids
        self.ids = self.df_stats_patches_sampled['id'].unique()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            list(
                tqdm(
                    executor.map(self.write_patches, self.ids),
                    desc='Writing sampled patches',
                    total=len(self.ids),
                )
            )

    def sample_patches(self, n_patches=None, n_bins=20, per_channel=False):
        """
        Sample patches from imgs_train for uniform distribution of mean intensity
        :param n_patches:
        :param n_bins:
        :param per_channel:
        :return:
        """
        if self.mode == 'grid':
            if n_patches is None:
                n_patches = len(self.df_stats_patches['file'].unique())
            indexes_sampled = []
            if per_channel:
                for channel_id in self.df_stats_patches['channel'].unique():
                    df = self.df_stats_patches[
                        self.df_stats_patches['channel'] == channel_id
                    ]
                    mean_intensity = df['mean']
                    # get quantiles
                    quantiles = np.histogram(mean_intensity, bins=n_bins)[1]
                    for i in tqdm(range(1, len(quantiles)), desc='Sampling patches'):
                        idx = np.where(
                            (mean_intensity >= quantiles[i - 1])
                            & (mean_intensity < quantiles[i])
                        )[0]
                        if len(idx) == 0:
                            continue
                        weight = 1 / (len(idx) / len(mean_intensity))
                        n_patches_quant = int(n_patches * weight)
                        # shuffle idx
                        np.random.shuffle(idx)
                        idx = idx[:n_patches_quant]
                        indexes_sampled.extend(df.iloc[idx].index.tolist())

            else:
                mean_intensity = self.df_stats_patches['mean']
                # get quantiles
                quantiles = np.histogram(mean_intensity, bins=n_bins)[1]
                for i in tqdm(range(1, len(quantiles)), desc='Sampling patches'):
                    idx = np.where(
                        (mean_intensity >= quantiles[i - 1])
                        & (mean_intensity < quantiles[i])
                    )[0]
                    if len(idx) == 0:
                        continue
                    weight = 1 / (len(idx) / len(mean_intensity))
                    n_patches_quant = int(n_patches * weight)
                    # shuffle idx
                    np.random.shuffle(idx)
                    idx = idx[:n_patches_quant]
                    indexes_sampled.extend(idx)
            # unique indexes
            indexes_sampled = list(set(indexes_sampled))
            self.df_stats_patches_sampled = self.df_stats_patches.iloc[
                indexes_sampled
            ].copy()
        if self.mode == 'segmented':
            if n_patches is None or n_patches > len(self.df_stats_patches):
                self.df_stats_patches_sampled = self.df_stats_patches.copy()
            else:
                self.df_stats_patches_sampled = self.df_stats_patches.sample(
                    n_patches, replace=False
                )

    def save_stats(self):
        """
        Save statistics
        :return:
        """
        self.df_stats.to_csv(os.path.join(self.dir_output, 'stats.csv'), index=False)
        self.df_stats_patches.to_csv(
            os.path.join(self.dir_output, 'stats_patches.csv'), index=False
        )
        if self.df_stats_patches_sampled is not None:
            self.df_stats_patches_sampled.to_csv(
                os.path.join(self.dir_output, 'stats_patches_sampled.csv'), index=False
            )

    def load_stats(self):
        """
        Load statistics
        :return:
        """
        file_stats = os.path.join(self.dir_output, 'stats.csv')
        file_stats_patches = os.path.join(self.dir_output, 'stats_patches.csv')
        file_stats_patches_sampled = os.path.join(
            self.dir_output, 'stats_patches_sampled.csv'
        )
        if os.path.exists(file_stats):
            self.df_stats = pd.read_csv(file_stats)
        if os.path.exists(file_stats_patches):
            self.df_stats_patches = pd.read_csv(file_stats_patches)
        if os.path.exists(file_stats_patches_sampled):
            self.df_stats_patches_sampled = pd.read_csv(file_stats_patches_sampled)

    def plot_stats(self, sampled=False):
        """
        Plot statistics
        :param sampled:
        :return:
        """
        # Group by channel
        if sampled and self.df_stats_patches_sampled is not None:
            df_grouped = self.df_stats_patches_sampled.copy()
        else:
            df_grouped = self.df_stats_patches.copy()
        df_grouped = df_grouped.groupby('channel')

        # Define colors for each channel
        colors = ['blue', 'green', 'red', 'purple']

        # Plot histograms for each channel on the same plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Channel Statistics')

        for (channel, group), color in zip(df_grouped, colors):
            axes[0, 0].hist(
                group['mean'],
                bins=30,
                alpha=0.7,
                label=f'Channel {channel}',
                color=color,
            )
            axes[0, 0].set_title('Mean')

            axes[0, 1].hist(
                group['std'],
                bins=30,
                alpha=0.7,
                label=f'Channel {channel}',
                color=color,
            )
            axes[0, 1].set_title('Standard Deviation')

            axes[1, 0].hist(
                group['quantile_high'],
                bins=30,
                alpha=0.7,
                label=f'Channel {channel}',
                color=color,
            )
            axes[1, 0].set_title('Quantile High')

            axes[1, 1].hist(
                group['quantile_low'],
                bins=30,
                alpha=0.7,
                label=f'Channel {channel}',
                color=color,
            )
            axes[1, 1].set_title('Quantile Low')

        for ax in axes.flat:
            ax.legend()
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


class DatasetMerger:
    """
    DatasetMerger
    """

    def __init__(self, datasets: list, dir_datasets: str):
        """
        Initialize DatasetMerger
        :param datasets:
        :param dir_datasets:
        """
        self.dir_datasets = dir_datasets
        self.datasets = datasets
        self.stats = None
        self.patch_stats = None

    def merge_datasets(self):
        """
        Merge datasets
        :return:
        """
        self.stats = []
        self.patch_stats = []
        for dataset in self.datasets:
            self.stats.append(
                pd.read_csv(
                    os.path.join(self.dir_datasets, dataset, 'stats.csv')
                ).assign(dataset=dataset)
            )
            self.patch_stats.append(
                pd.read_csv(
                    os.path.join(
                        self.dir_datasets, dataset, 'stats_patches_sampled.csv'
                    )
                ).assign(dataset=dataset, dir_datasets=self.dir_datasets)
            )
        self.stats = pd.concat(self.stats)
        self.patch_stats = pd.concat(self.patch_stats)
        # generate file paths
        self.patch_stats['file_path'] = self.patch_stats.apply(
            lambda x: os.path.join(
                x['dir_datasets'], x['dataset'], 'dataset', x['file']
            ),
            axis=1,
        )

    def get_files(self):
        """
        Get files
        :return:
        """
        return self.patch_stats['file_path'].unique().tolist()

    def get_scaling_stats(self):
        """
        Get scaling statistics
        :return:
        """
        quantiles_low = self.stats.groupby('channel')['quantile_low'].min()
        quantiles_high = self.stats.groupby('channel')['quantile_high'].max()
        return quantiles_low, quantiles_high


class NucleiPatchGenerator:
    """
    NucleiPatchGenerator
    """

    def __init__(self, dir_images):
        """
        Initialize NucleiPatchGenerator
        :param dir_images:
        """
        self.df_images = self.get_metadata(dir_images)
        self.patch_size = (128, 128)
        self.image_size = (3814, 3814)

    def get_metadata(self, path) -> pd.DataFrame:
        """
        Get metadata
        :param path:
        :return:
        """
        images = os.listdir(path)
        images = [image for image in images if '.tif' in image]
        regex = r'_(?P<well_id>[A-Z]\d{2})_T(?P<time_point>\d{4})F(?P<field_id>\d{3})L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})Z(?P<z_stack_id>\d{2})C(?P<channel_id>\d{2})\.tif$'
        df = pd.DataFrame({'file': images, 'dir_images': str(path)})
        df = df.join(df['file'].str.extractall(regex).groupby(level=0).last())
        df['id'] = df['well_id'] + '_' + df['z_stack_id']
        # remove rows that have nan in any column
        df = df[~df.isna().any(axis=1)]
        return df

    def load_image(self, file):
        """
        Load image
        :param file:
        :return:
        """
        image = io.imread(file)
        return image

    def extract_patch_from_x_y(self, img, x, y):
        """
        Extract patch from x and y
        :param img:
        :param x:
        :param y:
        :return:
        """
        x_max = x + self.patch_size[0]
        y_max = y + self.patch_size[1]
        img = img[y:y_max, x:x_max, :]
        return img

    def get_patches(
        self,
        well_id,
        df_nuclei,
        scale=True,
        quantiles_high=None,
        quantiles_low=None,
        channels=['01', '02', '03', '04'],
    ):
        """
        Get patches
        :param well_id:
        :param df_nuclei:
        :param scale:
        :param quantiles_high:
        :param quantiles_low:
        :return:
        """
        df_well_images = self.df_images[self.df_images['well_id'] == well_id]
        df_well_nuclei = df_nuclei[df_nuclei['well_id'] == well_id]
        # adjust x,y to patch size
        df_well_nuclei['x'] = (
            df_well_nuclei['centroid-1'] - self.patch_size[1] / 2
        ).astype(int)
        df_well_nuclei['y'] = (
            df_well_nuclei['centroid-0'] - self.patch_size[0] / 2
        ).astype(int)
        # filter out patches that exceed image boundaries
        x_min, x_max = 0, self.image_size[1] - self.patch_size[1]
        y_min, y_max = 0, self.image_size[0] - self.patch_size[0]
        df_well_nuclei = df_well_nuclei[
            (df_well_nuclei['x'] >= x_min)
            & (df_well_nuclei['x'] <= x_max)
            & (df_well_nuclei['y'] >= y_min)
            & (df_well_nuclei['y'] <= y_max)
        ]
        patches = []
        df = []
        # load images
        for z in tqdm(
            df_well_images['z_stack_id'].unique().tolist(),
            desc=f'Generating patches - {well_id}',
            total=df_well_images['z_stack_id'].unique().shape[0],
        ):
            df_z_images = df_well_images[df_well_images['z_stack_id'] == z]
            df_z_images = df_z_images.sort_values(by=['channel_id'], ascending=True)
            df_z_images = df_z_images[df_z_images['channel_id'].isin(channels)]
            df_z_nuclei = df_well_nuclei[df_well_nuclei['z'] == int(z)]
            imgs = np.asarray(
                [
                    self.load_image(os.path.join(dir_image, file))
                    for dir_image, file in zip(
                        df_z_images['dir_images'], df_z_images['file']
                    )
                ]
            )
            imgs = np.moveaxis(imgs, 0, -1)
            patches_z = [
                self.extract_patch_from_x_y(imgs, x, y)
                for x, y in zip(df_z_nuclei['x'], df_z_nuclei['y'])
            ]
            patches.extend(patches_z)
            df.append(df_z_nuclei)
        patches = np.asarray(patches)
        if scale:
            if quantiles_high is None or quantiles_low is None:
                raise ValueError('Quantiles need to be provided for scaling')
            patches = patches.astype(np.float32)
            print(patches.shape[-1])
            for i in range(patches.shape[-1]):
                patches[..., i] = np.clip(
                    (patches[..., i] - quantiles_low[i])
                    / (quantiles_high[i] - quantiles_low[i]),
                    0,
                    1,
                )
            patches = patches.astype(np.float32)

        df = pd.concat(df)
        return patches, df


class GridPatchGenerator:
    """
    GridPatchGenerator
    """

    def __init__(self, dir_images, patch_size=(128, 128), stride=128, quantiles=None):
        """
        Initialize GridPatchGenerator
        :param dir_images:
        :param patch_size:
        :param stride:
        :param quantiles:
        """
        self.dir_images = dir_images
        self.patch_size = patch_size
        self.stride = stride
        self.quantiles = quantiles
        self.image_size = (3814, 3814)
        self.channels = ['01', '02', '03', '04']
        self.df_images = self.get_metadata(dir_images)

    def load_image(self, file):
        """
        Load image
        :param file:
        :return:
        """
        image = io.imread(file)
        return image

    def get_metadata(self, path) -> pd.DataFrame:
        """
        Get metadata
        :param path:
        :return:
        """
        images = os.listdir(path)
        images = [image for image in images if '.tif' in image]
        regex = r'_(?P<well_id>[A-Z]\d{2})_T(?P<time_point>\d{4})F(?P<field_id>\d{3})L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})Z(?P<z_stack_id>\d{2})C(?P<channel_id>\d{2})\.tif$'
        df = pd.DataFrame({'file': images, 'dir_images': str(path)})
        df = df.join(df['file'].str.extractall(regex).groupby(level=0).last())
        df['id'] = df['well_id'] + '_' + df['z_stack_id']
        # remove rows that have nan in any column
        df = df[~df.isna().any(axis=1)]
        return df

    def get_patches(self, well_id, quantiles=None):
        """
        Get patches
        :param well_id:
        :param quantiles:
        :return:
        """
        df_well_images = self.df_images[self.df_images['well_id'] == well_id]
        patches = []
        df = []
        # load images
        for z in tqdm(
            df_well_images['z_stack_id'].unique().tolist(),
            desc=f'Generating patches - {well_id}',
            total=df_well_images['z_stack_id'].unique().shape[0],
        ):
            df_z_images = df_well_images[df_well_images['z_stack_id'] == z]
            df_z_images = df_z_images.sort_values(by=['channel_id'], ascending=True)

            imgs = np.asarray(
                [
                    self.load_image(os.path.join(dir_image, file))
                    for dir_image, file in zip(
                        df_z_images['dir_images'], df_z_images['file']
                    )
                ]
            )
            imgs = np.moveaxis(imgs, 0, -1)
            patches_tmp = view_as_windows(
                imgs,
                (self.patch_size[0], self.patch_size[0], len(self.channels)),
                self.stride,
            )
            # drop 3rd axis
            patches_tmp = np.squeeze(patches_tmp, axis=2)
            # get x and y indices of first and second axis
            indices = np.indices(patches_tmp.shape[:2])
            # flatten first two axis
            patches_tmp = patches_tmp.reshape(-1, *self.patch_size, len(self.channels))
            patches.append(patches_tmp)
            x = indices[0].flatten()
            y = indices[1].flatten()
            df.append(
                pd.DataFrame(
                    {'z_stack_id': z, 'well_id': well_id, 'x': x, 'y': y},
                    index=np.arange(0, patches_tmp.shape[0]),
                )
            )

        patches = np.concatenate(patches)
        df = pd.concat(df)
        patches = patches.astype(np.float32)
        if quantiles is not None:
            for i in range(patches.shape[-1]):
                patches[..., i] = np.clip(patches[..., i] / quantiles[i], 0, 1)
            patches = patches.astype(np.float32)
        return patches, df


class PatchGenerator(Sequence):
    """
    PatchGenerator
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
        quantiles_low=None,
        quantiles_high=None,
        conditions=None,
        return_conditions=False,
        **kwargs,
    ):
        """
        Patch generator
        :param list_ids:
        :param dir_dataset:
        :param batch_size:
        :param dim:
        :param n_channels:
        :param shuffle:
        :param kwargs:
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
        self.quantiles_low = quantiles_low
        self.quantiles_high = quantiles_high
        self.conditions = conditions
        self.return_conditions = return_conditions
        self.on_epoch_end()

    def __len__(self):
        """
        Get length
        :return:
        """
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """
        Get item
        :param index:
        :return:
        """
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        X = self.__data_generation(list_ids_temp)

        if self.scale:
            for i in range(X.shape[-1]):
                X[..., i] = np.clip(
                    (X[..., i] - self.quantiles_low[i])
                    / (self.quantiles_high[i] - self.quantiles_low[i]),
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
        On epoch end
        :return:
        """
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """
        Data generation
        :param list_ids_temp:
        :return:
        """
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # Generate data
        for i, idx in enumerate(list_ids_temp):
            # Store sample
            X[i,] = np.load(idx)

        return X


def extract_conditions(files, dir_datasets):
    """
    Extract conditions
    :param files:
    :param dir_datasets:
    :return:
    """
    conditions_dataset = []
    conditions_z = []
    for f in files:
        # remove dir_datasets from path
        f = f.replace(f'{dir_datasets}/', '')
        # extract digit between _ and _
        f = f.split('/')
        # extract condition
        condition_dataset = f[0]
        condition_z = f[-1].split('_')[1]
        conditions_dataset.append(condition_dataset)
        conditions_z.append(int(condition_z))

    return conditions_dataset, conditions_z


def setup_generators(
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
    Setup generators
    :param dir_datasets:
    :param conditional:
    :param batch_size:
    :param dim:
    :param n_channels:
    :param shuffle:
    :param scale:
    :param n_workers:
    :return:
    """
    datasets = os.listdir(dir_datasets)
    datasets = [
        d
        for d in datasets
        if os.path.isfile(os.path.join(dir_datasets, d, 'stats.csv'))
    ]
    dataset_merger = DatasetMerger(datasets=datasets, dir_datasets=dir_datasets)
    dataset_merger.merge_datasets()
    files = dataset_merger.get_files()
    quantiles_low, quantiles_high = dataset_merger.get_scaling_stats()
    conditions_dataset, conditions_z = extract_conditions(files, dir_datasets)
    df = pd.DataFrame({'file': files, 'dataset': conditions_dataset, 'z': conditions_z})
    # randomize
    df = df.sample(frac=1, random_state=42, replace=False)
    # get well dataset combinations and split into train and val
    df['well_id'] = df['file'].apply(lambda x: x.split('/')[-1].split('_')[0])
    df_well = df.groupby(['well_id', 'dataset']).count()
    df_well = df_well.reset_index().sample(frac=1, random_state=42, replace=False)
    # train val split
    n_train = int(df_well.shape[0] * 0.8)
    df_well['split'] = [
        'train' if i < n_train else 'val' for i in range(df_well.shape[0])
    ]
    # merge df with df_well_train
    df = pd.merge(
        df,
        df_well[['well_id', 'dataset', 'split']],
        on=['well_id', 'dataset'],
        how='left',
    )
    # drop remainders of split columns regarding batch_size grouped by split
    df = (
        df.groupby('split')
        .apply(lambda x: x.iloc[: -(x.shape[0] % batch_size)])
        .reset_index(drop=True)
    )
    files_train = df[df['split'] == 'train']['file']
    files_val = df[df['split'] == 'val']['file']
    # one hot encode conditions with sklearn
    enc = OneHotEncoder()
    cond = enc.fit_transform(df[['dataset', 'z']]).toarray()
    # convert conditions to numpy array
    cond_train = cond[df['split'] == 'train']
    cond_val = cond[df['split'] == 'val']

    generator_train = PatchGenerator(
        files_train.values,
        conditions=cond_train,
        batch_size=batch_size,
        dim=dim,
        n_channels=n_channels,
        shuffle=shuffle,
        scale=scale,
        quantiles_low=quantiles_low.values,
        quantiles_high=quantiles_high.values,
        return_conditions=conditional,
        workers=n_workers,
    )

    generator_val = PatchGenerator(
        files_val.values,
        conditions=cond_val,
        batch_size=batch_size,
        dim=dim,
        n_channels=n_channels,
        shuffle=shuffle,
        scale=scale,
        quantiles_low=quantiles_low.values,
        quantiles_high=quantiles_high.values,
        return_conditions=conditional,
        workers=n_workers,
    )

    return generator_train, generator_val, df, enc
