import anndata as ad
import numpy as np
import pandas as pd


class SpatialSubunitSampler:
    def __init__(
        self,
        adata: ad.AnnData,
        dim_subunit: tuple[int],
        min_obs: int,
        spatial_key: str,
        verbose: bool = False,
    ):
        self.adata = adata
        self.dim_subunit = dim_subunit
        self.min_obs = min_obs
        self.spatial_key = spatial_key
        self.verbose = verbose
        self.subunits = None

    def partition(self):
        """
        Partition into uniform cubes, filter by density, subsample dense cubes

            Parameters:
            -----------
            cube_size : float
                Edge length of each cube in µm
            min_nuclei_per_cube : int
                Discard cubes with fewer nuclei (empty/sparse regions)
            max_nuclei_per_cube : int
                Subsample cubes with more nuclei to this target
        """

        min_bounds = self.adata.obsm[self.spatial_key].min(axis=0)
        max_bounds = self.adata.obsm[self.spatial_key].max(axis=0)

        # Calculate number of cubes in each dimension
        extent = max_bounds - min_bounds
        n_subunits = np.ceil(extent / self.dim_subunit).astype(int)

        if self.verbose:
            print(f'Sample extent: {extent}')
            print(f'Grid dimensions: {n_subunits} subunits')
            print(f'Total potential subunits: {np.prod(n_subunits)}')

        # get subunit index for each cell position
        obs_indices = np.floor(
            (self.adata.obsm[self.spatial_key] - min_bounds) / self.dim_subunit
        ).astype(int)
        # obs_indices = np.clip(obs_indices, 0, n_subunits - 1)
        # assign cells to subunits
        self.subunits = {}
        for i, spatial_pos in enumerate(self.adata.obsm[self.spatial_key]):
            subunit_key = tuple(map(int, obs_indices[i]))
            if subunit_key not in self.subunits:
                # Calculate bounding box for this subunit
                subunit_min = min_bounds + np.array(subunit_key) * self.dim_subunit
                subunit_max = (
                    min_bounds + (np.array(subunit_key) + 1) * self.dim_subunit
                )
                self.subunits[subunit_key] = {
                    'obs_indices': [],  # List of observation indices
                    'obs_spatial': [],  # List of spatial coordinates
                    'bb_box': {
                        'min': subunit_min,  # Shape: (3,)
                        'max': subunit_max,  # Shape: (3,)
                    },
                }

            # Append to lists instead of overwriting
            self.subunits[subunit_key]['obs_indices'].append(self.adata.obs.index[i])
            self.subunits[subunit_key]['obs_spatial'].append(spatial_pos)

        # Convert lists to arrays and add subunit ids
        for i, subunit_key in enumerate(self.subunits):
            self.subunits[subunit_key]['id'] = i
            self.subunits[subunit_key]['obs_indices'] = np.array(
                self.subunits[subunit_key]['obs_indices']
            )
            self.subunits[subunit_key]['obs_spatial'] = np.array(
                self.subunits[subunit_key]['obs_spatial']
            )

    def filter(self):
        """
        Filter subunits based on minimum number of observations

        Args:
            min_obs (int): Minimum number of observations required

        Returns:
            None
        """
        self.subunits = {
            subunit_key: subunit_data
            for subunit_key, subunit_data in self.subunits.items()
            if len(subunit_data['obs_indices']) >= self.min_obs
        }

    def sample(self, max_obs: int):
        """
        Sample observations within each subunit based on max_obs threshold

        Uses the method specified in self.sample_method:
        - 'random': Random subsampling
        - 'fps': Farthest Point Sampling
        - 'uniform': Uniform voxel-based sampling

        Returns:
            None
        """
        self.max_obs = max_obs
        if self.max_obs is None:
            if self.verbose:
                print('No max_obs specified, skipping subsampling')
            return

        n_subsampled = 0
        total_before = sum(len(data['obs_indices']) for data in self.subunits.values())
        for _, subunit_data in self.subunits.items():
            n_obs = len(subunit_data['obs_indices'])
            if n_obs > self.max_obs:
                # Subsample this subunit
                keep_indices = self._random_sample(n_obs, self.max_obs)
                subunit_data['obs_indices'] = subunit_data['obs_indices'][keep_indices]
                subunit_data['obs_spatial'] = subunit_data['obs_spatial'][keep_indices]
                n_subsampled += 1

        total_after = sum(len(data['obs_indices']) for data in self.subunits.values())

        if self.verbose:
            print(
                f'Subsampled {n_subsampled} subunits with >{self.max_obs} observations'
            )
            print(f'Total observations: {total_before} → {total_after}')

    def _random_sample(self, n_points, n_samples):
        """Random subsampling"""
        return np.random.choice(n_points, n_samples, replace=False)

    def to_df(self):
        # Pre-calculate total number of rows
        total_obs = sum(len(data['obs_indices']) for data in self.subunits.values())

        # Pre-allocate arrays
        subunit_ids = []
        obs_indices = np.zeros(total_obs, dtype=int)

        idx = 0
        for subunit_id, subunit_data in self.subunits.items():
            n_obs = len(subunit_data['obs_indices'])
            subunit_ids.extend([str(subunit_id)] * n_obs)
            obs_indices[idx : idx + n_obs] = subunit_data['obs_indices']
            idx += n_obs

        # Create DataFrame
        df = pd.DataFrame(
            {
                'subunit_id': subunit_ids,
                'obs_index': obs_indices,
            }
        )
        df['obs_index'] = df['obs_index'].astype(str)
        df = df.set_index('obs_index')
        return df
