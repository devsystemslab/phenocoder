from __future__ import annotations

import hashlib

import anndata as ad
import numpy as np
import pandas as pd


def sample_seed(base: int, sample: object) -> int:
    """
    Derive a stable per-sample seed from a base seed and a sample name.

    Each sample is partitioned by its own sampler, so they cannot share one seed:
    a single int would hand every sample the identical draw sequence, correlating
    their subsamples, while a bare counter would make a sample's draw depend on how
    many samples happened to precede it.

    Uses blake2b rather than the builtin ``hash()``, whose string hashing is salted
    per process via ``PYTHONHASHSEED`` and so would vary between runs.

    Args:
        base (int): The run-level seed.
        sample (object): The sample identifier; stringified before hashing.

    Returns:
        int: A seed in ``[0, 2**32)``, stable across processes and platforms.
    """
    digest = hashlib.blake2b(f'{base}:{sample}'.encode(), digest_size=8).digest()
    return int.from_bytes(digest, 'big') % (2**32)


class SpatialSubunitSampler:
    def __init__(
        self,
        adata: ad.AnnData,
        dim_subunit: tuple[int],
        min_obs: int,
        spatial_key: str,
        verbose: bool = False,
        random_state: int | np.random.Generator | None = 0,
    ):
        self.adata = adata
        self.dim_subunit = dim_subunit
        self.min_obs = min_obs
        self.spatial_key = spatial_key
        self.verbose = verbose
        self.random_state = random_state
        # Pinned by default so that a run with max_obs set is reproducible, matching
        # the rest of the package (PCA random_state=0, SeedGenerator(42)). Pass
        # random_state=None for draws seeded from OS entropy.
        self._rng = np.random.default_rng(random_state)
        self.subunits = None

    def partition(self) -> None:
        """
        Partition the observations into a uniform grid of cubic spatial subunits.

        Divides the bounding box of the spatial coordinates (``adata.obsm[spatial_key]``)
        into cubes of edge length ``dim_subunit`` and assigns each observation to the cube
        it falls in. The result is stored on ``self.subunits`` as a dict keyed by the
        integer grid index of each cube, where each value holds the member observation
        indices, their spatial coordinates, a bounding box and an integer subunit id.

        Returns:
            None
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

    def filter(self) -> None:
        """
        Filter subunits based on minimum number of observations.

        Drops any subunit with fewer than ``self.min_obs`` observations (the
        threshold set at construction).

        Returns:
            None
        """
        self.subunits = {
            subunit_key: subunit_data
            for subunit_key, subunit_data in self.subunits.items()
            if len(subunit_data['obs_indices']) >= self.min_obs
        }

    def sample(self, max_obs: int | None) -> None:
        """
        Sample observations within each subunit based on max_obs threshold.

        Randomly subsamples observations in subunits that exceed the max_obs threshold.
        Subunits with fewer observations than max_obs are left unchanged.

        Args:
            max_obs (int): Maximum number of observations per subunit. Subunits exceeding this
                threshold will be randomly subsampled to this size.

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
        """Random subsampling, drawn from the instance RNG so runs are reproducible."""
        return self._rng.choice(n_points, n_samples, replace=False)

    def to_df(self) -> pd.DataFrame:
        """
        Build a per-observation table mapping each object to its spatial subunit.

        Flattens ``self.subunits`` into one row per observation, with the subunit
        assignment as a column and the observation index as the (string) index.

        Returns:
            pd.DataFrame: One row per observation, with a ``subunit_id`` column and
                the observation index (``obs_index``) as the DataFrame index.
        """
        # Observation names are arbitrary strings (e.g. cell barcodes like
        # "AAACCCA-1"), so these are collected as objects rather than coerced
        # through an integer array.
        subunit_ids = []
        obs_indices = []

        for subunit_id, subunit_data in self.subunits.items():
            names = subunit_data['obs_indices']
            subunit_ids.extend([str(subunit_id)] * len(names))
            obs_indices.extend(map(str, names))

        # Create DataFrame
        df = pd.DataFrame(
            {
                'subunit_id': subunit_ids,
                'obs_index': obs_indices,
            }
        )
        df = df.set_index('obs_index')
        return df
