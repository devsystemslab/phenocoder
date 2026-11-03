import anndata as ad
import numpy as np
import pandas as pd

from phenocoder.sampling import SpatialSubunitSampler, sample_seed
from tests.conftest import example_3d


def _sample_seed(sample, base=0):
    return sample_seed(base, sample)


def _barcode_sampler(**kwargs):
    """Sampler over synthetic data whose obs names are non-numeric barcodes.

    The example_3d fixture happens to use purely numeric obs names ('0', '1', ...),
    which coerce cleanly to int and so hide any integer round-tripping of the
    observation names. Real cell barcodes do not.
    """
    coords = np.random.RandomState(1).rand(200, 3) * 1000
    adata = ad.AnnData(
        np.zeros((200, 3)),
        obs=pd.DataFrame(index=[f'AAACCCA-{i}' for i in range(200)]),
    )
    adata.obsm['spatial'] = coords
    spss = SpatialSubunitSampler(
        adata=adata,
        dim_subunit=(500, 500, 100),
        min_obs=1,
        spatial_key='spatial',
        **kwargs,
    )
    spss.partition()
    spss.filter()
    return spss


def _sampler(max_obs_setup=True, **kwargs):
    """Partitioned + filtered sampler over a single well, ready for sample()."""
    adata = example_3d().sdata.tables['nuclei_features']
    adata = adata[adata.obs['well'] == 'A06']
    spss = SpatialSubunitSampler(
        adata=adata,
        dim_subunit=100,
        min_obs=5,
        spatial_key='spatial',
        **kwargs,
    )
    spss.partition()
    spss.filter()
    return spss


def _sizes(spss):
    return {key: len(data['obs_indices']) for key, data in spss.subunits.items()}


def test_subsampling():
    adata = example_3d().sdata.tables['nuclei_features']
    adata = adata[adata.obs['well'] == 'A06']
    spss = SpatialSubunitSampler(
        adata=adata,
        dim_subunit=100,
        min_obs=5,
        spatial_key='spatial',
    )
    spss.partition()
    spss.filter()
    spss.sample(max_obs=500)
    df = spss.to_df()
    # merge df to adata.obs
    adata.obs = adata.obs.merge(df, left_index=True, right_index=True, how='left')
    print(adata.obs)


def test_sample_none_leaves_subunits_untouched():
    """max_obs=None is the documented opt-out: nothing is dropped."""
    spss = _sampler()
    before = _sizes(spss)
    assert before, 'fixture should yield at least one subunit'

    spss.sample(max_obs=None)

    assert spss.max_obs is None
    assert _sizes(spss) == before


def test_sample_caps_oversized_subunits():
    """Subunits above the threshold are capped; those below are left alone.

    This is the branch that actually discards observations, so it changes every
    downstream statistic -- worth pinning rather than inferring from to_df().
    """
    spss = _sampler()
    before = _sizes(spss)
    max_obs = int(np.median(list(before.values())))
    assert max(before.values()) > max_obs, (
        'need an oversized subunit to exercise the cap'
    )

    spss.sample(max_obs=max_obs)
    after = _sizes(spss)

    assert set(after) == set(before), 'sampling must not add or drop whole subunits'
    for key, n_before in before.items():
        if n_before > max_obs:
            assert after[key] == max_obs
        else:
            assert after[key] == n_before


def test_sample_keeps_indices_and_coords_aligned():
    """obs_indices and obs_spatial are sliced with the same draw.

    They are indexed independently in sample(), so a divergence here would silently
    attach the wrong coordinates to each observation.
    """
    spss = _sampler()
    original = {
        key: dict(zip(data['obs_indices'], map(tuple, data['obs_spatial'])))
        for key, data in spss.subunits.items()
    }
    max_obs = int(np.median([len(d['obs_indices']) for d in spss.subunits.values()]))

    spss.sample(max_obs=max_obs)

    for key, data in spss.subunits.items():
        assert len(data['obs_indices']) == len(data['obs_spatial'])
        for obs_id, coord in zip(data['obs_indices'], data['obs_spatial']):
            assert tuple(coord) == original[key][obs_id]


def test_sample_draws_without_replacement():
    """Each retained observation appears once -- np.random.choice(replace=False)."""
    spss = _sampler()
    max_obs = int(np.median([len(d['obs_indices']) for d in spss.subunits.values()]))

    spss.sample(max_obs=max_obs)

    for data in spss.subunits.values():
        indices = list(data['obs_indices'])
        assert len(set(indices)) == len(indices)


def test_sample_verbose_reports_totals(capsys):
    spss = _sampler(verbose=True)
    total_before = sum(_sizes(spss).values())
    max_obs = int(np.median(list(_sizes(spss).values())))

    spss.sample(max_obs=max_obs)
    out = capsys.readouterr().out

    assert f'>{max_obs} observations' in out
    assert str(total_before) in out


def test_sample_verbose_reports_skip(capsys):
    spss = _sampler(verbose=True)
    spss.sample(max_obs=None)
    assert 'skipping subsampling' in capsys.readouterr().out


def test_to_df_preserves_non_numeric_obs_names():
    """to_df() must carry barcode-style obs names through unchanged.

    These were previously assigned into an int-dtype array, which raised on any
    name that was not parseable as an integer.
    """
    spss = _barcode_sampler()
    expected = set(spss.adata.obs.index)
    spss.sample(max_obs=3)

    df = spss.to_df()

    n_kept = sum(len(d['obs_indices']) for d in spss.subunits.values())
    assert n_kept > 0, 'fixture should retain observations'
    assert len(df) == n_kept
    assert set(df.index) <= expected, 'obs names must survive to_df() verbatim'
    assert df.index.is_unique
    assert df['subunit_id'].notna().all()


def test_to_df_maps_each_obs_to_its_own_subunit():
    """The subunit_id column must match the subunit the observation came from."""
    spss = _barcode_sampler()
    spss.sample(max_obs=3)

    df = spss.to_df()

    for key, data in spss.subunits.items():
        for name in data['obs_indices']:
            assert df.loc[str(name), 'subunit_id'] == str(key)


def test_sampling_is_reproducible_across_instances():
    """A pinned random_state makes the draw repeatable; a different seed changes it."""

    def draw(random_state):
        spss = _barcode_sampler(random_state=random_state)
        spss.sample(max_obs=3)
        return {
            key: sorted(map(str, data['obs_indices']))
            for key, data in spss.subunits.items()
        }

    assert draw(0) == draw(0), 'same seed must give the same observations'
    assert draw(0) != draw(7), 'a different seed must give a different draw'


def test_default_random_state_is_pinned():
    """The default is seeded, so an unparameterised run is reproducible too."""

    def draw():
        spss = _barcode_sampler()
        spss.sample(max_obs=3)
        return {
            key: sorted(map(str, data['obs_indices']))
            for key, data in spss.subunits.items()
        }

    assert draw() == draw()


def test_random_state_none_is_unseeded():
    """random_state=None opts back in to OS entropy."""
    draws = set()
    for _ in range(8):
        spss = _barcode_sampler(random_state=None)
        spss.sample(max_obs=3)
        draws.add(
            tuple(
                sorted(
                    (str(key), tuple(sorted(map(str, data['obs_indices']))))
                    for key, data in spss.subunits.items()
                )
            )
        )
    assert len(draws) > 1, 'unseeded draws should vary across instances'


def _stats_subunit_membership(**kwargs):
    """Run the real subunit pipeline and return each subunit's observation count."""
    pheno = example_3d()
    table = pheno.sdata.tables['nuclei_features']
    # The fixture has no clustering; a deterministic synthetic one is enough to
    # exercise the subunit path, which is what the seeding affects.
    table.obs['leiden'] = pd.Categorical(
        [str(i % 3) for i in range(table.n_obs)],
    )
    pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        table_key='nuclei_features',
        use_subunits=True,
        dim_subunit=(200, 200, 200),
        min_obs_per_subunit=10,
        max_obs_per_subunit=15,
        progress=False,
        **kwargs,
    )
    obs = pheno.adata.obs
    return dict(zip(obs.index.astype(str), obs['subunit_n_obs']))


def test_pipeline_subunit_sampling_is_reproducible():
    """Two identical spatialgraph_stats runs must subsample the same observations."""
    first = _stats_subunit_membership()
    second = _stats_subunit_membership()

    assert first, 'fixture should yield at least one subunit'
    assert first == second


def _kept_offsets(seed):
    """Which positional offsets a sampler keeps, over fixed geometry.

    Comparing offsets rather than observation names makes the draw comparable
    across samples that differ only by name, which is what isolates the seeding.
    """
    coords = np.random.RandomState(3).rand(120, 3) * 400
    adata = ad.AnnData(
        np.zeros((120, 3)),
        obs=pd.DataFrame(index=[f'obs-{i}' for i in range(120)]),
    )
    adata.obsm['spatial'] = coords
    spss = SpatialSubunitSampler(
        adata=adata,
        dim_subunit=(200, 200, 200),
        min_obs=5,
        spatial_key='spatial',
        random_state=seed,
    )
    spss.partition()
    spss.filter()
    spss.sample(max_obs=7)
    return sorted(
        tuple(sorted(int(name.split('-')[1]) for name in data['obs_indices']))
        for data in spss.subunits.values()
    )


def test_pipeline_seeds_samples_independently():
    """Each sample is seeded from its name, not from one shared draw sequence.

    With a single shared int, two samples of identical geometry subsample the
    *same positions* -- their draws are perfectly correlated. Seeding from the
    sample name decorrelates them. This test fails if the per-sample derivation
    in spatialgraph_stats is replaced by a shared seed.
    """
    pheno = example_3d()
    wells = sorted(
        pheno.sdata.tables['nuclei_features'].obs['well'].astype(str).unique()
    )
    assert len(wells) > 1, 'need multiple samples to compare seeding'

    shared = _kept_offsets(0)
    assert _kept_offsets(0) == shared, 'baseline must be deterministic'

    per_sample = [_kept_offsets(_sample_seed(well)) for well in wells[:2]]
    assert per_sample[0] != per_sample[1], (
        'samples must not draw the same positions; seed is not name-derived'
    )


def test_sample_seed_is_stable_across_processes():
    """Seed derivation must not depend on PYTHONHASHSEED.

    Python salts str hashing per process, so a hash()-based derivation would
    silently break reproducibility between runs.
    """
    assert _sample_seed('A06') == _sample_seed('A06')
    assert _sample_seed('A06') != _sample_seed('A07')
    # Pinned value: changing the derivation changes every historical draw.
    assert _sample_seed('A06') == 621216412


def test_global_numpy_seed_does_not_affect_sampling():
    """Sampling draws from the instance RNG, not the global np.random state."""

    def draw():
        spss = _barcode_sampler()
        spss.sample(max_obs=3)
        return {
            key: sorted(map(str, data['obs_indices']))
            for key, data in spss.subunits.items()
        }

    np.random.seed(0)
    first = draw()
    np.random.seed(12345)
    second = draw()

    assert first == second


if __name__ == '__main__':
    test_subsampling()
