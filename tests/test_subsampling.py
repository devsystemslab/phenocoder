import numpy as np

from phenocoder.sampling import SpatialSubunitSampler
from tests.conftest import example_3d


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


if __name__ == '__main__':
    test_subsampling()
