import numpy as np
import pytest
import scanpy as sc

from tests.conftest import example_3d

DESC_STATS = 'Computing spatial graph stats'
DESC_PARTITION = 'Partitioning samples'


@pytest.fixture(scope='module')
def clustered_pheno():
    """Phenocoder whose table carries leiden labels, ready for spatial stats."""
    pheno = example_3d()
    adata = pheno.sdata.tables['nuclei_features']
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5, flavor='igraph', n_iterations=2, directed=False)
    return pheno


def test_progress_bar_shown_by_default(clustered_pheno, capsys):
    """A progress bar is drawn for sample-level stats without opting in."""
    clustered_pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        table_key='nuclei_features',
        use_subunits=False,
    )
    # tqdm writes to stderr
    assert DESC_STATS in capsys.readouterr().err


def test_progress_bar_disabled(clustered_pheno, capsys):
    """progress=False suppresses the bar entirely."""
    clustered_pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        table_key='nuclei_features',
        use_subunits=False,
        progress=False,
    )
    captured = capsys.readouterr()
    assert DESC_STATS not in captured.err
    assert DESC_STATS not in captured.out


def test_progress_does_not_change_results(clustered_pheno):
    """Results are identical whether or not the progress bar is displayed."""
    kwargs = dict(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        table_key='nuclei_features',
        use_subunits=False,
    )
    clustered_pheno.spatialgraph_stats(progress=True, **kwargs)
    with_bar = clustered_pheno.adata.copy()
    clustered_pheno.spatialgraph_stats(progress=False, **kwargs)
    without_bar = clustered_pheno.adata.copy()

    assert with_bar.shape == without_bar.shape
    assert list(with_bar.var_names) == list(without_bar.var_names)
    np.testing.assert_allclose(with_bar.X, without_bar.X)


def test_progress_bars_in_subunit_path(clustered_pheno, capsys):
    """The subunit path shows both the partition and the stats bar."""
    clustered_pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        table_key='nuclei_features',
        use_subunits=True,
        dim_subunit=(200, 200, 200),
        min_obs_per_subunit=10,
    )
    err = capsys.readouterr().err
    assert DESC_PARTITION in err
    assert DESC_STATS in err


def test_subunit_results_match_without_progress(clustered_pheno):
    """The two-pass subunit refactor yields the same stats with the bar disabled."""
    kwargs = dict(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        table_key='nuclei_features',
        use_subunits=True,
        dim_subunit=(200, 200, 200),
        min_obs_per_subunit=10,
    )
    clustered_pheno.spatialgraph_stats(progress=True, **kwargs)
    with_bar = clustered_pheno.adata.copy()
    clustered_pheno.spatialgraph_stats(progress=False, **kwargs)
    without_bar = clustered_pheno.adata.copy()

    assert with_bar.shape == without_bar.shape
    # subunit identity and ordering must be preserved by the two-pass refactor
    assert list(with_bar.obs['subunit_key']) == list(without_bar.obs['subunit_key'])
    np.testing.assert_allclose(with_bar.X, without_bar.X)
