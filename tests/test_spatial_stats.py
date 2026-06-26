import pytest
import scanpy as sc

from phenocoder.spatial import SpatialGraphAnalyzer
from tests.conftest import example_3d


def _clustered_adata():
    """Single-well table with leiden labels, ready for spatial graph analysis."""
    adata = example_3d().sdata.tables['nuclei_features']
    adata = adata[adata.obs['well'] == 'A06'].copy()
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.05)
    return adata


def test_spatial_stats_all():
    """Default (stats=None) computes every stat group."""
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        index='5',
    )
    assert sga.stats == set(SpatialGraphAnalyzer.VALID_STATS)
    sga.run()
    df = sga.to_df()

    assert df.shape[0] == 1
    cols = ' '.join(df.columns)
    # a representative column from each stat group should be present
    assert 'stat:interactions' in cols
    assert 'stat:centrality' in cols
    assert 'stat:connectivity' in cols
    assert 'stat:moran_features' in cols
    assert 'stat:moran_clusters' in cols
    assert 'stat:chull' in cols


def test_spatial_stats_subset():
    """Only the selected stat groups are computed."""
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        index='5',
        stats=['interactions', 'connectivity'],
    )
    sga.run()
    df = sga.to_df()

    cols = ' '.join(df.columns)
    assert 'stat:interactions' in cols
    assert 'stat:connectivity' in cols
    # unselected groups must not appear
    assert 'stat:centrality' not in cols
    assert 'stat:moran_features' not in cols
    assert 'stat:moran_clusters' not in cols
    assert 'stat:chull' not in cols


def test_spatial_stats_chull_thresholds():
    """The chull group runs with custom min_nds / min_degree thresholds."""
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        index='5',
        stats=['chull'],
        chull_min_nds=15,
        chull_min_degree=4,
    )
    assert sga.chull_min_nds == 15
    assert sga.chull_min_degree == 4
    sga.run()
    df = sga.to_df()

    cols = ' '.join(df.columns)
    # only chull stats should be present
    assert 'stat:chull' in cols
    assert 'stat:interactions' not in cols


def test_spatial_stats_chull_min_nds_warning(capsys):
    """A chull_min_nds below 4 prints a degeneracy warning."""
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='5',
        stats=['chull'],
        chull_min_nds=3,
    )
    sga.get_chulls_connected_components(
        clusters=adata.obs['leiden'].unique().tolist(),
        radius=25,
        min_nds=sga.chull_min_nds,
        min_degree=sga.chull_min_degree,
    )
    out = capsys.readouterr().out
    assert 'chull_min_nds=3' in out
    assert 'below 4' in out


def test_spatial_stats_invalid_stat():
    """Unknown stat names raise a ValueError."""
    adata = _clustered_adata()
    with pytest.raises(ValueError, match='Unknown stats'):
        SpatialGraphAnalyzer(
            adata,
            cluster_key='leiden',
            spatial_key='spatial',
            radii=(25, 50),
            index='5',
            stats=['not_a_real_stat'],
        )


if __name__ == '__main__':
    test_spatial_stats_all()
    test_spatial_stats_subset()
    test_spatial_stats_chull_thresholds()
    test_spatial_stats_invalid_stat()

