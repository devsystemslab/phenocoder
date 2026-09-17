import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.spatial import QhullError

from phenocoder.spatial import SpatialGraphAnalyzer
from tests.conftest import example_3d


def _clustered_adata():
    """Single-well table with leiden labels, ready for spatial graph analysis."""
    adata = example_3d().sdata.tables['nuclei_features']
    adata = adata[adata.obs['well'] == 'A06'].copy()
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(
        adata, resolution=0.05, flavor='igraph', n_iterations=2, directed=False
    )
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
    assert 'stat:counts' in cols


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
    assert 'stat:counts' not in cols


def test_spatial_stats_counts():
    """The counts group reports per-cluster composition and the total cell count.

    Regression: get_counts existed but was in neither VALID_STATS nor the dispatch dict, so
    nothing ever called it -- and it raised TypeError when called, since
    Series.reset_index() has no `index` parameter.
    """
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        index='5',
        stats=['counts'],
    )
    sga.run()
    df = sga.to_df()

    assert df.shape[0] == 1
    counts = [c for c in df.columns if c.startswith('stat:counts_count_')]
    fracs = [c for c in df.columns if c.startswith('stat:counts_frac_')]
    n_clusters = len(adata.obs['leiden'].cat.categories)
    assert len(counts) == n_clusters + 1  # per-cluster, plus count_total
    assert len(fracs) == n_clusters

    # the per-cluster counts add up to the total, and the fractions to 1
    total = df['stat:counts_count_total'].iloc[0]
    assert total == adata.n_obs
    per_cluster = [c for c in counts if c != 'stat:counts_count_total']
    assert df[per_cluster].sum(axis=1).iloc[0] == total
    assert abs(df[fracs].sum(axis=1).iloc[0] - 1.0) < 1e-9


def test_counts_emitted_once_regardless_of_radii():
    """Composition does not depend on a radius, so it carries no radius prefix.

    Emitting it per radius would repeat identical values and give composition
    proportionally more weight in a downstream PCA.
    """
    adata = _clustered_adata()
    one = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='5',
        stats=['counts'],
    )
    one.run()
    three = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50, 100),
        index='5',
        stats=['counts'],
    )
    three.run()

    assert list(one.to_df().columns) == list(three.to_df().columns)
    assert not any('radius:' in c for c in three.to_df().columns)
    assert not three.to_df().columns.duplicated().any()


def test_counts_keeps_absent_clusters():
    """A cluster with no cells here still gets a zero column.

    The feature set has to follow the label set, not whichever labels this sample happens
    to contain, or reference mapping cannot line queries up against a reference.
    """
    adata = _clustered_adata()
    present = adata.obs['leiden'].cat.categories[0]
    subset = adata[adata.obs['leiden'] == present].copy()
    # keep the full label set even though only one label is present
    subset.obs['leiden'] = subset.obs['leiden'].cat.set_categories(
        adata.obs['leiden'].cat.categories
    )
    # two "clusters" are needed to clear the assert in get_spatial_stats, so pair counts
    # with nothing else and call get_counts directly
    sga = SpatialGraphAnalyzer(
        subset,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='5',
        stats=['counts'],
    )
    df = sga.get_counts()

    assert df.shape == (1, 2 * len(adata.obs['leiden'].cat.categories) + 1)
    assert df[f'count_{present}'].iloc[0] == subset.n_obs
    absent = adata.obs['leiden'].cat.categories[1]
    assert df[f'count_{absent}'].iloc[0] == 0
    assert df[f'frac_{absent}'].iloc[0] == 0.0


def test_counts_needs_no_neighbor_graph():
    """counts is radius-independent, so requesting it alone must not build the graph."""
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='5',
        stats=['counts'],
    )
    sga.run()
    assert 'spatial_connectivities' not in sga.adata.obsp


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


# --- get_chull ------------------------------------------------------------------------
#
# Whole-sample convex hull. Note this is *not* the method the run() pipeline uses -- that
# is get_chulls_connected_components -- so nothing else in the suite exercises it, and its
# three degenerate-input guards are the fragile part.

ZERO_CHULL = {'volume_chull': 0, 'area_chull': 0, 'density_chull': 0}


def _chull_analyzer(coords, index='s'):
    """Analyzer over bare coordinates, in the obs columns get_chull reads."""
    coords = np.asarray(coords, dtype=float)
    obs = pd.DataFrame(
        {
            'z': coords[:, 0],
            'centroid-0': coords[:, 1],
            'centroid-1': coords[:, 2],
        },
        index=[str(i) for i in range(len(coords))],
    )
    obs['leiden'] = pd.Categorical(['0'] * len(coords))
    adata = ad.AnnData(X=np.zeros((len(coords), 1), dtype=float), obs=obs)
    adata.obsm['spatial'] = coords
    return SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(50,),
        index=index,
    )


def _assert_zero_hull(df, index='s'):
    assert list(df.index) == [index]
    for col, value in ZERO_CHULL.items():
        assert df[col].iloc[0] == value


def test_get_chull_computes_hull_for_dense_cloud():
    """A dense 3D cloud yields a hull bounded by its box, with consistent density."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(80, 3))
    df = _chull_analyzer(coords).get_chull(radius=100, degree_threshold=5)

    assert list(df.index) == ['s']
    volume = df['volume_chull'].iloc[0]
    area = df['area_chull'].iloc[0]
    assert 0 < volume < 10**3  # cannot exceed the bounding cube
    assert area > 0
    # density is points-per-volume over the points that survived the degree filter
    assert df['density_chull'].iloc[0] == pytest.approx(80 / volume)


def test_get_chull_too_few_points():
    """Fewer than 4 points cannot span a 3D hull, so the zero frame is returned."""
    df = _chull_analyzer([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).get_chull()
    _assert_zero_hull(df)


def test_get_chull_degree_filter_empties_the_cloud():
    """Points further apart than the radius are all filtered out before the hull.

    The degree filter runs after the <4 check, so this is a separate guard.
    """
    coords = [[i * 1000.0, 0.0, 0.0] for i in range(6)]
    df = _chull_analyzer(coords).get_chull(radius=10, degree_threshold=5)
    _assert_zero_hull(df)


def test_get_chull_axis_aligned_planar_cloud():
    """A cloud flat along one axis is degenerate in 3D and returns the zero frame."""
    rng = np.random.default_rng(0)
    n = 40
    coords = np.column_stack(
        [np.full(n, 5.0), rng.uniform(0, 10, n), rng.uniform(0, 10, n)]
    )
    df = _chull_analyzer(coords).get_chull(radius=100, degree_threshold=5)
    _assert_zero_hull(df)


def test_get_chull_uses_its_index():
    """The sample identifier is carried onto the returned row."""
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 10, size=(40, 3))
    df = _chull_analyzer(coords, index='well_A06').get_chull(
        radius=100, degree_threshold=5
    )
    assert list(df.index) == ['well_A06']


def test_get_chull_tilted_planar_cloud_raises():
    """Coplanar but not axis-aligned: the degeneracy guard misses it and Qhull raises.

    The guard only rejects a *constant* z/centroid-0/centroid-1 column, so a cloud lying
    on a tilted plane (no column constant) reaches ConvexHull, which cannot build an
    initial simplex from coplanar points. get_chulls_connected_components catches
    QhullError for exactly this case; get_chull does not.

    Documented rather than fixed: coordinates landing exactly on a plane do not occur in
    experimental or simulation data, and get_chull is not on the run() path. This test
    pins the current behaviour so the difference between the two methods is visible and
    the case is not rediscovered from scratch. Adding the guard should turn this into the
    zero frame -- update the test then, it is not a tripwire.
    """
    rng = np.random.default_rng(1)
    n = 40
    u = rng.uniform(0, 10, n)
    v = rng.uniform(0, 10, n)
    coords = np.column_stack([u + v, u, v])  # z = x + y
    with pytest.raises(QhullError):
        _chull_analyzer(coords).get_chull(radius=100, degree_threshold=5)


if __name__ == '__main__':
    test_spatial_stats_all()
    test_spatial_stats_subset()
    test_spatial_stats_chull_thresholds()
    test_spatial_stats_invalid_stat()
