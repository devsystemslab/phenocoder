import anndata as ad
import numpy as np
import pandas as pd
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
    assert 'stat:distribution_interactions' in cols
    assert 'stat:distribution_features' in cols
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
    assert 'stat:distribution_interactions' not in cols
    assert 'stat:distribution_features' not in cols
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


def test_interaction_mean_recovers_the_graph_wide_sum():
    """The per-node mean times the source cluster's size is the graph-wide pair count.

    squidpy's interaction matrix is the sum over the per-node neighbor counts, so this is
    the invariant the per-node statistics are built on. It also pins the from/to
    orientation: a transposed assembly would still produce a full set of plausible-looking
    columns, and only an asymmetric pair count catches it.
    """
    adata = _clustered_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='5',
        stats=['interactions', 'distribution_interactions', 'counts'],
    )
    sga.run()
    row = sga.to_df().iloc[0]

    clusters = adata.obs['leiden'].cat.categories.tolist()
    for source in clusters:
        n_source = row[f'stat:counts_count_{source}']
        for target in clusters:
            mean = row[
                f'radius:25_stat:distribution_interactions_mean_{source}_{target}'
            ]
            total = row[f'radius:25_stat:interactions_interaction_{source}_{target}']
            assert mean * n_source == pytest.approx(total, abs=1e-6)


def test_interaction_distributions_toggle_independently_of_the_matrix():
    """Either interaction group can be computed without the other."""

    def _groups(stats):
        sga = SpatialGraphAnalyzer(
            _clustered_adata(),
            cluster_key='leiden',
            spatial_key='spatial',
            radii=(25,),
            index='5',
            stats=stats,
        )
        sga.run()
        return ' '.join(sga.to_df().columns)

    matrix_only = _groups(['interactions'])
    assert 'stat:interactions' in matrix_only
    assert 'stat:distribution_interactions' not in matrix_only

    distributions_only = _groups(['distribution_interactions'])
    assert 'stat:distribution_interactions' in distributions_only
    # the graph-wide matrix columns are prefixed stat:interactions_interaction_
    assert 'stat:interactions_' not in distributions_only


def test_percentiles_add_requested_columns():
    """Each requested percentile adds a full pair matrix; () leaves mean and sd only."""
    adata = _clustered_adata()
    clusters = adata.obs['leiden'].cat.categories.tolist()

    def _columns(percentiles):
        sga = SpatialGraphAnalyzer(
            adata,
            cluster_key='leiden',
            spatial_key='spatial',
            radii=(25,),
            index='5',
            stats=['distribution_interactions'],
            percentiles=percentiles,
        )
        sga.run()
        return sga.to_df().columns.tolist()

    default = _columns(())
    assert not [name for name in default if '_p10_' in name or '_p50_' in name]

    with_percentiles = _columns((10, 50))
    added = set(with_percentiles) - set(default)
    # one column per pair, for each percentile, on both counts and fractions
    assert len(added) == len(clusters) ** 2 * 2 * 2
    for source in clusters:
        for target in clusters:
            prefix = 'radius:25_stat:distribution_interactions'
            assert f'{prefix}_p10_{source}_{target}' in added
            assert f'{prefix}_frac_p50_{source}_{target}' in added


def test_spatial_stats_invalid_percentile():
    """A percentile outside [0, 100] raises, naming the 0-100 scale."""
    adata = _clustered_adata()
    with pytest.raises(ValueError, match=r'percentiles must lie'):
        SpatialGraphAnalyzer(
            adata,
            cluster_key='leiden',
            spatial_key='spatial',
            radii=(25,),
            index='5',
            percentiles=(10, 101),
        )


def _feature_adata(n=300, seed=0):
    """Two features on the same scale: one varying smoothly in space, one random."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 300, size=(n, 2))
    smooth = coords[:, 0]
    random = rng.normal(size=n) * smooth.std()
    adata = ad.AnnData(np.column_stack([smooth, random]))
    adata.var_names = ['smooth', 'random']
    adata.obs = pd.DataFrame(
        {'leiden': pd.Categorical(rng.choice(['A', 'B'], size=n))},
        index=[f'c{i}' for i in range(n)],
    )
    adata.obsm['spatial'] = coords
    return adata


def test_feature_stats_raw_block_is_radius_independent():
    """The raw-value descriptors are emitted once, the neighborhood ones per radius."""
    sga = SpatialGraphAnalyzer(
        _feature_adata(),
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        index='s',
        stats=['distribution_features'],
    )
    sga.run()
    columns = sga.to_df().columns.tolist()

    raw = [name for name in columns if '_feature_mean_' in name]
    assert raw == [
        'stat:distribution_features_feature_mean_smooth',
        'stat:distribution_features_feature_mean_random',
    ]
    # the neighborhood half does depend on the radius, so it appears once per radius
    nbhd = [name for name in columns if name.endswith('nbhd_mean_smooth')]
    assert nbhd == [
        'radius:25_stat:distribution_features_nbhd_mean_smooth',
        'radius:50_stat:distribution_features_nbhd_mean_smooth',
    ]


def test_feature_stats_raw_descriptors_match_the_matrix():
    """The raw family has an exact expected value, so assert it rather than its presence."""
    adata = _feature_adata()
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='s',
        stats=['distribution_features'],
        percentiles=(50,),
    )
    sga.run()
    row = sga.to_df().iloc[0]

    for position, var in enumerate(adata.var_names):
        values = adata.X[:, position]
        assert row[f'stat:distribution_features_feature_mean_{var}'] == pytest.approx(
            values.mean()
        )
        assert row[f'stat:distribution_features_feature_sd_{var}'] == pytest.approx(
            values.std()
        )
        assert row[f'stat:distribution_features_feature_p50_{var}'] == pytest.approx(
            np.percentile(values, 50)
        )


def test_feature_neighborhood_mean_includes_the_cell_itself():
    """Two cells in radius average each other; an isolated one reports its own value.

    Pins the self-loop convention borrowed from spatial_message_passing. Without it an
    isolated cell would divide by a zero degree, and filling 0 instead would invent a
    measurement indistinguishable from a real one for a feature centered near zero.
    """
    adata = ad.AnnData(np.array([[1.0], [3.0], [10.0]]))
    adata.var_names = ['f']
    adata.obs = pd.DataFrame(
        {'leiden': pd.Categorical(['A', 'A', 'A'])},
        index=['near1', 'near2', 'far'],
    )
    # the first two sit well inside the radius; the third is far outside it
    adata.obsm['spatial'] = np.array([[0.0, 0.0], [5.0, 0.0], [9000.0, 0.0]])

    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='s',
        stats=['distribution_features'],
        percentiles=(),
    )
    sga.run()
    row = sga.to_df().iloc[0]

    # neighborhood means are (1+3)/2 = 2 for both near cells, and 10 for the isolated one
    assert row['radius:25_stat:distribution_features_nbhd_mean_f'] == pytest.approx(
        (2.0 + 2.0 + 10.0) / 3
    )


def test_feature_neighborhood_averaging_keeps_smooth_feature_spread():
    """A spatially smooth feature keeps its spread; a random one loses most of it.

    This is the behavior that makes the neighborhood family worth having next to the raw
    one: both features have the same raw sd by construction.
    """
    sga = SpatialGraphAnalyzer(
        _feature_adata(n=2000),
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25,),
        index='s',
        stats=['distribution_features'],
    )
    sga.run()
    row = sga.to_df().iloc[0]

    retained = {}
    for var in ('smooth', 'random'):
        raw = row[f'stat:distribution_features_feature_sd_{var}']
        nbhd = row[f'radius:25_stat:distribution_features_nbhd_sd_{var}']
        retained[var] = nbhd / raw
    assert retained['smooth'] > 0.9
    assert retained['random'] < 0.5


def test_feature_stats_percentiles_follow_the_shared_argument():
    """Both halves of the group honour percentiles, and () leaves mean and sd only."""

    def _columns(percentiles):
        sga = SpatialGraphAnalyzer(
            _feature_adata(),
            cluster_key='leiden',
            spatial_key='spatial',
            radii=(25,),
            index='s',
            stats=['distribution_features'],
            percentiles=percentiles,
        )
        sga.run()
        return sga.to_df().columns.tolist()

    plain = _columns(())
    statistics = {name.rsplit('_', 2)[-2] for name in plain}
    assert statistics == {'mean', 'sd'}

    added = set(_columns((25, 75))) - set(plain)
    # two percentiles x two features, in each of the raw and neighborhood families
    assert len(added) == 2 * 2 * 2
    assert 'stat:distribution_features_feature_p25_smooth' in added
    assert 'radius:25_stat:distribution_features_nbhd_p75_random' in added


if __name__ == '__main__':
    test_spatial_stats_all()
    test_spatial_stats_subset()
    test_spatial_stats_chull_thresholds()
    test_spatial_stats_invalid_stat()


def test_spatialgraph_stats_min_clusters():
    """One-cluster samples are skipped by default and encoded with min_clusters=1."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import spatialdata as sd

    import phenocoder as phc

    rng = np.random.default_rng(0)
    labels = ['A'] * 40 + ['A'] * 20 + ['B'] * 20  # s0: one cluster; s1: two
    adata = ad.AnnData(rng.normal(size=(80, 4)))
    adata.obs = pd.DataFrame(
        {
            'leiden': pd.Categorical(labels, categories=['A', 'B']),
            'well': ['s0'] * 40 + ['s1'] * 40,
            'region': 'cells',
            'instance_id': np.arange(80),
        },
        index=[str(i) for i in range(80)],
    )
    adata.obsm['spatial'] = rng.uniform(0, 40, size=(80, 3))
    adata.var_names = [f'g{i}' for i in range(4)]
    table = sd.models.TableModel.parse(
        adata, region='cells', region_key='region', instance_key='instance_id'
    )

    for min_clusters, expected in ((2, ['s1']), (1, ['s0', 's1'])):
        pheno = phc.Phenocoder(table_key='cells', sample_key='well')
        pheno.add_sdata(sd.SpatialData(tables={'cells': table.copy()}))
        pheno.spatialgraph_stats(
            cluster_key='leiden',
            radii=(25,),
            table_key='cells',
            progress=False,
            min_clusters=min_clusters,
        )
        assert sorted(pheno.adata.obs_names) == expected
        assert pheno.stats_config['min_clusters'] == min_clusters
        assert np.isfinite(pheno.adata.X).all()
