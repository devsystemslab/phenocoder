import anndata as ad
import numpy as np
import pandas as pd
import pytest

from phenocoder.spatial import SpatialGraphAnalyzer

# Every cluster-aware stat group should key its features off the *declared* label set, so
# that two samples sharing a cluster_key produce the same feature vector even when one of
# them happens to contain no cells of some cluster.
CLUSTER_AWARE_GROUPS = [
    'interactions',
    'centrality',
    'connectivity',
    'moran_clusters',
]


def _adata(present, categories, n_per=20, spread=40.0, seed=0):
    """AnnData whose label column declares `categories` but only fills `present`."""
    rng = np.random.default_rng(seed)
    labels = [cluster for cluster in present for _ in range(n_per)]
    n = len(labels)
    adata = ad.AnnData(rng.normal(size=(n, 4)))
    adata.obs = pd.DataFrame(
        {'leiden': pd.Categorical(labels, categories=categories)},
        index=[f'c{i}' for i in range(n)],
    )
    adata.obsm['spatial'] = rng.uniform(0, spread, size=(n, 3))
    adata.var_names = [f'g{i}' for i in range(4)]
    return adata


def _analyzer(adata, stats, radius=25, index='s'):
    return SpatialGraphAnalyzer(
        adata=adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(radius,),
        index=index,
        stats=stats,
    )


def _features(adata, stats, radius=25, index='s'):
    """The single-row feature frame a sample contributes."""
    sga = _analyzer(adata, stats, radius=radius, index=index)
    sga.run()
    return sga.to_df()


@pytest.mark.parametrize('categories', [['A', 'B', 'C'], ['A', 'B', 'C', 'D']])
def test_absent_category_does_not_raise(categories):
    """A declared-but-empty cluster no longer kills the run.

    squidpy scores every category of the cluster column and average_clustering divides by
    the node count, so an empty category raised ZeroDivisionError. Via the broad except in
    spatialgraph_stats that dropped the entire sample -- every stat group, not just
    centrality -- and silently so for subunits unless verbose=True.
    """
    sga = _analyzer(_adata(['A', 'B'], categories), stats=['centrality'])
    sga.run()

    assert sga.to_df().shape[0] == 1


@pytest.mark.parametrize('group', CLUSTER_AWARE_GROUPS)
def test_feature_set_follows_the_label_set(group):
    """Every cluster-aware group emits columns for absent clusters too.

    Without this, a sample's feature set depends on which clusters it happens to contain,
    so samples sharing a cluster_key are no longer directly comparable.
    """
    categories = ['A', 'B', 'C', 'D']
    all_present = _features(_adata(['A', 'B', 'C', 'D'], categories), stats=[group])
    some_absent = _features(_adata(['A', 'B'], categories), stats=[group])

    assert list(all_present.columns) == list(some_absent.columns)


def test_interactions_keeps_pairs_between_two_absent_clusters():
    """The pair columns for two mutually-absent clusters survive.

    squidpy returns the full category x category matrix with an all-NaN row/column per
    empty category, and a plain .stack() dropped those entries. With one absent category
    the surviving mixed pairs (A_C, B_C) hid the problem; it only shows with two, where
    C_D and D_C vanished too.
    """
    categories = ['A', 'B', 'C', 'D']
    row = _features(_adata(['A', 'B'], categories), stats=['interactions']).iloc[0]

    # all 4 variants: raw and normalized, in both directions
    pairs = [name for name in row.index if '_C_D' in name or '_D_C' in name]
    assert len(pairs) == 4, f'expected 4 C/D pair columns, got {pairs}'
    assert all(row[name] == 0.0 for name in pairs)
    # and the full matrix is there: 2 normalizations x 4 x 4 pairs
    assert len(row) == 32


def test_moran_cluster_column_order_is_deterministic():
    """Moran's I columns follow the label set, not the computed values.

    squidpy's spatial_autocorr sorts its result by I descending, so without a reindex the
    column order varied between samples of one run (e.g. B,C,A,D for one and D,A,B,C for
    another). Nothing downstream was corrupted -- concat and align_features both join by
    name -- but the order carried no meaning and silently tracked the data.
    """
    categories = ['A', 'B', 'C', 'D']
    orders = [
        [
            column.split('moranI_')[-1]
            for column in _features(
                _adata(categories, categories, seed=seed),
                stats=['moran_clusters'],
                index=f's{seed}',
            ).columns
        ]
        for seed in range(4)
    ]

    assert all(order == categories for order in orders)


def test_absent_cluster_scores_zero_for_centrality():
    """Absent clusters score 0 for every centrality measure."""
    sga = _analyzer(_adata(['A', 'B'], ['A', 'B', 'C']), stats=['centrality'])
    sga.run()
    row = sga.results[25]['centrality'].iloc[0]

    absent = {name: value for name, value in row.items() if '_C_' in name}
    assert absent, 'expected columns for the absent cluster C'
    assert all(value == 0.0 for value in absent.values())
    # the clusters that are present still score above zero
    assert row['centrality_A_degree_centrality'] > 0


def test_absent_cluster_reports_zero_degree():
    """Absent clusters report a degree of 0 rather than dropping their columns."""
    sga = _analyzer(_adata(['A', 'B'], ['A', 'B', 'C']), stats=['connectivity'])
    sga.run()
    row = sga.results[25]['connectivity'].iloc[0]

    assert row['degree_C_mean'] == 0.0
    assert row['degree_C_std'] == 0.0
    assert row['degree_A_mean'] > 0


def test_present_cluster_scores_are_unaffected_by_absent_ones():
    """Restricting the categorical must not perturb the clusters that are present.

    get_centrality scores only the non-empty categories, so this pins that the graph -- and
    therefore each present cluster's score -- is identical to declaring just those
    categories in the first place.
    """
    shared = {'n_per': 20, 'spread': 40.0, 'seed': 7}
    baseline = _analyzer(_adata(['A', 'B'], ['A', 'B'], **shared), stats=['centrality'])
    with_absent = _analyzer(
        _adata(['A', 'B'], ['A', 'B', 'C', 'D'], **shared), stats=['centrality']
    )
    baseline.run()
    with_absent.run()

    row_baseline = baseline.results[25]['centrality'].iloc[0]
    row_absent = with_absent.results[25]['centrality'].iloc[0]
    for name, value in row_baseline.items():
        assert row_absent[name] == pytest.approx(value)


def test_global_degree_is_unchanged_by_the_per_cluster_rewrite():
    """The global mean/std cover all nodes, independently of the cluster bookkeeping."""
    adata = _adata(['A', 'B'], ['A', 'B', 'C'], seed=11)
    sga = _analyzer(adata, stats=['connectivity'])
    sga.run()
    row = sga.results[25]['connectivity'].iloc[0]

    degrees = np.asarray(sga.adata.obsp['spatial_connectivities'].sum(axis=0)).ravel()
    assert row['mean'] == pytest.approx(degrees.mean())
    assert row['std'] == pytest.approx(degrees.std())


def test_isolated_cluster_is_not_distinguishable_from_an_absent_one():
    """A present-but-isolated cluster also scores 0, which 'counts' disambiguates.

    Documents the known collision: degree_C_mean == 0 means either 'no cells of C here' or
    'cells of C here with no neighbors in radius'. count_C separates them, so this pins
    that counts is the thing to reach for rather than the zero itself.
    """
    rng = np.random.default_rng(5)
    # C exists but sits far outside the radius, so its single cell has degree 0
    coords = np.vstack(
        [
            rng.uniform(0, 25, size=(20, 3)),
            rng.uniform(0, 25, size=(20, 3)),
            [[9000.0, 0, 0]],
        ]
    )
    adata = ad.AnnData(rng.normal(size=(41, 4)))
    adata.obs = pd.DataFrame(
        {
            'leiden': pd.Categorical(
                ['A'] * 20 + ['B'] * 20 + ['C'], categories=['A', 'B', 'C']
            )
        },
        index=[f'c{i}' for i in range(41)],
    )
    adata.obsm['spatial'] = coords
    adata.var_names = [f'g{i}' for i in range(4)]

    sga = _analyzer(adata, stats=['connectivity', 'counts'])
    sga.run()

    assert sga.results[25]['connectivity'].iloc[0]['degree_C_mean'] == 0.0
    # identical to the absent case above -- only the count tells them apart
    assert sga.results_global.iloc[0]['count_C'] == 1
