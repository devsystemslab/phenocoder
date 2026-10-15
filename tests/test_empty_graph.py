import anndata as ad
import numpy as np
import pandas as pd
import pytest

from phenocoder.spatial import EmptyGraphError, SpatialGraphAnalyzer

CATEGORIES = ['A', 'B', 'C']
# Coordinates spread this far apart leave no cell within the test radius of another.
SPREAD_DISCONNECTED = 1_000_000.0
SPREAD_DENSE = 40.0
RADIUS = 25


def _adata(spread, n_per=20, seed=0):
    rng = np.random.default_rng(seed)
    labels = [cluster for cluster in CATEGORIES for _ in range(n_per)]
    n = len(labels)
    adata = ad.AnnData(rng.normal(size=(n, 6)))
    adata.obs = pd.DataFrame(
        {'leiden': pd.Categorical(labels, categories=CATEGORIES)},
        index=[f'c{i}' for i in range(n)],
    )
    adata.obsm['spatial'] = rng.uniform(0, spread, size=(n, 3))
    adata.var_names = [f'gene{i}' for i in range(6)]
    return adata


def _analyzer(adata, stats=None, radii=(RADIUS,)):
    return SpatialGraphAnalyzer(
        adata=adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=radii,
        index='s',
        stats=stats,
    )


def test_edgeless_graph_is_refused():
    """A graph with no edges raises instead of emitting an all-zero row.

    Such a sample passes the upstream gates in spatialgraph_stats (it has cells, and more
    than one cluster) but every graph-derived statistic collapses to 0, so the row it used
    to produce carried only cluster composition -- which does not depend on the radius.
    """
    with pytest.raises(EmptyGraphError, match='no edges'):
        _analyzer(_adata(SPREAD_DISCONNECTED)).run()


def test_error_names_the_radius_and_the_spatial_key():
    """The message has to be actionable: an edgeless graph is usually a unit mismatch."""
    with pytest.raises(EmptyGraphError) as excinfo:
        _analyzer(_adata(SPREAD_DISCONNECTED)).run()

    message = str(excinfo.value)
    assert f'radius {RADIUS}' in message
    assert 'spatial' in message
    assert 'units' in message


def test_empty_graph_error_is_a_value_error():
    """Subclassing ValueError keeps it catchable by callers that do not import it."""
    assert issubclass(EmptyGraphError, ValueError)


def test_connected_graph_is_unaffected():
    """The gate must not disturb the normal path."""
    sga = _analyzer(_adata(SPREAD_DENSE))
    sga.run()

    assert sga.to_df().shape[0] == 1
    assert sga.adata.obsp['spatial_connectivities'].sum() > 0


@pytest.mark.parametrize('stats', [['counts'], ['chull', 'counts']])
def test_radius_independent_groups_still_run_on_disconnected_data(stats):
    """counts and chull never build the neighbor graph, so they must not be refused.

    get_spatial_stats skips the graph construction entirely for these groups, so the gate
    is placed inside that branch -- checking unconditionally would raise on data these
    statistics describe perfectly well.

    stats=['chull'] on its own is excluded: scattered coordinates leave no connected
    component above chull_min_nds, so every result frame is empty and to_df() then fails
    its concat. That is a separate, pre-existing defect (to_df has no empty-results
    branch), unchanged by this gate, and it is not reached here because the gate does not
    fire for these groups.
    """
    sga = _analyzer(_adata(SPREAD_DISCONNECTED), stats=stats)
    sga.run()

    assert sga.to_df().shape[0] == 1


def test_chull_only_on_disconnected_data_does_not_hit_the_gate():
    """chull alone is still not refused -- it never builds a graph.

    Pins that the gate is not what breaks this case: run() succeeds, and only the
    pre-existing empty-concat in to_df() fails afterwards.
    """
    sga = _analyzer(_adata(SPREAD_DISCONNECTED), stats=['chull'])
    sga.run()  # must not raise EmptyGraphError

    assert all(frame.empty for frame in sga.results[RADIUS].values())


def test_any_empty_radius_refuses_the_whole_sample():
    """One unusable radius refuses the sample rather than emitting a partial row.

    Dropping just the offending radius would make the feature set depend on which radii
    happened to work, which is the ragged-feature-set problem the absent-cluster handling
    exists to avoid.
    """
    # radius 1 is empty on this data, radius 25 is not
    with pytest.raises(EmptyGraphError, match='radius 1 '):
        _analyzer(_adata(SPREAD_DENSE), stats=['centrality'], radii=(1, RADIUS)).run()


def test_moran_cluster_fallback_uses_cluster_names():
    """Called directly on an edgeless graph, the fallback names clusters, not features.

    run() refuses an edgeless graph before this branch is reached, so it only fires for a
    caller invoking the method directly. It used to return self.adata.var_names -- the
    *feature* names -- giving both the wrong names and the wrong column count (6 genes
    instead of 3 clusters).
    """
    from scipy.sparse import csr_matrix

    adata = _adata(SPREAD_DENSE)
    adata.obsp['spatial_connectivities'] = csr_matrix((adata.n_obs, adata.n_obs))
    sga = _analyzer(adata, stats=['moran_clusters'])

    columns = list(sga.get_moran_cluster().columns)

    assert columns == [f'moranI_{cluster}' for cluster in CATEGORIES]


def test_moran_feature_fallback_matches_its_normal_path():
    """The feature-level fallback keeps the moranI_ prefix the normal path applies."""
    from scipy.sparse import csr_matrix

    adata = _adata(SPREAD_DENSE)
    adata.obsp['spatial_connectivities'] = csr_matrix((adata.n_obs, adata.n_obs))
    sga = _analyzer(adata, stats=['moran_features'])

    columns = list(sga.get_moran().columns)

    assert columns == [f'moranI_gene{i}' for i in range(6)]


def test_spatialgraph_stats_skips_disconnected_samples_loudly(capsys):
    """An edgeless sample is skipped with a warning rather than contributing a row.

    The warning is not gated on verbose: an edgeless graph normally means radii does not
    match the coordinate units, which would otherwise silently discard every sample.
    """
    import scanpy as sc

    from tests.conftest import example_3d

    pheno = example_3d()
    adata = pheno.sdata.tables['nuclei_features']
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.5, flavor='igraph', n_iterations=2, directed=False)

    # radius 0.001 is far below the coordinate scale, so no sample has any edge
    with pytest.raises(ValueError, match='No spatial statistics computed'):
        pheno.spatialgraph_stats(
            cluster_key='leiden',
            spatial_key='spatial',
            radii=(0.001,),
            table_key='nuclei_features',
            use_subunits=False,
            progress=False,
        )

    out = capsys.readouterr().out
    assert 'skipped' in out
    assert 'no edges' in out
