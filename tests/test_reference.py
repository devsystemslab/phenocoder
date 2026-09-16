"""Tests for projecting a query into a frozen reference embedding space.

The guarantee under test: once a reference space is saved, mapping a query into it must not
move the reference axes, and a query whose features do not line up must fail loudly rather
than be projected onto the wrong coordinates.
"""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import phenocoder as phc
from phenocoder.reference import align_features, load_transform


def _toy_sdata(n_samples=12, n_per_sample=60, n_clusters=3, seed=0):
    """Build a small SpatialData table with spatial coords and cluster labels.

    The spatial-graph statistics need only coordinates and a categorical label column, so
    this skips patch extraction and the CVAE entirely -- which is also the simulation
    -reference path we care about.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_samples):
        # vary the cluster mix per sample so the samples are not identical
        weights = rng.dirichlet(np.ones(n_clusters))
        labels = rng.choice(n_clusters, size=n_per_sample, p=weights)
        coords = rng.uniform(0, 200, size=(n_per_sample, 3))
        for (x, y, z), lab in zip(coords, labels):
            rows.append(
                {
                    'well': f'w{s:02d}',
                    'leiden': str(lab),
                    'x': x,
                    'y': y,
                    'z': z,
                }
            )
    df = pd.DataFrame(rows)

    # a couple of numeric features so moran_features has something to chew on
    X = rng.normal(size=(len(df), 4))
    adata = ad.AnnData(X=X, obs=df.reset_index(drop=True))
    adata.obs_names = [str(i) for i in range(adata.n_obs)]
    adata.obs['leiden'] = pd.Categorical(
        adata.obs['leiden'], categories=[str(i) for i in range(n_clusters)]
    )
    adata.obsm['spatial'] = adata.obs[['x', 'y', 'z']].values
    adata.obs['region'] = 'cells'
    adata.obs['instance_id'] = np.arange(adata.n_obs)
    adata.var_names = [f'f{i}' for i in range(X.shape[1])]

    sdata = sd.SpatialData(
        tables={
            'cells': sd.models.TableModel.parse(
                adata,
                region='cells',
                region_key='region',
                instance_key='instance_id',
                overwrite_metadata=True,
            )
        }
    )
    return sdata


def _pheno(dir_project, seed=0, **kwargs):
    pheno = phc.Phenocoder(
        table_key='cells', sample_key='well', dir_project=dir_project
    )
    pheno.add_sdata(_toy_sdata(seed=seed, **kwargs))
    return pheno


def _fit_reference(tmp_path, seed=0, umap=True, n_dim=5):
    pheno = _pheno(tmp_path, seed=seed)
    pheno.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    pheno.spatialgraph_embedding(
        n_dim=n_dim,
        scale=True,
        umap=umap,
        n_neighbors=min(5, pheno.adata.shape[0] - 1),
        save_transform=True,
    )
    return pheno


# --- the core guarantee ---------------------------------------------------------------


def test_identity_roundtrip(tmp_path):
    """Mapping the reference's own statistics back through the saved space reproduces it.

    This is the test that proves the space is actually frozen: same input -> same
    coordinates, with nothing refitted.
    """
    ref = _fit_reference(tmp_path)
    fitted_pca = ref.adata.obsm['X_pca'].copy()
    fitted_umap = ref.adata.obsm['X_umap'].copy()

    # a fresh object holding the identical statistics
    query = _pheno(tmp_path)
    query.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    query.spatialgraph_map_query()

    np.testing.assert_allclose(query.adata.obsm['X_pca'], fitted_pca, atol=1e-5)
    np.testing.assert_allclose(query.adata.obsm['X_umap'], fitted_umap, atol=1e-5)


def test_reference_unchanged_by_query(tmp_path):
    """Projecting a query must not touch the stored transform."""
    ref = _fit_reference(tmp_path)
    path = ref.dir_reference / 'embedding_transform.joblib'
    before = load_transform(path)
    mean_before = before['scaler'].mean_.copy()
    comps_before = before['pca'].components_.copy()

    query = _pheno(tmp_path, seed=99)
    query.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    query.spatialgraph_map_query()

    after = load_transform(path)
    np.testing.assert_array_equal(after['scaler'].mean_, mean_before)
    np.testing.assert_array_equal(after['pca'].components_, comps_before)


def test_query_projects_into_reference_space(tmp_path):
    """A genuinely different query lands in the reference space with the right shape."""
    ref = _fit_reference(tmp_path, n_dim=5)
    query = _pheno(tmp_path, seed=7)
    query.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    query.spatialgraph_map_query()

    assert query.adata.obsm['X_pca'].shape == (query.adata.n_obs, 5)
    assert np.isfinite(query.adata.obsm['X_pca']).all()
    # different data -> different coordinates
    assert not np.allclose(
        query.adata.obsm['X_pca'][: ref.adata.n_obs], ref.adata.obsm['X_pca']
    )


# --- feature alignment ----------------------------------------------------------------


def test_align_features_reorders_columns():
    """Column order is restored, so a shuffled query is not projected onto wrong axes."""
    var_names = ['radius:25_stat:interactions_a', 'radius:25_stat:interactions_b']
    df = pd.DataFrame([[2.0, 1.0]], columns=var_names[::-1])
    out = align_features(df, var_names)
    assert list(out.columns) == var_names
    assert out.iloc[0].tolist() == [1.0, 2.0]


def test_align_features_missing_is_zero_filled():
    """Missing features are zero-filled, matching spatialgraph_stats' own fillna(0).

    AnnData subsetting drops unused categories, so a sample lacking a cluster emits no
    columns for it -- an absent interaction count genuinely means "never co-occurs here".
    Feature sets differ between samples even within a single run.
    """
    var_names = [
        'radius:25_stat:interactions_0_1',
        'radius:25_stat:centrality_0',
        'radius:25_stat:chull_volume',
    ]
    df = pd.DataFrame([[1.0]], columns=[var_names[0]])
    out = align_features(df, var_names)
    assert out.iloc[0].tolist() == [1.0, 0.0, 0.0]


def test_align_features_extra_dropped_with_warning():
    """Query-only features cannot be projected (no reference loading), so drop and warn.

    A cluster pair that co-occurs in the query but never in the reference produces a column
    the reference never had -- routine, not an error.
    """
    var_names = ['radius:25_stat:interactions_0_1']
    df = pd.DataFrame([[1.0, 9.0]], columns=var_names + ['radius:25_stat:chull_extra'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = align_features(df, var_names)
    assert list(out.columns) == var_names
    assert out.iloc[0].tolist() == [1.0]
    assert any('Dropping 1 of 2 query feature' in str(w.message) for w in caught)


def test_align_features_extra_can_be_made_fatal():
    var_names = ['radius:25_stat:interactions_0_1']
    df = pd.DataFrame([[1.0, 9.0]], columns=var_names + ['radius:25_stat:chull_extra'])
    with pytest.raises(ValueError, match='absent from the reference'):
        align_features(df, var_names, allow_extra=False)


def test_different_label_set_is_rejected(tmp_path):
    """A query clustered independently of the reference must fail loudly.

    This is the mismatch that matters: zero-filling would otherwise produce plausible-looking
    but meaningless coordinates. Caught by comparing label sets, not feature sets.
    """
    _fit_reference(tmp_path)

    query = _pheno(tmp_path, seed=3)
    table = query.sdata.tables['cells']
    # collapse cluster '2' into '1' and drop it from the categories entirely
    labels = table.obs['leiden'].astype(str).replace({'2': '1'})
    table.obs['leiden'] = pd.Categorical(labels, categories=['0', '1'])
    query.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    with pytest.raises(ValueError, match='cluster labels do not match'):
        query.spatialgraph_map_query()


def test_sample_missing_a_cluster_is_accepted(tmp_path):
    """A sample that simply lacks one cluster is normal, and must project fine.

    Distinct from the case above: the label *set* still matches, this sample just has no
    cells of one type. Its absent interaction counts are legitimately zero.
    """
    _fit_reference(tmp_path)

    query = _pheno(tmp_path, seed=11)
    table = query.sdata.tables['cells']
    # remove every cell of cluster '2' from one well, keeping the category list intact
    mask = ~((table.obs['well'] == 'w00') & (table.obs['leiden'] == '2'))
    query.sdata.tables['cells'] = table[mask.values].copy()
    query.sdata.tables['cells'].obs['leiden'] = pd.Categorical(
        query.sdata.tables['cells'].obs['leiden'].astype(str),
        categories=['0', '1', '2'],
    )
    query.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    query.spatialgraph_map_query()
    assert np.isfinite(query.adata.obsm['X_pca']).all()


def test_near_constant_reference_feature_is_clipped():
    """A feature that barely varied in the reference must not dominate the projection.

    Dividing a query deviation by a near-zero reference sd turns a trivial difference into
    hundreds of standard deviations. Real references have plenty of such features (706 of
    2693 were zero-variance on the bundled test data).
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from phenocoder.reference import apply_transform

    rng = np.random.default_rng(0)
    var_names = ['radius:25_stat:interactions_a', 'radius:25_stat:interactions_b']
    # feature 'b' is essentially constant in the reference
    ref = np.column_stack(
        [rng.normal(size=40), np.full(40, 1.0) + rng.normal(scale=1e-6, size=40)]
    )
    scaler = StandardScaler().fit(ref)
    pca = PCA(n_components=1, svd_solver='arpack', random_state=0).fit(
        scaler.transform(ref)
    )
    payload = {
        'scaler': scaler,
        'pca': pca,
        'umap': None,
        'var_names': var_names,
        'scale': True,
    }
    # query deviates slightly on the near-constant feature
    query = pd.DataFrame([[0.0, 1.5]], columns=var_names)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        X_clipped, _ = apply_transform(query, payload, clip=10.0)
    assert any('Clipped' in str(w.message) for w in caught)

    X_unclipped, _ = apply_transform(query, payload, clip=None)
    assert np.abs(X_clipped).max() < np.abs(X_unclipped).max()
    assert np.isfinite(X_clipped).all()


# --- configuration validation ---------------------------------------------------------


def test_radii_mismatch_is_rejected(tmp_path):
    """Different radii produce identically-named but incomparable features."""
    _fit_reference(tmp_path)
    query = _pheno(tmp_path, seed=5)
    query.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(30, 60), table_key='cells'
    )
    with pytest.raises(ValueError, match='radii'):
        query.spatialgraph_map_query()


def test_cluster_key_mismatch_is_rejected(tmp_path):
    _fit_reference(tmp_path)
    query = _pheno(tmp_path, seed=5)
    table = query.sdata.tables['cells']
    table.obs['other'] = table.obs['leiden']
    query.spatialgraph_stats(
        cluster_key='other', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    with pytest.raises(ValueError, match='cluster_key'):
        query.spatialgraph_map_query()


# --- guards ---------------------------------------------------------------------------


def test_batch_correction_rejected_in_save_path(tmp_path):
    """bbknn.ridge_regression has no transform-only form, so it cannot be saved."""
    pheno = _pheno(tmp_path)
    pheno.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    with pytest.raises(ValueError, match='incompatible with batch_correction'):
        pheno.spatialgraph_embedding(
            n_dim=5,
            save_transform=True,
            batch_correction=True,
            batch_key='well',
        )


def test_variable_features_rejected_in_save_path(tmp_path):
    pheno = _pheno(tmp_path)
    pheno.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    with pytest.raises(ValueError, match='variable_features'):
        pheno.spatialgraph_embedding(
            n_dim=5, save_transform=True, variable_features=True
        )


def test_save_transform_requires_stats_config(tmp_path):
    """Without a stats run we cannot record what the features mean."""
    pheno = _pheno(tmp_path)
    pheno.adata = ad.AnnData(X=np.random.default_rng(0).normal(size=(8, 6)))
    with pytest.raises(ValueError, match='requires spatialgraph_stats'):
        pheno.spatialgraph_embedding(n_dim=3, save_transform=True, umap=False)


def test_map_query_without_saved_transform(tmp_path):
    pheno = _pheno(tmp_path)
    pheno.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    with pytest.raises(FileNotFoundError, match='No saved embedding transform'):
        pheno.spatialgraph_map_query()


def test_map_query_requires_stats(tmp_path):
    _fit_reference(tmp_path)
    pheno = _pheno(tmp_path)
    with pytest.raises(ValueError, match='self.adata is None'):
        pheno.spatialgraph_map_query()


# --- default path is unchanged --------------------------------------------------------


def test_default_embedding_still_uses_scanpy(tmp_path):
    """save_transform=False must leave the existing behaviour (and numbers) alone."""
    pheno = _pheno(tmp_path)
    pheno.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    pheno.spatialgraph_embedding(
        n_dim=5, scale=True, umap=True, n_neighbors=min(5, pheno.adata.shape[0] - 1)
    )
    assert 'X_pca' in pheno.adata.obsm
    assert 'X_umap' in pheno.adata.obsm
    # scanpy's PCA records loadings in varm; the sklearn path does not
    assert 'PCs' in pheno.adata.varm
    assert not (pheno.dir_reference / 'embedding_transform.joblib').exists()
