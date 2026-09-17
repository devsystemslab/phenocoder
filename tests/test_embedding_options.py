"""Tests for the optional branches of ``Phenocoder.spatialgraph_embedding``.

``obs_keys`` and ``batch_correction`` each carry their own metadata lookup and validation,
and both reach into ``sdata.tables[table_key].obs`` to resolve per-sample values that the
sample-level adata does not have. Neither branch runs on the default path, so nothing else
in the suite exercises them.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import phenocoder as phc


def _toy_sdata(n_samples=12, n_per_sample=60, n_clusters=3, seed=0):
    """Small table with spatial coords, cluster labels and per-sample metadata.

    Mirrors ``tests.test_reference._toy_sdata`` but adds ``condition``, ``plate`` and
    ``donor`` columns that are constant within a sample -- the shape ``obs_keys`` and
    ``batch_key`` expect.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_samples):
        weights = rng.dirichlet(np.ones(n_clusters))
        labels = rng.choice(n_clusters, size=n_per_sample, p=weights)
        coords = rng.uniform(0, 200, size=(n_per_sample, 3))
        for (x, y, z), lab in zip(coords, labels):
            rows.append(
                {
                    'well': f'w{s:02d}',
                    'leiden': str(lab),
                    # constant within a sample, which is what groupby().first() assumes
                    'condition': 'treated' if s % 2 else 'control',
                    'plate': f'p{s % 2}',
                    'donor': f'd{s % 3}',
                    'x': x,
                    'y': y,
                    'z': z,
                }
            )
    df = pd.DataFrame(rows)

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

    return sd.SpatialData(
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


def _pheno_with_stats(dir_project, **sdata_kwargs):
    """A Phenocoder with sample-level spatial statistics already computed."""
    pheno = phc.Phenocoder(
        table_key='cells', sample_key='well', dir_project=dir_project
    )
    pheno.add_sdata(_toy_sdata(**sdata_kwargs))
    pheno.spatialgraph_stats(
        cluster_key='leiden', spatial_key='spatial', radii=(25, 50), table_key='cells'
    )
    return pheno


def _embed(pheno, **kwargs):
    kwargs.setdefault('n_dim', 5)
    kwargs.setdefault('umap', False)
    kwargs.setdefault('n_neighbors', min(5, pheno.adata.shape[0] - 1))
    pheno.spatialgraph_embedding(**kwargs)


# --- obs_keys -------------------------------------------------------------------------


def test_obs_keys_maps_metadata_onto_samples(tmp_path):
    """Per-sample metadata is carried from the source table onto the sample-level adata.

    spatialgraph_stats builds that adata with an empty obs, so without this the UMAP
    cannot be colored by condition at all.
    """
    pheno = _pheno_with_stats(tmp_path)
    _embed(pheno, obs_keys='condition')

    assert 'condition' in pheno.adata.obs.columns
    assert not pheno.adata.obs['condition'].isna().any()
    # w00, w02, ... are control; odd wells are treated
    expected = {
        sample: ('treated' if int(sample[1:]) % 2 else 'control')
        for sample in pheno.adata.obs.index
    }
    assert pheno.adata.obs['condition'].to_dict() == expected


def test_obs_keys_accepts_a_list(tmp_path):
    """Several columns can be carried at once."""
    pheno = _pheno_with_stats(tmp_path)
    _embed(pheno, obs_keys=['condition', 'donor'])

    assert 'condition' in pheno.adata.obs.columns
    assert 'donor' in pheno.adata.obs.columns
    assert not pheno.adata.obs[['condition', 'donor']].isna().any().any()


def test_obs_keys_string_and_list_agree(tmp_path):
    """A bare string is normalised to a one-element list."""
    as_string = _pheno_with_stats(tmp_path / 'a')
    _embed(as_string, obs_keys='condition')
    as_list = _pheno_with_stats(tmp_path / 'b')
    _embed(as_list, obs_keys=['condition'])

    pd.testing.assert_series_equal(
        as_string.adata.obs['condition'], as_list.adata.obs['condition']
    )


def test_obs_keys_unknown_column_is_rejected(tmp_path):
    """A typo'd key fails loudly rather than yielding a silently all-NaN column."""
    pheno = _pheno_with_stats(tmp_path)
    with pytest.raises(ValueError, match='obs_key "nope" not found'):
        _embed(pheno, obs_keys='nope')


def test_obs_keys_requires_sdata(tmp_path):
    """Without a source table there is nothing to look the metadata up in."""
    pheno = _pheno_with_stats(tmp_path)
    pheno.sdata = None
    with pytest.raises(ValueError, match='obs_keys requires self.sdata'):
        _embed(pheno, obs_keys='condition')


def test_obs_keys_requires_sample_key_in_table(tmp_path):
    """The sample column is what the per-sample values are grouped by."""
    pheno = _pheno_with_stats(tmp_path)
    pheno.sample_key = 'not_a_column'
    with pytest.raises(ValueError, match='sample_key "not_a_column" not found'):
        _embed(pheno, obs_keys='condition')


def test_obs_keys_maps_onto_subunit_level_adata(tmp_path):
    """Subunit-level adata carries sample_key as a column, not as the index.

    The two levels resolve the sample identifier differently, so a lookup that always
    used the index would attach NaN to every subunit.
    """
    pheno = phc.Phenocoder(table_key='cells', sample_key='well', dir_project=tmp_path)
    pheno.add_sdata(_toy_sdata())
    pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        table_key='cells',
        use_subunits=True,
        dim_subunit=(100, 100, 100),
        min_obs_per_subunit=10,
    )
    # the branch under test: sample_key present in obs, index is the subunit
    assert 'well' in pheno.adata.obs.columns
    assert pheno.adata.shape[0] > 0

    _embed(pheno, obs_keys='condition')

    assert not pheno.adata.obs['condition'].isna().any()
    # each subunit gets the condition of the sample it came from
    expected = pheno.adata.obs['well'].map(
        lambda w: 'treated' if int(str(w)[1:]) % 2 else 'control'
    )
    assert (pheno.adata.obs['condition'] == expected).all()


def test_obs_keys_does_not_disturb_the_embedding(tmp_path):
    """Carrying metadata is an obs-only operation; the coordinates must not move."""
    without = _pheno_with_stats(tmp_path / 'a')
    _embed(without)
    with_keys = _pheno_with_stats(tmp_path / 'b')
    _embed(with_keys, obs_keys='condition')

    np.testing.assert_allclose(
        without.adata.obsm['X_pca'], with_keys.adata.obsm['X_pca'], atol=1e-6
    )


# --- batch_correction -----------------------------------------------------------------


def test_batch_correction_runs_and_changes_the_embedding(tmp_path):
    """Ridge regression on the batch key alters .X, and so the PCA coordinates."""
    plain = _pheno_with_stats(tmp_path / 'a')
    _embed(plain)
    corrected = _pheno_with_stats(tmp_path / 'b')
    _embed(corrected, batch_correction=True, batch_key='plate')

    assert 'X_pca' in corrected.adata.obsm
    assert corrected.adata.obsm['X_pca'].shape == plain.adata.obsm['X_pca'].shape
    # the batch key is looked up from the source table and attached to obs
    assert 'plate' in corrected.adata.obs.columns
    assert set(corrected.adata.obs['plate']) == {'p0', 'p1'}
    assert not np.allclose(
        plain.adata.obsm['X_pca'], corrected.adata.obsm['X_pca'], atol=1e-6
    )


def test_batch_correction_with_confounder(tmp_path):
    """A confounder is also resolved from the source table and passed to bbknn."""
    pheno = _pheno_with_stats(tmp_path)
    _embed(pheno, batch_correction=True, batch_key='plate', confounder_key='donor')

    assert 'plate' in pheno.adata.obs.columns
    assert 'donor' in pheno.adata.obs.columns
    assert not pheno.adata.obs['donor'].isna().any()
    assert 'X_pca' in pheno.adata.obsm


def test_batch_correction_confounder_accepts_a_list(tmp_path):
    pheno = _pheno_with_stats(tmp_path)
    _embed(
        pheno,
        batch_correction=True,
        batch_key='plate',
        confounder_key=['donor', 'condition'],
    )
    assert 'donor' in pheno.adata.obs.columns
    assert 'condition' in pheno.adata.obs.columns


def test_batch_correction_uses_batch_key_already_in_obs(tmp_path):
    """When obs already carries the batch column, no table lookup is needed."""
    pheno = _pheno_with_stats(tmp_path)
    pheno.adata.obs['plate'] = [f'p{i % 2}' for i in range(pheno.adata.shape[0])]
    before = pheno.adata.obs['plate'].copy()

    _embed(pheno, batch_correction=True, batch_key='plate')

    pd.testing.assert_series_equal(pheno.adata.obs['plate'], before)
    assert 'X_pca' in pheno.adata.obsm


def test_batch_correction_requires_batch_key(tmp_path):
    pheno = _pheno_with_stats(tmp_path)
    with pytest.raises(ValueError, match='batch_key must be specified'):
        _embed(pheno, batch_correction=True)


def test_batch_correction_unknown_batch_key_is_rejected(tmp_path):
    pheno = _pheno_with_stats(tmp_path)
    with pytest.raises(ValueError, match='batch_key "nope" not found'):
        _embed(pheno, batch_correction=True, batch_key='nope')


def test_batch_correction_requires_sdata_for_lookup(tmp_path):
    """A batch key absent from obs can only come from the source table."""
    pheno = _pheno_with_stats(tmp_path)
    pheno.sdata = None
    with pytest.raises(ValueError, match='sdata/table_key not available'):
        _embed(pheno, batch_correction=True, batch_key='plate')


def test_batch_correction_unknown_table_key_is_rejected(tmp_path):
    pheno = _pheno_with_stats(tmp_path)
    pheno.table_key = 'not_a_table'
    with pytest.raises(ValueError, match='table_key "not_a_table" not found'):
        _embed(pheno, batch_correction=True, batch_key='plate')


def test_batch_correction_requires_sample_key_in_table(tmp_path):
    """Without the sample column the batch value cannot be resolved per sample."""
    pheno = _pheno_with_stats(tmp_path)
    pheno.sample_key = 'not_a_column'
    with pytest.raises(ValueError, match='sample_key "not_a_column" not found'):
        _embed(pheno, batch_correction=True, batch_key='plate')
