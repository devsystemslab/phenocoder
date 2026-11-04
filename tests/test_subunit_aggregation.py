import anndata as ad
import numpy as np
import pandas as pd
import pytest

import phenocoder as phc


def _subunit_pheno(counts=(4, 3), n_features=3, seed=0):
    """Phenocoder holding a hand-built subunit table, one entry per sample in counts.

    Built directly rather than via spatialgraph_stats: the aggregation only reads
    adata_subunits, so a small deterministic table makes the expected values checkable
    by hand and keeps these tests fast.
    """
    rng = np.random.default_rng(seed)
    samples = []
    for i, n in enumerate(counts):
        samples.extend([f'S{i:02d}'] * n)
    n_obs = len(samples)
    X = rng.normal(size=(n_obs, n_features))
    obs = pd.DataFrame(
        {
            'well': samples,
            'subunit_n_obs': np.arange(10, 10 + n_obs),
        },
        index=[f'sub{i}' for i in range(n_obs)],
    )
    adata = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=[f'feat{i}' for i in range(n_features)]),
    )
    return phc.Phenocoder(adata_subunits=adata, sample_key='well')


def test_aggregate_shapes_and_naming():
    """One row per sample, one column per feature x descriptor, desc: prefixed."""
    pheno = _subunit_pheno(counts=(4, 3), n_features=3)

    pheno.aggregate_subunits(descriptors=('mean', 'sd'))
    agg = pheno.adata_subunits_agg

    assert agg.shape == (2, 6)
    assert list(agg.obs['well']) == ['S00', 'S01']
    assert list(agg.var_names) == [
        'desc:mean_feat0',
        'desc:mean_feat1',
        'desc:mean_feat2',
        'desc:sd_feat0',
        'desc:sd_feat1',
        'desc:sd_feat2',
    ]


def test_aggregate_values_match_groupby():
    """Each descriptor equals the plain pandas aggregation of the subunit values.

    The prefix naming is what makes a wrong-column bug silent, so the values are
    checked per descriptor rather than only in aggregate.
    """
    pheno = _subunit_pheno(counts=(5, 4), n_features=2)
    df = pheno.adata_subunits.to_df()
    df['well'] = pheno.adata_subunits.obs['well'].to_numpy()
    grouped = df.groupby('well')

    pheno.aggregate_subunits(descriptors=('mean', 'median', 'sd', 'var', 'skew'))
    agg = pheno.adata_subunits_agg

    expected = {
        'mean': grouped.mean(),
        'median': grouped.median(),
        'sd': grouped.std(),
        'var': grouped.var(),
        'skew': grouped.skew(),
    }
    for label, frame in expected.items():
        for sample in ['S00', 'S01']:
            for feature in ['feat0', 'feat1']:
                got = agg[agg.obs['well'] == sample, f'desc:{label}_{feature}'].X[0, 0]
                assert np.isclose(got, frame.loc[sample, feature]), (
                    f'{label} mismatch for {sample}/{feature}'
                )


def test_percentiles_expand_to_one_column_each():
    pheno = _subunit_pheno(counts=(5,), n_features=2)
    df = pheno.adata_subunits.to_df()
    df['well'] = pheno.adata_subunits.obs['well'].to_numpy()

    pheno.aggregate_subunits(descriptors=('percentile',), percentiles=(0.1, 0.5, 0.9))
    agg = pheno.adata_subunits_agg

    assert agg.shape == (1, 6)
    assert 'desc:p10_feat0' in agg.var_names
    assert 'desc:p50_feat0' in agg.var_names
    assert 'desc:p90_feat0' in agg.var_names
    expected = df.groupby('well').quantile(0.9).loc['S00', 'feat1']
    got = agg[:, 'desc:p90_feat1'].X[0, 0]
    assert np.isclose(got, expected)


def test_descriptor_order_follows_argument():
    """Column order is deterministic, so a saved reference stays alignable."""
    pheno = _subunit_pheno(counts=(4,), n_features=1)

    pheno.aggregate_subunits(descriptors=('sd', 'mean'))

    assert list(pheno.adata_subunits_agg.var_names) == [
        'desc:sd_feat0',
        'desc:mean_feat0',
    ]


@pytest.mark.parametrize(
    ('descriptor', 'n_subunits'),
    [('sd', 1), ('skew', 2), ('kurt', 3)],
)
def test_underpowered_samples_are_refused(descriptor, n_subunits, capsys):
    """A sample with too few subunits is skipped rather than given a fabricated value.

    pandas returns NaN below these counts, and the downstream zero-fill would turn an
    unknown spread into a reported spread of exactly 0 -- indistinguishable from a
    genuinely uniform sample.
    """
    pheno = _subunit_pheno(counts=(6, n_subunits), n_features=2)

    pheno.aggregate_subunits(descriptors=(descriptor,), min_subunits=1)
    out = capsys.readouterr().out

    assert list(pheno.adata_subunits_agg.obs['well']) == ['S00']
    assert 'S01' in out
    assert not np.isnan(pheno.adata_subunits_agg.X).any()


def test_min_subunits_filters_independently_of_descriptors():
    pheno = _subunit_pheno(counts=(6, 3), n_features=2)

    pheno.aggregate_subunits(descriptors=('mean',), min_subunits=5)

    assert list(pheno.adata_subunits_agg.obs['well']) == ['S00']


def test_no_surviving_sample_raises():
    pheno = _subunit_pheno(counts=(2, 2), n_features=2)

    with pytest.raises(ValueError, match='No sample has'):
        pheno.aggregate_subunits(descriptors=('kurt',))


def test_invalid_inputs_raise():
    pheno = _subunit_pheno(counts=(5,), n_features=2)

    with pytest.raises(ValueError, match='Unknown descriptor'):
        pheno.aggregate_subunits(descriptors=('mean', 'mode'))
    with pytest.raises(ValueError, match='Duplicate descriptor'):
        pheno.aggregate_subunits(descriptors=('mean', 'mean'))
    with pytest.raises(ValueError, match='percentiles must lie'):
        pheno.aggregate_subunits(descriptors=('percentile',), percentiles=(1.5,))
    with pytest.raises(ValueError, match='at least one'):
        pheno.aggregate_subunits(descriptors=())


def test_missing_subunits_table_raises():
    pheno = phc.Phenocoder(sample_key='well')

    with pytest.raises(ValueError, match='use_subunits=True'):
        pheno.aggregate_subunits()


def test_obs_carries_subunit_counts():
    pheno = _subunit_pheno(counts=(4, 3), n_features=2)
    expected_obs = (
        pheno.adata_subunits.obs.groupby('well', observed=True)['subunit_n_obs']
        .sum()
        .to_dict()
    )

    pheno.aggregate_subunits(descriptors=('mean',))
    obs = pheno.adata_subunits_agg.obs

    assert list(obs['n_subunits']) == [4, 3]
    assert obs.loc['S00', 'subunit_n_obs'] == expected_obs['S00']
    assert obs.loc['S01', 'subunit_n_obs'] == expected_obs['S01']


def test_aggregation_leaves_source_tables_untouched():
    """The subunit table is the input, not scratch space."""
    pheno = _subunit_pheno(counts=(4, 3), n_features=2)
    before_shape = pheno.adata_subunits.shape
    before_X = pheno.adata_subunits.X.copy()
    before_cols = list(pheno.adata_subunits.obs.columns)

    pheno.aggregate_subunits(descriptors=('mean', 'sd'))

    assert pheno.adata_subunits.shape == before_shape
    assert np.array_equal(pheno.adata_subunits.X, before_X)
    assert list(pheno.adata_subunits.obs.columns) == before_cols
    assert pheno.adata is None, 'aggregation must not populate the sample-level table'


def test_aggregation_recorded_in_stats_config():
    """Reference mapping needs the descriptor set; align_features cannot infer it."""
    pheno = _subunit_pheno(counts=(5,), n_features=2)

    pheno.aggregate_subunits(descriptors=('mean', 'percentile'), percentiles=(0.25,))

    assert pheno.stats_config['aggregation'] == {
        'descriptors': ['mean', 'percentile'],
        'percentiles': [0.25],
    }


def test_embedding_target_writes_only_the_chosen_table():
    """target routes both the input and the result; the other tables stay clean.

    self.adata is reassigned internally for non-default targets, so a leak here would
    put the coordinates on the wrong table or leave self.adata pointing at the wrong one.
    """
    pheno = _subunit_pheno(counts=(5, 4, 4), n_features=4)
    sample_level = ad.AnnData(
        X=np.arange(12.0).reshape(3, 4) ** 2,
        obs=pd.DataFrame(index=['S00', 'S01', 'S02']),
        var=pd.DataFrame(index=[f'feat{i}' for i in range(4)]),
    )
    pheno.adata = sample_level
    pheno.aggregate_subunits(descriptors=('mean', 'sd'))

    pheno.spatialgraph_embedding(
        n_dim=2, target='subunits_agg', min_variance=0.0, n_neighbors=2, umap=False
    )

    assert 'X_pca' in pheno.adata_subunits_agg.obsm
    assert pheno.adata_subunits_agg.obsm['X_pca'].shape == (3, 2)
    assert 'X_pca' not in pheno.adata.obsm, 'sample-level table must be untouched'
    assert 'X_pca' not in pheno.adata_subunits.obsm, 'subunit table must be untouched'
    assert pheno.adata is sample_level, 'self.adata must be restored'


def test_resolve_target_rejects_unknown_and_unset():
    pheno = _subunit_pheno(counts=(4,), n_features=2)

    with pytest.raises(ValueError, match='Unknown target'):
        pheno._resolve_target('nope')
    with pytest.raises(ValueError, match='aggregate_subunits'):
        pheno._resolve_target('subunits_agg')
    assert pheno._resolve_target('subunits') == 'adata_subunits'
