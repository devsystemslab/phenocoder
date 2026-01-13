import shutil

import scanpy as sc
from spatialdata import SpatialData

from tests.conftest import example_3d


def test_workflow():
    pheno = example_3d()
    assert pheno.sdata is not None
    assert isinstance(pheno.sdata, SpatialData)
    pheno.generate_dataset(
        dataset='dataset_1',
        dir_dataset='tests/data/tmp/phenocoder',
        spatial_key_index='spatial_index',
    )
    pheno.initialize_model(n_latent_dim=32, n_dense_dim=64, conditions=['dataset', 'z'])
    pheno.train(n_epochs=5)
    pheno.encode(spatial_key_index='spatial_index')

    # Add clustering for spatial graph statistics
    sc.pp.pca(pheno.sdata.tables['phenocoder'])
    sc.pp.neighbors(pheno.sdata.tables['phenocoder'])
    sc.tl.leiden(pheno.sdata.tables['phenocoder'], resolution=1)

    # Test 1: Sample-level spatial graph statistics
    pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        table_key='phenocoder',
        use_subunits=False,
    )

    # Verify sample-level results
    assert pheno.adata is not None
    assert pheno.adata.shape[0] > 0  # At least one sample analyzed
    assert pheno.sample_key in pheno.adata.obs.columns or pheno.adata.obs.index.name == pheno.sample_key

    # Test 2: Subunit-level spatial graph statistics
    pheno.spatialgraph_stats(
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        table_key='phenocoder',
        use_subunits=True,
        dim_subunit=(200, 200, 200),
        min_obs_per_subunit=10,  # Lower threshold for test data
        max_obs_per_subunit=None,
        verbose=True,
    )

    # Verify subunit-level results
    assert pheno.adata is not None
    assert pheno.adata.shape[0] > 0  # At least one subunit analyzed
    assert pheno.sample_key in pheno.adata.obs.columns
    assert 'subunit_id' in pheno.adata.obs.columns
    assert 'subunit_key' in pheno.adata.obs.columns
    assert 'subunit_n_obs' in pheno.adata.obs.columns

    # Test 3: Spatial graph embedding on subunit-level data
    n_subunits = pheno.adata.shape[0]

    pheno.spatialgraph_embedding(
        n_dim=32,
        scale=True,
        variable_features=True,
        batch_correction=False,
        n_neighbors=min(15, n_subunits - 1),  # Ensure n_neighbors < n_subunits
        umap=True,
    )

    # Verify embedding results
    assert 'X_pca' in pheno.adata.obsm
    assert pheno.adata.obsm['X_pca'].shape[0] == n_subunits
    assert 'X_umap' in pheno.adata.obsm
    assert pheno.adata.obsm['X_umap'].shape[0] == n_subunits
    assert pheno.adata.obsm['X_umap'].shape[1] == 2

    shutil.rmtree('tests/data/tmp/phenocoder')


if __name__ == '__main__':
    test_workflow()
