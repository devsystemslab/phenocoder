import scanpy as sc

from phenocoder.spatial import SpatialGraphAnalyzer
from tests.conftest import example_3d


def test_spatial_stats():
    adata = example_3d().sdata.tables['nuclei_features']
    adata = adata[adata.obs['well'] == 'A06']
    # generate labels
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=0.05)
    # run spatial graph analysis
    sga = SpatialGraphAnalyzer(
        adata,
        cluster_key='leiden',
        spatial_key='spatial',
        radii=(25, 50),
        index='5',
    )
    sga.run()
    sga.to_df()


if __name__ == '__main__':
    test_spatial_stats()
