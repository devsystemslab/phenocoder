from phenocoder.sampling import SpatialSubunitSampler
from tests.conftest import example_3d


def test_subsampling():
    adata = example_3d().sdata.tables['nuclei_features']
    adata = adata[adata.obs['well'] == 'A06']
    spss = SpatialSubunitSampler(
        adata=adata,
        dim_subunit=100,
        min_obs=5,
        spatial_key='spatial',
    )
    spss.partition()
    spss.filter()
    spss.sample(max_obs=500)
    df = spss.to_df()
    # merge df to adata.obs
    adata.obs = adata.obs.merge(df, left_index=True, right_index=True, how='left')
    print(adata.obs)


if __name__ == '__main__':
    test_subsampling()
