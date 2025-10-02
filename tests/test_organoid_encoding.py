from phenocoder.phenocode import encode_nuclei_patches, encode_grid_patches, merge_adata
from phenocoder.plot import plot_organoid_test
import numpy as np
from skimage.color import label2rgb
from skimage.util import montage
from skimage import io
import matplotlib.pyplot as plt
import scanpy as sc


def test_nuclei_patch_encoding():
    adata_target = encode_nuclei_patches(
        well_ids=['A08', 'A09'],
        plate='004',
        dir_screen='/pstore/data/ihb-tumoroidscreen/data/processed/inhibitors',
        cycle='03',
        use_registered=True,
        label_type='target',
        dir_model='/pstore/data/ihb-tumoroidscreen/data/processed/inhibitors/phenocoder/2nd_cycle_nuclei/models/cond_latent_32_dense_64_dropout_0.25_beta_0.01_20241029-230904',
    )
    adata_source = encode_nuclei_patches(
        well_ids=['A08', 'A09'],
        plate='004',
        dir_screen='/pstore/data/ihb-tumoroidscreen/data/processed/inhibitors',
        cycle='01',
        use_registered=True,
        label_type='source',
        dir_model='/pstore/data/ihb-tumoroidscreen/data/processed/inhibitors/phenocoder/1st_cycle_nuclei/models/cond_latent_32_dense_64_dropout_0.25_beta_0.01_20241030-000353',
    )

    adata = merge_adata(adata_source, adata_target)
    plot_organoid_test(adata, res=0.5)
    plot_organoid_test(adata, res=0.5, layer='message_passing')


def test_grid_encoding():
    adata_grid_1 = encode_grid_patches(
        well='A08',
        plate='004',
        dir_screen='/pstore/data/ihb-tumoroidscreen/data/processed/inhibitors',
        grid_resolution=1,
        cycle='03',
        dir_model='/pstore/data/ihb-tumoroidscreen/data/processed/inhibitors/phenocoder/2nd_cycle_nuclei/models/cond_latent_32_dense_64_dropout_0.25_beta_0.01_20241029-230904',
        filter_encodable_conditions=False,
    )
    sc.tl.pca(adata_grid_1, n_comps=16)
    sc.pp.neighbors(adata_grid_1)
    sc.tl.leiden(adata_grid_1, resolution=0.5)
    img = np.empty(
        (
            adata_grid_1.obs['z'].max(),
            adata_grid_1.obs['x'].max() + 1,
            adata_grid_1.obs['y'].max() + 1,
        )
    )
    for i, row in adata_grid_1.obs.iterrows():
        img[row['z'] - 1, row['x'], row['y']] = int(row['leiden'])

    for i in np.unique(img):
        img_tmp = np.where(img == i, i, 0)
        img_tmp = label2rgb(img) * img_tmp[..., None]
        img_tmp = montage(img_tmp, channel_axis=-1)
        io.imshow(img_tmp)
        plt.show()

    adata_grid_2 = encode_grid_patches(
        well='A08',
        plate='HM004',
        dir_screen='/pstore/data/ihb-tumoroidscreen/data/processed/tumoroidscreen',
        cycle='03',
        dir_model='/pstore/data/ihb-tumoroidscreen/data/processed/tumoroidscreen/phenocoder/2nd_cycle/models/cond_latent_64_dense_128_dropout_0.25_20241004-123950',
        filter_encodable_conditions=False,
    )

    adata = merge_adata(adata_grid_1, adata_grid_2)


if __name__ == '__main__':
    test_nuclei_patch_encoding()
    # test_grid_encoding()
