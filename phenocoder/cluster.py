import numpy as np
import pynndescent
import muon as mu
import anndata as ad
import pandas as pd
import scanpy as sc
import rapids_singlecell as rsc
import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from numba import cuda
from tqdm import tqdm
# TODO: reduce number of used libraries. -> remove rapids, rmm


def majority_voting(x: np.ndarray, y: np.ndarray) -> int:
    """
    Majority voting for labels
    :param x:
    :param y:
    :return:
    """
    label_counts = np.unique(y[x], return_counts=True)
    return label_counts[0][np.argmax(label_counts[1])]


def build_search_tree(embedding: np.ndarray) -> pynndescent.NNDescent:
    """
    Build search tree
    :param embedding:
    :return:
    """
    print('Building search tree...')
    index = pynndescent.NNDescent(embedding)
    index.prepare()
    return index


def project_clustering(
    adata: ad.AnnData,
    adata_ref: ad.AnnData,
    chunk_size: int = 50000,
    k: int = 25,
    epsilon: float = 0.2,
):
    """
    :param adata:
    :param adata_ref:
    :param chunk_size:
    :param k:
    :param epsilon:
    :return:
    """
    index = build_search_tree(adata_ref.obsm['X_pca'])

    y_pred = np.asarray(adata_ref.obs['leiden'].values.astype(int))
    labels = []
    print()
    for i in tqdm(
        range(0, adata.obsm['X_pca'].shape[0], chunk_size), desc='Projecting clusters'
    ):
        neighbors = index.query(
            adata.obsm['X_pca'][i : i + chunk_size], k=k, epsilon=epsilon
        )[0]
        labels.append(np.asarray([majority_voting(x, y_pred) for x in neighbors]))
    adata.obs['leiden'] = np.concatenate(labels)
    return adata


def get_accuracy(adata: ad.AnnData, adata_sampled: ad.AnnData) -> float:
    """
    Get accuracy
    :param adata:
    :param adata_sampled:
    :return:
    """
    n_sampled = adata_sampled.shape[0]
    pred_true = adata.obs.loc[adata_sampled.obs.index]['leiden'] == adata_sampled.obs[
        'leiden'
    ].astype(int)
    return np.sum(pred_true) / n_sampled


def clustering(
    adata: ad.AnnData,
    dry_run: bool = False,
    resolution: float = 0.3,
    run_pca: bool = True,
    run_harmony: bool = False,
    n_comps: int = 6,
    use_gpu=True,
    reset_gpu=False,
    layer=None,
    var_subset=None,
) -> ad.AnnData:
    """
    Run clustering pipeline on adata
    :param adata:
    :param dry_run:
    :param resolution:
    :param run_pca:
    :param run_harmony:
    :param n_comps:
    :param use_gpu:
    :param reset_gpu:
    :param layer:
    :return:
    """
    if use_gpu:
        print('Setting up GPU...')
        if reset_gpu:
            device = cuda.get_current_device()
            device.reset()
        rmm.reinitialize(managed_memory=True, pool_allocator=False, devices=0)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print('Running clustering pipeline...')
        rsc.get.anndata_to_GPU(adata)
        if run_pca:
            print('Running PCA...')
            if n_comps >= adata.X.shape[1]:
                n_comps = adata.X.shape[1] - 1
            rsc.tl.pca(adata, n_comps=n_comps, layer=layer, mask_var=var_subset)
        if not dry_run:
            if run_harmony:
                print('Running harmony...')
                rsc.pp.harmony_integrate(adata, key=['plate_id'], max_iter_harmony=30)
                print('Computing neighbors...')
                rsc.pp.neighbors(adata, use_rep='X_pca_harmony')
            else:
                print('Computing neighbors...')
                rsc.pp.neighbors(adata, use_rep='X_pca')
            print('Running UMAP...')
            rsc.tl.umap(adata, n_components=2)
            print('Running leiden clustering...')
            rsc.tl.leiden(adata, resolution=resolution)
        rsc.get.anndata_to_CPU(adata)
    else:
        if run_pca:
            print('Running PCA...')
            if adata.X.shape[1] == 1:
                adata.obsm['X_pca'] = adata.X.copy()
            else:
                if n_comps >= adata.X.shape[1]:
                    n_comps = adata.X.shape[1] - 1
            sc.tl.pca(adata, n_comps=n_comps, layer=layer, mask_var=var_subset)
        if not dry_run:
            if run_harmony:
                print('Running harmony...')
                rsc.pp.harmony_integrate(adata, key=['plate_id'], max_iter_harmony=30)
                print('Computing neighbors...')
                sc.pp.neighbors(adata, use_rep='X_pca_harmony')
            else:
                print('Computing neighbors...')
                sc.pp.neighbors(adata, use_rep='X_pca')
            print('Running UMAP...')
            sc.tl.umap(adata, n_components=2)
            print('Running leiden clustering...')
            sc.tl.leiden(adata, resolution=resolution)

    return adata


def run_clustering(
    adata: ad.AnnData,
    subsampling: bool,
    frac: float,
    resolution: float,
    n_comps: int,
    harmony: bool,
    pca: bool = True,
    use_gpu: bool = True,
    var_subset: str = None,
) -> ad.AnnData:
    """
    Run clustering pipeline
    :param adata:
    :param subsampling:
    :param frac:
    :param resolution:
    :param n_comps:
    :param harmony:
    :param pca:
    :param use_gpu:
    :return:
    """

    if subsampling:
        print('Running clustering pipeline with subsampling...')
        adata = clustering(
            adata, dry_run=True, run_pca=True, n_comps=n_comps, use_gpu=use_gpu
        )
        adata_sampled = adata[
            np.random.choice(adata.obs.index, int(frac * adata.shape[0]), replace=False)
        ].copy()
        adata_sampled = clustering(
            adata_sampled,
            resolution=resolution,
            run_pca=False,
            n_comps=n_comps,
            use_gpu=use_gpu,
        )
        adata = project_clustering(adata, adata_sampled)
        accuracy = get_accuracy(adata, adata_sampled)
        adata.uns['label_transfer_accuracy'] = accuracy
        adata.obs['leiden'] = adata.obs['leiden'].astype(str)
        adata.obs['leiden'] = pd.Categorical(adata.obs['leiden'])
        adata.uns['adata_sampled'] = adata_sampled
    else:
        print('Running clustering pipeline...')
        adata = clustering(
            adata,
            resolution=resolution,
            run_pca=pca,
            run_harmony=harmony,
            n_comps=n_comps,
            use_gpu=use_gpu,
            var_subset=var_subset,
        )

    return adata


def run_clustering_pipeline(
    mdata: mu.MuData, use_gpu: bool, n_comps_pca: dict, res: dict
) -> mu.MuData:
    """
    Run clustering pipeline
    :param mdata:
    :param use_gpu:
    :param n_comps_pca:
    :param res:
    :return:
    """
    for mod in mdata.mod_names:
        print(f'Running clustering for {mod}...')
        mdata.mod[mod] = run_clustering(
            mdata[mod],
            subsampling=True,
            frac=0.1,
            resolution=res[mod],
            n_comps=n_comps_pca[mod],
            harmony=False,
            use_gpu=use_gpu,
        )
        mdata.update()
    return mdata
