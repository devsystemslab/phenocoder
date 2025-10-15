import numpy as np
import muon as mu
import pynndescent
import anndata as ad
import pandas as pd
import scanpy as sc
from tqdm import tqdm

# TODO: Add spatialdata import
# TODO: Refactor to work with sdata.tables instead of AnnData/MuData
# TODO: Store clustering results in sdata.tables instead of modifying AnnData in-place


def majority_voting(x: np.ndarray, y: np.ndarray) -> int:
    """Majority voting for labels.

    Parameters
    ----------
    x : np.ndarray
        Array of indices.
    y : np.ndarray
        Array of labels.

    Returns
    -------
    int
        Most frequent label among the selected indices.
    """
    label_counts = np.unique(y[x], return_counts=True)
    return label_counts[0][np.argmax(label_counts[1])]


def build_search_tree(embedding: np.ndarray) -> pynndescent.NNDescent:
    """Build search tree for nearest neighbor queries.

    Parameters
    ----------
    embedding : np.ndarray
        Input embedding array.

    Returns
    -------
    pynndescent.NNDescent
        Prepared search index.
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
    """Project clustering labels from reference to query data.

    Parameters
    ----------
    adata : ad.AnnData
        Query AnnData object to project labels to.
    adata_ref : ad.AnnData
        Reference AnnData object with existing labels.
    chunk_size : int, optional
        Number of cells to process in each chunk, by default 50000.
    k : int, optional
        Number of nearest neighbors to use, by default 25.
    epsilon : float, optional
        Precision parameter for approximate search, by default 0.2.

    Returns
    -------
    ad.AnnData
        Query AnnData object with projected leiden labels.
    """
    # TODO: Refactor to work with sdata tables - assumes adata.obsm['X_pca'] and adata.obs['leiden'] exist
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


def get_accuracy(adata: ad.AnnData, adata_sampled: ad.AnnData, label_key: str) -> float:
    """Calculate label transfer accuracy.

    Parameters
    ----------
    adata : ad.AnnData
        Full dataset with projected labels.
    adata_sampled : ad.AnnData
        Sampled dataset with ground truth labels.
    label_key : str
        Key for the label column to compare.

    Returns
    -------
    float
        Accuracy as fraction of correctly predicted labels.
    """
    n_sampled = adata_sampled.shape[0]
    pred_true = adata.obs.loc[adata_sampled.obs.index][label_key] == adata_sampled.obs[
        label_key
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
    """Run clustering pipeline on AnnData object.

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData object.
    dry_run : bool, optional
        If True, only run PCA without clustering steps, by default False.
    resolution : float, optional
        Leiden clustering resolution, by default 0.3.
    run_pca : bool, optional
        Whether to run PCA, by default True.
    run_harmony : bool, optional
        Whether to run Harmony integration, by default False.
    n_comps : int, optional
        Number of principal components, by default 6.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
    reset_gpu : bool, optional
        Whether to reset GPU memory, by default False.
    layer : str, optional
        Layer to use for analysis, by default None.
    var_subset : array-like, optional
        Subset of variables to use, by default None.

    Returns
    -------
    ad.AnnData
        AnnData object with clustering results.
    """
    # TODO: Refactor to work with sdata - uses scanpy methods directly on AnnData (sc.tl.pca, sc.pp.neighbors, etc.)
    # TODO: Results (PCA, UMAP, leiden) should be stored in sdata.tables
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
            sc.pp.harmony_integrate(adata, key=['plate_id'], max_iter_harmony=30)
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
    """Run clustering pipeline with optional subsampling.

    Parameters
    ----------
    adata : ad.AnnData
        Input AnnData object.
    subsampling : bool
        Whether to use subsampling strategy for large datasets.
    frac : float
        Fraction of cells to subsample.
    resolution : float
        Leiden clustering resolution.
    n_comps : int
        Number of principal components.
    harmony : bool
        Whether to run Harmony integration.
    pca : bool, optional
        Whether to run PCA, by default True.
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default True.
    var_subset : str, optional
        Subset of variables to use, by default None.

    Returns
    -------
    ad.AnnData
        AnnData object with clustering results.
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
    """Run clustering pipeline on MuData object.

    Parameters
    ----------
    mdata : mu.MuData
        Input MuData object with multiple modalities.
    use_gpu : bool
        Whether to use GPU acceleration.
    n_comps_pca : dict
        Dictionary mapping modality names to number of PCA components.
    res : dict
        Dictionary mapping modality names to clustering resolutions.

    Returns
    -------
    mu.MuData
        MuData object with clustering results for all modalities.
    """
    # TODO: Refactor to accept sdata instead of MuData
    # TODO: Return sdata-integrated result instead of MuData
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
