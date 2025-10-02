import scanpy as sc
import bbknn
import muon as mu
import anndata as ad
import pandas as pd
import numpy as np


def organoid_embedding(
    df: pd.DataFrame,
    df_plate_layouts: pd.DataFrame = None,
    n_comps_pca: int = 64,
    res: float = 0.5,
    batch_correction: bool = True,
    confounder: str = None,
) -> ad.AnnData:
    """
    Process organoid embedding
    :param df:
    :param df_plate_layouts:
    :param n_comps_pca:
    :param res:
    :param batch_correction:
    :param confounder:
    :return:
    """
    df = df.loc[:, ~df.columns.duplicated()]
    adata = ad.AnnData(df.drop(columns=["well_id", "plate_id"]))
    adata.obs.index = adata.obs.index.astype(str)
    adata.var_names = df.drop(columns=["well_id", "plate_id"]).columns.tolist()
    adata.obs = df.drop(columns=adata.var_names)
    adata.layers["raw"] = adata.X.copy()
    sc.pp.scale(adata)
    adata.X[np.isnan(adata.X)] = 0

    if df_plate_layouts is not None:
        adata.obs = pd.merge(
            adata.obs, df_plate_layouts, on=["well_id", "plate_id"], how="left"
        )
        adata.obs.index = adata.obs["well_id"] + "_" + adata.obs["plate_id"]
        if batch_correction:
            if confounder is None:
                confounder = []
            bbknn.ridge_regression(
                adata, batch_key=["plate_id"], confounder_key=confounder
            )
    sc.pp.highly_variable_genes(adata)
    if n_comps_pca > adata.obs.shape[0]:
        n_comps_pca = adata.obs.shape[0] - 1
    sc.pp.pca(adata, n_comps=n_comps_pca, use_highly_variable=True)

    if batch_correction:
        bbknn.bbknn(adata, batch_key="plate_id")
    else:
        sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=res)
    sc.tl.umap(adata, n_components=2)
    return adata


def run_organoid_embedding(
    spatial_dict: dict,
    df_plate_layouts: pd.DataFrame,
    batch_correction: bool = False,
    confounder: str = None,
    combine_modalities: bool = True,
    n_comps_pca=None,
    res=None,
) -> mu.MuData:
    """
    Organoid embedding
    :param spatial_dict:
    :param df_plate_layouts:
    :param batch_correction:
    :param confounder:
    :param n_comps_pca:
    :param res:
    :return:
    """

    adata_dict = {}
    for mod in spatial_dict.keys():
        print(f"Generating organoid embedding for {mod}...")
        adata_dict[mod] = organoid_embedding(
            spatial_dict[mod],
            df_plate_layouts,
            n_comps_pca=n_comps_pca["org_embedding"],
            res=res["org_embedding"],
            batch_correction=batch_correction,
            confounder=confounder,
        )

    if combine_modalities:
        print("Generating combined phenocoder organoid embeddings...")
        adata_dict["phenocoder_combined"] = organoid_embedding(
            pd.concat(
                [spatial_dict["phenocoder"], spatial_dict["phenocoder_msg"]], axis=1
            ),
            df_plate_layouts,
            n_comps_pca=n_comps_pca["org_embedding"],
            res=res["org_embedding"],
            batch_correction=batch_correction,
            confounder=confounder,
        )

        print("Generating combined imputed organoid embeddings...")
        adata_dict["imputed_combined"] = organoid_embedding(
            pd.concat(
                [
                    spatial_dict["imputed_nuclei_bytimepoints_False"],
                    spatial_dict["imputed_neighbors_bytimepoints_False"],
                ],
                axis=1,
            ),
            df_plate_layouts,
            n_comps_pca=n_comps_pca["org_embedding"],
            res=res["org_embedding"],
            batch_correction=batch_correction,
            confounder=confounder,
        )

        print("Generating combined organoid embeddings with all modalities...")
        adata_dict["all_combined"] = organoid_embedding(
            pd.concat(
                [
                    spatial_dict[mod]
                    for mod in spatial_dict.keys()
                    if "phenocoder" in mod or "imputed" in mod
                ],
                axis=1,
            ),
            df_plate_layouts,
            n_comps_pca=n_comps_pca["org_embedding"],
            res=res["org_embedding"],
            batch_correction=batch_correction,
            confounder=confounder,
        )

    mdata_org = mu.MuData(adata_dict)

    return mdata_org
