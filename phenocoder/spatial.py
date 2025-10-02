import warnings

from autoencoder.miniCAMPA.cluster import cluster_latent_space

warnings.filterwarnings("ignore", category=RuntimeWarning, module='squidpy')
warnings.filterwarnings("ignore", category=FutureWarning, module='pandas')

import sys
sys.path.append('/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen')
from whole_mount_tumoroid.phenocoder.utils import setup_dask_slurm_client

import pandas as pd
import numpy as np
import squidpy as sq
from tqdm import tqdm
from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
from scipy.spatial import ConvexHull
import anndata as ad
from dask.distributed import as_completed
import dask.array as da

def get_chull(adata: ad.AnnData, well: str, plate: str, radius: int = 100, degree_threshold: int = 5,
              filter_obs: bool = False, convert_units: bool = True) -> float:
    """
    Get convex hull volume
    :param well:
    :param plate:
    :param adata:
    :param radius:
    :param degree_threshold:
    :param filter_obs:
    :param convert_units:
    :return:
    """
    if filter_obs:
        adata = adata[adata.obs['plate'] == plate]
        adata = adata[adata.obs['well'] == well]
    coordinate_cols = ['z', 'centroid-0', 'centroid-1']
    df = adata.obs[coordinate_cols]
    if len(df) < 4:
        return pd.DataFrame({'volume_chull': 0, 'area_chull': 0, 'density_chull':0}, index=[well + '_' + plate])
    pts = df.to_numpy()
    # neighbor graph
    graph = nx.from_numpy_array(radius_neighbors_graph(pts, radius, mode='distance', include_self=False).toarray(),
                                create_using=nx.DiGraph)
    A = nx.adjacency_matrix(graph, weight=None)
    degrees = np.sum(A, axis=1)
    # filter df with degrees
    df_filtered = df[degrees > degree_threshold]
    if len(df_filtered) < 4:
        return pd.DataFrame({'volume_chull': 0, 'area_chull': 0, 'density_chull':0}, index=[well + '_' + plate])
    pts = df_filtered[coordinate_cols].to_numpy()
    for i in range(pts.shape[-1]):
        if len(np.unique(pts[:, i])) == 1:
            return pd.DataFrame({'volume_chull': 0, 'area_chull': 0, 'density_chull':0}, index=[well + '_' + plate])
    chull = ConvexHull(pts)
    df_results = pd.DataFrame({'volume_chull': chull.volume, 'area_chull': chull.area}, index=[well + '_' + plate])
    if convert_units:
        df_results['volume_chull'] = df_results['volume_chull'] * 0.322 ** 3 / 1000 ** 3
        df_results['area_chull'] = df_results['area_chull'] * 0.322 ** 2 / 1000 ** 2
    df_results['density_chull'] = len(pts) / df_results['volume_chull']
    return df_results

def get_chulls_connected_components(adata: ad.AnnData, well: str, plate: str, clusters:list, radius:int = 100, min_nds=10, convert_units=True) -> pd.DataFrame:
    """
    Calculate convex hull for connected components in subset of spatial graph.
    :param adata:
    :param well:
    :param plate:
    :param clusters:
    :param filter_obs:
    :param radius:
    :param min_nds:
    :param convert_units:
    :return:
    """
    adata = adata[adata.obs['plate_id'] == plate]
    adata = adata[adata.obs['well_id'] == well]
    # get center of mass
    graph_center = adata.obs[['z', 'centroid-0', 'centroid-1']].to_numpy().mean(axis=0)
    adata = adata[adata.obs['leiden'].isin(clusters)]
    df = adata.obs[['z', 'centroid-0', 'centroid-1']]
    if df.empty:
        return pd.DataFrame()
    pts = df.to_numpy()
    # neighbor graph
    G = nx.from_numpy_array(radius_neighbors_graph(pts, radius, mode='distance', include_self=False).toarray(),
                                create_using=nx.DiGraph).to_undirected()
    # filter out points that have less than 3 connections
    for node in list(G.nodes):
        if G.degree[node] < 3:
            G.remove_node(node)
    if len(G.nodes) <= min_nds:
        return pd.DataFrame()
    # selected connected components
    df_results = []
    for i, component in enumerate(list(nx.connected_components(G))):
        if len(component) >= min_nds:
            pts_component = pts[list(component)]
            # unique values in each dim:
            n_unique = [len(np.unique(pts_component[:, i])) == 1 for i in range(pts_component.shape[-1])]
            if np.any(n_unique):
                continue
            chull = ConvexHull(pts_component)
            center_component = pts_component.mean(axis=0)
            distance_center = np.linalg.norm(center_component - graph_center)
            df_component = pd.DataFrame({'volume_chull': chull.volume, 'area_chull': chull.area,'n_pts':len(pts_component), 'well_id': well, 'plate_id':plate, 'distance_center':distance_center}, index=[str(i)])
            if convert_units:
                df_component['distance_center'] = df_component['distance_center'] * 0.322 / 1000
                df_component['area_chull'] = df_component['area_chull'] * 0.322 ** 2 / 1000 ** 2
                df_component['volume_chull'] = df_component['volume_chull'] * 0.322 ** 3 / 1000 ** 3
            df_component['density_chull'] = df_component['volume_chull'] / df_component['n_pts']
            df_results.append(df_component)
    if len(df_results) == 0:
        return pd.DataFrame()
    df_results = pd.concat(df_results)
    # filter n_pts for min_nds
    df_results = df_results[df_results['n_pts'] >= min_nds]

    # check plot
    # for i, component in enumerate(list(nx.connected_components(G))):
    #     if len(component) < min_nds:
    #         for node in component:
    #             G.remove_node(node)
    # A = nx.adjacency_matrix(G, weight=None)
    # conn_comps = connected_components(A)
    # # plot graph and label colors with connected components
    # fig = plt.figure(figsize=(10, 10))
    # # use pts as coordinate layout
    # pos = {i: pts[i,1:] for i in range(len(pts))}
    # nx.draw(G, pos=pos, node_size=10, node_color=conn_comps[1], cmap='tab20',edge_color='k', alpha=0.5, arrows=False)
    # plt.show()
    return df_results

def get_centrality(adata: ad.AnnData, well: str, plate: str) -> pd.DataFrame:
    """
    Get centrality
    :param adata:
    :param well:
    :param plate:
    :return:
    """
    # centrality scores
    sq.gr.centrality_scores(adata, cluster_key="leiden", connectivity_key="spatial")
    df_centrality = pd.DataFrame(adata.uns['leiden_centrality_scores'])
    cluster_names = pd.Categorical(adata.obs['leiden'].cat.categories).tolist()
    df_centrality.index = cluster_names
    # pivot wide new column names are column_names-index
    df_centrality = df_centrality.stack().reset_index()
    df_centrality.columns = ['source', 'target', 'value']
    df_centrality.index = 'centrality_' + df_centrality['source'] + '_' + df_centrality['target']
    df_centrality = df_centrality.drop(columns=['source', 'target'])
    df_centrality = df_centrality.T
    df_centrality.index = [well + '_' + plate]
    return df_centrality


def get_interactions(adata: ad.AnnData, well: str, plate: str) -> pd.DataFrame:
    """
    Get interactions
    :param adata:
    :param well:
    :param plate:
    :return:
    """

    # interaction matrix
    results = []
    for i in (True, False):
        sq.gr.interaction_matrix(adata, cluster_key="leiden", normalized=i)
        df_interaction = pd.DataFrame(adata.uns['leiden_interactions'])
        cluster_names = pd.Categorical(adata.obs['leiden'].cat.categories).tolist()
        df_interaction.columns = cluster_names
        df_interaction.index = cluster_names
        df_interaction = df_interaction.stack().reset_index()
        df_interaction.columns = ['source', 'target', 'count']
        if i:
            df_interaction.index = 'interaction_norm_' + df_interaction['source'] + '_' + df_interaction['target']
        else:
            df_interaction.index = 'interaction_' + df_interaction['source'] + '_' + df_interaction['target']
        df_interaction = df_interaction.drop(columns=['source', 'target'])
        df_interaction = df_interaction.T
        df_interaction = df_interaction.reset_index(drop=True)
        df_interaction.index = [well + '_' + plate]
        results.append(df_interaction)
    df_interaction = pd.concat(results, axis=1)
    return df_interaction


def get_connectivity(adata: ad.AnnData, well: str, plate: str) -> pd.DataFrame:
    """
    Get connectivity
    :param adata:
    :param well:
    :param plate:
    :return:
    """
    # get mean connectivity
    degrees = adata.obsp['spatial_connectivities'].sum(axis=0)
    mean_degree = degrees.mean()
    std_degree = degrees.std()
    df_degree = pd.DataFrame({'mean': mean_degree, 'std': std_degree}, index=[well + '_' + plate])
    # for each leiden cluster
    cluster_names = pd.Categorical(adata.obs['leiden'].cat.categories).tolist()
    mean_degree_leiden = [degrees[:, adata.obs['leiden'] == cluster].mean() for cluster in cluster_names]
    std_degree_leiden = [degrees[:, adata.obs['leiden'] == cluster].std() for cluster in cluster_names]
    df_degree_leiden = pd.DataFrame({'mean': mean_degree_leiden, 'std': std_degree_leiden},
                                    index=cluster_names)
    # pivot wide and merge with df_degree
    df_degree_leiden = df_degree_leiden.stack().reset_index()
    df_degree_leiden.columns = ['source', 'metric', 'value']
    df_degree_leiden = df_degree_leiden.pivot_table(index='source', columns='metric', values='value')
    df_degree_leiden.index = 'degree_' + df_degree_leiden.index
    df_degree_leiden = df_degree_leiden.T
    df_degree_leiden = df_degree_leiden.stack().reset_index()
    df_degree_leiden.columns = ['source', 'metric', 'value']
    df_degree_leiden['metric_combined'] = df_degree_leiden['metric'] + '_' + df_degree_leiden['source']
    # drop source and metric
    df_degree_leiden = df_degree_leiden.drop(columns=['source', 'metric'])
    # pivot wide so 1 row remains
    df_degree_leiden = df_degree_leiden.pivot_table(columns='metric_combined', values='value')
    df_degree_leiden.index = [well + '_' + plate]
    df_degree = pd.concat([df_degree, df_degree_leiden], axis=1)
    return df_degree


def get_neighborhood_enrichment(adata: ad.AnnData, well: str, plate: str) -> pd.DataFrame:
    """
    Get neighborhood enrichment
    :param adata:
    :param well:
    :param plate:
    :return:
    """
    sq.gr.nhood_enrichment(adata, cluster_key="leiden", connectivity_key="spatial", show_progress_bar=False)
    df_nhood = pd.DataFrame(adata.uns['leiden_nhood_enrichment']['zscore'])
    cluster_names = pd.Categorical(adata.obs['leiden'].cat.categories).tolist()
    df_nhood.columns = cluster_names
    df_nhood.index = cluster_names
    df_nhood = df_nhood.stack().reset_index()
    df_nhood.columns = ['source', 'target', 'value']
    df_nhood.index = 'nhood_z_' + df_nhood['source'].astype(str) + '_' + df_nhood['target'].astype(str)
    df_nhood = df_nhood.drop(columns=['source', 'target'])
    df_nhood = df_nhood.T
    df_nhood.index = [well + '_' + plate]
    return df_nhood


def get_moran(adata: ad.AnnData, well: str, plate: str) -> pd.DataFrame:
    """
    Get moran I
    :param adata:
    :param well:
    :param plate:
    :return:
    """
    if np.sum(adata.obsp.get('spatial_connectivities').toarray()) == 0:
        df_moran = pd.DataFrame(0, index=[well + '_' + plate], columns=adata.var_names)
        return df_moran
    sq.gr.spatial_autocorr(
        adata,
        mode="moran",
        connectivity_key="spatial_connectivities",
        genes=adata.var_names,
        n_perms=100,
        n_jobs=1,
        show_progress_bar=False)
    df_moran = pd.DataFrame(adata.uns['moranI'])
    df_moran = df_moran[['I']].T
    df_moran.index = [well + '_' + plate]
    df_moran.columns = ['moranI_' + col for col in df_moran.columns]
    return df_moran

def get_moran_cluster(adata: ad.AnnData, well: str, plate: str) -> pd.DataFrame:
    # one hot encode adata.obs['leiden']
    df = pd.get_dummies(adata.obs['leiden']).astype(int)
    adata_cl = ad.AnnData(df, obs=adata.obs, obsm=adata.obsm,obsp=adata.obsp)
    if np.sum(adata.obsp.get('spatial_connectivities').toarray()) == 0:
        df_moran = pd.DataFrame(0, index=[well + '_' + plate], columns=adata.var_names)
        return df_moran
    sq.gr.spatial_autocorr(
        adata_cl,
        mode="moran",
        connectivity_key="spatial_connectivities",
        genes=adata_cl.var_names,
        n_perms=100,
        n_jobs=1,
        show_progress_bar=False)
    df_moran = pd.DataFrame(adata_cl.uns['moranI'])
    df_moran = df_moran[['I']].T
    df_moran.index = [well + '_' + plate]
    # add moranI prefix to all columns
    df_moran.columns = ['moranI_' + col for col in df_moran.columns]
    return df_moran

def get_spatial_stats(well: str, plate: str, adata: ad.AnnData, radii: tuple[int] = (25, 50, 100, 150),
                      radius_chull: int = 100,
                      layer: str = None, cluster_key: str = None) -> pd.DataFrame:
    """
    Get spatial stats
    :param well: str well id
    :param plate: str plate id
    :param adata: ad.AnnData
    :param radii: tuple[int] radii
    :param radius_chull: int radius chull
    :param layer: str layer
    :param cluster_key: str cluster key
    :return:
    """
    adata = adata[adata.obs['well_id'] == well]
    adata = adata[adata.obs['plate_id'] == plate]
    adata = adata.copy()
    if layer is not None:
        adata.X = adata.layers[layer].copy()
        # remove layer
        adata.layers[layer] = None
    if cluster_key is not None:
        if 'leiden' in adata.obs.columns:
            adata.obs = adata.obs.drop(columns=['leiden'])
        adata.obs['leiden'] = adata.obs[cluster_key].astype(str)
    # generate spatial obsm
    adata.obsm['spatial3d'] = adata.obs[['centroid-0', 'centroid-1', 'z']].values.copy()
    df = []
    for radius in radii:
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type='generic', spatial_key='spatial3d')

        # calculate spatial features
        df_spatial_features = pd.concat([get_interactions(adata, well, plate),
                                         get_centrality(adata, well, plate),
                                         get_connectivity(adata, well, plate),
                                         get_moran(adata, well, plate),
                                         get_moran_cluster(adata, well, plate),
                                         get_neighborhood_enrichment(adata, well, plate)], axis=1)
        # radius as suffix to all columns
        df_spatial_features.columns = [col + f'_{radius}' for col in df_spatial_features.columns]
        df.append(df_spatial_features)

    df = pd.concat(df, axis=1)
    df_chull = get_chull(adata, well, plate, radius=radius_chull)
    df = pd.concat([df, df_chull], axis=1)
    return df

def run_spatial_feature_processing(mdata, radii:tuple[int] = (25, 50, 100, 150), radius_chull:int = 100, use_dask:bool = False):
    """
    Run spatial feature processing
    :param mdata:
    :param radii:
    :param radius_chull:
    :return:
    """

    if use_dask:
        client, cluster = setup_dask_slurm_client()

    spatial_dict = {}
    df_qc = []
    for mod in mdata.mod_names:
        n_cluster = len(mdata[mod].obs['leiden'].unique())
        assert n_cluster > 1
        df_tmp = mdata[mod].obs.groupby(['well_id', 'plate_id','leiden'], observed=False).size().reset_index(
            name='count')
        df_tmp[f'detected_{mod}'] = df_tmp['count'] > 0
        df_tmp = df_tmp.groupby(['well_id','plate_id'], observed=False)[f'detected_{mod}'].agg("sum").reset_index()
        df_tmp[f'detected_{mod}'] = df_tmp[f'detected_{mod}'] > 1
        # set index
        df_tmp = df_tmp.set_index(['well_id','plate_id'])
        df_qc.append(df_tmp)

    df_qc = pd.concat(df_qc, axis=1).all(axis=1).reset_index(name='selected')

    for mod in mdata.mod_names:
        df = mdata[mod].obs.groupby(['well_id', 'plate_id'], observed=False).size().reset_index(
            name='cell_count')
        df = df[df['cell_count'] > 0]
        df = df.merge(df_qc[df_qc['selected']], on=['well_id', 'plate_id'], how='inner')
        df = df.drop(columns=['selected'])
        df.index = df['well_id'].astype(str) + '_' + df['plate_id'].astype(str)

        if use_dask:
            mdata_copy = mdata[mod].copy()
            #print("Scattering")
            #mdata_scatter = client.scatter(mdata_copy, broadcast=True)
            futures = []
            print("Computing")
            for well, plate in zip(df['well_id'], df['plate_id']):
                future = client.submit(get_spatial_stats, well, plate, mdata_copy, radii, radius_chull)
                futures.append(future)
            df_stats = pd.concat(client.gather(futures)).fillna(0)     
        else:
            print(mdata[mod].obs.index)
            print(df)
            df_stats = pd.concat([get_spatial_stats(well, plate, mdata[mod], radii, radius_chull) for well, plate in
                              tqdm(zip(df['well_id'], df['plate_id']),
                                   desc=f'Spatial statistics - {mod}',
                                   total=df.shape[0])]).fillna(0)

        df_stats.columns = [f'{mod}_stat_' + col for col in df_stats.columns]
        # add prefix to leiden
        df_counts = mdata[mod].obs.groupby(['well_id', 'plate_id', 'leiden'], observed=False)['leiden'].count().reset_index(name='count')
        df_counts['leiden'] = f'{mod}_' + df_counts['leiden'].astype(str)
        df_counts = df_counts.pivot_table(index=['well_id', 'plate_id'], columns='leiden', values='count', observed=False)
        df_counts.reset_index(inplace=True)
        df_counts.index = df_counts['well_id'].astype(str) + '_' + df_counts['plate_id'].astype(str)
        df_counts = df_counts.drop(columns=['well_id', 'plate_id'])
        # merge counts in
        df = pd.merge(df, df_counts, left_index=True, right_index=True, how='left')
        # merge stats in
        df = pd.merge(df, df_stats, left_index=True, right_index=True, how='inner')
        spatial_dict[mod] = df

    if use_dask:
        client.close()

    return spatial_dict


