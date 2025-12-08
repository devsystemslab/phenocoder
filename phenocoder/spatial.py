import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import squidpy as sq
from scipy.spatial import ConvexHull
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm


class SpatialGraphAnalyzer:
    def __init__(self, adata: ad.AnnData):
        self.adata = adata
        self.radii: tuple[int] = None

    def get_chull(
        self,
        index: str,
        radius: int = 100,
        degree_threshold: int = 5,
    ) -> float:
        """
        Calculate convex hull volume, area, and density for spatial data.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data object containing spatial coordinates.
        sample : str
            Sample identifier.
        sample_key : str
            Key in adata.obs identifying the sample column.
        radius : int, default 100
            Radius for neighbor graph construction.
        degree_threshold : int, default 5
            Minimum degree threshold for filtering points.
        filter_obs : bool, default False
            Whether to filter observations by sample.
        convert_units : bool, default True
            Whether to convert units using pixel_size.
        pixel_size : float, default 0.322
            Pixel size for unit conversion.

        Returns
        -------
        pd.DataFrame
            DataFrame containing convex hull volume, area, and density metrics.
        """
        coordinate_cols = ['z', 'centroid-0', 'centroid-1']
        df = self.adata.obs[coordinate_cols]
        if len(df) < 4:
            return pd.DataFrame(
                {'volume_chull': 0, 'area_chull': 0, 'density_chull': 0},
                index=[index],
            )
        pts = df.to_numpy()
        # neighbor graph
        graph = nx.from_numpy_array(
            radius_neighbors_graph(
                pts, radius, mode='distance', include_self=False
            ).toarray(),
            create_using=nx.DiGraph,
        )
        A = nx.adjacency_matrix(graph, weight=None)
        degrees = np.sum(A, axis=1)
        # filter df with degrees
        df_filtered = df[degrees > degree_threshold]
        if len(df_filtered) < 4:
            return pd.DataFrame(
                {'volume_chull': 0, 'area_chull': 0, 'density_chull': 0},
                index=[index],
            )
        pts = df_filtered[coordinate_cols].to_numpy()
        for i in range(pts.shape[-1]):
            if len(np.unique(pts[:, i])) == 1:
                return pd.DataFrame(
                    {'volume_chull': 0, 'area_chull': 0, 'density_chull': 0},
                    index=[index],
                )
        chull = ConvexHull(pts)
        df_results = pd.DataFrame(
            {'volume_chull': chull.volume, 'area_chull': chull.area},
            index=[index],
        )
        df_results['density_chull'] = len(pts) / df_results['volume_chull']
        return df_results

    def get_chulls_connected_components(
        self, clusters: list, cluster_key: str, radius: int = 100, min_nds=10
    ) -> pd.DataFrame:
        """
        Calculate convex hull for connected components in subset of spatial graph.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data object containing spatial coordinates.
        well : str
            Well identifier.
        plate : str
            Plate identifier.
        clusters : list
            List of cluster identifiers to include.
        radius : int, default 100
            Radius for neighbor graph construction.
        min_nds : int, default 10
            Minimum number of nodes for connected components.
        convert_units : bool, default True
            Whether to convert units using pixel_size.
        pixel_size : float, default 0.322
            Pixel size for unit conversion.
        plot : bool, default False
            Whether to plot the graph.

        Returns
        -------
        pd.DataFrame
            DataFrame containing convex hull metrics for each connected component.
        """
        # get center of mass
        graph_center = (
            self.adata.obs[['z', 'centroid-0', 'centroid-1']].to_numpy().mean(axis=0)
        )
        adata = self.adata[self.adata.obs[cluster_key].isin(clusters)]
        df = adata.obs[['z', 'centroid-0', 'centroid-1']]
        if df.empty:
            return pd.DataFrame()
        pts = df.to_numpy()
        # neighbor graph
        G = nx.from_numpy_array(
            radius_neighbors_graph(
                pts, radius, mode='distance', include_self=False
            ).toarray(),
            create_using=nx.DiGraph,
        ).to_undirected()
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
                n_unique = [
                    len(np.unique(pts_component[:, i])) == 1
                    for i in range(pts_component.shape[-1])
                ]
                if np.any(n_unique):
                    continue
                chull = ConvexHull(pts_component)
                center_component = pts_component.mean(axis=0)
                distance_center = np.linalg.norm(center_component - graph_center)
                df_component = pd.DataFrame(
                    {
                        'volume_chull': chull.volume,
                        'area_chull': chull.area,
                        'n_pts_chull': len(pts_component),
                        'distance_center_chull': distance_center,
                    },
                    index=[str(i)],
                )
                df_component['density_chull'] = (
                    df_component['volume_chull'] / df_component['n_pts']
                )
                df_results.append(df_component)
        if len(df_results) == 0:
            return pd.DataFrame()
        df_results = pd.concat(df_results)
        # filter n_pts for min_nds
        df_results = df_results[df_results['n_pts_chull'] >= min_nds]

    def get_interactions(self, cluster_key: str, index: str = None) -> pd.DataFrame:
        """
        Calculate interaction matrices between clusters.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data object with spatial connectivity.
        well : str
            Well identifier.
        plate : str
            Plate identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame containing normalized and raw interaction counts.
        """

        # interaction matrix
        results = []
        adata = self.adata.copy()
        for i in (True, False):
            sq.gr.interaction_matrix(adata, cluster_key=cluster_key, normalized=i)
            df_interaction = pd.DataFrame(adata.uns[f'{cluster_key}_interactions'])
            cluster_names = pd.Categorical(
                adata.obs[cluster_key].cat.categories
            ).tolist()
            df_interaction.columns = cluster_names
            df_interaction.index = cluster_names
            df_interaction = df_interaction.stack().reset_index()
            df_interaction.columns = ['from', 'to', 'count']
            if i:
                df_interaction.index = (
                    'interaction_norm_'
                    + df_interaction['from']
                    + '_'
                    + df_interaction['to']
                )
            else:
                df_interaction.index = (
                    'interaction_' + df_interaction['from'] + '_' + df_interaction['to']
                )
            df_interaction = df_interaction.drop(columns=['from', 'to'])
            df_interaction = df_interaction.T
            df_interaction = df_interaction.reset_index(drop=True)
            df_interaction.index = [index]
            results.append(df_interaction)
        df_interaction = pd.concat(results, axis=1)
        return df_interaction

    def get_moran(self, index) -> pd.DataFrame:
        """
        Calculate Moran's I spatial autocorrelation for features.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data object with spatial connectivity.
        well : str
            Well identifier.
        plate : str
            Plate identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame containing Moran's I values for each feature.
        """
        adata = self.adata.copy()
        if np.sum(adata.obsp.get('spatial_connectivities').toarray()) == 0:
            df_moran = pd.DataFrame(0, index=[index], columns=adata.var_names)
            return df_moran
        sq.gr.spatial_autocorr(
            adata,
            mode='moran',
            connectivity_key='spatial_connectivities',
            genes=adata.var_names,
            n_perms=100,
            n_jobs=1,
            show_progress_bar=False,
        )
        df_moran = pd.DataFrame(adata.uns['moranI'])
        df_moran = df_moran[['I']].T
        df_moran.index = [index]
        df_moran.columns = ['moranI_' + col for col in df_moran.columns]
        return df_moran

    def get_moran_cluster(
        self, index:str, cluster_key: str
    ) -> pd.DataFrame:
        """
        Calculate Moran's I spatial autocorrelation for cluster assignments.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data object with spatial connectivity and leiden clusters.
        well : str
            Well identifier.
        plate : str
            Plate identifier.

        Returns
        -------
        pd.DataFrame
            DataFrame containing Moran's I values for each cluster.
        """
        adata = self.adata.copy()
        # one hot encode cluster labels 
        df = pd.get_dummies(adata.obs[cluster_key]).astype(int)
        adata_cl = ad.AnnData(df, obs=adata.obs, obsm=adata.obsm, obsp=adata.obsp)
        if np.sum(adata.obsp.get('spatial_connectivities').toarray()) == 0:
            df_moran = pd.DataFrame(
                0, index=[index], columns=adata.var_names
            )
            return df_moran
        sq.gr.spatial_autocorr(
            adata_cl,
            mode='moran',
            connectivity_key='spatial_connectivities',
            genes=adata_cl.var_names,
            n_perms=100,
            n_jobs=1,
            show_progress_bar=False,
        )
        df_moran = pd.DataFrame(adata_cl.uns['moranI'])
        df_moran = df_moran[['I']].T
        df_moran.index = [index]
        # add moranI prefix to all columns
        df_moran.columns = ['moranI_' + col for col in df_moran.columns]
        return df_moran

    def get_spatial_stats(
        self,
        sample: str,
        sample_key: str,
        spatial_key: str,
        adata: ad.AnnData,
        radii: tuple[int] = (25, 50, 100, 150),
        radius_chull: int = 100,
        layer: str = None,
        cluster_key: str = None,
    ) -> pd.DataFrame:
        """
        Calculate comprehensive spatial statistics for a sample.

        Parameters
        ----------
        sample : str
            Sample identifier.
        sample_key : str
            Key in adata.obs identifying the sample column.
        adata : ad.AnnData
            Annotated data object containing spatial coordinates.
        radii : tuple[int], default (25, 50, 100, 150)
            Radii for spatial neighbor calculations.
        radius_chull : int, default 100
            Radius for convex hull calculations.
        layer : str, optional
            Layer to use for expression data.
        cluster_key : str, optional
            Key for cluster assignments.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all spatial statistics for the sample.
        """
        adata = adata[adata.obs[sample_key] == sample]
        adata = adata.copy()
        if layer is not None:
            adata.X = adata.layers[layer].copy()
            # remove layer
            adata.layers[layer] = None
        if cluster_key is None:
            raise ValueError('cluster_key must be provided')
        # generate spatial obsm
        df = []
        for radius in radii:
            sq.gr.spatial_neighbors(
                adata, radius=radius, coord_type='generic', spatial_key=spatial_key
            )

            # calculate spatial features
            df_spatial_features = pd.concat(
                [
                    self.get_interactions(self.adata, sample),
                    self.get_centrality(self.adata, sample),
                    self.get_connectivity(self.adata, sample),
                    self.get_moran(self.adata, sample),
                    self.get_moran_cluster(self.adata, sample),
                    self.get_neighborhood_enrichment(self.adata, sample),
                ],
                axis=1,
            )
            # radius as suffix to all columns
            df_spatial_features.columns = [
                col + f'_{radius}' for col in df_spatial_features.columns
            ]
            df.append(df_spatial_features)

        df = pd.concat(df, axis=1)
        df_chull = self.get_chull(adata, sample, radius=radius_chull)
        df = pd.concat([df, df_chull], axis=1)
        return df

    def run_spatial_feature_processing(
        self, mdata, radii: tuple[int] = (25, 50, 100, 150), radius_chull: int = 100
    ):
        """
        Run spatial feature processing across all modalities in MuData object.

        Parameters
        ----------
        mdata : MuData
            Multi-modal data object containing spatial coordinates and clusters.
        radii : tuple[int], default (25, 50, 100, 150)
            Radii for spatial neighbor calculations.
        radius_chull : int, default 100
            Radius for convex hull calculations.

        Returns
        -------
        dict
            Dictionary with modality names as keys and spatial feature DataFrames as values.
        """
        spatial_dict = {}
        df_qc = []
        for mod in mdata.mod_names:
            n_cluster = len(mdata[mod].obs['leiden'].unique())
            assert n_cluster > 1
            df_tmp = (
                mdata[mod]
                .obs.groupby(['well_id', 'plate_id', 'leiden'], observed=False)
                .size()
                .reset_index(name='count')
            )
            df_tmp[f'detected_{mod}'] = df_tmp['count'] > 0
            df_tmp = (
                df_tmp.groupby(['well_id', 'plate_id'], observed=False)[
                    f'detected_{mod}'
                ]
                .agg('sum')
                .reset_index()
            )
            df_tmp[f'detected_{mod}'] = df_tmp[f'detected_{mod}'] > 1
            # set index
            df_tmp = df_tmp.set_index(['well_id', 'plate_id'])
            df_qc.append(df_tmp)

        df_qc = pd.concat(df_qc, axis=1).all(axis=1).reset_index(name='selected')

        for mod in mdata.mod_names:
            df = (
                mdata[mod]
                .obs.groupby(['well_id', 'plate_id'], observed=False)
                .size()
                .reset_index(name='cell_count')
            )
            df = df[df['cell_count'] > 0]
            df = df.merge(
                df_qc[df_qc['selected']], on=['well_id', 'plate_id'], how='inner'
            )
            df = df.drop(columns=['selected'])
            df.index = df['well_id'].astype(str) + '_' + df['plate_id'].astype(str)
            df_stats = pd.concat(
                [
                    self.get_spatial_stats(well, plate, mdata[mod], radii, radius_chull)
                    for well, plate in tqdm(
                        zip(df['well_id'], df['plate_id']),
                        desc=f'Spatial statistics - {mod}',
                        total=df.shape[0],
                    )
                ]
            ).fillna(0)
            df_stats.columns = [f'{mod}_stat_' + col for col in df_stats.columns]
            # add prefix to leiden
            df_counts = (
                mdata[mod]
                .obs.groupby(['well_id', 'plate_id', 'leiden'], observed=False)[
                    'leiden'
                ]
                .count()
                .reset_index(name='count')
            )
            df_counts['leiden'] = f'{mod}_' + df_counts['leiden'].astype(str)
            df_counts = df_counts.pivot_table(
                index=['well_id', 'plate_id'],
                columns='leiden',
                values='count',
                observed=False,
            )
            df_counts.reset_index(inplace=True)
            df_counts.index = (
                df_counts['well_id'].astype(str)
                + '_'
                + df_counts['plate_id'].astype(str)
            )
            df_counts = df_counts.drop(columns=['well_id', 'plate_id'])
            # merge counts in
            df = pd.merge(df, df_counts, left_index=True, right_index=True, how='left')
            # merge stats in
            df = pd.merge(df, df_stats, left_index=True, right_index=True, how='inner')
            spatial_dict[mod] = df

        return spatial_dict
