import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import squidpy as sq
from scipy.spatial import ConvexHull
from sklearn.neighbors import radius_neighbors_graph


class SpatialGraphAnalyzer:
    """
    Analyzer for computing spatial graph statistics on spatial omics data.

    This class provides comprehensive analysis of spatial relationships between cells
    and cell clusters in spatial transcriptomics or imaging data. It constructs spatial
    neighborhood graphs at multiple radii and computes various statistics including
    interaction matrices, spatial autocorrelation, centrality scores, connectivity metrics,
    and convex hull properties.

    Typical workflow:
        1. Initialize with AnnData object and analysis parameters
        2. Call run() to compute all statistics across all specified radii
        3. Call to_df() to get results as a single DataFrame

    Attributes:
        adata (ad.AnnData): Annotated data object with spatial coordinates and cluster labels.
        cluster_key (str): Key in adata.obs containing cluster assignments.
        spatial_key (str): Key in adata.obsm containing spatial coordinates.
        radii (tuple[int]): Tuple of radii (in spatial units) for neighborhood calculations.
        index (str): Identifier for this sample/analysis (used as DataFrame index).
        results (dict): Computed statistics, populated after run() is called.

    Example:
        >>> analyzer = SpatialGraphAnalyzer(
        ...     adata=adata_sample,
        ...     cluster_key='leiden',
        ...     spatial_key='spatial',
        ...     radii=(25, 50, 100),
        ...     index='sample_001'
        ... )
        >>> analyzer.run()
        >>> df_stats = analyzer.to_df()
    """

    def __init__(
        self,
        adata: ad.AnnData,
        cluster_key: str,
        spatial_key: str,
        radii: tuple[int],
        index: str,
    ):
        self.adata = adata
        self.cluster_key = cluster_key
        self.spatial_key = spatial_key
        self.radii = radii
        self.index = index

    def get_chull(
        self,
        radius: int = 100,
        degree_threshold: int = 5,
    ) -> pd.DataFrame:
        """
        Calculate convex hull volume, area, and density for spatial data.

        Parameters
        ----------
        adata : ad.AnnData
            Annotated data object containing spatial coordinates.
        radius : int, default 100
            Radius for neighbor graph construction.
        degree_threshold : int, default 5
            Minimum degree threshold for filtering points.
        filter_obs : bool, default False
            Whether to filter observations by sample.

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
                index=[self.index],
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
                index=[self.index],
            )
        pts = df_filtered[coordinate_cols].to_numpy()
        for i in range(pts.shape[-1]):
            if len(np.unique(pts[:, i])) == 1:
                return pd.DataFrame(
                    {'volume_chull': 0, 'area_chull': 0, 'density_chull': 0},
                    index=[self.index],
                )
        chull = ConvexHull(pts)
        df_results = pd.DataFrame(
            {'volume_chull': chull.volume, 'area_chull': chull.area},
            index=[self.index],
        )
        df_results['density_chull'] = len(pts) / df_results['volume_chull']
        return df_results

    def get_chulls_connected_components(
        self, clusters: list, radius: int = 100, min_nds=10, min_degree=3
    ) -> pd.DataFrame:
        """
        Calculate convex hull for connected components in subset of spatial graph.

        Parameters
        ----------
        clusters : list
            List of cluster identifiers to include.
        cluster_key : str
            Key for cluster labels.
        radius : int, default 100
            Radius for neighbor graph construction.
        min_nds : int, default 10
            Minimum number of nodes for connected components.

        Returns
        -------
        pd.DataFrame
            DataFrame containing convex hull metrics for each connected component.
        """
        # get center of mass
        graph_center = self.adata.obsm[self.spatial_key].mean(axis=0)
        adata = self.adata[self.adata.obs[self.cluster_key].isin(clusters)]
        pts = adata.obsm[self.spatial_key].copy()
        if pts.shape[0] == 0:
            return pd.DataFrame()
        # neighbor graph
        G = nx.from_numpy_array(
            radius_neighbors_graph(
                pts, radius, mode='distance', include_self=False
            ).toarray(),
            create_using=nx.DiGraph,
        ).to_undirected()
        # filter out points that have less than min_degree connections
        for node in list(G.nodes):
            if G.degree[node] < min_degree:
                G.remove_node(node)
        if len(G.nodes) <= min_nds:
            return pd.DataFrame()
        # selected connected components
        df = []
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
                        'n_chull': len(pts_component),
                        'distance_center_chull': distance_center,
                    },
                    index=[str(i)],
                )
                df_component['density_chull'] = (
                    df_component['volume_chull'] / df_component['n_chull']
                )
                df.append(df_component)
        if len(df) == 0:
            return pd.DataFrame()

        df = pd.concat(df)

        # filter n_pts for min_nds
        df = df[df['n_chull'] >= min_nds]
        return df

    def get_interactions(self) -> pd.DataFrame:
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
        for i in (True, False):
            sq.gr.interaction_matrix(
                self.adata, cluster_key=self.cluster_key, normalized=i
            )
            df_interaction = pd.DataFrame(
                self.adata.uns[f'{self.cluster_key}_interactions']
            )
            cluster_names = pd.Categorical(
                self.adata.obs[self.cluster_key].cat.categories
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
            df_interaction.index = [self.index]
            results.append(df_interaction)
        df_interaction = pd.concat(results, axis=1)
        return df_interaction

    def get_moran(self) -> pd.DataFrame:
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
            df_moran = pd.DataFrame(0, index=[self.index], columns=adata.var_names)
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
        df_moran.index = [self.index]
        df_moran.columns = ['moranI_' + col for col in df_moran.columns]
        return df_moran

    def get_moran_cluster(self) -> pd.DataFrame:
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
        # one hot encode cluster labels
        df = pd.get_dummies(self.adata.obs[self.cluster_key]).astype(int)
        adata_cl = ad.AnnData(
            df, obs=self.adata.obs, obsm=self.adata.obsm, obsp=self.adata.obsp
        )
        if np.sum(self.adata.obsp.get('spatial_connectivities').toarray()) == 0:
            df_moran = pd.DataFrame(0, index=[self.index], columns=self.adata.var_names)
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
        df_moran.index = [self.index]
        # add moranI prefix to all columns
        df_moran.columns = ['moranI_' + col for col in df_moran.columns]
        return df_moran

    def get_centrality(self) -> pd.DataFrame:
        """
        Calculate centrality scores between clusters.

        Computes pairwise centrality scores that measure how central each cluster
        is relative to other clusters in the spatial graph.

        Returns
        -------
        pd.DataFrame
            DataFrame containing centrality scores for each cluster pair, with one row
            indexed by self.index and columns named 'centrality_{from}_{to}'.
        """
        # centrality scores
        sq.gr.centrality_scores(
            self.adata, cluster_key=self.cluster_key, connectivity_key='spatial'
        )
        df_centrality = pd.DataFrame(
            self.adata.uns[f'{self.cluster_key}_centrality_scores']
        )
        cluster_names = pd.Categorical(
            self.adata.obs[self.cluster_key].cat.categories
        ).tolist()
        df_centrality.index = cluster_names
        # pivot wide new column names are column_names-index
        df_centrality = df_centrality.stack().reset_index()
        df_centrality.columns = ['from', 'to', 'value']
        df_centrality.index = (
            'centrality_' + df_centrality['from'] + '_' + df_centrality['to']
        )
        df_centrality = df_centrality.drop(columns=['from', 'to'])
        df_centrality = df_centrality.T
        df_centrality.index = [self.index]
        return df_centrality

    def get_connectivity(self) -> pd.DataFrame:
        """
        Calculate connectivity statistics (degree) for the spatial graph.

        Computes the mean and standard deviation of node degrees (number of neighbors)
        both globally and per cluster.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row indexed by self.index containing:
            - 'mean': Mean degree across all nodes
            - 'std': Standard deviation of degree across all nodes
            - 'mean_degree_{cluster}': Mean degree for each cluster
            - 'std_degree_{cluster}': Standard deviation of degree for each cluster
        """
        # get mean connectivity
        degrees = self.adata.obsp['spatial_connectivities'].sum(axis=0)
        mean_degree = degrees.mean()
        std_degree = degrees.std()
        df_degree = pd.DataFrame(
            {'mean': mean_degree, 'std': std_degree}, index=[self.index]
        )
        # for each cluster
        cluster_names = pd.Categorical(
            self.adata.obs[self.cluster_key].cat.categories
        ).tolist()
        mean_degree_cluster = [
            degrees[:, self.adata.obs[self.cluster_key] == cluster].mean()
            for cluster in cluster_names
        ]
        str_degree_cluster = [
            degrees[:, self.adata.obs[self.cluster_key] == cluster].std()
            for cluster in cluster_names
        ]
        df_degree_cluster = pd.DataFrame(
            {'mean': mean_degree_cluster, 'std': str_degree_cluster},
            index=cluster_names,
        )
        # pivot wide and merge with df_degree
        df_degree_cluster = df_degree_cluster.stack().reset_index()
        df_degree_cluster.columns = ['from', 'metric', 'value']
        df_degree_cluster = df_degree_cluster.pivot_table(
            index='from', columns='metric', values='value'
        )
        df_degree_cluster.index = 'degree_' + df_degree_cluster.index
        df_degree_cluster = df_degree_cluster.T
        df_degree_cluster = df_degree_cluster.stack().reset_index()
        df_degree_cluster.columns = ['from', 'metric', 'value']
        df_degree_cluster['metric_combined'] = (
            df_degree_cluster['metric'] + '_' + df_degree_cluster['from']
        )
        df_degree_cluster = df_degree_cluster.drop(columns=['from', 'metric'])
        # pivot wide so 1 row remains
        df_degree_cluster = df_degree_cluster.pivot_table(
            columns='metric_combined', values='value'
        )
        df_degree_cluster.index = [self.index]
        df_degree = pd.concat([df_degree, df_degree_cluster], axis=1)
        return df_degree

    def get_counts(self) -> pd.DataFrame:
        """
        Calculate cell counts per cluster.

        Computes the number of cells in each cluster and the total number of cells.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row indexed by self.index containing:
            - 'cluster': Cluster label
            - 'count': Number of cells in the cluster
            - 'total': Total number of cells across all clusters
        """
        df_counts = (
            self.adata.obs.groupby(self.cluster_key)
            .size()
            .reset_index(index=[self.index])
        )
        df_counts.columns = ['cluster', 'count']
        df_counts['total'] = self.adata.obs.shape[0]
        df_counts.index = [self.index]

        return df_counts

    def get_spatial_stats(
        self,
        radius,
    ) -> dict:
        """
        Calculate spatial statistics for a sample.

        Parameters
        ----------
        radius : int
            Radius for spatial neighbor calculations.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all spatial statistics for the sample.
        """
        if self.cluster_key is None:
            raise ValueError('cluster_key must be provided')

        assert len(self.adata.obs[self.cluster_key].unique()) > 1

        sq.gr.spatial_neighbors(
            self.adata,
            radius=radius,
            coord_type='generic',
            spatial_key=self.spatial_key,
        )
        clusters = self.adata.obs[self.cluster_key].unique().tolist()

        dict = {
            'interactions': self.get_interactions(),
            'centrality': self.get_centrality(),
            'connectivity': self.get_connectivity(),
            'moran_features': self.get_moran(),
            'moran_clusters': self.get_moran_cluster(),
            'chull_all': self.get_chulls_connected_components(
                clusters=clusters, radius=radius
            ),
        }
        for cluster in clusters:
            dict[f'chull_cluster:{cluster}'] = self.get_chulls_connected_components(
                clusters=[cluster], radius=radius
            )

        # average all chull dataframes in results and add number of chulls as one column
        for result in [key for key in dict.keys() if 'chull' in key]:
            if dict[result].empty:
                continue
            n_obs = dict[result].shape[0]
            df_mean = pd.DataFrame(dict[result].mean(axis=0)).T
            df_mean.columns = [col + '_mean' for col in df_mean.columns]
            df_sd = pd.DataFrame(dict[result].std(axis=0)).T
            df_sd.columns = [col + '_sd' for col in df_sd.columns]
            dict[result] = pd.concat([df_mean, df_sd], axis=1)
            dict[result]['n_obs_chull'] = n_obs

        return dict

    def to_df(self):
        df = []
        for radius in self.results.keys():
            for result in self.results[radius].keys():
                # add radius as prefix to all columns
                if self.results[radius][result].empty:
                    continue
                self.results[radius][result].columns = [
                    f'radius:{radius}_' + f'stat:{result}_' + col
                    for col in self.results[radius][result].columns
                ]
                df.append(self.results[radius][result].reset_index(drop=True))
        df = pd.concat(df, axis=1)
        df.index = [self.index]
        return df

    def run(self) -> None:
        self.results = dict()
        for radius in self.radii:
            self.results[radius] = self.get_spatial_stats(radius)
