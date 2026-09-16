"""Persisting and reusing a fitted spatial-graph embedding space.

``Phenocoder.spatialgraph_embedding`` fits a scaler, a PCA and (optionally) a UMAP on the
per-sample statistics it is given. Ordinarily that is fine: you embed every sample together
and the axes mean whatever the pooled data says they mean.

It is *not* fine when the embedding is a **reference** — for example a space built from a
simulation parameter sweep, where PC1 might be a density gradient you swept deliberately.
Re-fitting on reference+query pooled moves those axes, and "where does this experimental
sample fall in simulated parameter space" stops having an answer.

This module stores the fitted transforms so a query can be projected into a frozen reference
space without refitting anything. The hard part is not persistence -- it is making sure the
query's feature vector actually lines up with the reference's.

Two things are checked, and it matters that they are separate:

- :func:`align_features` handles *shape*. It reindexes the query onto the reference's exact
  feature list and order, zero-filling what is missing and dropping what is extra. Both are
  routine: per-sample feature sets legitimately differ, because AnnData subsetting drops
  unused categories and a sample simply may not contain every cluster.
- :func:`check_stats_config` handles *meaning*. It compares the cluster label set, the radii
  and the stat groups. This is what catches a query clustered independently of the
  reference -- the case where zero-filling would otherwise produce plausible-looking but
  meaningless coordinates.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

TRANSFORM_FILENAME = 'embedding_transform.joblib'


def _stat_group(feature: str) -> str | None:
    """
    Recover the stat group from a feature name.

    Args:
        feature (str): Column name as produced by ``SpatialGraphAnalyzer.to_df``, of the
            form ``radius:{radius}_stat:{group}_{col}``.

    Returns:
        str | None: The stat group, or None if the name does not carry one.
    """
    marker = '_stat:'
    start = feature.find(marker)
    if start == -1:
        return None
    rest = feature[start + len(marker) :]
    return rest.split('_', 1)[0] or None


def align_features(
    df: pd.DataFrame,
    var_names: list[str],
    allow_extra: bool = True,
) -> pd.DataFrame:
    """
    Reindex a query's statistics onto the reference's feature set.

    Column *order* matters as much as membership: the stored PCA multiplies a raw matrix, so
    a query whose columns happen to be ordered differently would be silently projected wrong.
    ``spatialgraph_stats`` builds its column order from a dict iteration
    (``SpatialGraphAnalyzer.to_df``) and hides any divergence behind
    ``concat(join='outer').fillna(0)`` -- which only runs *within* one call. Reference and
    query are separate calls, so nothing lines them up for us.

    Missing features are zero-filled, matching that same ``fillna(0)``. This is expected
    rather than exceptional: AnnData subsetting drops unused categories, so a sample that
    happens not to contain a given cluster emits no columns for it, and feature sets differ
    substantially even between samples of one run. An interaction count that is absent
    genuinely means "these two clusters never co-occur here", i.e. zero.

    What this cannot detect is a query built from a *different label set*, where zero-filling
    would quietly produce wrong coordinates. That is checked separately and directly, by
    comparing cluster categories -- see :func:`check_stats_config`.

    Query-only features are dropped with a warning. The reference PCA has no loading for
    them, so they cannot contribute to the projection; this is equally routine (a cluster
    pair that co-occurs in the query but never in the reference produces a column the
    reference never had). The warning reports how much is being discarded, since a large
    fraction suggests the reference is too narrow to represent the query well.

    Args:
        df (pd.DataFrame): Query statistics, one row per sample/subunit.
        var_names (list[str]): Reference feature names, in reference order.
        allow_extra (bool, optional): Kept for callers that want query-only features to be
            a hard error rather than a warning; pass False to raise instead of dropping.
            Defaults to True.

    Returns:
        pd.DataFrame: ``df`` reindexed to exactly ``var_names``, in that order.

    Raises:
        ValueError: If the query has features absent from the reference and
            ``allow_extra`` is False.
    """
    reference = list(var_names)

    extra = [f for f in df.columns if f not in set(reference)]
    if extra:
        if not allow_extra:
            raise ValueError(
                f'Query has {len(extra)} feature(s) absent from the reference, and '
                f'allow_extra=False. The reference PCA has no loading for them, so they '
                f'cannot be projected.\nFirst extra: {extra[:10]}'
            )
        warnings.warn(
            f'Dropping {len(extra)} of {len(df.columns)} query feature(s) absent from the '
            f'reference; they cannot be projected. A large fraction suggests the reference '
            f'does not span the query well. First: {extra[:5]}',
            stacklevel=2,
        )

    # Absent features become 0, matching spatialgraph_stats' own fillna(0). Reindexing
    # also restores the reference column order.
    return df.reindex(columns=reference, fill_value=0.0)


def save_transform(
    path: str | Path,
    scaler: Any,
    pca: Any,
    umap_model: Any,
    var_names: list[str],
    cluster_key: str,
    cluster_categories: list[str],
    radii: tuple[int, ...],
    stats_groups: list[str] | None,
    scale: bool,
) -> Path:
    """
    Persist a fitted embedding space.

    Args:
        path (str | Path): Destination file, or a directory to write
            ``embedding_transform.joblib`` into.
        scaler: Fitted ``StandardScaler``, or None if ``scale`` was False.
        pca: Fitted ``PCA``.
        umap_model: Fitted ``umap.UMAP``, or None if UMAP was not computed.
        var_names (list[str]): Reference feature names, in order.
        cluster_key (str): obs column the statistics were computed from.
        cluster_categories (list[str]): Full label set behind ``cluster_key``.
        radii (tuple[int, ...]): Radii the statistics were computed at.
        stats_groups (list[str] | None): Stat groups computed, or None for all.
        scale (bool): Whether scaling was applied.

    Returns:
        Path: The file written.
    """
    path = Path(path)
    if path.suffix == '':
        path = path / TRANSFORM_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'scaler': scaler,
        'pca': pca,
        'umap': umap_model,
        'var_names': list(var_names),
        'cluster_key': cluster_key,
        'cluster_categories': list(cluster_categories),
        'radii': tuple(radii),
        'stats_groups': list(stats_groups) if stats_groups is not None else None,
        'scale': scale,
    }
    joblib.dump(payload, path)
    return path


def load_transform(path: str | Path) -> dict:
    """
    Load a saved embedding space.

    Args:
        path (str | Path): File written by :func:`save_transform`, or the directory
            holding it.

    Returns:
        dict: The stored payload.

    Raises:
        FileNotFoundError: If no transform exists at ``path``.
    """
    path = Path(path)
    if path.is_dir():
        path = path / TRANSFORM_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f'No saved embedding transform at {path}. Fit one first with '
            'spatialgraph_embedding(..., save_transform=True).'
        )
    return joblib.load(path)


def check_stats_config(
    payload: dict,
    cluster_key: str | None,
    radii: tuple[int, ...] | None,
    stats_groups: list[str] | None,
    cluster_categories: list[str] | None = None,
) -> None:
    """
    Verify the query's statistics were computed the same way as the reference's.

    This is where a genuine mismatch is caught. Feature-set comparison cannot do it:
    per-sample feature sets legitimately differ (see :func:`align_features`), so missing
    columns are zero-filled rather than rejected. The label set behind those columns is the
    thing that actually has to match, along with the radii -- statistics at different radii
    carry identical column names but incomparable values.

    Args:
        payload (dict): Loaded reference transform.
        cluster_key (str | None): Query's cluster key, or None to skip the check.
        radii (tuple[int, ...] | None): Query's radii, or None to skip.
        stats_groups (list[str] | None): Query's stat groups, or None to skip.
        cluster_categories (list[str] | None, optional): Query's full cluster label set, or
            None to skip. Defaults to None.

    Raises:
        ValueError: On any mismatch.
    """
    if cluster_categories is not None:
        ref_cats = list(payload.get('cluster_categories') or [])
        if ref_cats and list(cluster_categories) != ref_cats:
            only_query = sorted(set(cluster_categories) - set(ref_cats))
            only_ref = sorted(set(ref_cats) - set(cluster_categories))
            raise ValueError(
                f'Query cluster labels do not match the reference. Reference has '
                f'{len(ref_cats)} categories, query has {len(cluster_categories)}.'
                + (f' Only in query: {only_query[:10]}.' if only_query else '')
                + (f' Only in reference: {only_ref[:10]}.' if only_ref else '')
                + ' Reference mapping needs a shared label set -- transfer the reference '
                'labels onto the query rather than clustering it independently.'
            )

    if cluster_key is not None and cluster_key != payload['cluster_key']:
        raise ValueError(
            f'Query used cluster_key={cluster_key!r} but the reference was built with '
            f'{payload["cluster_key"]!r}.'
        )
    if radii is not None and tuple(radii) != tuple(payload['radii']):
        raise ValueError(
            f'Query used radii={tuple(radii)} but the reference was built with '
            f'{tuple(payload["radii"])}. Statistics at different radii share column names '
            f'but are not comparable.'
        )
    if stats_groups is not None and payload['stats_groups'] is not None:
        if sorted(stats_groups) != sorted(payload['stats_groups']):
            raise ValueError(
                f'Query computed stats={sorted(stats_groups)} but the reference used '
                f'{sorted(payload["stats_groups"])}.'
            )


def apply_transform(
    df: pd.DataFrame,
    payload: dict,
    allow_extra: bool = True,
    umap: bool = True,
    clip: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Project query statistics into a saved reference space.

    Nothing here is fitted -- the stored scaler, PCA and UMAP are applied as-is, so the
    reference coordinates are unaffected by whatever the query contains.

    Args:
        df (pd.DataFrame): Query statistics, one row per sample/subunit.
        payload (dict): Loaded reference transform.
        allow_extra (bool, optional): Drop query-only features with a warning rather than
            raising. Defaults to True.
        umap (bool, optional): Also project into the reference UMAP, if one was saved.
            Defaults to True.
        clip (float | None, optional): Clip scaled values to +/- this many standard
            deviations before projecting. Guards against features whose reference variance
            is near zero (see Note). Pass None to disable. Defaults to 10.0.

    Returns:
        tuple: ``(X_pca, X_umap)``; ``X_umap`` is None when no UMAP was saved or
            ``umap=False``.

    Note:
        Reference mapping is sensitive to features that barely varied across the reference.
        Dividing a query's deviation by a near-zero reference standard deviation can turn a
        trivial difference into hundreds of standard deviations, letting a handful of
        near-constant features dominate the projection (and, in the extreme, overflow).
        ``clip`` bounds that. The same features are harmless in the pooled fit path, because
        there the scale is derived from the very data being scaled.
    """
    aligned = align_features(df, payload['var_names'], allow_extra=allow_extra)
    X = aligned.to_numpy(dtype=np.float64)

    if payload['scale'] and payload['scaler'] is not None:
        X = payload['scaler'].transform(X)
        # Constant-variance reference features scale to NaN; zero them as the fit path does.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if clip is not None:
            n_clipped = int(np.sum(np.abs(X) > clip))
            if n_clipped:
                warnings.warn(
                    f'Clipped {n_clipped} scaled value(s) to +/-{clip} sd. These come from '
                    f'features with near-zero variance in the reference, where a small '
                    f'query difference scales to a huge value. Pass clip=None to disable.',
                    stacklevel=2,
                )
            X = np.clip(X, -clip, clip)

    X_pca = payload['pca'].transform(X)

    X_umap = None
    if umap and payload.get('umap') is not None:
        X_umap = payload['umap'].transform(X_pca)

    return X_pca, X_umap
