import anndata as ad
import numpy as np
import pytest

from phenocoder.spatial import spatial_message_passing

# A star graph: node 0 at the origin, four leaves 10 units away on the axes. With radius 11
# the centre neighbors every leaf and the leaves neighbor only the centre, so the degrees
# differ (5 vs 2 once self-loops are added) and row- and column-normalization disagree.
STAR_COORDS = np.array(
    [[0, 0, 0], [10, 0, 0], [-10, 0, 0], [0, 10, 0], [0, -10, 0]], dtype=float
)
STAR_RADIUS = 11


def _star(values):
    """AnnData on the star graph, with one latent dimension holding ``values``."""
    adata = ad.AnnData(np.asarray(values, dtype=float).reshape(-1, 1))
    adata.obsm['spatial'] = STAR_COORDS.copy()
    return adata


def test_message_passing_averages_over_the_neighborhood():
    """Each smoothed latent is the unweighted mean over the node and its neighbors.

    Pins the row-normalization D_inv @ (A + I). Column-normalizing instead -- (A + I) @
    D_inv -- gives the centre 2.0 here, because it sums four degree-discounted leaves
    rather than averaging them.
    """
    adata = spatial_message_passing(_star([0, 1, 1, 1, 1]), radius=STAR_RADIUS)
    smoothed = adata.layers['spatial_message_passing'].ravel()

    # centre: mean(0, 1, 1, 1, 1) = 0.8; each leaf: mean(1, 0) = 0.5
    assert smoothed == pytest.approx([0.8, 0.5, 0.5, 0.5, 0.5])


def test_message_passing_returns_a_numeric_layer():
    """The layer holds floats, not an object array.

    np.diag(D_inv).dot(A) type-checks and raises nothing, but dispatches to NumPy, which
    treats the sparse operand as opaque and returns an object-dtype array of sparse
    matrices. That corrupts the layer silently, so assert the dtype explicitly.
    """
    adata = spatial_message_passing(_star([0, 1, 1, 1, 1]), radius=STAR_RADIUS)
    layer = adata.layers['spatial_message_passing']

    assert layer.dtype == np.float64
    assert layer.shape == adata.X.shape


def test_message_passing_stays_within_the_input_range():
    """Averaging is a convex combination, so it cannot leave the input range.

    This is the property the column-normalized version violated: the degree of a hub
    leaked into its latent values and pushed them past the inputs.
    """
    rng = np.random.default_rng(0)
    values = rng.uniform(-1, 1, size=5)
    adata = spatial_message_passing(_star(values), radius=STAR_RADIUS)
    smoothed = adata.layers['spatial_message_passing']

    assert smoothed.min() >= values.min() - 1e-12
    assert smoothed.max() <= values.max() + 1e-12


def test_message_passing_preserves_a_constant_signal():
    """A constant latent is unchanged by averaging -- the rows of A' sum to 1."""
    adata = spatial_message_passing(_star([3, 3, 3, 3, 3]), radius=STAR_RADIUS)

    smoothed = adata.layers['spatial_message_passing'].ravel()
    assert smoothed == pytest.approx([3.0] * 5)


def test_message_passing_isolated_node_keeps_its_own_value():
    """With no neighbors in radius, the self-loop makes smoothing the identity."""
    adata = ad.AnnData(np.array([[1.0], [2.0], [3.0]]))
    # 1000 units apart, far outside the radius below
    adata.obsm['spatial'] = np.array(
        [[0, 0, 0], [1000, 0, 0], [2000, 0, 0]], dtype=float
    )

    adata = spatial_message_passing(adata, radius=10)

    smoothed = adata.layers['spatial_message_passing'].ravel()
    assert smoothed == pytest.approx([1.0, 2.0, 3.0])


def test_message_passing_requires_spatial_coordinates():
    """Missing coordinates raise a ValueError naming the keys that are present.

    This used to be guarded by ``if adata.obsm['spatial'] is None``, which could never be
    true: subscripting a missing obsm key raises KeyError, so the guard -- and the x/y/z
    obs-column fallback behind it -- was unreachable.
    """
    adata = ad.AnnData(np.ones((4, 2)))
    adata.obsm['spatial_2d'] = np.ones((4, 2))

    with pytest.raises(ValueError, match="'spatial' not found in adata.obsm"):
        spatial_message_passing(adata, radius=10)


def test_message_passing_error_lists_available_obsm_keys():
    """The message points at what is there, so a wrong key is easy to spot."""
    adata = ad.AnnData(np.ones((4, 2)))
    adata.obsm['spatial_index'] = np.ones((4, 3))

    with pytest.raises(ValueError, match='spatial_index'):
        spatial_message_passing(adata, radius=10)


def test_message_passing_handles_multiple_latent_dimensions():
    """Smoothing acts per latent dimension, independently."""
    X = np.array([[0.0, 10.0], [1.0, 20.0], [1.0, 20.0], [1.0, 20.0], [1.0, 20.0]])
    adata = ad.AnnData(X)
    adata.obsm['spatial'] = STAR_COORDS.copy()

    adata = spatial_message_passing(adata, radius=STAR_RADIUS)

    smoothed = adata.layers['spatial_message_passing']
    assert smoothed[:, 0] == pytest.approx([0.8, 0.5, 0.5, 0.5, 0.5])
    assert smoothed[:, 1] == pytest.approx([18.0, 15.0, 15.0, 15.0, 15.0])
