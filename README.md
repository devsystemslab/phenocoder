# Phenocoder

A machine-learning framework that combines conditional variational autoencoders with spatial graph analysis to learn unsupervised phenotypic embeddings of complex tissue architectures from microscopy images.

## Overview

Phenocoder learns compressed morphological representations of cells/nuclei directly from
microscopy images using (conditional) convolutional variational autoencoders, then analyzes
how those representations are organized in space. It is built around the
[SpatialData](https://spatialdata.scverse.org/) ecosystem: images, segmentation labels and
per-object tables live in a single `SpatialData` object, and the `Phenocoder` class drives the
full workflow on top of it:

1. **`generate_dataset`** — extract image patches centered on each segmented object and write
   them (plus per-channel intensity statistics) to disk.
2. **`initialize_model`** — build a `CVAE` or conditional `CondCVAE` and the train/validation
   data generators.
3. **`train`** — fit the model with early stopping, learning-rate scheduling and TensorBoard
   logging.
4. **`encode`** — embed every object into the learned latent space, optionally smoothing the
   latents over each object's spatial neighborhood (message passing). Results are written to
   `sdata.tables['phenocoder']`.
5. **`spatialgraph_stats`** — compute spatial neighborhood-graph statistics per sample (or per
   spatial subunit) from clustered latents.
6. **`spatialgraph_embedding`** — embed the per-sample/per-subunit statistics (PCA + UMAP, with
   optional batch correction) for sample-level comparison.

## Features

- **Convolutional VAE (`CVAE`)**: encode multi-channel image patches into a compact latent space.
- **Conditional VAE (`CondCVAE`)**: condition the encoder/decoder on one or more metadata
  columns (e.g. dataset, z-slice, donor) via one-hot encoding.
- **SpatialData-native**: works directly on `SpatialData` images, labels and tables.
- **Flexible patch extraction**: configurable patch size, 2D or per-z-slice 3D sampling, and
  global or per-sample intensity normalization.
- **Spatial message passing**: aggregate latents over a physical-distance neighborhood graph.
- **Spatial graph analysis**: interaction matrices, Moran's I, centrality, connectivity and
  convex-hull statistics at sample or subunit resolution.
- **Beta-VAE support**: `beta` weights the KL-divergence term against the reconstruction loss,
  tuning the reconstruction-vs-regularization trade-off (higher `beta` → stronger latent
  regularization; lower `beta` → higher reconstruction fidelity).
- Built on Keras 3 with the TensorFlow backend.

## Installation

```bash
# Clone the repository
git clone https://github.com/devsystemslab/phenocoder.git
cd phenocoder

# Install with uv (recommended)
uv pip install .

# Or with pip
pip install .
```

## Requirements

- Python >= 3.10, < 3.13
- TensorFlow >= 2.19.0 (CUDA on Linux, Metal on macOS)
- SpatialData >= 0.5.0
- Additional dependencies listed in `pyproject.toml`

## Quick Start

> **Runnable example:** [`examples/example_workflow.py`](examples/example_workflow.py) executes
> the entire pipeline end to end on the small example dataset bundled with the repository
> (`tests/data/3d/`) — no external data required. Run it with `python examples/example_workflow.py`.
> The full narrative walkthrough is in the [documentation](https://phenocoder.readthedocs.io).

Phenocoder operates on a `SpatialData` object whose images are keyed per sample as
`f"{image_key}_{sample}"` (e.g. `IF_well01`) and whose table holds per-object spatial
coordinates in `obsm`. The snippet below mirrors the end-to-end integration test in
`tests/test_workflow.py`.

```python
import scanpy as sc
from phenocoder import Phenocoder

# Configure how to read the SpatialData object
pheno = Phenocoder(
    table_key="nuclei_features",  # table in sdata.tables with per-object obs/obsm
    sample_key="well",            # obs column identifying each sample
    image_key="IF",               # images are stored as f"{image_key}_{sample}"
)
pheno.add_sdata(sdata)

# 1. Extract patches around each object and write them to disk
pheno.generate_dataset(
    dataset="dataset_1",
    dir_dataset="data/phenocoder",
    patch_size=(32, 32),
    spatial_key_index="spatial_index",  # obsm key with (y, x, z) integer coords
)

# 2. Build a conditional CVAE (conditioned on dataset + z-slice)
pheno.initialize_model(
    n_latent_dim=32,
    n_dense_dim=64,
    conditions=["dataset", "z"],   # pass [] for a plain (non-conditional) CVAE
    input_shape=(32, 32, 4),
)

# 3. Train
pheno.train(n_epochs=10)

# 4. Encode every object into latent space, smoothing latents over a 50-unit
#    spatial neighborhood. Results are stored in sdata.tables['phenocoder'].
pheno.encode(spatial_key_index="spatial_index", spatial_message_passing_radius=50)

# 5. Cluster the latents (standard scanpy), then compute spatial graph statistics
sc.pp.pca(pheno.sdata.tables["phenocoder"])
sc.pp.neighbors(pheno.sdata.tables["phenocoder"])
sc.tl.leiden(
    pheno.sdata.tables["phenocoder"],
    resolution=0.5,
    flavor="igraph",
    n_iterations=2,
    directed=False,
)

pheno.spatialgraph_stats(
    cluster_key="leiden",
    spatial_key="spatial",
    radii=(25, 50),
    table_key="phenocoder",
)

# 6. Embed the per-sample statistics for comparison (stored in pheno.adata)
pheno.spatialgraph_embedding(n_dim=32, scale=True, umap=True)
```

### Per-sample vs. global scaling

Intensity normalization can be computed per sample (each sample scaled to its own intensity
range, the default) or globally across all samples. Whatever you choose at
`generate_dataset` time **must** be passed again to `encode`, so training and inference scale
identically:

```python
pheno.generate_dataset(..., scale_per_sample=True, scale_percentile=1)
pheno.encode(..., scale_per_sample=True, scale_percentile=1)
```

### Subunit-level statistics

For large or 3D samples, `spatialgraph_stats` can partition each sample into spatial subunits
(cubes) and compute statistics per subunit:

```python
pheno.spatialgraph_stats(
    cluster_key="leiden",
    radii=(25, 50),
    table_key="phenocoder",
    use_subunits=True,
    dim_subunit=(200, 200, 200),
    min_obs_per_subunit=10,
)
```

### Standalone spatial graph analysis

The spatial graph analysis does **not** depend on the CVAE. `spatialgraph_stats` runs on any
table in the `SpatialData` object that has spatial coordinates in `.obsm` and a categorical
label column in `.obs` — the labels can come from any source (Phenocoder latents, Leiden
clustering of the raw morphometric features, marker-based cell-type annotations, manual regions,
…). This lets you run the spatial statistics on their own, without training a model:

```python
import scanpy as sc
from phenocoder import Phenocoder

pheno = Phenocoder(table_key="nuclei_features", sample_key="well", image_key="IF")
pheno.add_sdata(sdata)

# Cluster the raw feature table directly (no CVAE involved).
adata = pheno.sdata.tables["nuclei_features"]
sc.pp.scale(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.05, flavor="igraph", n_iterations=2, directed=False)

# Run the spatial statistics on those labels
pheno.spatialgraph_stats(
    cluster_key="leiden",          # any categorical .obs column
    spatial_key="spatial",         # coordinates in .obsm
    radii=(25, 50),
    table_key="nuclei_features",   # any table in sdata.tables
)
```

By default every stat group is computed. Pass `stats=[...]` to select a subset — valid groups
are `interactions`, `centrality`, `connectivity`, `moran_features`, `moran_clusters` and
`chull`:

```python
pheno.spatialgraph_stats(
    cluster_key="leiden",
    spatial_key="spatial",
    radii=(25, 50),
    table_key="nuclei_features",
    stats=["interactions", "connectivity"],  # only these groups
)
```

## Model Architecture

### CVAE (Convolutional Variational Autoencoder)

- **Encoder**: a stack of strided `Conv2D` layers (downsampling) → `Flatten` → `Dense` →
  `z_mean` and `z_log_var`, with the reparameterization trick producing the latent sample `z`.
- **Decoder**: `Dense` → `Reshape` → stacked `Conv2DTranspose` layers (upsampling) → a final
  `Conv2DTranspose` with sigmoid activation reconstructing all input channels.
- **Loss**: per-channel binary cross-entropy reconstruction loss + `beta` × KL divergence.

### CondCVAE (Conditional CVAE)

Extends `CVAE` by concatenating one-hot encoded condition labels into the encoder (after the
flattened features) and the decoder (with the latent vector). The number of condition columns
chosen in `initialize_model` determines the one-hot dimension; the fitted encoder is saved
alongside the model so the same encoding is reused at inference.

## Configuration

Key parameters of `Phenocoder.initialize_model`:

| Parameter      | Default                  | Description                                              |
| -------------- | ------------------------ | ------------------------------------------------------- |
| `n_latent_dim` | —                        | Dimensionality of the latent space.                     |
| `n_dense_dim`  | —                        | Size of the dense layer between conv and latent layers. |
| `conditions`   | —                        | obs columns used as conditions; `[]` → plain `CVAE`.    |
| `input_shape`  | `(128, 128, 4)`          | Patch shape `(height, width, channels)`.                |
| `conv_layers`  | `(8, 16, 32, 64, 128)`   | Filters per convolutional layer.                        |
| `dropout`      | `0.25`                   | Dropout rate.                                            |
| `beta`         | `0.01`                   | KL-divergence weight (beta-VAE).                         |
| `batch_size`   | `64`                     | Training batch size.                                     |

> Note: `input_shape` must be consistent with the `patch_size` used in `generate_dataset`
> (same height/width, plus the channel count).

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ruff format .

# Lint
ruff check .
```

## Project Status

Phenocoder is under active development. Some methods still contain dataset-specific
assumptions (noted in their docstrings) that are being generalized.

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 DevSystems Lab.

## Citation

If you use Phenocoder in your research, please cite this repository:

```bibtex
@software{phenocoder,
  author  = {Christoph Harmel},
  title   = {Phenocoder: unsupervised phenotypic embedding of tissue architectures via conditional variational autoencoders and spatial graph analysis},
  year    = {2026},
  url     = {https://github.com/devsystemslab/phenocoder}
}
```
