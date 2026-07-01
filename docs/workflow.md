# Example workflow

The {class}`~phenocoder.Phenocoder` class drives a six-step pipeline that turns raw microscopy
images into spatially-aware morphological embeddings. Each step maps to a method on the class.

Phenocoder operates on a single [`SpatialData`](https://spatialdata.scverse.org/) object that
bundles the intensity images, the segmentation label masks and a per-object feature table. The
example below runs the full pipeline end to end and mirrors the integration test in
`tests/test_workflow.py` (the SpatialData object it uses is built in `tests/conftest.py`).

## Input data

The starting point is the output of a typical microscopy segmentation pipeline. In the bundled
3D test dataset (`tests/data/3d/`) that is three kinds of file, organised per **sample** (here a
microtiter-plate *well*, e.g. `A06`, `G04`, `H03`):

| Input | Location | Format | Description |
| ----- | -------- | ------ | ----------- |
| **Intensity images** | `imgs/` | image data, one per z-slice per channel | Raw microscopy signal. 4 channels (`C01`–`C04`) x n z-slices per well. The channel and z-index are encoded in the filename. |
| **Label masks** | `labels/` | image data, one per z-slice | Integer segmentation masks where each object (nucleus) has a unique label id, matching the `label` column of the table. |
| **Feature table** | `tables/` | tabular data, one per z-slice | Per-object regionprops-style measurements. Columns: `label`, `centroid-0`, `centroid-1`, `area`, `eccentricity`, `intensity_mean-0…3`, `major_axis_length`, `minor_axis_length`, `well`, `z_stack`. |

You do **not** need this exact on-disk layout — Phenocoder only cares about the assembled
`SpatialData` object described next. Any pipeline (e.g. `regionprops` on a 3D label image, a
CellProfiler export, …) that can produce per-object measurements plus the matching image and
label arrays will work.

## Building the `SpatialData` object

Phenocoder locates data in the `SpatialData` object through the constructor keys (`image_key`,
`table_key`, `sample_key`) plus the spatial coordinates stored in the table's `.obsm`:

- **Images** keyed `f"{image_key}_{sample}"` (e.g. `IF_A06`) — an
  {class}`~spatialdata.models.Image3DModel` with shape `(c, z, y, x)`. Patch extraction reads
  the image arrays directly from here.
- **A table** keyed `table_key` (e.g. `nuclei_features`) — an `AnnData` whose `.obs` carries the
  per-object metadata (including the `sample_key` column) and whose `.obsm` carries the spatial
  coordinates. This drives the whole workflow: one row per object.
- **Spatial coordinates** in the table's `.obsm`:
  - `spatial` — float `(y, x, z)` physical coordinates, used for the neighbourhood graph.
  - `spatial_index` — integer `(y, x, z)` coordinates, used to index into the image arrays when
    extracting patches (this is the `spatial_key_index` passed to `generate_dataset` / `encode`).
- **Label masks** keyed `f"nuclei_{sample}"` — a {class}`~spatialdata.models.Labels3DModel` with
  shape `(z, y, x)`. These are how the objects are defined and how the feature table is produced
  upstream; each object's `label` id is stored in the table as `instance_key` (`instance_id`)
  and linked via the SpatialData `region` metadata. Phenocoder's patch extraction itself indexes
  the images by coordinate rather than reading the label arrays, but including the labels keeps
  the object a well-formed, self-describing `SpatialData`.

The table's `AnnData` is split so that the learned/analysed features live in `X` and the
descriptive morphometrics + keys live in `.obs`. The snippet below is the essence of the
`example_3d()` fixture in `tests/conftest.py` (per-slice tables concatenated, aggregated per
object, then packed into a `SpatialData`):

```python
import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd
from skimage import io

# 1. Assemble the per-object table (AnnData) --------------------------------
#    (here: concatenate the per-slice CSVs and average per object)
df = pd.concat([pd.read_csv(f) for f in table_files])
df = df.groupby(["label", "well"]).mean().reset_index()

# obs = descriptive metadata + keys; X = feature matrix
features_obs = [
    "area", "eccentricity", "major_axis_length", "minor_axis_length",
    "centroid-0", "centroid-1", "z_stack", "well",
]
features_X = [c for c in df.columns if c not in features_obs]
adata = ad.AnnData(X=df[features_X], obs=df[features_obs])

# Spatial coordinates in obsm
adata.obsm["spatial"] = adata.obs[["centroid-0", "centroid-1", "z_stack"]].values
adata.obsm["spatial_index"] = (
    adata.obs[["centroid-0", "centroid-1", "z_init"]].values.astype(int)
)

# Keys SpatialData needs to link the table to the labels
adata.obs["instance_id"] = adata.obs.index.astype(int)  # matches label ids
adata.obs["region"] = "nuclei"

# 2. Build the per-sample image and label models ---------------------------
images_dict, labels_dict = {}, {}
for well in adata.obs["well"].unique():
    # imgs: array of shape (channels, z, y, x) for this well
    imgs = load_channel_zstacks(well)          # your loader (see conftest.py)
    imgs_label = load_label_zstack(well)       # (z, y, x)

    images_dict[f"IF_{well}"] = sd.models.Image3DModel.parse(
        imgs, c_coords=["C01", "C02", "C03", "C04"]
    )
    # optional 2D maximum-intensity projection, handy for plotting
    images_dict[f"IF_MIP_{well}"] = sd.models.Image2DModel.parse(
        imgs.max(axis=1), c_coords=["C01", "C02", "C03", "C04"]
    )
    labels_dict[f"nuclei_{well}"] = sd.models.Labels3DModel.parse(imgs_label)

# 3. Assemble the SpatialData object ---------------------------------------
sdata = sd.SpatialData(
    images=images_dict,
    labels=labels_dict,
    tables={
        "nuclei_features": sd.models.TableModel.parse(
            adata,
            region="nuclei",
            region_key="region",
            instance_key="instance_id",
        )
    },
)
```

## Configuring Phenocoder

With the `SpatialData` object in hand, tell Phenocoder which keys to read:

```python
import scanpy as sc
from phenocoder import Phenocoder

pheno = Phenocoder(
    table_key="nuclei_features",  # table in sdata.tables with per-object obs/obsm
    sample_key="well",            # obs column identifying each sample
    image_key="IF",               # images are stored as f"{image_key}_{sample}"
)
pheno.add_sdata(sdata)
```

## 1. Generate the patch dataset

{meth}`~phenocoder.Phenocoder.generate_dataset` extracts an image patch centered on each
segmented object and writes the patches (plus per-channel intensity statistics) to disk. Patch
size, the spatial index used for 2D/per-z-slice sampling, and the intensity-normalization
strategy are all configurable here.

```python
pheno.generate_dataset(
    dataset="dataset_1",
    dir_dataset="data/phenocoder",
    patch_size=(32, 32),
    spatial_key_index="spatial_index",  # obsm key with (y, x, z) integer coords
)
```

Key parameters:

- `patch_size` — patch `(height, width)`; must match the height/width of the model's
  `input_shape`.
- `spatial_key_index` — the `obsm` key holding integer `(y, x, z)` coordinates.
- `scale_per_sample` / `scale_percentile` — how intensities are normalized (see
  [Per-sample vs. global scaling](#per-sample-vs-global-scaling) below).

## 2. Initialize the model

{meth}`~phenocoder.Phenocoder.initialize_model` builds either a plain `CVAE` or a conditional
`CondCVAE` (when `conditions` is non-empty) together with the train/validation data generators.
See [Model architecture](architecture.md) for the network details and the full parameter table.

```python
# Build a conditional CVAE (conditioned on dataset + z-slice)
pheno.initialize_model(
    n_latent_dim=32,
    n_dense_dim=64,
    conditions=["dataset", "z"],   # pass [] for a plain (non-conditional) CVAE
    input_shape=(32, 32, 4),
)
```

## 3. Train

{meth}`~phenocoder.Phenocoder.train` fits the model with early stopping, learning-rate
scheduling and TensorBoard logging.

```python
pheno.train(n_epochs=10)
```

## 4. Encode

{meth}`~phenocoder.Phenocoder.encode` embeds every object into the learned latent space and
writes the result to `sdata.tables['phenocoder']`. Optionally it smooths each object's latent
vector over its physical-distance spatial neighborhood (**message passing**) via
`spatial_message_passing_radius`.

```python
# Encode every object, smoothing latents over a 50-unit spatial neighborhood.
pheno.encode(spatial_key_index="spatial_index", spatial_message_passing_radius=50)
```

```{important}
The scaling options (`scale_per_sample`, `scale_percentile`) passed to `encode` **must** match
those used in `generate_dataset`, so training and inference normalize identically.
```

## 5. Spatial graph statistics

After clustering the latents (standard scanpy: PCA → neighbors → Leiden),
{meth}`~phenocoder.Phenocoder.spatialgraph_stats` computes spatial neighborhood-graph
statistics per sample — interaction matrices, Moran's I, centrality, connectivity and
convex-hull statistics.

```python
# Cluster the latents (standard scanpy)
sc.pp.pca(pheno.sdata.tables["phenocoder"])
sc.pp.neighbors(pheno.sdata.tables["phenocoder"])
sc.tl.leiden(pheno.sdata.tables["phenocoder"], resolution=0.5)

# Compute spatial graph statistics
pheno.spatialgraph_stats(
    cluster_key="leiden",
    spatial_key="spatial",
    radii=(25, 50),
    table_key="phenocoder",
)
```

For large or 3D samples it can partition each sample into spatial subunits and compute
statistics per subunit — see [Subunit-level statistics](#subunit-level-statistics) below.

## 6. Spatial graph embedding

{meth}`~phenocoder.Phenocoder.spatialgraph_embedding` embeds the per-sample (or per-subunit)
statistics with PCA + UMAP, optionally with batch correction, producing a sample-level
representation for comparison. Results are stored in `pheno.adata`.

```python
pheno.spatialgraph_embedding(n_dim=32, scale=True, umap=True)
```

## Per-sample vs. global scaling

Intensity normalization can be computed per sample (each sample scaled to its own intensity
range, the default) or globally across all samples. Whatever you choose at
`generate_dataset` time **must** be passed again to `encode`, so training and inference scale
identically:

```python
pheno.generate_dataset(..., scale_per_sample=True, scale_percentile=1)
pheno.encode(..., scale_per_sample=True, scale_percentile=1)
```

## Subunit-level statistics

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

## Standalone spatial graph analysis

The spatial graph analysis does **not** depend on the CVAE.
{meth}`~phenocoder.Phenocoder.spatialgraph_stats` runs on any table in the `SpatialData` object
that has spatial coordinates in `.obsm` and a categorical label column in `.obs` — the labels
can come from **any source**: Phenocoder latents, Leiden clustering of the raw morphometric
features, marker-based cell-type annotations, manual region labels, and so on. This means you
can run the spatial statistics on their own, without generating patches or training a model.

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
sc.tl.leiden(adata, resolution=0.05)

# Run the spatial statistics on those labels
pheno.spatialgraph_stats(
    cluster_key="leiden",          # any categorical .obs column
    spatial_key="spatial",         # coordinates in .obsm
    radii=(25, 50),
    table_key="nuclei_features",   # any table in sdata.tables
)
```

By default every stat group is computed. Pass `stats=[...]` to select a subset — the valid
groups are `interactions`, `centrality`, `connectivity`, `moran_features`, `moran_clusters` and
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
