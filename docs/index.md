# Phenocoder

A deep learning library for **unsupervised morphometric spatial phenotyping** of microscopy
image data.

Phenocoder learns compressed morphological representations of cells/nuclei directly from
microscopy images using (conditional) convolutional variational autoencoders, then analyzes
how those representations are organized in space. It is built around the
[SpatialData](https://spatialdata.scverse.org/) ecosystem: images, segmentation labels and
per-object tables live in a single `SpatialData` object, and the {class}`~phenocoder.Phenocoder`
class drives the full workflow on top of it.

## The workflow at a glance

1. **{meth}`~phenocoder.Phenocoder.generate_dataset`** — extract image patches centered on each
   segmented object and write them (plus per-channel intensity statistics) to disk.
2. **{meth}`~phenocoder.Phenocoder.initialize_model`** — build a `CVAE` or conditional
   `CondCVAE` and the train/validation data generators.
3. **{meth}`~phenocoder.Phenocoder.train`** — fit the model with early stopping, learning-rate
   scheduling and TensorBoard logging.
4. **{meth}`~phenocoder.Phenocoder.encode`** — embed every object into the learned latent space,
   optionally smoothing the latents over each object's spatial neighborhood (message passing).
5. **{meth}`~phenocoder.Phenocoder.spatialgraph_stats`** — compute spatial neighborhood-graph
   statistics per sample (or per spatial subunit) from clustered latents.
6. **{meth}`~phenocoder.Phenocoder.spatialgraph_embedding`** — embed the per-sample/per-subunit
   statistics (PCA + UMAP, with optional batch correction) for sample-level comparison.

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
- **Beta-VAE support**: tune the KL weight (`beta`) for more disentangled representations.
- Built on Keras 3 with the TensorFlow backend.

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
workflow
architecture
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```
