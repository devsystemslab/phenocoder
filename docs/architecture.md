# Model architecture

## Project layout

Everything a Phenocoder run produces lives under a single `dir_project`:

```
dir_project/
├── <dataset>/              patches.csv, stats.csv, *.npy
├── models/<model_name>/    config.yaml, model.weights.h5, oh_encoder.joblib
├── tensorboard_logs/<model_name>/
├── reference/              saved spatial-graph embedding transforms
└── zarr/                   conventional location for the SpatialData store
```

Set it once, in the constructor:

```python
pheno = Phenocoder(..., dir_project="data/phenocoder")
```

Setting it up front (rather than as a side effect of `generate_dataset`) matters for
workflows that never extract image patches — for example a simulation-derived reference
that only runs the spatial-graph steps.

`config.yaml` records dataset *names*, not absolute paths, and
{meth}`~phenocoder.Phenocoder.load_model` recovers `dir_project` from the config's own
location. A project directory can therefore be moved or shared without editing anything.

Datasets default to `dir_project/<dataset>`. Pass `dir_dataset=` to
{meth}`~phenocoder.Phenocoder.generate_dataset` to place one elsewhere (a scratch disk, or a
dataset shared read-only between projects); the override is recorded per dataset and does not
change `dir_project`. `zarr/` is a naming convention only — Phenocoder does not currently read
or write the store itself.

## CVAE (Convolutional Variational Autoencoder)

- **Encoder**: a stack of strided `Conv2D` layers (downsampling) → `Flatten` → `Dense` →
  `z_mean` and `z_log_var`, with the reparameterization trick producing the latent sample `z`.
- **Decoder**: `Dense` → `Reshape` → stacked `Conv2DTranspose` layers (upsampling) → a final
  `Conv2DTranspose` with sigmoid activation reconstructing all input channels.
- **Loss**: per-channel binary cross-entropy reconstruction loss + `beta` × KL divergence.

See {class}`~phenocoder.model.CVAE`.

## CondCVAE (Conditional CVAE)

Extends `CVAE` by concatenating one-hot encoded condition labels into the encoder (after the
flattened features) and the decoder (with the latent vector). The number of condition columns
chosen in `initialize_model` determines the one-hot dimension; the fitted encoder is saved
alongside the model so the same encoding is reused at inference.

See {class}`~phenocoder.model.CondCVAE`.

## Configuration

Key parameters of {meth}`~phenocoder.Phenocoder.initialize_model`:

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

```{note}
`input_shape` must be consistent with the `patch_size` used in `generate_dataset`
(same height/width, plus the channel count).
```
