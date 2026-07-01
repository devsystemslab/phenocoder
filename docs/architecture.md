# Model architecture

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
