# Phenocoder

A deep learning library for encoding phenotypic image data using Variational Autoencoders (VAEs).

## Overview

Phenocoder provides convolutional variational autoencoder (CVAE) models for learning compressed latent representations of microscopy images. The library supports both standard and conditional VAEs, enabling both unsupervised feature learning and class-conditional image generation.

## Features

- **Convolutional Variational Autoencoder (CVAE)**: Encode high-dimensional image data into compact latent representations
- **Conditional CVAE (CondCVAE)**: Conditional generation with class labels for controlled latent space learning
- **Flexible Architecture**: Configurable convolutional layers, latent dimensions, and regularization
- **Beta-VAE Support**: Adjustable beta parameter for disentangled representation learning
- **Keras 3.0**: Built on modern Keras with TensorFlow backend

## Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Requirements

- Python >= 3.10
- TensorFlow 2.19.0
- SpatialData >= 0.5.0
- Additional dependencies listed in `pyproject.toml`

## Quick Start

```python
from phenocoder import Phenocoder

# Initialize model
model = Phenocoder(
    input_shape=(128, 128, 4),
    latent_dim=128,
    conditional=True,
    n_classes=2
)

# Train model
model.train(data, epochs=100, batch_size=64)

# Encode images to latent space
embeddings = model.encode(images)

# Reconstruct images
reconstructed = model.decode(embeddings)
```

## Model Architecture

### CVAE (Convolutional Variational Autoencoder)

- **Encoder**: Series of convolutional layers with strided convolutions for downsampling
- **Latent Space**: Learned mean and log-variance for probabilistic encoding
- **Decoder**: Transposed convolutions for upsampling and reconstruction
- **Loss**: Reconstruction loss (binary cross-entropy) + KL divergence with adjustable beta weighting

### CondCVAE (Conditional CVAE)

Extends CVAE with class-conditional generation by concatenating one-hot encoded class labels to both the encoder and decoder pathways.

## Configuration

Key hyperparameters:

- `input_shape`: Input image dimensions (height, width, channels)
- `latent_dim`: Dimensionality of latent space (default: 128)
- `dense_dim`: Dense layer size between conv and latent layers (default: 128)
- `conv_layers`: Tuple of channel sizes for convolutional layers (default: (8, 16, 32, 64, 128))
- `dropout`: Dropout rate for regularization (default: 0.5)
- `beta`: Beta-VAE weighting parameter for KL divergence (default: 1)

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

This project is currently under active refactoring to remove dataset-specific code and create a more generic, reusable API.

## License

[Add license information]

## Citation

[Add citation information if applicable]
