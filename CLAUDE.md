# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the source code for the book "Deep Learning from Scratch 5" (ゼロから作る Deep Learning ❺) published by O'Reilly Japan. The codebase implements various deep learning concepts and models from scratch, covering statistical distributions, neural networks, VAEs, and diffusion models.

## Common Commands

### Environment Setup
- `uv sync` - Install dependencies using UV package manager
- `uv run python <script>` - Run Python scripts using UV virtual environment

### Running Code
- `uv run python stepXX/<script>.py` - Execute individual Python scripts from root directory
- `pytest` - Run tests (pytest is configured in pyproject.toml)

### Key Libraries Used
- PyTorch 2.7.1 (main deep learning framework)
- NumPy 2.3.0 (numerical computing)
- Matplotlib 3.10.3 (plotting and visualization)
- torchvision 0.22.1 (computer vision utilities)
- scipy 1.15.3 (scientific computing)
- tqdm 4.67.1 (progress bars)

## Project Structure

The repository is organized into 10 steps, each building upon previous concepts:

- `step01`: Normal distributions and basic statistics (norm_dist.py, sample_avg.py)
- `step02`: Maximum likelihood estimation (fit.py, generate.py, hist.py)
- `step03`: Multivariate normal distributions (mle.py, plot_3d.py, numpy_matrix.py)
- `step04`: Gaussian Mixture Models (gmm.py, old_faithful.py)
- `step05`: EM Algorithm (em.py, generate.py)
- `step06`: Neural Networks with PyTorch (neuralnet.py, vision.py, regression.py)
- `step07`: Variational Autoencoders (vae.py) - Implements encoder/decoder architecture with reparameterization trick
- `step08`: Hierarchical Variational Autoencoders (hvae.py)
- `step09`: Diffusion Models (diffusion_model.py, simple_unet.py) - Implements full diffusion process with UNet architecture
- `step10`: Advanced Diffusion (conditional.py, classifier_free_guidance.py)

The `notebooks/` directory contains Jupyter notebook versions of all code for cloud execution.

## Architecture Notes

- **VAE Implementation**: Step 7 uses proper encoder-decoder architecture with Gaussian latent variables and KL divergence regularization
- **Diffusion Models**: Step 9 implements complete diffusion process including forward noise addition, reverse denoising with UNet, and positional encoding for timesteps
- **PyTorch Integration**: From step 6 onwards, code uses PyTorch for automatic differentiation and GPU acceleration
- **Data Handling**: MNIST dataset is automatically downloaded to `data/` directory when needed

## Development Notes

### Python Development
- Python 3.12+ required
- Code follows educational structure with clear, readable implementations
- Each step can be run independently 
- Visualization is integral to most scripts - many generate plots to demonstrate concepts
- Some scripts contain TODO comments indicating areas for enhancement

### Rust Development
- Rust edition 2021 required
- Workspace structure with shared dependencies
- Each step implemented as separate package under workspace
- Pure Rust implementation using ndarray for numerical computing

#### Rust Commands
- `cargo check -p stepXX` - Check compilation without building
- `cargo run -p stepXX` - Run step package
- `cargo run -p stepXX -- -e 10` - Run with 10 epochs
- `cargo test -p stepXX` - Run tests for specific step
- `cargo build --workspace` - Build all packages

#### Rust Dependencies
- `burn` 0.17 - Modern deep learning framework with autodiff
- `burn-ndarray` 0.17 - NdArray backend for burn
- `burn-autodiff` 0.17 - Automatic differentiation support  
- `clap` 4.0 - Command line argument parsing
- `anyhow` 1.0 - Error handling

#### Implementation Notes
- Simplified architecture focusing on core concepts
- No GPU acceleration (CPU-only implementation)
- Dummy data generation for testing (actual MNIST loading can be added)
- Type-safe command line interfaces
- Comprehensive error handling with `Result<T, E>`

#### Deep Learning Framework Options
**Current Implementation**: `burn` 0.17 with automatic differentiation
- ✅ Modern Rust ML framework with full autodiff support
- ✅ Type-safe neural network modules
- ✅ Automatic gradient computation and optimization
- ✅ PyTorch-like API with Rust safety
- ✅ Successfully compiles and runs (as of burn 0.17)

**Alternative Implementations**:
- Pure `ndarray` with manual gradient computation
  - ✅ Stable and reliable
  - ✅ Educational clarity  
  - ❌ No automatic differentiation
  - ❌ Manual optimization required
- `tch` - PyTorch Rust bindings
  - ❌ Requires libtorch installation
  - ❌ Complex setup and distribution
- `candle` - HuggingFace Rust framework  
  - ❌ Version dependency conflicts with rand crate

**Burn Framework Notes**:
- burn 0.13/0.14 had bincode dependency issues (resolved in 0.17)
- Requires `burn-autodiff` for gradient computation
- Uses `Autodiff<NdArray<f32>>` backend for automatic differentiation
- Training loop requires explicit type constraints for `Backend::FloatElem`