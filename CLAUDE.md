# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the source code for the book "Deep Learning from Scratch 5" (ゼロから作る Deep Learning ❺) published by O'Reilly Japan. The codebase implements various deep learning concepts and models from scratch, covering statistical distributions, neural networks, VAEs, and diffusion models.

## Common Commands

### Environment Setup
- `uv sync` - Install dependencies using UV package manager
- `uv run python <script>` - Run Python scripts using UV virtual environment

### Running Code
- `cd stepXX && python <script>.py` - Execute individual Python scripts in each step directory
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

- Python 3.12+ required
- Code follows educational structure with clear, readable implementations
- Each step can be run independently 
- Visualization is integral to most scripts - many generate plots to demonstrate concepts
- Some scripts contain TODO comments indicating areas for enhancement