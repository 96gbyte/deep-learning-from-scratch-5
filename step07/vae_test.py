from vae import train_vae


def test_vae_training():
    """Test that VAE training completes without errors."""
    train_vae(epochs=1, show_plots=False)