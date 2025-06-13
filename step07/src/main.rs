use clap::Parser;
use step07::train_vae;

#[derive(Parser)]
#[command(about = "Train a Variational Autoencoder (VAE) on MNIST dataset")]
struct Args {
    /// Number of training epochs
    #[arg(short = 'e', long, default_value = "30")]
    epochs: usize,

    /// Skip plotting results
    #[arg(long)]
    no_plots: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    train_vae(args.epochs, !args.no_plots)?;
    Ok(())
}
