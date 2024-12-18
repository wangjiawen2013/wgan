#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]

mod ndarray {
    use burn::{
        backend::{ndarray::{NdArray, NdArrayDevice}, Autodiff},
        optim::RmsPropConfig,
    };
    use wgan::{
        cli::{Cli, Commands},
        training::{train, TrainingConfig},
    };

    pub fn run(cli: Cli) {
        let device = NdArrayDevice::Cpu;

        match cli.command {
            Commands::Train {
                artifact_dir,
                train_path,
                valid_path,
                num_epochs,
                num_critic,
                batch_size,
                num_workers,
                seed,
                lr,
                latent_dim,
                image_size,
                channels,
                clip_value,
                sample_interval,
            } => {
                train::<Autodiff<NdArray>>(
                    &artifact_dir,
                    &train_path,
                    &valid_path,
                    num_epochs,
                    num_critic,
                    batch_size,
                    num_workers,
                    seed,
                    lr,
                    latent_dim,
                    image_size,
                    channels,
                    clip_value,
                    sample_interval,
                    device,
                );
            },
            //Commands::Generate {artifact_dir, infer_path} => {
            //    inference::infer::<NdArray>(artifact_dir, infer_path, device);
            //},
        }
    }
}


use clap::Parser;
use wgan::cli::Cli;

fn main() {
    let cli = Cli::parse();
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-netlib",
    ))]
    ndarray::run(cli);
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run(cli);
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run(cli);
    #[cfg(feature = "wgpu")]
    wgpu::run(cli);
}