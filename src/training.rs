use crate::dataset::MnistBatcher;
use crate::model::{GeneratorConfig, DiscriminatorConfig, Clip};
use burn::optim::{RmsPropConfig, Optimizer, GradientsParams};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    prelude::*,
    //record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::{Distribution, backend::AutodiffBackend},
};
use std::path::Path;
use image::{buffer::ConvertBuffer, Rgb32FImage, RgbImage, error::ImageResult};

#[derive(Config)]
pub struct TrainingConfig {
    pub model_g: GeneratorConfig,
    pub model_d: DiscriminatorConfig,
    pub optimizer: RmsPropConfig,
    pub num_epochs: usize,
    pub num_critic: usize,
    pub batch_size: usize,
    pub num_workers: usize,
    pub seed: u64,
    pub lr: f64,
    pub latent_dim: usize,
    pub clip_value: f32,
    pub sample_interval: usize,
}

// Create the directory to save the model and model config
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Save the generated images
// The images format is [B, H, W, C]
fn save_image<B: Backend, Q: AsRef<Path>>(images: Tensor<B, 4>, nrow: u32, path: Q) -> ImageResult<()> {
    let ncol = match images.dims()[0] as u32 % nrow {
        0 => images.dims()[0] as u32 / nrow,
        _ => images.dims()[0] as u32 / nrow + 1,
    };
    let width = images.dims()[2] as u32;
    let height = images.dims()[1] as u32;
    let channels = match images.dims()[3] {
        1 => 3,
        3 => 1,
        _ => panic!("Wrong channels number"),
    };

    let mut imgbuf = RgbImage::new(nrow*width, ncol*height);
    // Write images into a nrow*ncol grid layout
    for row in 0..nrow {
        for col in 0..ncol {
            let image: Tensor<B, 3> = images.clone().slice([(row*col) as usize..(row*col+1) as usize]).squeeze(0);
            // The Rgb32 should be in range 0.0-1.0
            let image = image.into_data().iter::<f32>().collect::<Vec<f32>>();
            // Supports both 1 and 3 channels image
            let image = image.into_iter().flat_map(|n| std::iter::repeat(n).take(channels)).collect();
            let image = Rgb32FImage::from_vec(width, height, image).unwrap();
            let image: RgbImage = image.convert();
            for (x, y, pixel) in image.enumerate_pixels() {
                imgbuf.put_pixel(row*width+x, col*height+y, *pixel);
            }
        }
    }
    imgbuf.save(path)
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str, train_path: &str, valid_path: &str, num_epochs: usize,
    num_critic: usize, batch_size: usize, num_workers: usize, seed: u64, lr: f64,
    latent_dim: usize, image_size: usize, channels: usize, clip_value: f32,
    sample_interval: usize, device: B::Device) {

    create_artifact_dir(artifact_dir);
    // Create the configuration
    let config_model_g = GeneratorConfig::new(latent_dim, image_size, channels);
    let config_model_d = DiscriminatorConfig::new(latent_dim, image_size, channels);
    let config_optimizer = RmsPropConfig::new();
    let config = TrainingConfig::new(
        config_model_g, config_model_d, config_optimizer, num_epochs, num_critic, batch_size,
        num_workers, seed, lr, latent_dim, clip_value, sample_interval
    );
    // Create the Clip module mapper
    let mut clip = Clip {min: -config.clip_value, max: config.clip_value};

    // Save training config
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(config.seed);

    // Create the model and optimizer
    let mut generator = config.model_g.init::<B>(&device);
    let mut discriminator = config.model_d.init::<B>(&device);
    let mut optimizer_g = config.optimizer.init();
    let mut optimizer_d = config.optimizer.init();

    // Create the batcher
    let batcher_train = MnistBatcher::<B>::new(device.clone());

    // Create the dataloaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let mut batches_done = 0;
    // Iterate over our training for X epochs
    for epoch in 1..config.num_epochs + 1 {
        // Implement our training loop
        for (iteration, batch) in dataloader_train.iter().enumerate() {
            // Generate a batch of fake images from noise (standarded normal distribution)
            let noise = Tensor::<B, 2>::random([config.batch_size, config.latent_dim], Distribution::Normal(0.0, 1.0), &device);
            // datach: do not update gerenator, only discriminator is updated
            let fake_images = generator.forward(noise.clone()).detach();  // [batch_size, channels*height*width]
            let fake_images = fake_images.reshape([batch_size, channels, image_size as usize, image_size as usize]);
            // Adversarial loss
            let loss_d = -discriminator.forward(batch.images).mean() + discriminator.forward(fake_images).mean();

            // Gradients for the current backward pass
            let grads = loss_d.backward();
            // Gradients linked to each parameter of the discriminator
            let grads = GradientsParams::from_grads(grads, &discriminator);
            // Update the discriminator using the optimizer
            discriminator = optimizer_d.step(config.lr, discriminator, grads);
            // Clip parameters (weights) of discriminator
            discriminator = discriminator.map(&mut clip);

            // Train the generator every num_critic iterations
            if iteration % config.num_critic == 0 {
                // Generate a batch of images
                let gen_images = generator.forward(noise.clone());
                let gen_images = gen_images.reshape([batch_size, channels, image_size as usize, image_size as usize]);
                // Adversarial loss. Minimize it to make the fake images as truth
                let loss_g = -discriminator.forward(gen_images).mean();
                
                let grads = loss_g.backward();
                let grads = GradientsParams::from_grads(grads, &generator);
                generator = optimizer_g.step(config.lr, generator, grads);

                // Print the progression
                let batch_num = match dataloader_train.num_items() % config.batch_size {
                    0 => dataloader_train.num_items() / config.batch_size,
                    _ => dataloader_train.num_items() / config.batch_size + 1,
                };
                println!(
                    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]",
                    epoch, config.num_epochs, iteration, batch_num, loss_d.into_scalar(), loss_g.into_scalar()
                );
            }
            //  If at save interval => save the first 25 generated image
            if batches_done % config.sample_interval == 0 {
                let gen_images = generator.forward(noise);
                // The Rgb32 images should be in range 0.0-1.0
                let gen_images = gen_images.reshape([batch_size, channels, image_size as usize, image_size as usize]);
                // [B, C, H, W] to [B, H, C, W] to [B, H, W, C]
                let gen_images = gen_images.swap_dims(2, 1).swap_dims(3, 2).slice([0..25]);
                let path = format!("image{}.png", batches_done);
                let path = Path::new(&path);
                save_image::<B, _>(gen_images, 5, path).unwrap();
            }
            batches_done += 1;
        }
    }
}