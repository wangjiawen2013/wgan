use burn::{
    module::{Module, ModuleMapper, ParamId},
    nn::BatchNorm,
    prelude::*,
};


/// Layer block of generator model
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    fc: nn::Linear<B>,
    bn: nn::BatchNorm<B, 1>,
    leakyrelu: nn::LeakyRelu,
}

impl<B: Backend> LayerBlock<B> {
    pub fn new(input: usize, output: usize, device: &B::Device) -> Self {
        let fc = nn::LinearConfig::new(input, output)
            .with_bias(true)
            .init(device);
        let bn: BatchNorm<B, 1> = nn::BatchNormConfig::new(1).with_epsilon(0.8).init(device);
        let leakyrelu = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();

        Self {
            fc,
            bn,
            leakyrelu,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.fc.forward(input);  // output: [Batch, x]
        let output: Tensor<B, 3> = output.unsqueeze_dim(1);  // output:[ Batch, 1, x]
        let output = self.bn.forward(output);  // output: [Batch, 1, x]
        let output = self.leakyrelu.forward(output);  // output: [Batch, 1, x]
        
        output.flatten(1, 2)  // [Batch, x]
    }
}


// Generator model
#[derive(Module, Debug)]
pub struct Generator<B: Backend> {
    layer1: LayerBlock<B>,
    layer2: LayerBlock<B>,
    layer3: LayerBlock<B>,
    layer4: LayerBlock<B>,
    fc: nn::Linear<B>,
    tanh: nn::Tanh, 
}

impl <B: Backend> Generator<B> {
    /// Applies the forward pass on the input tensor by specified order
    pub fn forward(&self, noise: Tensor<B, 2>) -> Tensor<B, 2> {
        let output = self.layer1.forward(noise);
        let output = self.layer2.forward(output);
        let output = self.layer3.forward(output);
        let output = self.layer4.forward(output);
        let output = self.fc.forward(output);
        let output = self.tanh.forward(output);
        
        output  // return [batch_size, channels*height*width]
    }
}


/// Discriminator model
#[derive(Module, Debug)]
pub struct Discriminator<B: Backend> {
    fc1: nn::Linear<B>,
    leakyrelu1: nn::LeakyRelu,
    fc2: nn::Linear<B>,
    leakyrelu2: nn::LeakyRelu,
    fc3: nn::Linear<B>,
}

impl<B: Backend> Discriminator<B> {
    /// Applies the forward pass on the input tensor by specified order.
    /// The input image shape is [batch, channels, height, width]
    pub fn forward(&self, images: Tensor<B, 4>) -> Tensor<B, 2> {
        // Full connection for each batch
        let output = images.flatten(1, 3);  // output: [batch, channels*height*width]
        let output = self.fc1.forward(output);  // output: [batch, 512]
        let output = self.leakyrelu1.forward(output);  // output: [batch, 512]
        let output = self.fc2.forward(output);  // output: [batch, 256]
        let output = self.leakyrelu2.forward(output);  // output: [batch, 256]
        
        self.fc3.forward(output)  // output: [batch, 1]
    }
}

// Use generator model config to construct a generative model
#[derive(Config, Debug)]
pub struct GeneratorConfig {
    latent_dim: usize,  // dimensionality of the latent sapce
    image_size: usize,
    channels: usize,
}

impl GeneratorConfig {
    /// "init" is used to create other objects, while "new" is usally used to create itself.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Generator<B> {
        // Construct the initialized generator
        let layer1 = LayerBlock::new(self.latent_dim, 128, device);
        let layer2 = LayerBlock::new(128, 256, device);
        let layer3 = LayerBlock::new(256, 512, device);
        let layer4 = LayerBlock::new(512, 1024, device);
        let fc = nn::LinearConfig::new(1024, self.channels*self.image_size*self.image_size)
            .with_bias(true).init(device);

        let generator = Generator {
            layer1,
            layer2,
            layer3,
            layer4,
            fc,
            tanh: nn::Tanh::new(),
        };

        generator
    }
}


// Use discriminator config to construct a adverserial model
#[derive(Config, Debug)]
pub struct DiscriminatorConfig {
    latent_dim: usize,  // dimensionality of the latent sapce
    image_size: usize,
    channels: usize,
}

impl DiscriminatorConfig {
    /// "init" is used to create other objects, while "new" is usally used to create itself.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Discriminator<B> {
        // Construct the initialized discriminator
        let fc1 = nn::LinearConfig::new(self.channels*self.image_size*self.image_size, 512).init(device);
        let leakyrelu1 = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();
        let fc2 = nn::LinearConfig::new(512, 256).init(device);
        let leakyrelu2 = nn::LeakyReluConfig::new().with_negative_slope(0.2).init();
        let fc3 = nn::LinearConfig::new(256, 1).init(device);
        
        let discriminator = Discriminator {
            fc1,
            leakyrelu1,
            fc2,
            leakyrelu2,
            fc3,
        };

        discriminator
    }
}

/// Clip module mapper to clip all module parameters between a range of values
#[derive(Module, Clone, Debug)]
pub struct Clip {
    pub min: f32,
    pub max: f32,
}

impl<B: Backend> ModuleMapper<B> for Clip {
    /// Map an float tensor in the module. Each ParamId and it's tensor value will be transfered
    /// to map_float and the modified value will be retured.
    fn map_float<const D: usize> (&mut self, _id: ParamId, tensor: Tensor<B, D>) -> Tensor <B, D> {
        tensor.clamp(self.min, self.max)
    }
}