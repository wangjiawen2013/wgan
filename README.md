This is an re-implentation of pytorch wgan (https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan/wgan.py) in rust burn. Please note that this is only a draft manuscript and some problems are still needed to adressed. Just copy the following

### Usage:
`cargo run --release --features ndarray -- train --artifact-dir /home/wangjw/data/work/projects/wgan/output --train-path /home/wangjw/data/work/projects/mnist/train --valid-path /home/wangjw/data/work/projects/test --num-epochs 3 --batch-size 64 --num-workers 3 --lr 0.00005 --latent-dim 100 --image-size 28 --channels 1 --num-critic 10 --clip-value 0.01  --sample-interval 100`
