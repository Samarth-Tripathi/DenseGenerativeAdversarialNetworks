Fisher GAN
===============

Code accompanying the paper ["Fisher GAN"](https://arxiv.org/abs/1705.09675) [arxiv:1705.09675]

Note: This code was obtained by running ```git apply diff.txt``` on [martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN) using diff found in arXiv LaTeX source of arxiv:1705.09675

## A few notes

- The first time running on the LSUN dataset it can take a long time (up to an hour) to create the dataloader. After the first run a small cache file will be created and the process should take a matter of seconds. The cache is a list of indices in the lmdb database (of LSUN)

## Prerequisites

- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.

Two main empirical claims:

### Less compute overhead than Wasserstein GAN



### Better Inception scores




## Reproducing LSUN experiments

**With DCGAN:**

```bash
python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda
```

**With MLP:**

```bash
python main.py --mlp_G --ngf 512
```

Generated samples will be in the `samples` folder.

If you plot the value `-Loss_D`, then you can reproduce the curves from the paper. The curves from the paper (as mentioned in the paper) have a median filter applied to them:

```python
med_filtered_loss = scipy.signal.medfilt(-Loss_D, dtype='float64'), 101)
```

More improved README in the works.
