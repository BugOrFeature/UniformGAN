# UniformGAN
UniformGAN: generative adversarial network in copula space, built on CTAB-GAN and copulaGAN.

## Paper

The paper can be found in paper/UniformGAN.pdf

## Description

One of the challenges faced in synthetic data generation is aptly modeling the raw data; transforming it into numerical, and specifying the hyper-parameters such as which columns are categorical, mixed type, numerical or log distributed is a non-trivial task. Another difficult task is making estimations about the underlying distributions of the data and how these different distributions are correlated.


## Troubleshooting

If your dataset has large number of columns, you may encounter the problem that our currnet code cannot encode all of your data since CTAB-GAN will wrap the encoded data into an image-like format. What you can do is changing the line 341 and 348 in model/synthesizer/ctabgan_synthesizer.py. The number in the slide list

sides = [4, 8, 16, 24, 32]

is the side size of image. You can enlarge the list to [4, 8, 16, 24, 32, 64] or [4, 8, 16, 24, 32, 64, 128] for accepting a larger dataset.

## Setup

pip install -r requirements.txt
python run.py
