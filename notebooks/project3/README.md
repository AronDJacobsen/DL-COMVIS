# Notebooks for the GAN project

For project 3. More or less all of these notebooks operate on `projected_w.npy` vectors as output from [styleGAN2-ADA-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). 

None of these are runnable, as the styleGAN2-ADA repo must be cloned separately and run using first a [FFHQ-align script](https://github.com/happy-jihye/FFHQ-Alignment),  then from styleGAN2 the generate.py and projector scripts.

Guide on configuring styleGAN2-ADA (HPC)

all dependencies are in requirements.txt. Note that extra dependencies are Python 3.7
`pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html` and `pip install urllib3==1.26.6 imageio==2.31.1` Although it probably runs on newer versions of pytorch.
```
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```
```
git clone https://github.com/happy-jihye/FFHQ-Alignment.git
```
