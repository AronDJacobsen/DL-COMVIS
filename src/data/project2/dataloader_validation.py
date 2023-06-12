# python src/data/project2/dataloader_validation.py

from src.data.project2.dataloader import get_loaders
import matplotlib.pyplot as plt

loaders = get_loaders(
    dataset='DRIVE', 
    batch_size=6, 
    seed=0, 
    num_workers=1,
    augmentations={'rotate': True, 'flip': True}
)

img, mask = next(iter(loaders[0]['train']))
print(img.shape, mask.shape)

N = 4
fig, ax = plt.subplots(N, 2, figsize=(10, 5))
for i in range(N):
    ax[i, 0].imshow(img[i, :, :, :].permute(1, 2, 0))
    ax[i, 1].imshow(mask[i, :, :].permute(1,2,0), cmap='gray')
plt.savefig('train_DRIVE.png')

img, mask = next(iter(loaders[0]['test']))
print(img.shape, mask.shape)

fig, ax = plt.subplots(N, 2, figsize=(10, 5))
for i in range(N):
    ax[i, 0].imshow(img[i, :, :, :].permute(1, 2, 0))
    ax[i, 1].imshow(mask[i, :, :].permute(1,2,0), cmap='gray')
plt.savefig('test_DRIVE.png')

loaders = get_loaders(
    dataset='PH2', 
    batch_size=6, 
    seed=0, 
    num_workers=1,
    augmentations={'rotate': True, 'flip': True}
)

img, mask = next(iter(loaders['train']))
print('train')
print(img.shape, mask.shape)
print(img, mask)

N = 4
fig, ax = plt.subplots(N, 2, figsize=(10, 5))
for i in range(N):
    ax[i, 0].imshow(img[i, :, :, :].permute(1, 2, 0))
    ax[i, 1].imshow(mask[i, :, :].permute(1,2,0), cmap='gray')
plt.savefig('train_PH2.png')

img, mask = next(iter(loaders['test']))
print('test')
print(img.shape, mask.shape)
print(img, mask)

fig, ax = plt.subplots(N, 2, figsize=(10, 5))
for i in range(N):
    ax[i, 0].imshow(img[i, :, :, :].permute(1, 2, 0))
    ax[i, 1].imshow(mask[i, :, :].permute(1,2,0), cmap='gray')
plt.savefig('test_PH2.png')
