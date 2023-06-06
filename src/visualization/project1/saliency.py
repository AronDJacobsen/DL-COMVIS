import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt

from src.models.project1.models import get_model
from src.data.project1.dataloader import get_loaders, get_normalization_constants
from src.utils import invertNormalization

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/dtu/datasets1/02514/hotdog_nothotdog',
                            help="Path to data set.")
    parser.add_argument("--network_name", type=str,
                            help="Network name - either 'efficientnet_b4' or one of the self-implemented ones.")
    parser.add_argument("--model_path", type=str, default='DL-COMVIS/logs/test1234/test/version_1/checkpoints/epoch=9_val_loss=0.7083.ckpt', help='Path to saved model file')
    parser.add_argument("--verbose", type=bool, default=False,
                        help="Determines console logging.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Pseudo-randomness.")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size.")
    # parser.add_argument("--lr", type=float, default=1e-4,
                        # help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="The optimizer to be used.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training the model.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers in the dataloader.")

    return parser.parse_args()

args = parse_arguments()


model0 = get_model(network_name=args.network_name)(args)
# train_mean, train_std = get_normalization_constants(root=args.data_path, seed=args.seed)

train_mean = torch.tensor([0.5132, 0.4369, 0.3576])
train_std = torch.tensor([0.0214, 0.0208, 0.0223])

# Define transforms for training
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),             # flips "left-right"
    # transforms.RandomVerticalFlip(p=1.0),             # flips "upside-down"
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(60, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=train_mean, 
        std=train_std, 
    )
])

# Define transforms for test and validation
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=train_mean, 
        std=train_std, 
    )
])

loaders = get_loaders(
    root=args.data_path, 
    batch_size=args.batch_size, 
    seed=args.seed, 
    train_transforms=train_transforms, 
    test_transforms=test_transforms, 
    num_workers=args.num_workers,
)

class2idx = loaders['train'].dataset.subset.dataset.class_to_idx
idx2class = {v: k for k, v in class2idx.items()}

model = model0.load_from_checkpoint(args.model_path, args=args)
model.eval()

x, y = next(iter(loaders['test']))
x0, y0 = x.to(model.device), y.to(model.device)
# we can only do a single image at the time since otherwise the x.requires_grad_() will fail
fig, ax = plt.subplots(3, 2)
for i in range(3):
    x, y = x0[i+3].reshape(1,3,224,224), y0[i+3]

    x.requires_grad_()
    y_hat = model(x)
    output_idx = y_hat.argmax(dim=1)
    output_max = y_hat[0, output_idx]

    # Do backpropagation to get the derivative of the output based on the image
    output_max.backward()

    # get the saliency map
    saliency, _ = torch.max(x.grad.data.abs(), dim=1) 
    saliency = saliency.reshape(224, 224)

    # Reshape the image
    x, y = x.cpu(), y.cpu()
    image = x.reshape(-1, 224, 224)

    # Visualize the image and the saliency map
    img_back_transformed = invertNormalization(train_mean, train_std)(image).detach().numpy().transpose(1, 2, 0)
    img_back_transformed = img_back_transformed.clip(0, 1)
    ax[i, 0].imshow(img_back_transformed)
    ax[i, 0].set_title(r"$\hat{y}$"+f": {idx2class[y_hat.argmax().item()]}, y: {idx2class[y.item()]}")
    ax[i, 0].axis('off')
    ax[i, 1].imshow(saliency.cpu(), cmap='gray')
    ax[i, 1].axis('off')
plt.tight_layout()
fig.suptitle('Image and Saliency Map')
plt.savefig("src/visualization/project1/saliency_map.png",dpi=300)
