import torch
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt

from src.models.project1.models import get_model
from src.data.project1.dataloader import get_loaders, get_normalization_constants
from src.utils import invertNormalization
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

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
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--optimizer", type=str, default='Adam',
                        help="The optimizer to be used.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training the model.")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers in the dataloader.")
    parser.add_argument("--min_lr", type=float, default=1e-08,
                        help="Minimum allowed learning rater.")
    parser.add_argument("--max_lr", type=float, default=1,
                        help="Maximum allowed learning rate.")
    parser.add_argument("--initial_lr_steps", type=int, default=1000,
                        help="Number of initial steps for finding learning rate.")

    return parser.parse_args()

args = parse_arguments()

model = get_model(network_name=args.network_name)(args)
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

model = model.load_from_checkpoint(args.model_path, args=args)
model.eval()

train_correct = 0
y_true = []
y_pred = []
img_paths = loaders['test'].dataset.imgs
class_names = np.array([i.split('/')[-1].split(' ')[0] for i, _ in img_paths])

for x, y in loaders['test']:
    x, y = x.to(model.device), y.to(model.device)
    
    y_hat = model(x)
    y_hat = torch.argmax(y_hat, dim=1)
    y_true.extend(y.cpu().tolist())
    y_pred.extend(y_hat.cpu().tolist())
    # accuracy
    train_correct += (y==y_hat).sum().cpu().item()

accuracy = train_correct / len(loaders['test'].dataset)
print(f'Accuracy: {accuracy:.4f}')

# binary confusion matrix
conf_mat = confusion_matrix(y_true, y_pred, normalize='true')
tn, fp, fn, tp = conf_mat.ravel()
tnr, fpr, fnr, tpr = tn/(tn+fp), fp/(tn+fp), fn/(fn+tp), tp/(fn+tp)
print("TNR \t & FPR  \t & FNR  \t & TPR  \\\\")
print(f'{tnr:.3f} \t & {fpr:.3f} \t & {fnr:.3f} \t & {tpr:.3f}')

fig, ax = plt.subplots(1)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=loaders['test'].dataset.classes)
disp.plot(cmap='hot')
plt.savefig('src/models/project1/confusion_matrix_binary.png',dpi=300)

y_pred = np.array(y_pred)
y_true = np.array(y_true)
# multi-class confusion matrix
correct_idx = y_pred == y_true
incorrect_idx = y_pred != y_true
correct_classified_true_labels = class_names[correct_idx]
incorrect_classified_true_labels = class_names[incorrect_idx]

fig, ax = plt.subplots(1,2, sharex=True)
ax[0].barh(*np.unique(correct_classified_true_labels, return_counts=True))
ax[0].set_title('Correctly classified')
ax[1].barh(*np.unique(incorrect_classified_true_labels, return_counts=True))
ax[1].set_title('Incorrectly classified')
fig.tight_layout()
plt.savefig('src/models/project1/confusion_matrix_multiclass.png', dpi=300)