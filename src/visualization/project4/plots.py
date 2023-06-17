
import matplotlib.pyplot as plt
import matplotlib.patches as patches
    
def plot_example_images(image, extracted_bboxes, extracted_pred_bboxes, bboxes, pred_bboxes)
    fig, ax = plt.subplots(2, 2, figsize=(12, 12)) 
    ax[0, 0].imshow(modified_image)
    ax[0, 0].set_title("Original image")

    ax[0, 1].imshow(extracted_bboxes[0].permute(1,2,0))
    ax[0, 1].set_title(f"{self.id2cat[category_ids[0][0]]}")

    ax[1, 0].imshow(extracted_pred_bboxes[0].permute(1,2,0))
    ax[1, 0].set_title("One proposed bbox")

    ax[1, 1].imshow(modified_image)
    for bbox in pred_bboxes[:10]:
        rect = patches.Rectangle(
            (bbox[0].item(), bbox[1].item()), 
            width=(bbox[2] - bbox[0]).item(), 
            height=(bbox[3] - bbox[1]).item(), 
            linewidth=2, 
            edgecolor='b', 
            facecolor='none'
        )
        ax[1, 1].add_patch(rect)

    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox[0].item(), bbox[1].item()), 
            width=(bbox[2] - bbox[0]).item(), 
            height=(bbox[3] - bbox[1]).item(), 
            linewidth=3, 
            edgecolor='r', 
            facecolor='none'
        )
        ax[1, 1].add_patch(rect)
    ax[1, 1].set_title("GT and randomly sampled proposed bboxes")

    plt.tight_layout()
    fig.savefig(f"/work3/s194253/02514/test_{idx}_super={self.use_super_categories}.png")
