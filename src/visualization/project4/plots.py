
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(2, 2, figsize=(12, 12)) 
ax[0, 0].imshow(img.permute(1,2,0))
ax[0, 0].set_title("Original image")

ax[0, 1].imshow(regions[0].permute(1,2,0))
ax[0, 1].set_title(f"{cat_id[0]}")

ax[1, 0].imshow(pred_regions[0].permute(1,2,0))
ax[1, 0].set_title("One proposed bbox")

ax[1, 1].imshow(img.permute(1,2,0))
for i, bbox in enumerate(pred_bboxes):
    rect = patches.Rectangle(
        (bbox[0].item(), bbox[1].item()), 
        width=(bbox[2] - bbox[0]).item(), 
        height=(bbox[3] - bbox[1]).item(), 
        linewidth=2, 
        edgecolor=f"C{pred_boxes[i].item()}", 
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
fig.savefig(f"/work3/s194253/02514/test_new.png")
