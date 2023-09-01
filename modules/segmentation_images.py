import matplotlib.pyplot as plt
import numpy as np

# Color palette to map each class to a RGB value
color_palette = [
    [128, 64, 128],  # 0: road - maroon
    [244, 35, 232],  # 1: sidewalk - pink
    [70, 70, 70],  # 2: building - dark gray
    [102, 102, 156],  # 3: wall - purple
    [190, 153, 153],  # 4: fence - light brown
    [153, 153, 153],  # 5: pole - gray
    [250, 170, 30],  # 6: traffic light - orange
    [220, 220, 0],  # 7: traffic sign - yellow
    [0, 255, 0],  # 8: vegetation - dark green
    [152, 251, 152],  # 9: terrain - light green
    [70, 130, 180],  # 10: sky - blue
    [220, 20, 60],  # 11: person - red
    [255, 0, 0],  # 12: rider - bright red
    [0, 0, 142],  # 13: car - dark blue
    [0, 0, 70],  # 14: truck - navy blue
    [0, 60, 100],  # 15: bus - dark teal
    [0, 80, 100],  # 16: train - dark green
    [0, 0, 230],  # 17: motorcycle - blue
    [119, 11, 32]  # 18: bicycle - dark red
]

def visualize_results(city, image_id, image, segmentation, gvi, num):
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), sharey=True)

    # Display the widened panorama image
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    # Map the segmentation result to the color palette
    seg_color = np.zeros(segmentation.shape + (3,), dtype=np.uint8)
    for label, color in enumerate(color_palette):
        seg_color[segmentation == label] = color

    # Display the colored segmentation result
    ax2.imshow(seg_color)
    ax2.set_title("Segmentation")
    ax2.axis("off")
    
    fig.savefig("results/{}/sample_images/{}-{}.png".format(city, image_id, num), bbox_inches='tight', dpi=100)


def save_images(city, image_id, images, pickles, gvi):
    num = 0

    for image, segmentation in zip(images, pickles):
        num += 1
        visualize_results(city, image_id, image, segmentation, gvi, num)