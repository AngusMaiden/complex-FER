import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

def inspect_batch(batches, emotion_list):
    for batch in batches.take(1):
        plt.figure(figsize = (15, 10))
        imgs, labels = batch
        for i, img in enumerate(imgs):
            plt.subplot(4,8,i+1)
            img = img.numpy() * 255
            plt.imshow(img.astype('uint8'))
            plt.axis('off')
            plt.title(emotion_list[np.argmax(labels[i])])
        plt.tight_layout()
        plt.show()

def inspect_layers(model):
    tab = []
    for i, layer in enumerate(model.layers):
        tab.append([i, layer.name, layer.trainable])
    print(tabulate(tab))
