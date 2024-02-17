import matplotlib.pyplot as plt
import numpy as np
import itertools
import io
from PIL import Image
import torch
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
import random
import pandas as pd

#####用于画图###
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    image = transforms.F.to_tensor(Image.open(buf))
    return image

#####画混淆矩阵######
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)
                       [:, np.newaxis], decimals=5)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


######特征可视化-限定数量######
def visualize_features(features: torch.Tensor, targets: torch.Tensor, filename: str, select_n=6000):####展示的特征个数######
    features = features.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # label2feats = {}
    # for i in range(targets.shape[0]):
    #     k = targets[i]
    #     if k not in label2feats:
    #         label2feats[k] = []

    #     label2feats[k].append(features[i])
    
    # selected_feats = []
    # selected_labels = []
    # for k in label2feats:
    #     selected_feats.extend(random.sample(label2feats[k], select_n))
    #     selected_labels.extend([k] * select_n)
    
    # selected_feats = np.vstack(selected_feats)

    # pca = PCA(n_components=2)
    # pca.fit(selected_feats)
    # selected_feats = pca.transform(selected_feats)

    pca = PCA(n_components=2)
    pca.fit(features)
    feats_2d = pca.transform(features)

    label2feats = {}
    for i in range(targets.shape[0]):
        k = targets[i]
        if k not in label2feats:
            label2feats[k] = []

        label2feats[k].append(feats_2d[i])
    
    selected_feats = []
    selected_labels = []
    for k in label2feats:
        n_samples = min(select_n, len(label2feats[k]))
        selected_feats.extend(random.sample(label2feats[k], n_samples))
        selected_labels.extend([k] * n_samples)
    
    selected_feats = np.vstack(selected_feats)

    figure = plt.figure()
    plt.scatter(selected_feats[:, 0], selected_feats[:, 1], c=selected_labels)
    
    
    #####存数据##############
    x = selected_feats[:, 0]
    y = selected_feats[:, 1]
    l = selected_labels
    
    df = pd.DataFrame({"x":x,"y":y,'l':l})
    writer = df.to_excel(filename)
 
    #plt.savefig('./feature_map.png', format='png')
    return figure


######特征可视化-全部数量######
def visualize_features_all(features: torch.Tensor, targets: torch.Tensor, filename: str):####展示的特征个数######
    features = features.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()


    pca = PCA(n_components=2)
    pca.fit(features)
    feats_2d = pca.transform(features)

    figure = plt.figure()
    plt.scatter(feats_2d[:, 0], feats_2d[:, 1], c=targets)
    
    #####存数据##############
    x = feats_2d[:, 0]
    y = feats_2d[:, 1]
    l = targets
    
    df = pd.DataFrame({"x":x,"y":y,'l':l})
    writer = df.to_excel(filename)

    #plt.savefig('./future/feature_map.png', format='png')

    return figure