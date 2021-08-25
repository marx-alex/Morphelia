import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import os


def annotate_cells(adata, ann_var, channel1_path,
                   channel2_path=None,
                   channel3_path=None,
                   loc_x_var='Primarieswithoutborder_Location_Center_X',
                   loc_y_var='Primarieswithoutborder_Location_Center_Y',
                   cmap='tab10',
                   **kwargs):
    """Annotate image of cells with variable from view of an anndata object.

    adata (anndata.AnnData): Multidimensional morphological data.
    ann_var (str): Name of variable to annotate.
    channel1_path (str): Path to channel 1.
    channel2_path (str): Path to channel 2.
    channel3_path (str): Path to channel 3.
    loc_x_var (str): Variable with x location of cells/ objects.
    loc_y_var (str): Variable with y location of cells/ objects.
    cmap (str): Matplotlib colormap to be used.
    **kwargs (dict): Keyword arguments passed to matplotlib.pyplot.scatter
    """
    # open images
    imgs = []
    for path in [channel1_path, channel2_path, channel3_path]:
        if path is not None:
            if os.path.exists(path):
                img = io.imread(path)
                imgs.append(img)
            else:
                raise OSError(f"Path not found: {path}")

    # location
    if isinstance(loc_x_var, np.ndarray):
        loc_x = loc_x_var
    elif loc_x_var in adata.var_names:
        loc_x = adata[:, loc_x_var].X
    elif loc_x_var in adata.obs.columns:
        loc_x = adata.obs[loc_x_var].to_numpy()
    else:
        raise ValueError(f"Variable for x location not found: {loc_x_var}")

    if isinstance(loc_y_var, np.ndarray):
        loc_y = loc_x_var
    elif loc_y_var in adata.var_names:
        loc_y = adata[:, loc_y_var].X
    elif loc_y_var in adata.obs.columns:
        loc_y = adata.obs[loc_y_var].to_numpy()
    else:
        raise ValueError(f"Variable for y location not found: {loc_y_var}")

    # color
    if isinstance(ann_var, np.ndarray):
        ann = ann_var
    elif ann_var in adata.var_names:
        ann = adata[:, ann_var].X
    elif ann_var in adata.obs.columns:
        ann = adata.obs[ann_var].to_numpy()
    else:
        raise ValueError(f"Variable for annotation not found: {ann_var}")

    data = pd.DataFrame({loc_x_var: loc_x, loc_y_var: loc_y,
                         ann_var: ann})
    if _isint(data.loc[0, ann_var]):
        data['ann_code'] = data[ann_var].apply(lambda x: int(x))
        data = data.sort_values('ann_code')
        data['ann_code'] = data['ann_code'].apply(lambda x: plt.get_cmap(cmap).colors[x])
    else:
        ord_enc = OrdinalEncoder()
        data['ann_code'] = ord_enc.fit_transform(data[[ann_var]])
        data['ann_code'] = data[ann_var].apply(lambda x: int(x))
        data = data.sort_values('ann_code')
        data['ann_code'] = data['ann_code'].apply(lambda x: plt.get_cmap(cmap).colors[x])

    handles = set(zip(data['ann_code'].tolist(), data[ann_var].tolist()))
    sorted_handles = sorted(handles, key=lambda tup: tup[1])

    # legend
    cells = []
    for col, label in sorted_handles:
        cell = mpatches.Patch(color=col, label=label)
        cells.append(cell)

    kwargs.setdefault('facecolor', 'none')
    kwargs.setdefault('alpha', 0.8)
    kwargs.setdefault('s', 1000)
    kwargs.setdefault('linewidths', 3)

    width = 12 * len(imgs)

    plt.figure(figsize=(width, 12))

    for ix, image in enumerate(imgs):

        plt.subplot(1, len(imgs), ix+1)
        plt.imshow(image, cmap='gray'), plt.axis('off')
        plt.scatter(x=data[loc_x_var],
                    y=data[loc_y_var],
                    edgecolor=data['ann_code'],
                    **kwargs)
        plt.legend(handles=cells)

    plt.suptitle(f"Annotation by {ann_var}", fontsize=24)


def _isint(s):
    """Test if s is string that contains a number."""
    try:
        int(s)
        return True
    except ValueError:
        return False

