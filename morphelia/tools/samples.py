# import internal libraries
import os
import re
import logging

# import external libraries
from tqdm import tqdm
import numpy as np
import pandas as pd
from skimage import io
from skimage import color
from skimage import exposure
from skimage.util import img_as_ubyte, img_as_float

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def group_samples(adata, file_path, identifier, group_var='leiden',
                  out_path='./samples',
                  loc_x_var='Primarieswithoutborder_Location_Center_X',
                  loc_y_var='Primarieswithoutborder_Location_Center_Y',
                  img_meta='^(?P<Row>[A-H]).*(?P<Column>[0-9]{2}).*(?P<Field>[0-9]{2}).*',
                  img_suffix='.tif',
                  channel_dict={'TexasRed': 'red', 'DAPI': 'blue'},
                  well_var='Metadata_Well',
                  field_var='Metadata_Field',
                  max_patches_per_group=30,
                  patch_size=(300, 300),
                  enhance_contrast=True,
                  verbose=False):
    """Collects information about images from a given experiment and stores patches
    from cells for different groups.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        file_path (str): Path to file with images from a single plate.
        identifier (dict): Identifier for adata query to get view on adata object
            with relevant cells. Should point to a single plate.
        group_var (str): Variables to use for grouping.
        out_path (str): Location to use for saving cell patches.
        loc_x_var (str): Variable with x location of cells/ objects.
        loc_y_var (str): Variable with y location of cells/ objects.
        img_meta (str): Regular expression to use for parsing file names.
            Should contain the groups: Row, Column and Field.
        img_suffix (str): Format of images.
        channel_dict (dict): Channels with colors to use for patches.
        well_var (str): Variable from adata object that stores wells.
        field_var (str): Variable from adata object that stores fields.
        max_patches_per_group (int): Maximum number of patches to be stored per group.
            If None, stores all cell that are found.
        patch_size (tuple): Size of cell patches.
        enhance_contrast (bool): Use skimage.exposure.equalize_adapthist to enhance contrast
            of patches.
        verbose (bool)
    """
    # checking variables
    assert group_var in adata.obs.columns, f'Grouping variable not in adata annotations: {group_var}'
    assert well_var in adata.obs.columns, f'Well variable not in adata annotations: {well_var}'
    assert field_var in adata.obs.columns, f'Field variable not in adata annotations: {field_var}'
    assert loc_x_var in adata.obs.columns, f'X location variable not in adata annotations: {loc_x_var}'
    assert loc_y_var in adata.obs.columns, f'Y location variable not in adata annotations: {loc_y_var}'

    ######################
    # First: Load files from image directory
    ######################
    # check for file existence
    assert os.path.exists(file_path), 'Input path does not exist'
    if not os.path.exists(out_path):
        try:
            os.mkdir(out_path)
        except OSError:
            print(f"Can not use output directory: {out_path}")

    img_df = _load_files(file_path, img_meta, img_suffix, channel_dict, verbose)

    ######################
    # Second: get view on cells from same plate
    ######################
    assert isinstance(identifier, dict), f"expect identifier to be dict, instead got {type(identifier)}"
    query = ' and '.join([f'{k} == {repr(v)}' for k, v in identifier.items()])
    adata_view = adata.obs.query(query)

    # stop if view is empty
    if len(adata_view) == 0:
        raise ValueError(f"identifier incorrect, resulting view on the adata object is empty")

    # drop all wells and fields that were not found in image file
    wells_fields = list(zip(img_df['well'], img_df['field']))
    wells_fields_mask = list(zip(adata_view[well_var], adata_view[field_var]))
    wells_fields_mask = [field_id in wells_fields for field_id in wells_fields_mask]
    adata_view = adata_view[wells_fields_mask]

    # stop if view is empty
    if len(adata_view) == 0:
        raise ValueError(f"No cells found from images from {file_path}")

    ######################
    # Third: Iterate over groups and store patches
    ######################
    def cut_groups(x_df, max_ix):
        if len(x_df) > max_ix:
            return x_df[:max_ix]
        else:
            return x_df

    if max_patches_per_group is not None:
        adata_view = adata_view.groupby(group_var).apply(lambda x: cut_groups(x, max_patches_per_group))
        
    # count group items
    groups = adata_view.groupby(group_var, as_index=False)
    adata_view['group_counts'] = groups.cumcount()

    y_patch_r = patch_size[0] / 2
    x_patch_r = patch_size[1] / 2

    for (well, field), field_df in tqdm(adata_view.groupby([well_var, field_var])):

        # get image_paths and open images
        img_path_df = img_df[(img_df['well'] == well) & (img_df['field'] == field)]
        if len(img_path_df) != len(channel_dict.keys()):
            raise ValueError("wells and fields in image directory are not unique")

        img = _load_images(img_path_df, path_var='file_path', channel_var='channel', channel_dict=channel_dict,
                           adapt_hist=enhance_contrast)

        # iterate over each sample in field_df and get patch
        for index, cell in field_df.iterrows():
            min_y, max_y, min_x, max_x = (int(cell[loc_y_var] - y_patch_r),
                                          int(cell[loc_y_var] + y_patch_r),
                                          int(cell[loc_x_var] - x_patch_r),
                                          int(cell[loc_x_var] + x_patch_r))

            if min_y < 0:
                min_y = 0
            if min_x < 0:
                min_x = 0
            if max_y > img.shape[0]:
                max_y = img.shape[0]
            if max_x > img.shape[1]:
                max_x = img.shape[1]

            patch = img[min_y:max_y, min_x:max_x, :]

            # save patch
            group = cell[group_var]
            row = img_path_df.iloc[0, img_path_df.columns.get_loc('row')]
            column = img_path_df.iloc[0, img_path_df.columns.get_loc('column')]
            gc = cell['group_counts']
            patch_name = f"{row} - {column}(fld {field} gr {group}_{gc})"

            _save_patch(patch, out_path, group, patch_name)

    return None


def _load_files(file_path,
                img_meta='^(?P<Row>[A-H]).*(?P<Column>[0-9]{2}).*(?P<Field>[0-9]{2}).*',
                img_suffix='.tif',
                channel_dict={'TexasRed': 'red', 'DAPI': 'blue'},
                verbose=False):
    """Loads image files from a given directory as dataframe.

    Args:
        file_path (str): Path to file with images from experiment.
        img_meta (str): Regular expression to use for parsing file names.
            Should contain the groups: Row, Column and Field.
        img_suffix (str): Format of images.
        channel_dict (dict): Channels and colors to use for patches.
        verbose (bool)
    """
    # compile regular expression to parse filenames
    pattern = re.compile(str(img_meta + img_suffix))

    # query and store file_ids, well names and field_ids
    file_names = []
    fields = []
    rows = []
    cols = []
    for root, dirs, files in os.walk(file_path):
        for file in files:
            match = pattern.match(file)
            if pattern.match(file):
                file_names.append(file)
                fields.append(int(match.group('Field')))
                rows.append(str(match.group('Row')))
                cols.append(int(match.group('Column')))

    assert len(fields) != 0, f'No images that match pattern: {pattern}'
    # create a data frame with all image information
    df = pd.DataFrame({'file_name': file_names, 'row': rows, 'column': cols, 'field': fields})

    # add well name
    df['well'] = df['row'] + df['column'].map(str)

    # extract channel from file_id
    def find_channel(s, chan):
        for ch in chan:
            if ch in s:
                return ch
        return np.nan

    try:
        channel = list(channel_dict.keys())
    except TypeError:
        print(f'channel_dict should be dict, instead got {type(channel_dict)}')
    df['channel'] = df.file_name.apply(lambda x: find_channel(x, channel))

    # drop missing values
    if verbose:
        logger.info(f"Files without a channel from the given channel list: {df.file_name[df.isnull().any(axis=1)].tolist()}")

    df = df.dropna()

    # extract complete file paths
    df['file_path'] = df.file_name.apply(lambda x: os.path.join(file_path, x))

    return df


def _save_patch(patch, out_path, group_name, file_name, suffix=".png"):
    """Saves patch to directory with specification of groups.

    Args:
        patch (numpy.ndarray): Cell patch.
        out_path (str): Path to output directory.
        group_name (str): Name of patch group.
        file_name (str): Name for image file.
        suffix (str): Valid image format.
    """
    file_name = file_name + suffix
    if os.path.exists(os.path.join(out_path, group_name)):
        io.imsave(os.path.join(out_path, group_name, file_name), arr=patch, check_contrast=False)
    else:
        os.mkdir(os.path.join(out_path, group_name))
        try:
            io.imsave(os.path.join(out_path, group_name, file_name), arr=patch, check_contrast=False)
        except OSError:
            print(f"{file_name} could not be stored to {out_path}/{group_name}")

    return None


def _load_images(img_df, path_var, channel_var, channel_dict, adapt_hist=False):
    """Loads images from a dataframe that contains file paths and
    channel information.

    Args:
        img_df (pandas.DataFrame): Image data frame
        path_var (str): Variable for image paths.
        channel_var (str): Variable for channel names.
        channel_dict (dict): Channels with colors to use for patches.
        adapt_hist (bool): CLAHE contrast enhancement.
    """
    # check colors
    all_colors = ['blue', 'yellow', 'red', 'green', 'magenta']
    if any(x not in all_colors for x in channel_dict.values()):
        raise ValueError(f'one or more colors from channel_dict not supported, '
                         f'supported colors: {all_colors}')

    # get rgb color multipliers
    red_multiplier = [1, 0, 0]
    yellow_multiplier = [1, 1, 0]
    green_multiplier = [1, 0, 1]
    blue_multiplier = [0, 0, 1]
    magenta_multiplier = [1, 0, 1]

    multipliers = {'red': red_multiplier,
                   'yellow:': yellow_multiplier,
                   'green': green_multiplier,
                   'blue': blue_multiplier,
                   'magenta': magenta_multiplier}

    imgs = []
    for index, row in img_df.iterrows():
        img_grayscale = io.imread(row[path_var])
        img_grayscale = img_as_float(img_grayscale)

        img_color = color.gray2rgb(img_grayscale)

        # colorize
        col = channel_dict[row[channel_var]]
        img_color = img_color * multipliers[col]
        imgs.append(img_color)

    # merge channels
    img_merged = sum(imgs)

    if adapt_hist:
        img = exposure.equalize_adapthist(img_merged)
        img = img_as_ubyte(img)
    else:
        img = img_as_ubyte(img_merged)

    return img


