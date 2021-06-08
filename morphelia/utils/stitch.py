# internal libraries
import collections

# external libraries
import numpy as np
from scipy.stats import zscore
from sklearn.metrics.pairwise import euclidean_distances
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.optim
import pandas as pd


# TODO: include method for easy and advanced stitching
# TODO: Warn if too less objects in overlap regions
# TODO: don't store morphome

class Stitcher(object):
    """Stitches objects of tiles of wells together.

    Args:
        morphome (pd.DataFrame): All morphological data.
        tile_grid (tuple): Rows and columns of tile grid in well
        stitch_obj (str): Object used for stitching (should be prefix of columns)
        size (tuple): Size of images (pixel in y- and x-direction)
        overlap (float): Overlap between tiles in percentage
        stitch_features (iterable or None): list of feature names to use for matching objects.
                If None all features are used for matching.
        tile_var (str): Columns with tile identifiers
        tcol_var (str): Column name for tile column
        trow_var (str): Column name for tile row
        min_inliers (int): minimum keypoints for stitching
        tile_reading (str): Reading method of microscope: horizontal,
                horizontal_serp, vertical, vertical_serp
        group_vars (iterable): Variables to use for grouping wells.
        ransac_reproj_threshold (float): max reprojection error in RANSAC
            to consider a point as an inlier, the higher, the more tolerant
            RANSAC is, defaults to 7.0
        optim_n_iter (int): Number of iterations for global optimization
        optim_lr (float): Learning rate for globar optimization
        output_iter (iterable): Iteration steps for outputs
        loc_id (str): Unique identifier for object location variables (x and y)
    """

    def __init__(self, morphome, tile_grid=(5, 5), stitch_obj="Cells", size=(2048, 2028), overlap=0.2,
                 stitch_features=("AreaShape_Compactness", "AreaShape_Eccentricity", "AreaShape_MajorAxisLength"),
                 tile_var="Metadata_Field", trow_var="Metadata_TileRow", tcol_var="Metadata_TileCol",
                 min_inliers=4, tile_reading="horizontal", group_vars=("BatchNumber", "PlateNumber", "Metadata_Well"),
                 ransac_reproj_threshold=50.0, optim_n_iter=10000, optim_lr=1,
                 output_iter=(0, 100, 200, 300, 500, 1000, 2000, 4000, 6000, 8000, 9000, 9500, 9999),
                 loc_id="Location_Center_"):
        # initialize morphome
        self.morphome = morphome
        self.tile_grid = tile_grid
        self.tile_reading = tile_reading
        self.tile_grid_dict = self.get_tile_grid_dict()
        # undirected graph for image tiles
        self.tile_graph = self.build_graph()
        # store stitching parameters
        self.stitch_obj = stitch_obj

        assert len(size) == 2, f"Not two values for tile size: {size}."
        self.size = size

        assert overlap <= 1, f"Overlap should be between 0 and 1. Given: {overlap}."
        try:
            self.overlap = float(overlap)
        except KeyError:
            f"Overlap value can not be converted to float: {overlap}, {type(overlap)}"

        if stitch_features is not None:
            try:
                iter(stitch_features)
            except TypeError:
                print(f"Features for stitching are neither None nor iterable: {stitch_features}.")
            self.stitch_features = ["_".join([self.stitch_obj, feat]) for feat in stitch_features]

        self.tile_var = tile_var
        self.trow_var = trow_var
        self.tcol_var = tcol_var
        self.group_vars = list(group_vars)
        self.min_inliers = min_inliers
        self.ransac_reproj_threshold = ransac_reproj_threshold
        # parameters for graph based optimization
        self.optim_n_iter = optim_n_iter
        self.optim_lr = optim_lr
        self.output_iter = output_iter
        # store location variables
        self.loc_x = None
        self.loc_y = None
        self.loc_id = loc_id
        # create object to store figures
        self.figures = []

    @classmethod
    def from_morphdata(cls, md):
        """Initializes class with MorphData object.

        Args:
            md (morphome.MorphData)
        """
        return cls(morphome=md.morphome, tile_var=md.tile_var, tile_grid=md.tile_grid,
                   tcol_var=md.tcol_var, trow_var=md.trow_var)

    def get_tile_grid_dict(self):
        """Create Dictionary with row and column for each tile in a grid."""
        # extract rows and columns
        assert len(self.tile_grid) == 2, f"Grid should be a tuple with two integers for rows and columns of tiles."
        tile_rows, tile_cols = self.tile_grid

        # create a dictionary with ImageNumbers as keys and TileRow and TileCol as items
        if self.tile_reading == "horizontal":
            col_ls = list(range(1, tile_cols + 1)) * tile_rows
            row_ls = [row for row in range(1, tile_rows + 1) for _ in range(tile_cols)]
        elif self.tile_reading == "vertical":
            row_ls = list(range(1, tile_rows + 1)) * tile_cols
            col_ls = [col for col in range(1, tile_cols + 1) for _ in range(tile_rows)]
        elif self.tile_reading == "horizontal_serp":
            row_ls = [row for row in range(1, tile_rows + 1) for _ in range(tile_cols)]
            col_ls = (list(range(1, tile_cols + 1)) + list(range(1, tile_cols + 1))[::-1]) * (tile_rows // 2)
            if len(col_ls) == 0:
                col_ls = list(range(1, tile_cols + 1))
            elif (tile_rows % 2) != 0:
                col_ls = col_ls + list(range(1, tile_cols + 1))
        elif self.tile_reading == "vertical_serp":
            col_ls = [col for col in range(1, tile_cols + 1) for _ in range(tile_rows)]
            row_ls = (list(range(1, tile_rows + 1)) + list(range(1, tile_rows + 1))[::-1]) * (tile_cols // 2)
            if len(row_ls) == 0:
                row_ls = list(range(1, tile_rows + 1))
            elif (tile_rows % 2) != 0:
                row_ls = row_ls + list(range(1, tile_rows + 1))
        else:
            reading_methods = ['horizontal', 'horizontal_serp', 'vertical', 'vertical_serp']
            raise ValueError(f"{self.tile_reading} not in reading methods: {reading_methods}")

        tiles = list(range(1, (tile_rows * tile_cols) + 1))
        tile_grid_dict = dict(zip(tiles, list(zip(row_ls, col_ls))))

        return tile_grid_dict

    def build_graph(self):
        """Builds an undirected graph that describes an image's neighbours.

        Returns:
            collections.defaultdict(list): graph
        """
        # build graph of neighbouring tiles
        tile_graph = collections.defaultdict(list)

        for img, (row, col) in self.tile_grid_dict.items():
            for n_img, (n_row, n_col) in self.tile_grid_dict.items():
                if n_row == row and (n_col == col + 1 or n_col == col - 1):
                    if img not in tile_graph[n_img]:
                        tile_graph[img].append(n_img)
                if n_col == col and (n_row == row + 1 or n_row == row - 1):
                    if img not in tile_graph[n_img]:
                        tile_graph[img].append(n_img)

        return tile_graph

    def stitch(self, vis={"batch": [1], "plate": [1,2,3,4,5,6], "well": ["E07"]}):
        """Iterate over plates and wells of a Cellprofiler output and stitch tiles.

        This is the main function of the object. It iterates over every well, builds
        a graph that connects tiles and their neighbors and links neighboring tiles
        by finding its relative translations. Finally it estimates absolute translations
        of tiles relative to the whole well and applies those translations to all
        features with x and y locations stored.

        Args:
            morphome (pd.DataFrame): Cellprofiler output with following columns:
                PlateNumber: Number of plates
                Well: Name of Well
                vis (dict): Groups to use for visualization.
        """
        # check that columns for tile row and column exist
        if self.tcol_var not in self.morphome.columns or self.trow_var not in self.morphome.columns:
            raise KeyError(f"Column variables for tile position are not the data frame."
                           f"Tile Row: {self.trow_var}, Tile Column: {self.tcol_var}")
        # get location variables for x and y locations
        locvars_x = [col for col in self.morphome.columns if "Location" in col and "X" in col]
        locvars_y = [col for col in self.morphome.columns if "Location" in col and "Y" in col]
        assert len(locvars_x) != 0 and len(locvars_y) != 0, f"No columns found for locations."
        loc_x = [var for var in locvars_x if (self.stitch_obj in var) and (self.loc_id in var)]
        loc_y = [var for var in locvars_y if (self.stitch_obj in var) and (self.loc_id in var)]
        assert len(loc_x) == 1 and len(loc_y) == 1, \
            f"No or more than one location variable for object: {self.stitch_obj}, " \
            f"X Location: {loc_x}, Y location: {loc_y}."
        self.loc_x = loc_x[0]
        self.loc_y = loc_y[0]

        # check that variables for grouping exist
        assert len([col for col in self.morphome.columns if any(
            matcher.lower() in col.lower() for matcher in self.group_vars)]) == len(
            self.group_vars), f"Grouping variables not in morphome data: {self.group_vars}"
        # check variables for grouping
        assert len(self.group_vars) == 3, f"Grouping variables should be three: Batch number," \
                                          f"Plate Number and Well Number. ({self.group_vars})"
        stitched_wells = []
        for i, ((batch, plate, well), well_df) in enumerate(self.morphome.groupby(self.group_vars)):
            print(f"Start stitching batch {batch}, plate {plate} and well {well}.")

            if self.stitch_features is not None:
                feat = self.stitch_features.copy()
                feat.extend([self.tile_var, self.loc_x, self.loc_y])
                tile_links = self.build_links(well_df=well_df[feat], tile_var=self.tile_var)
            else:
                # build tile links
                tile_links = self.build_links(well_df=well_df, tile_var=self.tile_var)

            # get initiation values for x and y values of tiles
            xs_init = [(self.tile_grid_dict[key][1] - 1) * self.size[1] for key in sorted(self.tile_grid_dict.keys())]
            ys_init = [(self.tile_grid_dict[key][0] - 1) * self.size[0] for key in sorted(self.tile_grid_dict.keys())]

            # globally optimize links
            iters, losses, trans, params = optimize_rel_trans(
                nodes=sorted(self.tile_grid_dict.keys()),
                links=tile_links,
                xs_init=xs_init,
                ys_init=ys_init,
                n_iter=self.optim_n_iter,
                lr=self.optim_lr,
                output_iter=self.output_iter)

            # cache shape of stitched tiles
            cached_tiles = np.zeros((self.tile_grid[0] * self.size[0], self.tile_grid[1] * self.size[1]))
            well_df["Metadata_Duplicate"] = 0
            # update locations by optimized translations
            for node, (row, col) in self.tile_grid_dict.items():
                well_df.loc[(well_df[self.trow_var] == row) & (well_df[self.tcol_var] == col), locvars_x] = well_df.loc[
                    (well_df[self.trow_var] == row) & (well_df[self.tcol_var] == col), locvars_x].add(
                    trans[-1][node][0], axis=0)
                well_df.loc[(well_df[self.trow_var] == row) & (well_df[self.tcol_var] == col), locvars_y] = well_df.loc[
                    (well_df[self.trow_var] == row) & (well_df[self.tcol_var] == col), locvars_y].add(
                    trans[-1][node][1], axis=0)

                # indicate duplicates
                for index, r in well_df[(well_df[self.trow_var] == row) & (well_df[self.tcol_var] == col)].iterrows():
                    if cached_tiles[int(r[self.loc_y]), int(r[self.loc_x])] == 1:
                        well_df.loc[index, "Metadata_Duplicate"] = 1
                # update cached regions
                x_min, x_max, y_min, y_max = (int(trans[-1][node][0]), int(trans[-1][node][0] + self.size[1]),
                                              int(trans[-1][node][1]), int(trans[-1][node][1] + self.size[0]))
                cached_tiles[y_min:y_max, x_min:x_max] = 1

            well_df = well_df[well_df["Metadata_Duplicate"] == 0]
            well_df = well_df.drop("Metadata_Duplicate", axis=1)

            if batch in vis["batch"] and plate in vis["plate"] and well in vis["well"]:
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                colors = plt.cm.jet(np.random.rand(len(trans[-1])))
                for i, (node, tr) in enumerate(trans[-1].items()):
                    # create a rectangle patch
                    rect = patches.Rectangle((tr[0], tr[1]), self.size[0], self.size[1], linewidth=1,
                                             edgecolor='k', facecolor='none')
                    ax.add_patch(rect)
                    ax.text((tr[0] + self.size[0] // 2), (tr[1] + self.size[1] // 2), str(node), fontsize=12)
                    ax.scatter(well_df[well_df[self.tile_var] == node][self.loc_x],
                               well_df[well_df[self.tile_var] == node][self.loc_y],
                               facecolors='none', lw=2, alpha=0.9, label=str(node),
                               edgecolors=colors[i])
                    ax.legend(title="Tile Objects", bbox_to_anchor=(1, 1), loc="upper left")
                    fig.suptitle(f"Stitching for Well {well}, Plate {plate} and Batch {batch}", fontsize=16)

                # store figures
                self.figures.append((fig, (batch, well, plate)))

            stitched_wells.append(well_df)
            print(f"Done stitching batch {batch}, plate {plate} and well {well}.")

        self.morphome = pd.concat(stitched_wells, ignore_index=True)

        return self.morphome

    def build_links(self, well_df, tile_var, verbose=False):
        """Build links between image tiles per well.

        Args:
            well_df (pandas.DataFrame): objects of a single well
            tile_var (str): column with tile ids
            verbose (bool)
        """
        # store tile links
        tile_links = {}

        # get positions of tiles
        pairs = [(t, n, well_df[well_df[tile_var] == t], well_df[well_df[tile_var] == n],
                  self.tile_grid_dict[t], self.tile_grid_dict[n])
                 for t, ns in sorted(self.tile_graph.items()) for n in ns]

        # estimate transforms for every tile - neighbor pair
        result = [((t, n), self.stitch_pair(t_kps, n_kps, pos=(t_pos, n_pos)))
                  for t, n, t_kps, n_kps, t_pos, n_pos in pairs]

        # optimize pairwise relative translations
        result = self.trans_optimize(result)

        # collect into dictionary
        tile_links.update(dict(result))

        # make links symmetric
        for f, ns in self.tile_graph.items():
            for n in ns:
                if f < n:
                    tile_links[(n, f)] = (None if tile_links[(f, n)] is None
                                          else -tile_links[(f, n)])

        if verbose:
            print('Links: ', tile_links)

        return tile_links

    def stitch_pair(self, t_kps, n_kps, pos, mask=True, verbose=False):
        """Find the translation between two adjacent tiles.

        Args:
            t_kps (pandas.DataFrame): Keypoints of the first tile
            n_kps (pandas.DataFrame): Keypoints of the adjacent tile
            pos (tuple of two tuples of two ints): Row and column of both tiles
            mask (bool): True if mask should be applied
            verbose (bool)

        Returns:
            Relative translation (numpy.ndarray)
        """
        # mask keypoints
        if mask:
            t_kps, n_kps = self.roi_mask(t_kps, n_kps, pos)

            if verbose:
                print(f"{len(t_kps)} objects identified at position {pos[0]}")
                print(f"{len(n_kps)} objects identified at position {pos[1]}")

        # find matching keypoints between tile and its neighbor
        # if enough keypoints are found, else take default transformation
        if len(t_kps.index) >= self.min_inliers and len(n_kps.index) >= self.min_inliers:
            matches = self.match_kps(t_kps, n_kps, pos=pos)

            if verbose:
                print(f"{len(matches)} matches at position {pos}")

            # with all matches, estimate affine transform /w RANSAC
            pts_t = np.array([loc[0] for loc in matches])
            pts_n = np.array([loc[1] for loc in matches])
            transform, inliers = cv.estimateAffinePartial2D(
                pts_n, pts_t,
                method=cv.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_threshold)

            x_shift, y_shift = transform[0, 2], transform[1, 2]
            translation = np.array([x_shift, y_shift])

        # else get default transformation
        else:
            translation = self.default_translation(pos)

        if verbose:
            print(f"x-shift: {translation[0]}, y-shift: {translation[1]}")

        return translation

    def trans_optimize(self, trans, thresh=1.96, verbose=False):
        """Optimize a list of pairwise translations.

        Args:
            trans (list): List of pairwise translations
            thresh (int): threshold to detect outliers
                (thresh times standard deviation)
            verbose (bool)

        Returns:
            list: Optimized pairwise translations
        """
        if verbose:
            print(f'Translations before optimization: {trans}')

        # Sort by vertical and horizontal translations
        v = []
        h = []
        for (f, n), t in trans:
            if t is not None:
                if self.tile_grid_dict[f][0] == self.tile_grid_dict[n][0]:
                    h.append(t)
                else:
                    v.append(t)

        if len(v) == 0 or len(h) == 0:
            raise AssertionError('No vertical or horizontal tranlations calculated!')

        # stack lists
        v = np.stack(v)
        h = np.stack(h)

        # calculate z-scores for v and h
        vz = np.abs(zscore(v, axis=0))
        hz = np.abs(zscore(h, axis=0))
        # if std is 0 z-scores become np.nan
        # replace nan by 0
        vz = np.nan_to_num(vz)
        hz = np.nan_to_num(hz)

        # delete outliers from translations
        # outlier is defined to be outside of threshold
        v = v[((vz > -thresh) & (vz < thresh)).all(axis=1)]
        h = h[((hz > -thresh) & (hz < thresh)).all(axis=1)]
        assert v.size != 0 and h.size != 0, \
            'Translation optimization not possible, to many outliers!'

        # get mean, max and min translations for both directions
        v_mean, v_std = v.mean(axis=0), v.std(axis=0)
        v_min, v_max = v.min(axis=0), v.max(axis=0)
        h_mean, h_std = h.mean(axis=0), h.std(axis=0)
        h_min, h_max = h.min(axis=0), h.max(axis=0)

        if verbose:
            print(f'Mean horizontal translation: {h_mean}, '
                  f'Mean vertical translation: {v_mean}')
            print(f'Standard deviation for horizontal translations: {h_std}, '
                  f'Standard deviation for vertical translations: {v_std}')

        # iterate again over translations and replace
        # outliers and Nones with mean values
        optim_trans = []
        for (f, n), t in trans:
            # horizontal translations
            if self.tile_grid_dict[f][0] == self.tile_grid_dict[n][0]:
                if t is None or (t < h_min).any() or (t > h_max).any():
                    optim_trans.append(((f, n), h_mean))
                else:
                    optim_trans.append(((f, n), t))
            # vertical translations
            else:
                if t is None or (t < v_min).any() or (t > v_max).any():
                    optim_trans.append(((f, n), v_mean))
                else:
                    optim_trans.append(((f, n), t))

        return optim_trans

    def roi_mask(self, t_kps, n_kps, pos):
        """Filter all keypoints for keypoints in overlapping regions.

        Args:
            t_kps (pandas.DataFrame): Keypoints of a single tile
            n_kps (pandas.DataFrame): Keypoints of adjancent tile
            pos (tuple with two tuples with two ints): Positions of both tiles

        Returns:
            t_kps, n_kps (tuple of two pandas.DataFrame)
        """
        # function needs exactly two positions
        assert len(pos) == 2, 'Mask for keypoint detection needs two image positions.' \
                              f'{len(pos)} are given.'

        # unpack positions
        pos_t, pos_n = pos
        row_t, col_t = pos_t
        row_n, col_n = pos_n

        # get min and max values for locations of keypoints in both tiles
        if row_n == row_t:
            if col_n > col_t:
                y_min_t, y_max_t, x_min_t, x_max_t = 0, self.size[0], int(self.size[1] * (1 - self.overlap)), self.size[
                    1]
                y_min_n, y_max_n, x_min_n, x_max_n = 0, self.size[0], 0, int(self.size[1] * self.overlap)
            if col_n < col_t:
                y_min_t, y_max_t, x_min_t, x_max_t = 0, self.size[0], 0, int(self.size[1] * self.overlap)
                y_min_n, y_max_n, x_min_n, x_max_n = 0, self.size[0], int(self.size[1] * (1 - self.overlap)), self.size[
                    1]
        elif col_n == col_t:
            if row_n > row_t:
                y_min_t, y_max_t, x_min_t, x_max_t = 0, int(self.size[0] * self.overlap), 0, self.size[1]
                y_min_n, y_max_n, x_min_n, x_max_n = int(self.size[0] * (1 - self.overlap)), self.size[0], 0, self.size[
                    1]
            if row_n < row_t:
                y_min_t, y_max_t, x_min_t, x_max_t = int(self.size[0] * (1 - self.overlap)), self.size[0], 0, self.size[
                    1]
                y_min_n, y_max_n, x_min_n, x_max_n = 0, int(self.size[0] * self.overlap), 0, self.size[1]
        else:
            raise ValueError(f'Arguments for positions incorrect: {pos}')

        # filter keypoint dataframes
        t_kps = t_kps[(t_kps[self.loc_y] > y_min_t) & (t_kps[self.loc_y] < y_max_t)]
        t_kps = t_kps[(t_kps[self.loc_x] > x_min_t) & (t_kps[self.loc_x] < x_max_t)]
        n_kps = n_kps[(n_kps[self.loc_y] > y_min_n) & (n_kps[self.loc_y] < y_max_n)]
        n_kps = n_kps[(n_kps[self.loc_x] > x_min_n) & (n_kps[self.loc_x] < x_max_n)]

        return t_kps, n_kps

    def match_kps(self, t_kps, n_kps, pos, by="location"):
        """Matches keypoints in two dataframes.

        Args:
            t_kps (pandas.DataFrame)
            n_kps (pandas.DataFrame)
            pos (tuple of two tuples of two ints): row and column of tile
            by (str): matching either by location or numerical features
                "location": uses y location when vertical and x location when horizontal neighbor
                "num_feat": distance between all numerical features

        Returns:
            (list of tuples): locations of tile objects and its matches in
                the adjacent tile
        """
        matching_methods = ["location", "num_feat"]
        assert by in matching_methods, f"Value for by ({by}) none of {matching_methods}"

        if by == "num_feat":
            # get numerical columns
            t_num_cols = t_kps.select_dtypes(include=[np.number]).columns
            n_num_cols = n_kps.select_dtypes(include=[np.number]).columns

            # get distance matrix
            dist = euclidean_distances(t_kps[t_num_cols], n_kps[n_num_cols])

        elif by == "location":
            # check if tiles are aligned horizontal or vertical
            if pos[0][0] == pos[1][0]:  # horizontal
                dist = euclidean_distances(t_kps[self.loc_y].values.reshape(-1, 1),
                                           n_kps[self.loc_y].values.reshape(-1, 1))
            else:  # vertical
                dist = euclidean_distances(t_kps[self.loc_x].values.reshape(-1, 1),
                                           n_kps[self.loc_x].values.reshape(-1, 1))

        # get indices of closest neighbors
        t_nbs = np.argmin(dist, axis=1)
        n_nbs = np.argmin(dist, axis=0)
        # store nearest neighbor for every object
        nns = []
        for i, nb in enumerate(t_nbs):
            if n_nbs[nb] == i:
                nns.append(nb)
            else:
                nns.append(None)

        # collect locations of matches
        matched = []
        for (ix, row), nn in zip(t_kps.iterrows(), nns):
            if nn is not None:
                t_loc = (row[self.loc_x], row[self.loc_y])
                n_loc = (n_kps.iloc[nn][self.loc_x], n_kps.iloc[nn][self.loc_y])
                matched.append((t_loc, n_loc))

        return matched

    def default_translation(self, pos):
        """Takes the position of two tiles and estimates translation.

        Args:
            pos (tuple of two tuples of two ints): positions (row, column) of both tiles

        Returns:
            Relative translation (numpy.ndarray)
        """
        # neighbor tile can be located above, under, left or right of a tile
        (row_t, col_t), (row_n, col_n) = pos

        # default shift values
        x_def = (1 - self.overlap) * self.size[1]
        y_def = (1 - self.overlap) * self.size[0]

        if row_t == row_n:
            if col_n > col_t:
                x_shift = x_def
                y_shift = 0
            else:
                x_shift = -x_def
                y_shift = 0
        elif col_t == col_n:
            if row_n > row_t:
                x_shift = 0
                y_shift = -y_def
            else:
                x_shift = 0
                y_shift = y_def
        else:
            raise BrokenPipeError(f"No default shift for position {pos}.")

        # get relative translation
        translation = np.array([x_shift, y_shift])

        return translation


def optimize_rel_trans(nodes, links, xs_init, ys_init, n_iter=1000, lr=1, output_iter=None, verbose=False):
    """Globally optimizes pairwise relative translations.

    The loss function is the mean of distance between two relative
    translations between pairs of tiles, as estimated in the
    stitching procedures.

    Args:
        nodes (list of int): ids of images in the graph
        links (dict): {(int, int): affine.Affine}
            transforms between pairs of images that are estimated in the
            stitching procedures
        xs_init, ys_init (list of float [len(nodes),]):
            initial values for x and y translation shifts
            these refer to the centroids
        n_iter (int): number of iterations
        lr (float): param specific learning rates
            for the adam optimizer
        output_iter (list of int): iterations where loss and affines are
            returned as outputs, if None, this defaults to [last iteration]
        verbose (bool)

    Returns:
        tuple (list of int, list of float, list of list of affine.Affine):
            iterations, losses at those iterations,
            transforms estimated at those iterations
    """
    if output_iter is None:
        # output results from last iteration
        output_iter = [n_iter - 1]

    # prepare for init
    rel_true_tensor = []  # collects true relative links
    trans_i_idx = []  # collects integer indices for i in link(i, j)
    trans_j_idx = []  # collects integer indices for j in link(i, j)
    for (i_idx, j_idx), trans in links.items():
        if trans is not None:
            rel_true_tensor.append(torch.from_numpy(trans).float().T)
            trans_i_idx.append(nodes.index(i_idx))
            trans_j_idx.append(nodes.index(j_idx))

    if len(rel_true_tensor) == 0:
        raise ValueError('No links available for optimization.')
    rel_true_tensor = torch.stack(rel_true_tensor)  # [n_links, 2]

    # initialize leaf nodes
    xs = torch.tensor(xs_init, dtype=torch.float, requires_grad=True)
    ys = torch.tensor(ys_init, dtype=torch.float, requires_grad=True)

    # initialize optimizer and scheduler
    optimizer_xy = torch.optim.Adam([xs, ys], lr=lr)

    # prepare to collect output
    if (n_iter - 1) not in output_iter:
        output_iter.append(n_iter - 1)
    output_loss = []
    output_trans = []
    output_params = {}

    # iterate n_iter times
    for k in range(n_iter):
        optimizer_xy.zero_grad()
        # compute absolute translations
        trans = torch.stack([  # [n_nodes, 2]
            xs, ys]).T

        # extract i, j translations to estimate relative translations
        trans_i = trans[trans_i_idx, ...]  # [n_links, 2]
        trans_j = trans[trans_j_idx, ...]  # [n_links, 2]

        # get estimated relative translations
        rel_est_tensor = trans_j - trans_i  # [n_links, 2]

        # compute loss on each link (between true/estimated relative translations)
        # loss is the mean squared distance between points
        # in the true versus estimated relative translation
        losses = ((rel_est_tensor - rel_true_tensor) ** 2
                  ).mean(axis=0)  # -> [1, 2]
        loss = losses.sum()  # -> [1,]

        if verbose:
            if (k + 1) % 200 == 0:
                print('Iter: {}; Loss: {:.3f}'.format(k, loss.item()))
        # output affines and loss
        if k in output_iter:
            output_loss.append(loss.item())
            trans_dict = {}
            for node, trans in list(zip(nodes, trans.detach().numpy())):
                trans_dict[node] = trans
            output_trans.append(trans_dict)
        # back propagate
        loss.backward()
        optimizer_xy.step()

    # output transformation params
    output_params['loc'] = list(zip(xs.tolist(), ys.tolist()))

    return output_iter, output_loss, output_trans, output_params
