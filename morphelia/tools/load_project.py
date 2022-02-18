# import internal libraries
import re
import os
import pathlib
import warnings

# import external libraries
import pandas as pd
import anndata as ad

__all__ = ["MorphData",
           "LoadPlate"]


class LoadPlate:
    def __init__(self, path,
                 filenames=("Cells", "Primarieswithoutborder", "Cytoplasm"),
                 obj_sfx=".csv",
                 merge_method='object',
                 obj_well_var="Metadata_Well",
                 meta_var="Metadata",
                 image_var="ImageNumber",
                 obj_var="ObjectNumber",
                 obj_delimiter=",",
                 treat_file=None,
                 treat_well_var="well",
                 treat_sfx=".csv",
                 treat_delimiter=",",
                 observation_ids='cp'
                 ):
        self.path = pathlib.Path(path)

        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.obj_sfx = obj_sfx

        avail_merge_methods = ['relation', 'object']
        merge_method = merge_method.lower()
        assert (
                merge_method in avail_merge_methods
        ), f"merge_method must be one of {avail_merge_methods}, " \
           f"instead got {merge_method}"
        self.merge_method = merge_method

        self.obj_well_var = obj_well_var
        self.meta_var = meta_var
        self.image_var = image_var
        self.obj_var = obj_var
        self.obj_delimiter = obj_delimiter
        self.merge_vars = [image_var, obj_var, obj_well_var]
        self.treat_file = treat_file
        self.treat_well_var = treat_well_var
        self.treat_sfx = treat_sfx
        self.treat_delimiter = treat_delimiter
        self.plate = None

        self.observation_ids = []
        if observation_ids == 'cp':
            self.observation_ids = ['ImageNumber',
                                    'ObjectNumber',
                                    'Metadata',
                                    'Location',
                                    'Parent']
        elif isinstance(observation_ids, list):
            self.observation_ids = observation_ids

    def load(self):
        """Merge and load objects from a single plate.
        """
        dfs = {}
        parent_graph = {}   # if method is relation

        for file in self.filenames:
            _df = pd.read_csv(self.path.joinpath(file).with_suffix(self.obj_sfx), sep=self.obj_delimiter)
            assert (
                all(col in list(_df.columns) for col in self.merge_vars)
            ), f"Variables needed for merging must be in all files: {self.merge_vars}"

            new_cols = {col: "_".join([file, col]) for col in _df.columns
                        if (col not in self.merge_vars)
                        and (self.meta_var not in col)}
            _df = _df.rename(columns=new_cols)
            dfs[file] = _df

            # cache parents
            parents = [col.split("_")[2] for col in _df.columns if "parent" in col.lower()]
            parent_graph[file] = parents

        if self.merge_method == 'relation':
            plate_df = self.relational_merge(dfs, parent_graph)

        elif self.merge_method == 'object':
            plate_df = None
            for obj in self.filenames:
                if plate_df is not None:
                    meta_cols = [col for col in dfs[obj].columns if
                                 (self.meta_var in col) and (col not in self.merge_vars)]
                    plate_df = plate_df.merge(
                        dfs[obj].drop(meta_cols, axis=1),
                        left_on=list(self.merge_vars),
                        right_on=list(self.merge_vars))
                else:
                    plate_df = dfs[obj]
        else:
            plate_df = None

        # add treatment file
        if self.treat_file is not None:
            plate_df = self.add_annotation(plate_df)

        self.plate = plate_df
        return plate_df

    def add_annotation(self,
                       plate_df):
        treat = pd.read_csv(self.path.joinpath(self.treat_file).with_suffix(self.treat_sfx), sep=self.treat_delimiter)

        treat[self.treat_well_var] = treat[self.treat_well_var].apply(standard_well)

        # rename columns of treatment dataframe
        new_treat_cols = {value: ("_".join([self.meta_var, value]) if
                                  value != self.treat_well_var else self.obj_well_var)
                          for value in treat.columns}
        treat.rename(columns=new_treat_cols, inplace=True)

        # standardize well variable
        plate_df[self.obj_well_var] = plate_df[self.obj_well_var].apply(standard_well)

        plate_df = plate_df.merge(treat,
                                  left_on=[self.obj_well_var],
                                  right_on=[self.obj_well_var])

        return plate_df

    def relational_merge(self,
                         df_dict, parent_graph):

        # find best path through all nodes of parent_graph
        def graph_paths(graph, start_node, visited):
            if start_node in graph.keys():
                visited.append(start_node)
                for node in graph[start_node]:
                    if node not in visited:
                        graph_paths(graph, node, visited.copy())
            if len(path_list) == 0 or len(visited) > len(path_list[-1]):
                path_list.append(visited)

        path_list = []
        for obj in parent_graph.keys():
            graph_paths(parent_graph, obj, [])
        assert (
            len(path_list) > 0
        ), f"Plate files can not be merged relational, because there are no common parents: {parent_graph}"

        best_path = path_list[-1]

        # merge objects with best_path
        plate_df = None
        parent = None
        for obj in reversed(best_path):
            if plate_df is not None:
                parent_obj = "_".join([obj, "Parent", parent])
                meta_cols = [col for col in df_dict[obj].columns if (
                        self.meta_var in col) and (col not in self.merge_vars)]
                plate_df = plate_df.merge(
                    df_dict[obj].drop(meta_cols, axis=1).rename(
                        columns={self.obj_var: "_".join([obj, self.obj_var])}),
                    left_on=list(self.merge_vars),
                    right_on=[self.image_var, parent_obj, self.obj_well_var])
            else:
                plate_df = df_dict[obj]
            parent = obj

        return plate_df

    def to_anndata(self,
                   observation_ids=None):
        """Create AnnData object from dataframe.

        Args:
            observation_ids (iterable): Identifiers in columns to store as annotations.
                Takes default Cellprofiler variables if None.
        """
        # append observation identifiers if given
        if observation_ids is not None:
            if isinstance(observation_ids, str):
                observation_ids = [observation_ids]
            self.observation_ids = self.observation_ids + observation_ids

        assert isinstance(self.plate, pd.DataFrame), "No data loaded as DataFrame"
        assert len(self.observation_ids) > 0, "No observations found"
        obs_cols = [col for col in self.plate.columns if
                    any(matcher.lower() in col.lower() for matcher in self.observation_ids)]
        # variable annotation
        var = pd.DataFrame(index=self.plate.drop(obs_cols, axis=1).columns)
        # create AnnData object
        adata = ad.AnnData(X=self.plate.drop(obs_cols, axis=1).to_numpy(),
                           obs=self.plate[obs_cols], var=var)

        self.plate = adata
        return adata


def standard_well(s):
    """Standardize well names.
    Name should be like E07 or E7.
    """
    row = re.split(r"(\d+)", s)[0]
    col = str(int(re.split(r"(\d+)", s)[1]))
    return row + col


class MorphData(object):
    """Multidimensional morphological data.

    Stores all experiment-associated morphological data.
    MorphData is designed to read morphological outputs from
    microscopy experiments from multiple batches, plates and objects.
    Typically the output is created by a Cellprofiler pipeline.
    More about Cellprofiler can be found here:
    https://cellprofiler.org/

    Args:
        morphome (pandas.DataFrame): contains at least the following keys
            'ObjectNumber': Number of described object in image
            'ImageNumber': Number of image in well
            'Well': Name of Well
        tile_grid (tuple): Grid of tiles: rows, columns
        tile_reading (str): Reading method of microscope: horizontal,
            horizontal_serp, vertical, vertical_serp
        tile_var (str): Variable with field/ tile ids
        add_tile_pos (bool): If True, annotations for tile positions are added.
            Has to be True for stitching.
        tcol_var (str): Column name for tile column
        trow_var (str): Column name for tile row
        obj_delimiter (str): Delimiter for for object .csv files
        treat_delimiter (str): Delimiter for treatment .csv file
        datadict (dict): Dictionary with input files.
        obs_ids (list): List with strings that identify observational features.

    """

    def __init__(self, morphome=None, tile_grid=(5, 5), tile_reading="horizontal",
                 tile_var="Metadata_Field", add_tile_pos=False,
                 trow_var="Metadata_TileRow", tcol_var="Metadata_TileCol",
                 obj_delimiter=",", treat_delimiter=",", datadict=None, obs_ids=None):
        """
        Args:
            morphome (pandas.DataFrame): Cellprofiler output
        """
        # initialize variables
        self.tile_grid = tile_grid
        self.tile_reading = tile_reading
        self.tile_var = tile_var
        self.tcol_var = tcol_var
        self.trow_var = trow_var
        self.treat_delimiter = treat_delimiter
        self.obj_delimiter = obj_delimiter
        # initialize self.morphome
        if add_tile_pos is True and morphome is not None:
            self.morphome = self.add_tile_pos(morphome)
        else:
            self.morphome = morphome

        # store identifiers for annotation columns
        if obs_ids is None:
            self.obs_ids = ['Number', 'Center', 'Box', 'Parent', 'Child',
                            'Euler', 'Count', 'Metadata', 'Location']
        else:
            self.obs_ids = obs_ids

        # dictionary of data files
        self.datadict = datadict

    def from_csv(self, exp, treat_file=None, exp_well_var="Metadata_Well", treat_well_var="well",
                 files=("Cells.csv", "Primarieswithoutborder.csv", "Cytoplasm.csv"),
                 merge_method='relation',
                 meta_var="Metadata", exp_image_var="ImageNumber", exp_obj_var="ObjectNumber",
                 obj_delimiter=",", treat_delimiter=",", to_disk=None, output=None,
                 **kwargs):
        """Extracts phenotypic data per object from Cellprofiler Output 'Export to Spreadsheet'
        The output directory could look like the following:

        exp
        |----batch1
        |       |----plate1
        |       |       |----Cells.csv
        |       |       |----Primarieswithoutborder.csv
        |       |       |----Cytoplasm.csv
        |       |
        |       |----plate2
        |       ...
        |       |----platen
        |
        |----batch2
        ...
        |----batchn

        Args:
            exp (str): Path to experiment directory.
                Can contain subdirectories with plate data or a single plate.
            treat_file (str): Name of treatment file. Is ignored if None.
            exp_well_var (str): Name of well variable in experiment files.
            treat_well_var (str): Name of well variable in treatment file.
            files (iterable of str): Filenames to merge into dataframe.
                If None, all csv or txt files will be merged.
            merge_method (str): Method to use to merge object data. Can be one of:
                relation: Merge by information about parent and child objects.
                object_id: Merge by ImageNumber and ObjectNumber.
                Has to be True for stitching.
            meta_var (str): Identifier for columns with metadata that is the same
                for all objects.
            exp_image_var (str): Name of image variable in experiment files.
            exp_obj_var (str): Name of object variable in experiment files.
            obj_delimiter (str): Delimiter for for object .csv files.
            treat_delimiter (str): Delimiter for treatment .csv file.
            to_disk (str): Can be used to save memory. Batches will be saved directly
                as .hdf5 files without constructing a large dataframe.
                'plate': Save each plate independently.
                'batch': Save each batch independently.
            output (str): Location used to save hdf5 files. Only relevant if
                to_disk is True.

        Returns:
            pandas.DataFrame
        """
        # assert experiment path exists
        assert os.path.exists(pathlib.Path(exp)), f"Path {exp} does not exist"
        # assertion if data is saved directly
        if to_disk in ['plate', 'batch']:
            assert output is not None, "Select output path if to_disk is True"
            assert os.path.exists(pathlib.Path(output)), f"Path for output does not exist: {output}"
        elif to_disk is not None:
            raise ValueError(f"to_disk should be either 'plate', 'batch' or None: {to_disk}.")
        # create list of dictionaries
        # each dictionary represents one batch with one or more plates
        datadict = self._create_datadict(exp, files)

        assert len(list(datadict.keys())) != 0, f"{files} not found in {exp} " \
                                                f"or files are not .csv or .txt"

        # merging method
        avail_merge_methods = ['relation', 'object_id']
        merge_method = merge_method.lower()
        assert merge_method in avail_merge_methods, f"merge_method must be one of {avail_merge_methods}, " \
                                                    f"instead got {merge_method}"

        # create final dataframe
        morphome_list = []
        # store missing plates
        missing_plates = []
        plate_err = False

        # iterate over batches
        for batch_i, batch in enumerate(sorted(datadict.keys())):
            print('Reading Morphome Data from CSV...')
            batchdata_list = []
            # iterate over plates
            for plate_i, plate_path in enumerate(sorted(datadict[batch].keys())):

                # create parent graph
                parent_graph = {}
                # cache dataframes of objects
                plate_dfs = {}

                for file in datadict[batch][plate_path]:
                    open_df = pd.read_csv(os.path.join(plate_path, file), sep=obj_delimiter)

                    # check if merge columns in dataframe
                    merge_vars = [exp_image_var, exp_obj_var, exp_well_var]
                    if not all(col in list(open_df.columns) for col in merge_vars):
                        plate_err = True
                        warnings.warn(f"{merge_vars} not found in {os.path.join(plate_path, file)}")
                        break
                    # check if nan values in meta_vars
                    if open_df[merge_vars].isnull().values.any():
                        plate_err = True
                        warnings.warn(f"Merge variables corrupt: {merge_vars} (NaN): {os.path.join(plate_path, file)}")
                        break

                    # make column names unique
                    uid = file.split('.')[0]
                    if uid in parent_graph.keys():
                        plate_err = True
                        warnings.warn(f"File exists more than one time: {file}")
                        break

                    new_cols = {col: "_".join([uid, col]) for col in open_df.columns
                                if (col not in merge_vars)
                                and (meta_var not in col)}
                    open_df = open_df.rename(columns=new_cols)
                    plate_dfs[uid] = open_df

                    # cache parents
                    parents = [col.split("_")[2] for col in open_df.columns if "parent" in col.lower()]
                    parent_graph[uid] = parents

                if plate_err:
                    missing_plates.append((plate_i, plate_path))
                    break

                # find best path through all nodes of parent_graph
                def graph_paths(graph, start_node, visited):
                    if start_node in graph.keys():
                        visited.append(start_node)
                        for node in graph[start_node]:
                            if node not in visited:
                                graph_paths(graph, node, visited.copy())
                    if len(path_list) == 0 or len(visited) > len(path_list[-1]):
                        path_list.append(visited)

                path_list = []
                for obj in parent_graph.keys():
                    graph_paths(parent_graph, obj, [])
                if len(path_list) == 0:
                    warnings.warn(f"Files can not be merged: {files}, no common parents: {parent_graph}")
                    missing_plates.append((plate_i, plate_path))
                    break

                best_path = path_list[-1]

                # merge objects with best_path
                plate_df = None
                parent = None
                for obj in reversed(best_path):
                    if plate_df is not None:
                        parent_obj = "_".join([obj, "Parent", parent])
                        meta_cols = [col for col in plate_dfs[obj].columns if (meta_var in col) and (col not in merge_vars)]
                        plate_df = plate_df.merge(
                            plate_dfs[obj].drop(meta_cols, axis=1).rename(
                                columns={exp_obj_var: "_".join([obj, exp_obj_var])}),
                            left_on=list(merge_vars),
                            right_on=[exp_image_var, parent_obj, exp_well_var])
                    else:
                        plate_df = plate_dfs[obj]
                    parent = obj

                # add treatment file to plate_df
                if treat_file is not None:
                    def stand_well(s):
                        """Standardize well names.
                        Name should be like E07 or E7.
                        """
                        row = re.split("(\d+)", s)[0]
                        col = str(int(re.split("(\d+)", s)[1]))
                        return row + col

                    # read treatment
                    treat_path = os.path.join(plate_path, treat_file)
                    if not os.path.isfile(treat_path):
                        warnings.warn(f"Treatment file {treat_path} does not exist")
                        missing_plates.append((plate_i, plate_path))
                        break
                    treat = pd.read_csv(treat_path, sep=treat_delimiter)
                    try:
                        treat[treat_well_var] = treat[treat_well_var].apply(stand_well)
                    except:
                        warnings.warn(f"{treat_well_var} not in columns of treatment file {treat_file}: {treat.columns} "
                                      f"Or variables for wells are corrupted.")
                        missing_plates.append((plate_i, plate_path))
                        break

                    # rename columns of treatment dataframe
                    new_treat_cols = {value: ("_".join([meta_var, value]) if value != treat_well_var else exp_well_var)
                                      for value in treat.columns}
                    treat.rename(columns=new_treat_cols, inplace=True)

                    # standardize well variable
                    plate_df[exp_well_var] = plate_df[exp_well_var].apply(stand_well)
                    try:
                        plate_df = plate_df.merge(treat, left_on=[exp_well_var],
                                                  right_on=[exp_well_var])
                    except:
                        warnings.warn(f"Treatment file could not be merged on {treat_well_var}")
                        missing_plates.append((plate_i, plate_path))
                        break

                # insert information about plate number
                plate_df.insert(loc=0, column="PlateNumber", value=(plate_i + 1))
                # insert information about batch number
                plate_df.insert(loc=0, column="BatchNumber", value=(batch_i + 1))
                # concatenate batchdata_list
                if to_disk != 'plate':
                    batchdata_list.append(plate_df)
                else:
                    name = f"morph_data_batch{batch_i + 1}_plate{plate_i + 1}.h5ad"
                    self.save_anndata(md=plate_df, output=output, name=name)

            if to_disk != 'plate':
                batch_df = pd.concat(batchdata_list, ignore_index=True)
                if to_disk != 'batch':
                    morphome_list.append(batch_df)
                else:
                    name = f"morph_data_batch{batch_i + 1}.h5ad"
                    self.save_anndata(md=batch_df, output=output, name=name)

        # concatenate finale morphome data
        if len(morphome_list) > 0:
            morphome = pd.concat(morphome_list, ignore_index=True)
        else:
            morphome = None

        self.__init__(morphome=morphome, treat_delimiter=treat_delimiter, obj_delimiter=obj_delimiter,
                      datadict=datadict, **kwargs)

        return self

    @staticmethod
    def _create_datadict(exp_path, files):
        """Create dictionary with data from a image-based experiment with
        experiment, trial, plate and object data as keys.

        Args:
            exp_path (str): Path to experiment data.
            files (iterable): Object files that store data.

        Returns:
            (dict)
        """
        datadict = {}
        for path, subdirs, filenames in os.walk(pathlib.Path(exp_path)):
            if files is not None:
                csv_files = [filename for filename in filenames
                             if (filename.endswith(".csv") or filename.endswith(".txt"))
                             and filename in files]
            else:
                csv_files = [filename for filename in filenames
                             if (filename.endswith(".csv") or filename.endswith(".txt"))]

            if len(csv_files) != 0:
                batch = path.split(os.sep)[-2]
                if batch not in datadict.keys():
                    datadict[batch] = {}
                datadict[batch][path] = csv_files

        return datadict

    def add_tile_pos(self, morphome):
        # add row and columns of tiles to the morphome depending on the reading method
        # extract rows and columns
        assert len(self.tile_grid) == 2, f"Grid should be a tuple with two integers for rows and columns of tiles."
        tile_rows, tile_cols = self.tile_grid
        # assert self.tile_var is in morphome
        assert self.tile_var in morphome.columns, f"{self.tile_var} is not found in data."

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
        tile_grid = dict(zip(tiles, list(zip(row_ls, col_ls))))

        # add new columns to morphome
        morphome[self.trow_var] = morphome[self.tile_var].map(lambda x: tile_grid[x][0])
        morphome[self.tcol_var] = morphome[self.tile_var].map(lambda x: tile_grid[x][1])

        return morphome

    def to_anndata(self, md=None, obs_ids=None):
        """Takes self.morphome and creates an AnnData object.

        This stores morphological data as a numpy.array separate
        from annotations.

        Args:
            md (pandas.DataFrame): Multidimensional morphological data.
            obs_ids (iterable): Identifiers in columns to store as annotations.
                Takes default Cellprofiler variables if None.
        """
        # append observation identifiers if given
        if obs_ids is not None:
            if isinstance(obs_ids, str):
                obs_ids = [obs_ids]
            self.obs_ids = self.obs_ids + obs_ids

        # take data from class if md is None
        if md is None:
            assert self.morphome is not None, "No morphome initialized in this class."
            obs_cols = [col for col in self.morphome.columns if
                        any(matcher.lower() in col.lower() for matcher in self.obs_ids)]
            # variable annotation
            var = pd.DataFrame(index=self.morphome.drop(obs_cols, axis=1).columns)
            # create AnnData object
            md = ad.AnnData(X=self.morphome.drop(obs_cols, axis=1).to_numpy(), obs=self.morphome[obs_cols], var=var)

        else:
            obs_cols = [col for col in md.columns if
                        any(matcher.lower() in col.lower() for matcher in self.obs_ids)]
            # variable annotation
            var = pd.DataFrame(index=md.drop(obs_cols, axis=1).columns)
            # create AnnData object
            md = ad.AnnData(X=md.drop(obs_cols, axis=1).to_numpy(), obs=md[obs_cols], var=var)

        return md

    def save_anndata(self, md, output, name=None):
        """Saves morphological data as anndata.AnnData object in HDF5 format.

        Args:
            md (anndata.AnnData or pandas.DataFrame): Multidimensional morphological data.
            output (str): Path to output directory.
            name (str): Name to use as filename.
        """
        # convert dataframe to anndata
        if isinstance(md, pd.DataFrame):
            md = self.to_anndata(md=md)
        elif isinstance(md, ad.AnnData):
            pass
        else:
            raise TypeError(f"Data should be either pandas.DataFrame or anndata.AnnData.")

        # get output path
        output = pathlib.Path(output)
        if not os.path.exists(output):
            try:
                os.makedirs(output)
            except OSError:
                print(f"Output directory does not exist: {output}")

        # use name if given
        if name is None:
            name = 'morph_data.h5ad'

        # save
        md.write(os.path.join(output, name))

        return None
