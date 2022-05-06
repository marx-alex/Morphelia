# import internal libraries
import re
import pathlib
from typing import Union, List, Tuple, Optional

# import external libraries
import pandas as pd
import anndata as ad

__all__ = ["LoadPlate"]


class LoadPlate:
    """Class to Load morphological information from a cellprofiler output of a single plate.

    This class contains the logic to combine morphological data from a directory
    with CellProfiler output into an annotated dataframe (AnnData).
    Optionally, annotations from a CSV-file can be merged as additional annotations.

    The file structure might be as following:

    |----plate
    |      |----Cells.csv
    |      |----Primarieswithoutborder.csv
    |      |----Cytoplasm.csv
    |      |----Treatment.csv

    Parameters
    ----------
    path : str, pathlib.Path
        Path to plate directory
    filenames : str or list of str or tuple of str
        Files in plate directory to load
    obj_sfx : str
        File extension of object files (filenames)
    merge_method : str
        Chose one of the two merge methods:
        `object`: Merge by object number
        `relation`: Merge by object relations
    obj_well_var : str
        Name of well variable in object files
    meta_var : str
        Prefix to assign to metadata variables
    image_var : str
        Name of image variable in object files
    obj_var : str
        Name of object variable in object files
    obj_delimiter : str
        Delimiter of variables in object files
    treat_file : str, optional
        Name of treatment file
    treat_sfx : str
        File extension of treatment file
    treat_delimiter : str
        Delimiter of variables in treatment file
    observation_ids : list of str or str
        List of CellProfiler observations variables to store as annotations.
        If `observation_ids` is `cp`, a list of default variables is taken.
    """

    def __init__(
        self,
        path: str,
        filenames: Union[str, Tuple[str], List[str]] = (
            "Cells",
            "Primarieswithoutborder",
            "Cytoplasm",
        ),
        obj_sfx: str = ".csv",
        merge_method: str = "object",
        obj_well_var: str = "Metadata_Well",
        meta_var: str = "Metadata",
        image_var: str = "ImageNumber",
        obj_var: str = "ObjectNumber",
        obj_delimiter: str = ",",
        treat_file: Optional[str] = None,
        treat_well_var: str = "well",
        treat_sfx: str = ".csv",
        treat_delimiter: str = ",",
        observation_ids: Union[List[str], str] = "cp",
    ) -> None:
        self.path = pathlib.Path(path)

        if isinstance(filenames, str):
            filenames = [filenames]
        self.filenames = filenames
        self.obj_sfx = obj_sfx

        avail_merge_methods = ["relation", "object"]
        merge_method = merge_method.lower()
        assert merge_method in avail_merge_methods, (
            f"merge_method must be one of {avail_merge_methods}, "
            f"instead got {merge_method}"
        )
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
        if observation_ids == "cp":
            self.observation_ids = [
                "ImageNumber",
                "ObjectNumber",
                "Metadata",
                "Location",
                "Parent",
            ]
        elif isinstance(observation_ids, list):
            self.observation_ids = observation_ids

    def load(self):
        """Merge and load objects from a single plate.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        AssertionError
            If `merge_vars` are not in a dataframe
        """
        dfs = {}
        parent_graph = {}  # if method is relation

        for file in self.filenames:
            _df = pd.read_csv(
                self.path.joinpath(file).with_suffix(self.obj_sfx),
                sep=self.obj_delimiter,
            )
            assert all(
                col in list(_df.columns) for col in self.merge_vars
            ), f"Variables needed for merging must be in all files: {self.merge_vars}"

            new_cols = {
                col: "_".join([file, col])
                for col in _df.columns
                if (col not in self.merge_vars) and (self.meta_var not in col)
            }
            _df = _df.rename(columns=new_cols)
            dfs[file] = _df

            # cache parents
            parents = [
                col.split("_")[2] for col in _df.columns if "parent" in col.lower()
            ]
            parent_graph[file] = parents

        if self.merge_method == "relation":
            plate_df = self._relational_merge(dfs, parent_graph)

        elif self.merge_method == "object":
            plate_df = None
            for obj in self.filenames:
                if plate_df is not None:
                    meta_cols = [
                        col
                        for col in dfs[obj].columns
                        if (self.meta_var in col) and (col not in self.merge_vars)
                    ]
                    plate_df = plate_df.merge(
                        dfs[obj].drop(meta_cols, axis=1),
                        left_on=list(self.merge_vars),
                        right_on=list(self.merge_vars),
                    )
                else:
                    plate_df = dfs[obj]
        else:
            plate_df = None

        # add treatment file
        if self.treat_file is not None:
            plate_df = self.add_annotation(plate_df)

        self.plate = plate_df
        return plate_df

    def add_annotation(self, plate_df: pd.DataFrame):
        """Add annotations defined by `treat_file` to `plate`.

        Parameter
        ---------
        plate_df : pandas.DataFrame
            Open dataframe

        Returns
        -------
        pandas.DataFrame
            Dataframe with annotations
        """
        treat = pd.read_csv(
            self.path.joinpath(self.treat_file).with_suffix(self.treat_sfx),
            sep=self.treat_delimiter,
        )

        treat[self.treat_well_var] = treat[self.treat_well_var].apply(_standard_well)

        # rename columns of treatment dataframe
        new_treat_cols = {
            value: (
                "_".join([self.meta_var, value])
                if value != self.treat_well_var
                else self.obj_well_var
            )
            for value in treat.columns
        }
        treat.rename(columns=new_treat_cols, inplace=True)

        # standardize well variable
        plate_df[self.obj_well_var] = plate_df[self.obj_well_var].apply(_standard_well)

        plate_df = plate_df.merge(
            treat, left_on=[self.obj_well_var], right_on=[self.obj_well_var]
        )

        return plate_df

    def _relational_merge(self, df_dict, parent_graph):

        # find best path through all nodes of parent_graph
        def _graph_paths(graph, start_node, visited):
            if start_node in graph.keys():
                visited.append(start_node)
                for node in graph[start_node]:
                    if node not in visited:
                        _graph_paths(graph, node, visited.copy())
            if len(path_list) == 0 or len(visited) > len(path_list[-1]):
                path_list.append(visited)

        path_list = []
        for obj in parent_graph.keys():
            _graph_paths(parent_graph, obj, [])
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
                meta_cols = [
                    col
                    for col in df_dict[obj].columns
                    if (self.meta_var in col) and (col not in self.merge_vars)
                ]
                plate_df = plate_df.merge(
                    df_dict[obj]
                    .drop(meta_cols, axis=1)
                    .rename(columns={self.obj_var: "_".join([obj, self.obj_var])}),
                    left_on=list(self.merge_vars),
                    right_on=[self.image_var, parent_obj, self.obj_well_var],
                )
            else:
                plate_df = df_dict[obj]
            parent = obj

        return plate_df

    def to_anndata(
        self, observation_ids: Optional[Union[List[str], Tuple[str]]] = None
    ):
        """Create AnnData object from dataframe.

        Parameters
        ----------
        observation_ids : list of str or tuple of str, optional
            Identifiers in columns to store as annotations.
            Default CellProfiler variables are used if None.

        Returns
        -------
        anndata.AnnData
            Merged and annotated AnnData object

        Raises
        ------
        AssertionError
            If no plate data is loaded into the class
        AssertionError
            If no observation are found in plate data
        """
        # append observation identifiers if given
        if observation_ids is not None:
            if isinstance(observation_ids, str):
                observation_ids = [observation_ids]
            self.observation_ids = self.observation_ids + observation_ids

        assert isinstance(self.plate, pd.DataFrame), "No data loaded as DataFrame"
        assert len(self.observation_ids) > 0, "No observations found"
        obs_cols = [
            col
            for col in self.plate.columns
            if any(matcher.lower() in col.lower() for matcher in self.observation_ids)
        ]
        # variable annotation
        var = pd.DataFrame(index=self.plate.drop(obs_cols, axis=1).columns)
        # create AnnData object
        adata = ad.AnnData(
            X=self.plate.drop(obs_cols, axis=1).to_numpy(),
            obs=self.plate[obs_cols],
            var=var,
        )

        self.plate = adata
        return adata


def _standard_well(s):
    """Standardize well names.
    Name should be like E07 or E7.
    """
    row = re.split(r"(\d+)", s)[0]
    col = str(int(re.split(r"(\d+)", s)[1]))
    return row + col
