import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def correct_bleaching(adata,
                      channels,
                      treat_var='Metadata_Treatment',
                      time_var='Metadata_Time',
                      exp_curve='mono',
                      ctrl='ctrl',
                      correct_X=True,
                      ignore_weak_fits=None,
                      verbose=False):
    """
    Correction of Photobleaching as described by Vicente et al 2007 J. Phys.: Conf. Ser. 90 012068.
    Every intensity dependend feature is fit to a mono- or bi-exponential curve using
    non-linear least squares. The bleaching curve is then normalized.
    Each measured value is then divided by its corresponding value from the bleaching curve.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        channels (list): List of channel names that are in variable names.
        treat_var (str): Treatment variable.
        time_var (str): Time variable.
        exp_curve (str): Exponential curve to use for curve fitting. One of:
            'bi': bi-exponential curve
            'mono': mono-exponential curve
        ctrl (str): Name for control condition in treat_var
        correct_X (bool): Return anndata object with corrected .X
        ignore_weak_fits (float): Don't correct features with fits that have a R-squared value
            below a given value.
        verbose (bool)
    """
    min_vals = None
    if (adata.X < 0).any():
        warnings.warn('Negative values encountered in .X. Attempting to correct .X anyway.')
        min_vals = np.min(adata.X, axis=0)

        adata.X = adata.X + min_vals[None, :]

    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"
    assert time_var in adata.obs.columns, f"time_var not in .obs: {time_var}"

    # choose exponential curve
    avail_exp_curves = ['mono', 'bi']
    exp_curve = exp_curve.lower()
    assert exp_curve in avail_exp_curves, f"exp_curve must be one of {avail_exp_curves}, " \
                                          f"instead got {exp_curve}"
    if exp_curve == 'mono':
        func = mono_exp
    elif exp_curve == 'bi':
        func = bi_exp

    if isinstance(channels, str):
        channels = [channels]
    else:
        try:
            channels = list(channels)
        except:
            raise ValueError(
            f"channels must be of type str, list or tuple, instead got {type(channels)}")

    assert (
        len([var for var in adata.var_names if any((ch in var) for ch in channels)]) > 0
    ), f"no variables found with given channels: {channels}"

    # subset to control condition
    ctrl_adata = adata[adata.obs[treat_var] == ctrl, :].copy()
    assert (
        len(ctrl_adata) > 0
    ), f"no cells with control condition {ctrl} in treatment variable {treat_var}"

    # get unique timepoints
    time_points = list(np.sort(np.unique(adata.obs[time_var].to_numpy())))

    # store aggregated control data
    ctrl_df = np.zeros((len(time_points), len(adata.var_names)))
    for tp in time_points:
        tp_agg = np.nanmean(ctrl_adata[ctrl_adata.obs[time_var] == tp, :].X, axis=0)
        ctrl_df[tp, :] = tp_agg

    ctrl_df = pd.DataFrame(ctrl_df, index=time_points, columns=adata.var_names)

    # store theoretical values
    F_ = np.zeros(adata.X.shape)
    F_ctrl = np.zeros(ctrl_df.shape)

    # iterate through intensity variables and get theoretical values
    for ix, var in enumerate(adata.var_names):
        # only fit curves for intensity based variables
        if any(ch in var for ch in channels):
            try:
                popt, _ = curve_fit(func, time_points, ctrl_df[var])
                # get theoretical values
                f_ = np.vectorize(func)(adata.obs[time_var], *popt)
                F_[:, ix] = f_
                # get theoretical values for aggregated controls
                f_ctrl = np.vectorize(func)(time_points, *popt)
                F_ctrl[:, ix] = f_ctrl
            except:
                F_[:, ix] = 1
                F_ctrl[:, ix] = ctrl_df.loc[:, var]
        else:
            F_[:, ix] = 1
            F_ctrl[:, ix] = ctrl_df.loc[:, var]

    # calculate r squared
    residuals = ctrl_df.to_numpy() - F_ctrl
    ss_res = np.sum((residuals ** 2), axis=0)
    ss_tot = np.sum((ctrl_df.to_numpy() - np.mean(ctrl_df.to_numpy(), axis=0)) ** 2, axis=0)
    r_squared = 1 - (ss_res / ss_tot)

    if verbose:
        output = pd.DataFrame({'variable': adata.var_names, 'r_squared': r_squared})
        print(output.to_string())

    target = 0.8
    if ignore_weak_fits is not None:
        target = ignore_weak_fits
    if np.sum(r_squared < target) > 0:
        weak_mask = np.argwhere(r_squared < target)
        warnings.warn(
            f"R-Squared is partially below {target}, you may want to change to bi-exponential curve. Variables with low R-Squared: {adata.var_names[weak_mask.flatten()]}"
        )

    if ignore_weak_fits is not None:
        weak_mask = np.argwhere(r_squared < ignore_weak_fits).flatten()
        F_[:, weak_mask] = 1

    # normalize theoretical data
    F_ = F_ / np.max(F_, axis=0)

    # calculate corrected values for F_
    F = adata.X.copy() / F_

    if min_vals is not None:
        F = F - min_vals[None, :]
        adata.X = adata.X - min_vals[None, :]

    if correct_X:
        adata.var['R-squared'] = r_squared
        adata.X = F
        return adata

    return F, ctrl_df


def correct_bleached_var(adata,
                         var_name,
                         treat_var='Metadata_Treatment',
                         time_var='Metadata_Time',
                         exp_curve='mono',
                         ctrl='ctrl',
                         correct_adata=False,
                         ignore_weak_fit=None,
                         return_r_squared=False,
                         verbose=False):
    """
    Correction of Photobleaching as described by Vicente et al 2007 J. Phys.: Conf. Ser. 90 012068.
    A given variable is fit to a mono- or bi-exponential curve using
    non-linear least squares. The bleaching curve is then normalized.
    Each measured value is then divided by its corresponding value from the bleaching curve.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        var_name (list): Name of variable to correct.
        treat_var (str): Treatment variable.
        time_var (str): Time variable.
        exp_curve (str): Exponential curve to use for curve fitting. One of:
            'bi': bi-exponential curve
            'mono': mono-exponential curve
        ctrl (str): Name for control condition in treat_var
        correct_adata (bool): Return anndata object with corrected variable.
        ignore_weak_fit (float): Don't correct features with fits that have a R-squared value
            below a given value.
        return_r_squared: Return corrected object and the R-squared value.
        verbose (bool)
    """
    if var_name in adata.obs.columns:
        x = adata.obs[var_name].to_numpy().flatten()
    elif var_name in adata.var_names:
        x = adata[:, var_name].X.flatten().copy()
    else:
        raise ValueError(f'var_name not in .obs or .var_names: {var_name}')

    min_val = None
    if (x < 0).any():
        warnings.warn('Negative values encountered in x, attempting to correct .x anyway.')
        min_val = np.min(x)

        x = x + min_val

    assert treat_var in adata.obs.columns, f"treat_var not in .obs: {treat_var}"
    assert time_var in adata.obs.columns, f"time_var not in .obs: {time_var}"

    # choose exponential curve
    avail_exp_curves = ['mono', 'bi']
    if exp_curve == 'mono':
        func = mono_exp
    elif exp_curve == 'bi':
        func = bi_exp
    else:
        raise ValueError(f"exp_curve must be one of {avail_exp_curves}, instead got {exp_curve}")

    # subset to control condition
    ctrl_mask = adata.obs[treat_var] == ctrl
    assert (
            np.sum(ctrl_mask) > 0
    ), f"no cells with control condition {ctrl} in treatment variable {treat_var}"

    # get unique timepoints
    x_time = adata.obs[time_var].to_numpy().flatten()
    time_points = np.sort(np.unique(x_time))

    # store aggregated control data
    y_ctrl = []
    for tp in time_points:
        mask = np.logical_and(ctrl_mask, x_time == tp)
        tp_agg = np.nanmean(x[mask])
        y_ctrl.append(tp_agg)
    y_ctrl = np.array(y_ctrl)

    # only fit curves for intensity based variables
    try:
        popt, _ = curve_fit(func, time_points, y_ctrl)
        # get theoretical values
        f_ = np.vectorize(func)(x_time, *popt)
        # get theoretical values for aggregated controls
        f_ctrl = np.vectorize(func)(time_points, *popt)
    except:
        f_ = np.ones(x_time.shape)
        f_ctrl = np.ones(time_points.shape)

    # calculate r squared
    residuals = y_ctrl - f_ctrl
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_ctrl - np.mean(y_ctrl)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    if verbose:
        print(f'R-Squared: {r_squared}')

    if ignore_weak_fit is not None:
        f_ = np.ones(f_.shape)

    # normalize theoretical data
    f_ = f_ / np.max(f_)

    # calculate corrected values for F_
    f = x / f_

    if min_val is not None:
        f = f - min_val

    if correct_adata:
        if var_name in adata.obs.columns:
            adata.obs[var_name] = f
        elif var_name in adata.var_names:
            adata[:, var_name].X = f

        if return_r_squared:
            return adata, r_squared

    if return_r_squared:
        return f, r_squared
    return f


def mono_exp(x, a, b):
    return b * np.exp(-a * x)


def bi_exp(x, a1, b1, a2, b2):
    return (b1 * np.exp(-a1 * x)) + (b2 * np.exp(-a2 * x))