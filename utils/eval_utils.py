import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import ipdb


def smoothing(x, y, smooth=None):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    x_sm = np.array(x)
    y_sm = np.array(y)
    sp1 = UnivariateSpline(x_sm, y_sm)
    ynew = sp1(x_sm)
    return x_sm, ynew

def _plot_df_to_ax(ax, df, xcol, fillinf=np.inf, smooth=None, s=8, show_cov=True,
                   yhat_label='yhat',
                   PI_label='PI'):
    df = df.sort_values([xcol], ascending=True)
    ybar = df['y'].mean()

    # scatter plot missed
    miss_idx = df.index[((df['y'] < df['lo']) | (df['y'] > df['hi']))]
    miss_y = df.loc[miss_idx, 'y']
    miss_x = df.loc[miss_idx, xcol]
    ax.scatter(miss_x, miss_y, label='missed', alpha=0.5, s=s, c='red', marker='x')

    #scatter plot covered
    cover_idx = df.index[((df['y'] >= df['lo']) & (df['y'] <= df['hi']))]
    covered_y = df.loc[cover_idx, 'y']
    covered_x = df.loc[cover_idx, xcol]
    if show_cov:
        label = f'covered ({len(cover_idx) / len(df) * 100:.1f}%)'
    else:
        label = f'covered'
    ax.scatter(covered_x, covered_y, label=label, alpha=0.5, s=s, c='gray', marker='.')


    lo = df['lo'].groupby(df[xcol]).mean().clip(ybar - fillinf, ybar + fillinf)
    hi = df['hi'].groupby(df[xcol]).mean().clip(ybar - fillinf, ybar + fillinf)
    if isinstance(smooth, bool) and not smooth:
        x,lo_new,hi_new = lo.index, lo, hi
        smooth = None
    else:
        x, lo_new = smoothing(lo.index, lo.values, smooth=smooth)
        _, hi_new = smoothing(x, hi, smooth=smooth)
    ax.fill_between(x, lo_new, hi_new, color='b', alpha=.1, label=PI_label)

    yhat = df['yhat'].groupby(df[xcol]).mean()
    try:
        x, yhat_smooth = smoothing(yhat.index, yhat.values, smooth=smooth)
        ax.plot(x, yhat, color='black', alpha=1, label=yhat_label)
    except Exception as err:
        print("plot_df: %s"%err)
        ax.plot(yhat.index, yhat.values, color='black', alpha=1, label=yhat_label)


def plot_df(df, xcol, title='', fillinf=np.inf, smooth=None, cutoff=0.001, s=8, save_path=None):
    #xcol='yhat'
    fig, ax = plt.subplots(ncols=1, nrows=1)
    _plot_df_to_ax(ax, df, xcol, fillinf=fillinf, smooth=smooth, s=s)
    ax.set_xlabel(xcol)
    ax.set_ylabel('y')
    ax.set_title(title)
    ub, lb = df['y'].quantile(1 - cutoff), df['y'].quantile(cutoff)
    ax.set_ylim((lb - 0.1 * (ub - lb), ub + 0.1 * (ub - lb)))
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=400)


def clean_df(df, title="", fillinf=None, alpha=0.05, plot=True, smooth=None):
    assert all([c in df.columns for c in ['lo', 'hi', 'y', 'yhat']])
    df = df.copy()
    df['width'] = df['hi'] - df['lo']
    df['cover?'] = ((df['y'] > df['lo']) & (df['y'] < df['hi'])).map(int)
    df['resid'] = (df['y']-df['yhat']).abs()
    print(f"The {100 * (1 - alpha):.0f}% confidence interval covers {df['cover?'].mean() * 100:.2f}% of the time")
    print(df['width'][df['width'] < np.inf].describe())

    if plot:
        xcol = 'x' if 'x' in df.columns else 'yhat'
        plot_df(df.copy(), xcol, fillinf=fillinf, title=title, smooth=smooth)
    return df.sort_values('yhat') #added width, cover? and resid columns

def summ_df(df, alpha=0.05, resid_bound=None, fill_inf_for_corr=False, non_inf_idx=None, measures=None):
    from sklearn.metrics import roc_auc_score, r2_score
    df = df.copy()
    df['width'] = df['hi'] - df['lo']
    df['cover?'] = ((df['y'] > df['lo']) & (df['y'] < df['hi'])).map(int)
    df['resid'] = (df['y'] - df['yhat']).abs()

    if non_inf_idx is None:
        non_inf_idx = df.index[df['width'] < np.inf]
    if resid_bound is not None: df = df[df['resid'] < resid_bound]
    thres = df['resid'].quantile(1 - alpha)
    y_true = df['resid'] > thres
    ypred = df['width']
    if fill_inf_for_corr:
        ypred[ypred == np.infty] = np.max(ypred[ypred < np.infty]) * 2

    res_ser = pd.Series({})
    if measures is None or 'AUROC' in measures:
        res_ser['AUROC'] = roc_auc_score(y_true, ypred.rank())
    if measures is None or 'resid' in measures:
        res_ser['resid'] = df['resid'].mean()
    if measures is None or 'cover' in measures:
        res_ser['cover'] =df['cover?'].mean() * 100
    if measures is None or 'Corr(noninf)' in measures:
        res_ser['Corr(noninf)'] = df['resid'].loc[non_inf_idx].corr(ypred.loc[non_inf_idx], method='pearson')
    if measures is None or 'cover(noninf)' in measures:
        res_ser['cover(noninf)'] = df.loc[non_inf_idx]['cover?'].mean() * 100
    if measures is None or 'cover(tail)' in measures:
        sorted_y = df['y'].sort_values(inplace=False)
        extreme_idx = np.concatenate([sorted_y.index[:int(0.1 * len(df))], sorted_y.index[int(0.9 * len(df)):]])
        res_ser['cover(tail)'] = df.loc[extreme_idx]['cover?'].mean() * 100
    if measures is None or 'MSE' in measures:
        res_ser['MSE'] = df['resid'].map(np.square).mean()
    if measures is None or 'R2' in measures:
        res_ser['R^2'] = r2_score(df['y'], df['yhat'])
    if measures is None or 'mean_width(noninf)' in measures:
        res_ser['mean_width(noninf)'] = df.loc[non_inf_idx]['width'].mean()
    if measures is None or 'cnt_width(noninf)' in measures:
        res_ser['cnt_width(noninf)'] = len(non_inf_idx)
    return res_ser #y_true, ypred, auroc

#=======================================Visualize in tables

def merge_mean_std_tables(mean_df, std_df, prec1=4, prec2=4):
    format_ = "{:.%df}({:.%df})"%(prec1, prec2)
    if isinstance(mean_df, pd.DataFrame):
        ndf=  pd.DataFrame(index=mean_df.index, columns=mean_df.columns)
        for c in mean_df.columns:
            ndf[c] = merge_mean_std_tables(mean_df[c], std_df[c], prec1, prec2)
        return ndf
    nser = pd.Series("", index=mean_df.index)
    for i in nser.index:
        nser[i] = format_.format(mean_df[i], std_df[i])
    return nser


def ttest_from_stats(m1, s1, n1, m2, s2, n2, alternative='two-sided', **kwargs):
    import scipy.stats
    kwargs.setdefault('equal_var', False)
    assert alternative == 'two-sided', "ttest_from_stats: this scipy version only supports two-sided"
    return scipy.stats.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2, **kwargs)

def pivot_with_mean_std_tstat(flat_df, columns, index, values, axis=0, eps=1e-6, **kwargs):
    #compare across axis
    mdf = flat_df.pivot_table(columns=columns, index=index, values=values, aggfunc='mean')
    sdf = flat_df.pivot_table(columns=columns, index=index, values=values, aggfunc='std').reindex(mdf.index).fillna(0.)
    #ipdb.set_trace()
    #if len(sdf) < len(mdf): sdf = pd.DataFrame(0, index=mdf.index, columns=mdf.columns) #no standard deviation case
    df = merge_mean_std_tables(mdf, sdf, **kwargs)
    cdf = flat_df.pivot_table(columns=columns, index=index, values=values, aggfunc='size')
    if axis == 1: df, cdf, mdf, sdf, = df.T, cdf.T, mdf.T, sdf.T
    for c in df.columns:
        if len(mdf) > 1:
            for ascend in [True, False]:
                ser = mdf[c].sort_values(inplace=False, ascending=ascend)
                extreme = [0]
                j = 1
                while j < len(ser) and abs(ser.iloc[j] - ser.iloc[0]) < eps:
                    extreme += [j]
                    j += 1
                j = min(len(ser) - 1, j)
                i1, i2 = ser.index[0], ser.index[j]
                t_stat, pval = ttest_from_stats(mdf.loc[i1, c], sdf.loc[i1, c], cdf.loc[i1, c],
                                                mdf.loc[i2, c], sdf.loc[i2, c], cdf.loc[i2, c])
                for ii in extreme:
                    df.loc[ser.index[ii], c] += "[{:.2e}{}{}]".format(pval, "<" if ascend else ">", i2)
    if axis == 1: df, cdf, mdf, sdf, = df.T, cdf.T, mdf.T, sdf.T
    return df


def summary_methods(dfs, alpha=0.1, resid_bound=None, fill_inf_for_corr=False, filter_idx=False, drop_by_exp=False, measures=None):
    #dfs is a dict indexed by methods
    cidx = None
    index_ = None
    for m, df in dfs.items():
        if index_ is None: index_ = df['index']
        try:
            assert df['index'].equals(index_)
        except:
            assert df['index'].astype(int).equals(index_.astype(int))
        w = df['hi'] - df['lo']
        cidx_cur = w[w<np.inf].index
        cidx = cidx_cur if cidx is None else cidx_cur.intersection(cidx)
    if not filter_idx: cidx = None
    summ = {}
    for m, df in dfs.items():
        try:
            summ[m] = summ_df(df, alpha, resid_bound, fill_inf_for_corr, non_inf_idx=cidx, measures=measures)
        except Exception as err:
            if drop_by_exp: return None
    summ = pd.DataFrame(summ).stack(dropna=False).reset_index().rename(columns={"level_0": 'measure', 'level_1': 'method', 0: 'val'})
    return summ

def summary_by_dataset(dfs,alpha=0.1, resid_bound=None, fill_inf_for_corr=False, filter_idx=False,
                       dropnan='method', quiet=False, measures=None):
    """
    :param dfs:
    :param alpha:
    :param resid_bound:
    :param fill_inf_for_corr:
    :param filter_idx: filter_idx restricts all methods to compute ``noninf'' measures on the subset of data for which all CIs are finite.
                If you want to check the width, etc. without such filtering, set filter_idx to False.
    :param dropnan:
    :param quiet:
    :param measures:
    :return:
    """
    import tqdm
    #dfs is a dict indexed by method, inside is what's passed to summary_exps
    bdf = []
    dfs_exp_i = {}
    for method, dfs_ in dfs.items():
        for exp, df in dfs_.items():
            if exp not in dfs_exp_i: dfs_exp_i[exp] = {}
            dfs_exp_i[exp][method] = df
    for exp, dfs_ in (dfs_exp_i.items() if quiet else tqdm.tqdm(dfs_exp_i.items() )):
        summ = summary_methods(dfs_, alpha, resid_bound, fill_inf_for_corr, filter_idx=filter_idx, drop_by_exp=dropnan=='exp', measures=measures)
        if summ is None:
            continue
        summ['exp'] = exp
        bdf.append(summ)
    bdf = pd.concat(bdf, ignore_index=True)
    if dropnan == 'method':
        method_cnts = bdf.groupby('method')['exp'].nunique()
        good_methods = method_cnts[method_cnts == method_cnts.max()].index
        bdf = bdf[bdf['method'].isin(good_methods)]
    return bdf, pivot_with_mean_std_tstat(bdf, columns=['method'], index=['measure'], values='val', axis=1)


def _get_all_dfs(datasets, methods, seeds,  get_df_func, alphas=[0.1, 0.5]):
    from collections import defaultdict
    import itertools
    import tqdm
    all_res = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    n = len(datasets) * len(methods) * len(seeds) * len(alphas)
    for alpha, dataset, method, seed in tqdm.tqdm(itertools.product(alphas, datasets, methods, seeds), total=n, desc='reading all results'):
        tdf = get_df_func(dataset, method, seed, alpha)
        if tdf is not None:
            all_res[alpha][dataset][method][seed] = tdf
    return all_res

def _get_all_summsers(all_res):
    from collections import defaultdict
    import itertools
    import tqdm
    all_sers = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    bdf = []
    for alpha,df1 in all_res.items():
        for dataset, df2 in df1.items():
            for method, df3 in df2.items():
                for seed, df4 in df3.items():
                    all_sers[(alpha, dataset, method, seed)] = summ_df(df4, alpha, resid_bound=None, fill_inf_for_corr=False, non_inf_idx=None)
    return all_sers


def summary_key_metrics_all_datasets_and_methods(datasets, methods, seeds, get_df_func, all_res=None,
                                                 width_methods=['Method', 'MADSplit', 'CQR']):
    """
    This is used to gather all experiment results. See notebooks/Exp_Summary.ipynb
    :param datasets:
    :param methods:
    :param seeds:
    :param get_df_func(dataset, method, seed, alpha) -> df with columns {lo,hi,yhat,y,index} etc.
    :return:
    """
    from collections import defaultdict
    if all_res is None: all_res = _get_all_dfs(datasets, methods, seeds, get_df_func)

    #return all_res
    width_methods = [w for w in width_methods if w in methods]
    summ_res = defaultdict(defaultdict)
    for dataset in datasets:
        print("Aggregating results on %s"%dataset)
        summ_alpha5 = summary_by_dataset(all_res[0.5][dataset], alpha=0.5, measures=['AUROC'])[1].reindex(columns=methods)
        summ_alpha1 = summary_by_dataset(all_res[0.1][dataset], alpha=0.1, measures=['cover', 'cover(tail)', 'resid'])[1].reindex(columns=methods)
        summ_alpha5_noninf = summary_by_dataset(all_res[0.5][dataset], alpha=0.5, filter_idx=True, measures=['cnt_width(noninf)', 'mean_width(noninf)'])[1].reindex(columns=methods)
        summ_alpha1_noninf = summary_by_dataset(all_res[0.1][dataset], alpha=0.1, filter_idx=True, measures=['cnt_width(noninf)', 'mean_width(noninf)'])[1].reindex(columns=methods)

        #for AUROC
        summ_res['AUROC'][dataset] = summ_alpha5.loc['AUROC']

        # for MAD, MCR, TCR
        summ_res['MCR'][dataset] = summ_alpha1.loc['cover']
        summ_res['TCR'][dataset] = summ_alpha1.loc['cover(tail)']
        summ_res['MAD'][dataset] = summ_alpha1.loc['resid']

        # for efficiency
        summ_res['Count 50'][dataset] = summ_alpha5_noninf.loc['cnt_width(noninf)']
        summ_res['Width 50'][dataset] = summ_alpha5_noninf.loc['mean_width(noninf)']
        summ_res['Count 90'][dataset] = summ_alpha1_noninf.loc['cnt_width(noninf)']
        summ_res['Width 90'][dataset] = summ_alpha1_noninf.loc['mean_width(noninf)']
        if len(width_methods) > 0:
            summ_alpha5_noninf_good = summary_by_dataset({m:all_res[0.5][dataset][m] for m in width_methods}, alpha=0.5, filter_idx=True, measures=['cnt_width(noninf)', 'mean_width(noninf)'])[1].reindex(columns=width_methods)
            summ_alpha1_noninf_good = summary_by_dataset({m:all_res[0.1][dataset][m] for m in width_methods}, alpha=0.1, filter_idx=True, measures=['cnt_width(noninf)', 'mean_width(noninf)'])[1].reindex(columns=width_methods)
            summ_res['Count 50 Good'][dataset] = summ_alpha5_noninf_good.loc['cnt_width(noninf)']
            summ_res['Width 50 Good'][dataset] = summ_alpha5_noninf_good.loc['mean_width(noninf)']
            summ_res['Count 90 Good'][dataset] = summ_alpha1_noninf_good.loc['cnt_width(noninf)']
            summ_res['Width 90 Good'][dataset] = summ_alpha1_noninf_good.loc['mean_width(noninf)']

        # infinite
        noninf_methods = [m for m in methods if m != 'Method']
        summ_alpha1_inf = summary_by_dataset({m: all_res[0.1][dataset][m] for m in noninf_methods}, alpha=0.1, filter_idx=False)[1].reindex( columns=noninf_methods)
        summ_alpha5_inf = summary_by_dataset({m: all_res[0.5][dataset][m] for m in noninf_methods}, alpha=0.5, filter_idx=False)[1].reindex(columns=noninf_methods)
        summ_res['Width 50 Inf'][dataset] = summ_alpha5_inf.loc['mean_width(noninf)']
        summ_res['Count 50 Inf'][dataset] = summ_alpha5_inf.loc['cnt_width(noninf)']
        summ_res['Width 90 Inf'][dataset] = summ_alpha1_inf.loc['mean_width(noninf)']
        summ_res['Count 90 Inf'][dataset] = summ_alpha1_inf.loc['cnt_width(noninf)']
    summ_res = {k: pd.DataFrame(v).T.reindex(datasets) for k,v in summ_res.items()}
    return summ_res, all_res


def summ_table_to_latex(summ_df,
                        scale = 1,
                        number_format = None,
                        marker_func_from_str=None, marker_func_from_mean_std=None,
                        underscore_from_str=None, underscore_from_mean_std=None,
                        pad=0,
                        table_name='',
                        midrule="\\midrule"):
    """
    This is used to generate latex tables. See notebooks/Exp_Summary.ipynb
    :param summ_df:
    :param scale:
    :param number_format:
    :param marker_func_from_str:
    :param marker_func_from_mean_std:
    :param pad:
    :param table_name:
    :param midrule:
    :return:
    """

    ncols = summ_df.shape[1]
    nrows = summ_df.shape[0]
    if marker_func_from_str is None: marker_func_from_str = lambda x: False
    if marker_func_from_mean_std is None: marker_func_from_mean_std = lambda m,std: False
    if underscore_from_str is None: underscore_from_str = lambda x: False
    if underscore_from_mean_std is None: underscore_from_mean_std = lambda m,std: False
    if number_format is None: number_format = ":.2f"
    new_df = [["" for _ in range(ncols)] for _ in range(nrows)]

    for i in range(nrows):
        for j in range(ncols):
            curr_nf = number_format[j] if isinstance(number_format, list) else number_format
            mark_bf = False
            mark_ud = False
            v = summ_df.iloc[i, j]
            if pd.isnull(v):
                new_df[i][j] = "\\textendash"
                continue
            mark_bf = mark_bf or marker_func_from_str(v)
            mark_ud = mark_ud or underscore_from_str(v)
            v = v.split("[")[0]
            pos = v.index("(")
            mean_ = float(v[:pos].strip()) * scale
            std_ = float(v[pos+1:v.index(")")].strip()) * scale
            mark_bf = mark_bf or marker_func_from_mean_std(mean_, std_)
            mark_ud = mark_ud or underscore_from_mean_std(mean_, std_)

            mean_str = ("{%s}" % curr_nf).format(mean_)
            if mark_bf: mean_str = "\\textbf{%s}"%mean_str
            if mark_ud: mean_str = "\\underline{%s}" % mean_str
            std_str = ("{%s}" % curr_nf).format(std_)
            new_df[i][j] = f"{mean_str}$\pm${std_str}"
            new_df[i][j] = ("{:<%d}"%pad).format(new_df[i][j])
    latex_str = ("{:<%d} & "%pad).format(table_name) + " & ".join([("{:<%d}"%pad).format(_s) for _s in summ_df.columns]) + "\\\\\n"
    if midrule is not None:
        latex_str += midrule + "\n"
    for i, idx in enumerate(summ_df.index):
        latex_str += ("{:<%d} & "%pad).format(idx) + " & ".join(new_df[i]) + "\\\\\n"
    return latex_str