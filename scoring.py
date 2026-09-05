import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Optional,Union
from tqdm import tqdm
import plotting
import partitioning
import data_loading


class ScoreCache:
    """Reuse raw sequence scores while each mode keeps its own metric logic."""

    def __init__(self, loader: data_loading.Simple_Loader) -> None:
        self.loader = loader
        self._real_scores = {}
        self._generated_scores = {}
        self._subsequences = {}
        self._horizon_generated_scores = {}

    def score_real(self, scoring_fn):
        if scoring_fn not in self._real_scores:
            self._real_scores[scoring_fn] = partitioning.score_real(
                self.loader, scoring_fn
            )
        return self._real_scores[scoring_fn]

    def score_generated(self, scoring_fn):
        if scoring_fn not in self._generated_scores:
            self._generated_scores[scoring_fn] = partitioning.score_gen(
                self.loader, scoring_fn
            )
        return self._generated_scores[scoring_fn]

    def score_real_gen(self, scoring_fn):
        return self.score_real(scoring_fn), self.score_generated(scoring_fn)

    def subsequences(self, horizon_length: int):
        if horizon_length not in self._subsequences:
            self._subsequences[horizon_length] = tuple(
                partitioning.get_subseqs(sequence, subseq_len=horizon_length)
                for sequence in self.loader
            )
        return self._subsequences[horizon_length]

    def generated_horizon_scores(self, scoring_fn, horizon_length: int):
        key = (scoring_fn, horizon_length)
        if key not in self._horizon_generated_scores:
            self._horizon_generated_scores[key] = partitioning.score_gen(
                self.subsequences(horizon_length), scoring_fn
            )
        return self._horizon_generated_scores[key]


def get_kwargs(score_config, conditional=False):
    kwargs = {
        "discrete": score_config.get("discrete", False),
        "bin_method": score_config.get("bin_method", None),
        "quantiles": score_config.get("quantiles", None),
        "n_bins": score_config.get("n_bins", None),
        "thresholds": score_config.get("thresholds", None),
    }
    if not conditional:
        kwargs["group_scores"] = score_config.get("group_scores", True)

    # default to quartiles if no quantiles, n_bins or thresholds are specified
    if kwargs["bin_method"] is None \
        and (kwargs["quantiles"] is None) \
        and (kwargs["n_bins"] is None) \
        and (kwargs["thresholds"] is None):
            if conditional:
                # kwargs["quantiles"] = [0.25, 0.5, 0.75]
                # use deciles to condition on by default
                kwargs["quantiles"] = np.arange(0.1, 1.0, 0.1)
            else:
                # fd: Freedman Diaconis Estimator
                # see: https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
                kwargs["bin_method"] = 'fd'

    return kwargs


def score_data(
    loader: data_loading.Simple_Loader,
    scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    *,
    group_scores: bool = True,
    return_plot_fn=False,
    score_cache: ScoreCache | None = None,
    **kwargs,
):
    """
    """
    if score_cache is None:
        scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn)
    else:
        scores_real, scores_gen = score_cache.score_real_gen(scoring_fn)

    score_df = partitioning.get_score_table(scores_real, scores_gen, None, None)
    score_values = score_df.score.to_numpy(dtype=float)
    finite = np.isfinite(score_values)
    nonfinite = {}
    for score_type in ('real', 'generated'):
        type_mask = score_df.type.to_numpy() == score_type
        total = int(type_mask.sum())
        dropped = int((type_mask & ~finite).sum())
        nonfinite[score_type] = {
            'total': total,
            'score_nonfinite': dropped,
            'kept': total - dropped,
            'dropped_rate': dropped / total if total else np.nan,
        }

    score_df = score_df.loc[finite].copy()
    scores_real = score_df.loc[score_df.type == 'real', 'score'].to_numpy()
    scores_gen = (score_df.loc[
        score_df.type == 'generated', 'score'
    ].to_numpy(),)

    if group_scores:
        groups_real, groups_gen, thresholds = partitioning.group_by_score(
            scores_real, scores_gen,
            return_thresholds=True,
            **kwargs
        )
        thresholds = np.unique(thresholds)
    else:
        groups_real, groups_gen, thresholds = None, None, 'auto'

    score_df = partitioning.get_score_table(scores_real, scores_gen, groups_real, groups_gen)
    score_df.attrs['nonfinite'] = nonfinite

    if return_plot_fn:
        plot_fn = lambda title, ax: plotting.hist(
            scores_real,
            scores_gen,
            bins=thresholds,
            title=title,
            xlabel=title,
            ax=ax,
        )
        return score_df, plot_fn
    else:
        return score_df


def score_data_cond(
    loader: data_loading.Simple_Loader,
    scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    scoring_fn_cond: Callable[[pd.DataFrame, pd.DataFrame], float],
    *,
    return_plot_fn: bool = False,
    score_kwargs: dict = {},
    score_cond_kwargs: dict = {},
    score_cache: ScoreCache | None = None,
):
    """
    Grouping is done based on the scoring_fn_cond
    and scores are based on the scoring_fn.
    Conditional analysis can then be done using the returned score_df
    by grouping on 'group' and calculating metrics on 'score'.
    The additional colum 'subgroup' gives the bin of the conditional scores,
    where the binning is done based on the scoring_fn and score_kwargs.
    """
    # calc scores to be conditioned on
    if score_cache is None:
        scores_real, scores_gen = partitioning.score_real_gen(loader, scoring_fn_cond)
    else:
        scores_real, scores_gen = score_cache.score_real_gen(scoring_fn_cond)

    # calc scores of interest (to be conditioned on prior scores)
    if score_cache is None:
        eval_real, eval_gen = partitioning.score_real_gen(loader, scoring_fn)
    else:
        eval_real, eval_gen = score_cache.score_real_gen(scoring_fn)

    score_df = partitioning.get_score_table(eval_real, eval_gen, None, None)
    score_df_cond = partitioning.get_score_table(scores_real, scores_gen, None, None)
    assert score_df.type.equals(score_df_cond.type)
    score_df = score_df.drop(columns='group')
    score_df['score_cond'] = score_df_cond.score

    score_values = score_df.score.to_numpy(dtype=float)
    condition_values = score_df.score_cond.to_numpy(dtype=float)
    condition_finite = np.isfinite(condition_values)
    score_finite = np.isfinite(score_values)
    nonfinite = {}
    for score_type in ('real', 'generated'):
        type_mask = score_df.type.to_numpy() == score_type
        total = int(type_mask.sum())
        condition_dropped = int((type_mask & ~condition_finite).sum())
        score_dropped = int(
            (type_mask & condition_finite & ~score_finite).sum()
        )
        nonfinite[score_type] = {
            'total': total,
            'condition_nonfinite': condition_dropped,
            'score_nonfinite': score_dropped,
            'kept': total - condition_dropped - score_dropped,
            'dropped_rate': (
                (condition_dropped + score_dropped) / total if total else np.nan
            ),
        }

    score_df = score_df.loc[condition_finite & score_finite].copy()
    real_scores = score_df.loc[score_df.type == 'real', 'score'].to_numpy()
    generated_scores = score_df.loc[
        score_df.type == 'generated', 'score'
    ].to_numpy()
    real_conditions = score_df.loc[
        score_df.type == 'real', 'score_cond'
    ].to_numpy()
    generated_conditions = score_df.loc[
        score_df.type == 'generated', 'score_cond'
    ].to_numpy()
    groups_real, groups_gen, thresholds = partitioning.group_by_score(
        real_conditions,
        (generated_conditions,),
        return_thresholds=True,
        **score_cond_kwargs,
    )
    score_df = partitioning.get_score_table(
        real_scores,
        (generated_scores,),
        groups_real,
        groups_gen,
    )
    score_df['score_cond'] = np.concatenate(
        (real_conditions, generated_conditions)
    )

    # second grouping for each of the first groups (binning the conditional scores):
    score_df['subgroup'] = -1
    sub_dfs = [df[1] for df in score_df.groupby('group')]
    new_dfs = []
    for df in sub_dfs:
        real_scores = df.loc[df.type == 'real', 'score']
        gen_scores = df.loc[df.type == 'generated', 'score']
        groups_real, groups_gen = partitioning.group_by_score(
            real_scores.values,
            gen_scores.values,
            **score_kwargs
        )
        df = df.copy()
        # df['subgroup'] = -1
        df.loc[real_scores.index, 'subgroup'] = groups_real
        df.loc[gen_scores.index, 'subgroup'] = groups_gen
        new_dfs.append(df)
    score_df = pd.concat(new_dfs)
    score_df.attrs['nonfinite'] = nonfinite

    if return_plot_fn:
        plot_fn = lambda var_eval, var_cond, bins, binwidth: plotting.facet_grid_hist(
            score_df,
            var_eval=var_eval,
            var_cond=var_cond,
            filter_groups_below_weight=0.01,
            bins=bins,
            binwidth=binwidth,
        )
        return score_df, plot_fn
    else:
        return score_df


def score_prediction_horizons(
    loader: data_loading.Simple_Loader,
    scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    horizon_length: int,
    group_scores: bool = True,
    score_cache: ScoreCache | None = None,
    **kwargs
):
    """
    Computes a table unconditional scores of all real data with
    """
    # unconditional scoring of all real data
    if score_cache is None:
        scores_real = partitioning.score_real(loader, scoring_fn)
    else:
        scores_real = score_cache.score_real(scoring_fn)
    # score_df_real = partitioning.get_score_table(scores_real, None, None, None)

    # split generated data into subsequences based on generation horizon
    if score_cache is None:
        subseqs = tuple(
            partitioning.get_subseqs(s, subseq_len=horizon_length) for s in loader
        )
        scores_gen = partitioning.score_gen(subseqs, scoring_fn)
    else:
        subseqs = score_cache.subsequences(horizon_length)
        scores_gen = score_cache.generated_horizon_scores(scoring_fn, horizon_length)
    # enumerate the subsequences (===prediction horizon)
    _, groups_gen = partitioning.group_by_subseq(subseqs)
    score_df_gen = partitioning.get_score_table(
        scores_real=None,
        scores_gen=scores_gen,
        groups_real=None,
        groups_gen=groups_gen,
    )
    # print('score_df_gen')
    # display(score_df_gen)

    score_dfs = []
    for horizon_group, subtable in score_df_gen.groupby('group'):
        # print(horizon_group)
        # display(subtable)
        # compute score groups for each prediction horizon for generated data
        scores_gen = subtable.score.values
        # print('scores_real:', scores_real)

        if group_scores:
            groups_real, groups_gen, thresholds = partitioning.group_by_score(
                scores_real, (scores_gen,),
                return_thresholds=True, **kwargs)
        else:
            groups_real, groups_gen = None, None

        # print('groups_gen', groups_gen)
        score_df = partitioning.get_score_table(
            scores_real, (scores_gen,), groups_real, groups_gen)
        score_dfs.append(score_df)

    return score_dfs


def compute_metrics(
    loader: data_loading.Simple_Loader,
    scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
    # metric_fn: Callable[[pd.DataFrame], float] | \
    #            Iterable[Callable[[pd.DataFrame], float]],
    metric_fn: dict[str, Callable[[pd.DataFrame], float]],
    scoring_fn_cond: Optional[
                        Callable[[pd.DataFrame, pd.DataFrame], float]
                    ] = None,
    score_kwargs: dict = {},
    score_cond_kwargs: dict = {},
    score_cache: ScoreCache | None = None,
) -> tuple[float, pd.DataFrame, Callable]:
    # unconditional scoring
    if scoring_fn_cond is None:
        score_df, plot_fn = score_data(
            loader,
            scoring_fn,
            return_plot_fn=True,
            score_cache=score_cache,
            **score_kwargs,
        )
        metric = {m_name: m(score_df) for m_name, m in metric_fn.items()}

    # conditional scoring
    else:
        score_df, plot_fn = score_data_cond(
            loader, scoring_fn, scoring_fn_cond,
            return_plot_fn=True, score_kwargs=score_kwargs,
            score_cond_kwargs=score_cond_kwargs,
            score_cache=score_cache,
        )

        # calc. loss for each of the conditional distributions
        lens = []
        losses = []
        # len_and_losses = []
        for name, group in score_df.groupby('group'):
            # len_and_losses.append([len(group), metric_fn(group)[0]])
            lens.append(len(group))
            # use the subgroup for binning now:
            group.group = group.subgroup
            losses.append(
                np.stack(
                    tuple(mfn(group)[2] for mfn in metric_fn.values()),
                    axis=-1
                )
            )

        # calculate weights by normalizing the number of observations
        weights = np.array(lens, dtype=float)
        weights /= weights.sum()
        losses = np.array(losses).T
        # print('weights:', weights, 'losses:', losses.shape, losses)
        # sum over all groups
        # shape: (num metrics, n_bootstrap + 1, num groups)
        #     -> (num metrics, n_bootstrap + 1)
        metric = np.nansum(losses * weights, axis=-1)
        # print('metric:', metric)

        # get the percentiles of the bootstrapped loss values
        # TODO: make this an argument
        ci_alpha = 0.01
        q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
        # shape: (num metrics, n_bootstrap + 1)
        ci = np.nanpercentile(metric, q, axis=-1).T
        # print('ci', ci)
        # metric = (metric[:, 0], ci, metric)
        metric = {
            m_name: (m[0], ci_, m) for m_name, ci_, m
                in zip(metric_fn.keys(), ci, metric)
        }
        # print('metric:', metric)

    return metric, score_df, plot_fn


def _replace_gen_with_real_scores(horizon_df):
    df_real1 = horizon_df[horizon_df["type"] == 'real'].copy()
    df_real2 = df_real1.copy()
    # pretend real data is generated to calculate baseline divergence
    df_real2['type'] = 'generated'
    df_ = pd.concat([df_real1, df_real2])
    return df_


def calc_baseline_errors_by_score(
    scoring_dfs_horizon_all_scores: dict[str, list[pd.DataFrame]],
    metric_fn: Callable[[pd.DataFrame], float],
) -> dict[str, list[float]]:
    baseline_errors_by_score = {
        score_name: [
            # TODO: generalize this to other metrics
            metric_fn(
                _replace_gen_with_real_scores(df),
                # we want one-sided: 0.99
                # --> upper limit of the two-sided intervals for 0.98
                ci_alpha=0.98
            ) for df in scoring_dfs_horizon
        ] for score_name, scoring_dfs_horizon in scoring_dfs_horizon_all_scores.items()
    }
    return baseline_errors_by_score


def compute_divergence_metrics(
        loader: data_loading.Simple_Loader,
        scoring_fn: Callable[[pd.DataFrame, pd.DataFrame], float],
        metric_fn: Union[Callable[[pd.DataFrame], float] , \
                   Iterable[Callable[[pd.DataFrame], float]]],
        horizon_length: int,
        score_cache: ScoreCache | None = None,
        **kwargs,
    ):
    """
    """
    score_dfs_horizon = score_prediction_horizons(
        loader,
        scoring_fn,
        horizon_length,
        score_cache=score_cache,
        # quantiles=[0.25, 0.5, 0.75],
        **kwargs,
    )
    # L1 scores for each horizon distribution
    loss_horizons = [metric_fn(sdf) for sdf in score_dfs_horizon]

    plot_fn = lambda title, ax: plotting.error_divergence_plot(
        loss_horizons,
        horizon_length,
        title=title,
        xlabel='Prediction Horizon [messages]',
        ylabel='L1 score',
        ax=ax,
    )
    return loss_horizons, score_dfs_horizon, plot_fn


def run_benchmark(
    loader: data_loading.Simple_Loader,
    scoring_config_dict: dict,
    default_metric: Callable[[pd.DataFrame], float],
    divergence: bool = False,
    divergence_horizon: int = 50,
    score_cache: ScoreCache | None = None,
) -> tuple[
    dict[str, float],
    dict[str, pd.DataFrame],
    dict[str, Callable]
]:

    scores = {}
    score_dfs = {}
    plot_fns = {}

    for score_name, score_config in scoring_config_dict.items():
        print("Calculating scores and metrics for: ", score_name, end="\r\n")

        # conditional scoring
        if score_config.get("eval", None) is not None:
            score_cond_config = score_config["cond"]
            score_config = score_config["eval"]
            score_cond_kwargs = get_kwargs(score_cond_config, conditional=True)
            score_kwargs = get_kwargs(score_config, conditional=True)
            score_fn_cond = score_cond_config["fn"]
        # unconditional scoring
        else:
            score_cond_kwargs = {}
            score_fn_cond = None
            score_kwargs = get_kwargs(score_config)

        # print('score_cond_kwargs:', score_cond_kwargs)
        metric_fns = score_config.get("metric_fns", default_metric)

        if divergence:
            scores[score_name], score_dfs[score_name], plot_fns[score_name] = \
                compute_divergence_metrics(
                    loader,
                    score_config["fn"],
                    metric_fns,
                    divergence_horizon,
                    score_cache=score_cache,
                    **get_kwargs(score_config)
                )

        else:
            scores[score_name], score_dfs[score_name], plot_fns[score_name] = \
                compute_metrics(
                    loader,
                    score_config["fn"],
                    metric_fns,
                    score_fn_cond,
                    score_kwargs=score_kwargs,
                    score_cond_kwargs=score_cond_kwargs,
                    score_cache=score_cache,
                )

    return scores, score_dfs, plot_fns


def summary_stats(
        scores,
        bootstrap: bool = True,
        ci_alpha: float = 0.01,
        n_bootstrap: int = 1000,
        rng_np: np.random.Generator = np.random.default_rng(12345),
    ):
    return_dict = {}
    # for each metric:
    values_list = list(scores.values())
    for i, metric_name in enumerate(values_list[0].keys()):
        loss_vals = np.array([s[metric_name][0] for s in scores.values()])

        # loss_vals = np.array(
        #     [[score[2] for score in sdict.values()[0]] for sdict in scores.values()])
        aggr_mean, aggr_median, aggr_iqm = _calc_summary_stats(loss_vals)
        # print(aggr_mean, aggr_median, aggr_iqm)

        if bootstrap:
            # draw single score from each bootstrap sample per metric
            losses_bootstrap = np.array([
                [rng_np.choice(s[metric_name][2])
                    for s in scores.values()]
                for _ in range(n_bootstrap)
            ])
            # print(losses_bootstrap)
            bs_mean, bs_median, bs_iqm = _calc_summary_stats(losses_bootstrap)
            q = np.array([ci_alpha/2 * 100, 100 - ci_alpha/2*100])
            ci_mean = np.percentile(bs_mean, q)
            ci_median = np.percentile(bs_median, q)
            ci_iqm = np.percentile(bs_iqm, q)

            # return (aggr_mean, ci_mean), (aggr_median, ci_median), (aggr_iqm, ci_iqm)
            return_dict[metric_name] = (
                (aggr_mean, ci_mean),
                (aggr_median, ci_median),
                (aggr_iqm, ci_iqm),
            )
        # return aggr_mean, aggr_median, aggr_iqm
        else:
            return_dict[metric_name] = aggr_mean, aggr_median, aggr_iqm

    return return_dict

def _calc_summary_stats(loss_vals):
    aggr_mean = np.mean(loss_vals, axis=-1)
    aggr_median = np.median(loss_vals, axis=-1)

    q25, q75 = np.percentile(loss_vals, [25, 75], axis=-1, keepdims=True)
    # iqr = q75 - q25
    aggr_iqm = np.ma.masked_array(
        loss_vals,
        mask=(loss_vals < q25) | (loss_vals > q75)
    ).mean(axis=-1).__array__()

    return aggr_mean, aggr_median, aggr_iqm
