"""Presentation figures for the Week 3 historical experiments."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(directory, folds, summary, statcast, ages, importance,
                 trees, bins, calibration_scores, thresholds, selected):
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f"{name}.png", dpi=160)
        plt.close(fig)

    color = "#22647a"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for experiment, group in folds.groupby("experiment_id"):
        ax.plot(group.validation_season.astype(str), group.average_precision, marker="o",
                label=experiment)
    reference = folds.drop_duplicates("validation_season")
    ax.plot(reference.validation_season.astype(str), reference.validation_prevalence,
            "--", color="black", label="Prevalence")
    ax.set(title="Average Precision by feature season", xlabel="Validation season",
           ylabel="Average Precision")
    ax.legend(ncol=2, bbox_to_anchor=(1, 1), loc="upper left")
    save(fig, "average_precision_by_fold")

    ordered = summary.set_index("experiment_id")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, label in zip(
        axes, ("average_precision", "roc_auc", "brier_score"),
        ("Average Precision", "ROC-AUC", "Brier score ↓"),
    ):
        ax.bar(ordered.index, ordered[f"mean_{metric}"], yerr=ordered[f"std_{metric}"],
               color=color, capsize=3)
        ax.set_title(label)
    fig.suptitle("Mean historical performance ± one fold SD")
    save(fig, "mean_performance")

    fig, ax = plt.subplots(figsize=(8, 4))
    shortlist = ordered.sort_values("mean_precision_at_20", ascending=False)
    ax.bar(shortlist.index, shortlist.mean_precision_at_20, yerr=shortlist.std_precision_at_20,
           color=color, capsize=4)
    ax.axhline(reference.validation_prevalence.mean(), linestyle="--", color="black",
               label="Mean season prevalence")
    ax.set(title="Seasonal shortlist hit rate ± one fold SD", ylabel="Precision@20")
    ax.legend()
    save(fig, "precision_at_20")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, measure, label in zip(axes, ("average_precision", "precision_at_20"),
                                  ("Δ Average Precision", "Δ Precision@20")):
        statcast.pivot(index="model_name", columns="validation_season",
                      values=f"delta_{measure}").plot.bar(ax=ax, rot=15)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="", ylabel=label)
        ax.legend(title="Feature season")
    fig.suptitle("Incremental performance with Statcast")
    save(fig, "statcast_differences")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    age_plot = ages.pivot(index="age_group", columns="experiment_id", values="average_precision")
    age_plot = age_plot.reindex(["≤24", "25–29", "≥30"])
    age_plot.plot.bar(ax=ax, rot=0)
    prevalence = ages.groupby("age_group").prevalence.first().reindex(age_plot.index)
    ax.scatter(np.arange(len(prevalence)), prevalence, marker="D", color="black", label="Prevalence")
    ax.set(title="Pooled historical model performance by hitter age", xlabel="Age group",
           ylabel="Average Precision")
    ax.legend()
    save(fig, "age_group_performance")

    for experiment, filename, label in [
        ("M2", "logistic_coefficients", "Standardized coefficient"),
        (trees[0], "permutation_importance", "Average Precision drop"),
    ]:
        subset = importance.loc[importance.experiment_id.eq(experiment)].head(12).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(subset.feature, subset.mean_importance, xerr=subset.fold_std,
                color=np.where(subset.mean_importance < 0, "#ba6a35", color), capsize=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=f"{experiment} feature importance ± one fold SD", xlabel=label)
        save(fig, filename)

    if len(trees) == 2:
        fig, ax = plt.subplots(figsize=(9, 7))
        table = importance.loc[importance.experiment_id.isin(trees)].pivot(
            index="feature", columns="experiment_id", values="mean_importance"
        )
        order = table.max(axis=1).sort_values().index
        table = table.reindex(order)
        for experiment, offset, marker in zip(trees, (-.12, .12), ("o", "s")):
            available = table[experiment].notna()
            ax.scatter(table.loc[available, experiment], np.flatnonzero(available) + offset,
                       marker=marker, label=experiment)
        ax.set_yticks(np.arange(len(table)), table.index)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title="Same tree family with and without Statcast",
               xlabel="Mean validation Average Precision drop", ylabel="")
        ax.legend(title="Experiment")
        save(fig, "statcast_importance")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax, distribution, cutoff = axes
    ax.plot([0, 1], [0, 1], "--", color="gray")
    view = bins.loc[bins.experiment_id.eq(selected) & bins["count"].gt(0)]
    for method, group in view.groupby("method"):
        ax.plot(group.mean_probability, group.observed_frequency, marker="o", label=method)
        for row in group.itertuples():
            ax.annotate(str(row.count), (row.mean_probability, row.observed_frequency),
                        xytext=(4, 3 if method == "raw" else -10), textcoords="offset points",
                        fontsize=7)
    ax.set(title=f"{selected} reliability (bin counts)", xlabel="Mean estimated probability",
           ylabel="Observed breakout frequency", xlim=(0, 1), ylim=(0, 1))
    if not view.empty:
        ax.legend()
    group = calibration_scores.loc[calibration_scores.experiment_id.eq(selected)
                                   & calibration_scores.method.eq("raw")]
    for label, part in group.groupby("breakout"):
        distribution.hist(part.predicted_probability, bins=np.linspace(0, 1, 11), density=True,
                          histtype="step", linewidth=2, label="Breakout" if label else "Non-breakout")
    distribution.set(title="Raw scores by outcome", xlabel="Probability", ylabel="Density")
    if not group.empty:
        distribution.legend()
    view = thresholds.loc[thresholds.experiment_id.eq(selected)]
    for metric in ("precision", "recall", "f1"):
        cutoff.plot(view.threshold, view[metric], marker="o", label=metric)
    cutoff.set(title="Retrospective cutoff tradeoff", xlabel="Probability threshold", ylim=(0, 1))
    cutoff.legend()
    save(fig, "calibration_and_thresholds")
