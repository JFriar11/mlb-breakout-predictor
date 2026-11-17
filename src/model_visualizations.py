"""Plot retrospective validation results from saved prediction scores."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve


def plot_validation(predictions, metrics, selected, destination):
    destination.mkdir(parents=True, exist_ok=True)
    reference = metrics.loc["constant", "prevalence"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(destination / f"{name}.png", dpi=160)
        plt.close(fig)

    roc_fig, roc = plt.subplots(figsize=(7, 5))
    pr_fig, pr = plt.subplots(figsize=(7, 5))
    for experiment, frame in predictions.items():
        if experiment == "constant":
            continue
        y, scores = frame.breakout, frame.predicted_probability
        fpr, tpr, _ = roc_curve(y, scores)
        precision, recall, _ = precision_recall_curve(y, scores)
        roc.plot(fpr, tpr, label=f"{experiment} ({metrics.loc[experiment, 'roc_auc']:.3f})")
        pr.step(recall, precision, where="post",
                label=f"{experiment} ({metrics.loc[experiment, 'ap']:.3f})")
    roc.plot([0, 1], [0, 1], "--", color="gray", label="Constant score")
    pr.axhline(reference, linestyle="--", color="gray", label=f"Prevalence ({reference:.3f})")
    roc.set(title="Validation ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    pr.set(title="Validation precision–recall curves", xlabel="Recall", ylabel="Precision")
    roc.legend(title="Experiment (ROC-AUC)")
    pr.legend(title="Experiment (AP)")
    save(roc_fig, "roc_curves")
    save(pr_fig, "precision_recall_curves")

    choices = list(dict.fromkeys(["M1", selected]))
    fig, axes = plt.subplots(1, len(choices), figsize=(4.5 * len(choices), 4), squeeze=False)
    for experiment, ax in zip(choices, axes[0]):
        frame = predictions[experiment]
        ConfusionMatrixDisplay.from_predictions(
            frame.breakout, frame.predicted_class, labels=[0, 1],
            display_labels=["Non-breakout", "Breakout"], colorbar=False, ax=ax,
        )
        ax.set_title(experiment)
    fig.suptitle("Validation confusion matrices at probability ≥ 0.50")
    save(fig, "confusion_matrices")

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics.ap.plot.bar(ax=ax, rot=0, color="#2c6b91")
    ax.set(title="Validation Average Precision", xlabel="Experiment", ylabel="Average Precision")
    save(fig, "average_precision")

    measures = {
        "ap": "Average Precision", "roc_auc": "ROC-AUC", "brier": "Brier score ↓",
        "precision_at_20": "Precision@20", "recall": "Recall at 0.50",
    }
    fig, axes = plt.subplots(1, len(measures), figsize=(14, 4))
    for (metric, label), ax in zip(measures.items(), axes):
        values = pd.DataFrame({
            "Traditional": metrics.loc[["M2", "M4"], metric].to_numpy(),
            "Traditional + Statcast": metrics.loc[["M3", "M5"], metric].to_numpy(),
        }, index=["Logistic", "Forest"])
        values.plot.bar(ax=ax, rot=0, legend=False, color=["#2c6b91", "#d78940"])
        ax.set_title(label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .92), ncol=2)
    fig.suptitle("Validation performance with and without Statcast")
    fig.subplots_adjust(top=.72)
    fig.tight_layout(rect=(0, 0, 1, .79))
    fig.savefig(destination / "statcast_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    frame = predictions[selected]
    for label, name in [(0, "Non-breakout"), (1, "Breakout")]:
        scores = frame.loc[frame.breakout.eq(label), "predicted_probability"]
        ax.hist(scores, bins=np.linspace(0, 1, 21), alpha=.55, density=True,
                label=f"{name} (n={len(scores)})")
    ax.set(title=f"Validation probability distributions ({selected})",
           xlabel="Predicted breakout probability", ylabel="Density")
    ax.legend()
    save(fig, "probability_distributions")

    fig, axes = plt.subplots(1, len(choices), figsize=(5.2 * len(choices), 4.5), squeeze=False)
    for experiment, ax in zip(choices, axes[0]):
        frame = predictions[experiment]
        bins = pd.cut(frame.predicted_probability, np.linspace(0, 1, 11), include_lowest=True)
        groups = frame.groupby(bins, observed=True).agg(
            score=("predicted_probability", "mean"),
            observed=("breakout", "mean"),
            count=("breakout", "size"),
        )
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        ax.plot(groups.score, groups.observed, "o-", label=experiment)
        for row in groups.itertuples():
            ax.annotate(str(row.count), (row.score, row.observed),
                        xytext=(4, 5), textcoords="offset points")
        ax.set(xlim=(0, 1), ylim=(0, 1), title=experiment,
               xlabel="Mean predicted probability", ylabel="Observed breakout rate")
        ax.legend(loc="lower right")
    fig.suptitle("Validation reliability (labels show observations per nonempty bin)")
    save(fig, "calibration")
