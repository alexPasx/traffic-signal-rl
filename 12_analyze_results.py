from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "comparison"
OUTPUT = ROOT / "results" / "figures"

ORDER = ["fixed", "adaptive", "rl_wait", "rl_fair"]

LABELS = [
    "Fixed",
    "Rule-based",
    "RL: waiting",
    "RL: extra penalty",
]

COLORS = ["#94a3b8", "#3b82f6", "#0d9488", "#f59e0b"]


def validate(episodes, vehicles, directions):
    assert len(episodes) == 80, "Λάθος πλήθος επεισοδίων."

    assert not episodes.duplicated(
        ["controller", "seed"]
    ).any(), "Διπλά επεισόδια."

    assert not vehicles.duplicated(
        ["controller", "seed", "vehicle_id"]
    ).any(), "Διπλά οχήματα."

    assert not directions.duplicated(
        ["controller", "seed", "direction"]
    ).any(), "Διπλές κατευθύνσεις."

    assert not episodes.isna().any().any(), (
        "Κενές τιμές στα επεισόδια."
    )

    assert set(episodes["controller"]) == set(ORDER)

    for name in ORDER:
        seeds = set(
            episodes.loc[
                episodes["controller"] == name,
                "seed",
            ]
        )

        assert seeds == set(range(1000, 1020)), (
            f"Λάθος seeds για {name}."
        )

    # Μετατρέπουμε με ελεγχόμενο τρόπο το arrived σε boolean.
    arrived_text = (
        vehicles["arrived"]
        .astype(str)
        .str.lower()
    )

    assert arrived_text.isin(["true", "false"]).all()

    vehicles["arrived"] = arrived_text.eq("true")

    assert vehicles["observed_wait_s"].ge(0).all()

    assert vehicles["direction"].isin(
        ["n", "s", "e", "w"]
    ).all()

    # Επαναϋπολογισμός μετρήσεων από τις εγγραφές οχημάτων.
    rebuilt = (
        vehicles.groupby(["controller", "seed"])
        .agg(
            entered=("vehicle_id", "size"),
            arrived=("arrived", "sum"),
            mean_wait_s=("observed_wait_s", "mean"),
            p95_wait_s=(
                "observed_wait_s",
                lambda x: x.quantile(0.95),
            ),
            max_wait_s=("observed_wait_s", "max"),
            percent_over_30s=(
                "observed_wait_s",
                lambda x: 100 * x.gt(30).mean(),
            ),
            stopped_vehicle_seconds=(
                "observed_wait_s", "sum"
            ),
        )
    )

    original = episodes.set_index(
        ["controller", "seed"]
    ).sort_index()

    rebuilt = rebuilt.reindex(original.index)

    for column in [
        "entered",
        "arrived",
        "mean_wait_s",
        "p95_wait_s",
        "max_wait_s",
        "percent_over_30s",
    ]:
        assert np.allclose(
            original[column],
            rebuilt[column],
        ), f"Ασυμφωνία στη μέτρηση: {column}"

    assert np.allclose(
        original["total_wait_vehicle_seconds"],
        rebuilt["stopped_vehicle_seconds"]
        + original["pending_vehicle_seconds"],
    ), "Ασυμφωνία στη συνολική αναμονή."

    assert np.allclose(
        original["mean_stopped"] * 600,
        rebuilt["stopped_vehicle_seconds"],
    ), "Ασυμφωνία στο μέσο πλήθος σταματημένων."

    # Στο παρόν σύνολο περιμένουμε μηδενικά teleports.
    assert original["teleport_events"].eq(0).all()

    assert (
        original["entered"]
        == original["arrived"] + original["remaining"]
    ).all(), "Ασυμφωνία αριθμού οχημάτων."

    rebuilt_directions = (
        vehicles.groupby(
            ["controller", "seed", "direction"]
        )
        .agg(
            vehicle_count=("vehicle_id", "size"),
            mean_wait_s=("observed_wait_s", "mean"),
            p95_wait_s=(
                "observed_wait_s",
                lambda x: x.quantile(0.95),
            ),
            max_wait_s=("observed_wait_s", "max"),
        )
        .sort_index()
    )

    original_directions = directions.set_index(
        ["controller", "seed", "direction"]
    ).sort_index()

    assert (
        rebuilt_directions.index.equals(
            original_directions.index
        )
    ), "Ασυμφωνία στις κατευθύνσεις."

    for column in [
        "vehicle_count",
        "mean_wait_s",
        "p95_wait_s",
        "max_wait_s",
    ]:
        assert np.allclose(
            rebuilt_directions[column],
            original_directions[column],
        ), f"Ασυμφωνία κατευθύνσεων: {column}"

    print("Έλεγχοι συμφωνίας CSV: OK")


def comparison_plot(episodes):
    metrics = [
        ("mean_wait_s", "Mean observed waiting time (s)"),
        ("p95_wait_s", "95th percentile waiting time (s)"),
        ("arrived", "Completed trips"),
        (
            "total_wait_vehicle_seconds",
            "Total waiting, including insertion (vehicle-s)",
        ),
    ]

    grouped = episodes.groupby("controller")

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 9),
    )

    for ax, (metric, title) in zip(
        axes.flat, metrics
    ):
        means = grouped[metric].mean().reindex(ORDER)
        stds = grouped[metric].std().reindex(ORDER)

        bars = ax.bar(
            LABELS,
            means,
            color=COLORS,
            yerr=stds,
            capsize=4,
        )

        ax.bar_label(
            bars,
            labels=[f"{x:.1f}" for x in means],
            padding=8,
        )

        ax.set_title(title, loc="left", fontsize=12)

        ax.set_ylim(
            0,
            float((means + stds).max()) * 1.22,
        )

        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Traffic signal control: four-controller comparison",
        fontsize=18,
    )

    fig.text(
        0.06, 0.02,
        "20 matched simulation seeds; 600 s per run. "
        "Error bars: sample SD across simulation seeds.\n"
        "Waiting is observed up to the simulation cutoff. "
        "One trained model per RL variant.",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 0.94])

    fig.savefig(
        OUTPUT / "controller_comparison.png",
        dpi=200,
    )

    fig.savefig(
        OUTPUT / "controller_comparison.pdf"
    )

    plt.close(fig)


def direction_plot(directions):
    direction_order = ["n", "s", "e", "w"]

    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(4)
    width = 0.19

    for index, name in enumerate(ORDER):
        subset = directions[
            directions["controller"] == name
        ]

        grouped = subset.groupby("direction")[
            "mean_wait_s"
        ]

        means = grouped.mean().reindex(direction_order)
        stds = grouped.std().reindex(direction_order)

        ax.bar(
            positions + (index - 1.5) * width,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            label=LABELS[index],
            color=COLORS[index],
        )

    ax.set_xticks(
        positions,
        ["North", "South", "East", "West"],
    )

    ax.set_ylabel("Mean observed waiting time (s)")

    ax.set_title(
        "Waiting by origin direction",
        loc="left",
        fontsize=16,
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.text(
        0.09, 0.02,
        "Means and sample SD across 20 seeds. "
        "Entered vehicles only; insertion waiting excluded.",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    fig.savefig(
        OUTPUT / "waiting_by_direction.png",
        dpi=200,
    )

    fig.savefig(
        OUTPUT / "waiting_by_direction.pdf"
    )

    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(DATA / "episodes.csv")
    vehicles = pd.read_csv(DATA / "vehicles.csv")
    directions = pd.read_csv(
        DATA / "direction_by_seed.csv"
    )

    validate(episodes, vehicles, directions)

    metrics = [
        "mean_wait_s",
        "p95_wait_s",
        "max_wait_s",
        "percent_over_30s",
        "arrived",
        "total_wait_vehicle_seconds",
    ]

    table = (
        episodes.groupby("controller")[metrics]
        .mean()
        .reindex(ORDER)
    )

    table.to_csv(OUTPUT / "verified_means.csv")

    comparison_plot(episodes)
    direction_plot(directions)

    print("\n--- ΕΠΑΛΗΘΕΥΜΕΝΟΙ ΜΕΣΟΙ ΟΡΟΙ ---")
    print(table.round(2).to_string())

    print(f"\nΓραφήματα αποθηκεύτηκαν στο: {OUTPUT}")


if __name__ == "__main__":
    main()