
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from importlib import import_module


model = import_module("06_1_reward_modulated_stdp")

DELAYS = [0, 10, 25, 50, 100, 200]
EPSILONS = [0, 0.1, 0.2, 0.5]

SEEDS = range(10)
N_EPOCHS = 80

BASE_DELAY = 25
BASE_EPSILON = 0.2


# 1. VALUTAZIONE DELLA RETE

def evaluate(weights):

    rng = np.random.default_rng(0)
    correct = 0

    for group in range(2):

        _, action, outcome = model.simulate(
            group,
            weights,
            rng,
            epsilon=0,
            reward_enabled=False,
            reward_delay=0
        )

        correct += int(outcome == 1)

    return correct / 2


# 2. SINGOLO ADDESTRAMENTO

def run_experiment(seed, delay, epsilon):

    rng_init = np.random.default_rng(seed)
    rng_train = np.random.default_rng(42)

    weights = rng_init.uniform(
        0.35, 0.45,
        size=(2, 8)
    )

    scores = [evaluate(weights)]

    for epoch in range(N_EPOCHS):

        for group in rng_train.permutation(2):

            weights, _, _ = model.simulate(
                group,
                weights,
                rng_train,
                epsilon=epsilon,
                reward_enabled=True,
                reward_delay=delay
            )

        scores.append(evaluate(weights))

    scores = np.array(scores)

    # Epoca dalla quale la rete rimane
    # corretta fino alla fine del training

    stable_epoch = np.nan

    if scores[-1] == 1:

        failures = np.flatnonzero(scores < 1)

        stable_epoch = (
            failures[-1] + 1
            if failures.size > 0
            else 0
        )

    return {
        "initial_accuracy": scores[0],
        "final_accuracy": scores[-1],
        "success": scores[-1] == 1,
        "stable_epoch": stable_epoch
    }


# 3. ANALISI SU PIÙ SEED

def parameter_experiment(delay, epsilon):

    runs = []

    for seed in SEEDS:

        result = run_experiment(
            seed, delay, epsilon
        )

        runs.append(result)

    initial = np.array([
        r["initial_accuracy"] for r in runs
    ])

    final = np.array([
        r["final_accuracy"] for r in runs
    ])

    success = np.array([
        r["success"] for r in runs
    ])

    epochs = np.array([
        r["stable_epoch"] for r in runs
    ])

    median_epoch = (
        np.nanmedian(epochs)
        if np.any(success)
        else np.nan
    )

    return {
        "initial_accuracy": initial.mean(),
        "mean_accuracy": final.mean(),
        "std_accuracy": final.std(),
        "success_rate": success.mean(),
        "median_epoch": median_epoch
    }


# 4. STUDIO PARAMETRICO

results = {}

configurations = list(dict.fromkeys(
    [(d, BASE_EPSILON) for d in DELAYS]
    + [(BASE_DELAY, e) for e in EPSILONS]
))

for delay, epsilon in configurations:

    result = parameter_experiment(
        delay, epsilon
    )

    results[(delay, epsilon)] = result

    print(
        f"Delay={delay:3d} | "
        f"Epsilon={epsilon:.1f} | "
        f"Initial={result['initial_accuracy']:.0%} | "
        f"Final={result['mean_accuracy']:.0%} | "
        f"Success={result['success_rate']:.0%} | "
        f"Stable epoch={result['median_epoch']:.1f}",
        flush=True
    )


# 5. ESTRAZIONE DEI RISULTATI

delay_results = [
    results[(d, BASE_EPSILON)]
    for d in DELAYS
]

epsilon_results = [
    results[(BASE_DELAY, e)]
    for e in EPSILONS
]


# 6. VISUALIZZAZIONE

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 8),
    layout="constrained"
)

for column, x, data, xlabel in [
    (0, DELAYS, delay_results, "Reward delay"),
    (1, EPSILONS, epsilon_results, "Epsilon")
]:

    mean_accuracy = [
        r["mean_accuracy"] for r in data
    ]

    success_rates = [
        r["success_rate"] for r in data
    ]

    median_epochs = [
        r["median_epoch"] for r in data
    ]

    ax = axes[0, column]

    ax.plot(
        x, mean_accuracy,
        marker="o",
        label="Mean accuracy"
    )

    ax.plot(
        x, success_rates,
        marker="s",
        linestyle="--",
        label="Success rate"
    )

    ax.set_ylim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Performance")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, column]

    ax.plot(
        x, median_epochs,
        marker="o"
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median stable epoch")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)

axes[0, 0].set_title("Effect of reward delay")
axes[0, 1].set_title("Effect of exploration")

axes[1, 0].set_title("Learning speed vs delay")
axes[1, 1].set_title("Learning speed vs exploration")

Path("figures").mkdir(exist_ok=True)

fig.savefig(
    "figures/rstdp_parameter_study.png",
    dpi=200
)

plt.show()
