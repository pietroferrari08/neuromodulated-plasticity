
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PARAMETRI

A_PLUS = 0.025
A_MINUS = 0.030

TAU_PLUS = 10
TAU_MINUS = 10

INITIAL_WEIGHT = 0.5
N_PAIRS = 20


# 1. REGOLA STDP

def stdp_update(delta_t):

    if delta_t > 0:
        return A_PLUS * np.exp(-delta_t / TAU_PLUS)

    elif delta_t < 0:
        return -A_MINUS * np.exp(delta_t / TAU_MINUS)

    return 0.0


# 2. FINESTRA TEMPORALE STDP

negative_times = np.linspace(-30, -0.01, 300)
positive_times = np.linspace(0.01, 30, 300)

negative_changes = [
    stdp_update(dt) for dt in negative_times
]

positive_changes = [
    stdp_update(dt) for dt in positive_times
]


# 3. EVOLUZIONE DEL PESO

def simulate_learning(delta_t):

    weight = INITIAL_WEIGHT
    history = [weight]

    for _ in range(N_PAIRS):

        weight += stdp_update(delta_t)

        weight = np.clip(weight, 0, 1)

        history.append(weight)

    return history


potentiation = simulate_learning(5)
depression = simulate_learning(-5)


# 4. RISULTATI

print("Peso iniziale:", INITIAL_WEIGHT)

print(
    f"Peso finale LTP: {potentiation[-1]:.4f}"
)

print(
    f"Peso finale LTD: {depression[-1]:.4f}"
)


# 5. GRAFICI

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(
    negative_times, negative_changes,
    label="LTD"
)

axes[0].plot(
    positive_times, positive_changes,
    label="LTP"
)

axes[0].axhline(0, color="gray", linestyle="--")
axes[0].axvline(0, color="gray", linestyle="--")

axes[0].set_xlabel("t_post - t_pre")
axes[0].set_ylabel("Weight change")
axes[0].set_title("STDP learning window")
axes[0].legend()

axes[1].plot(potentiation, label="LTP")
axes[1].plot(depression, label="LTD")

axes[1].set_xlabel("Spike pairs")
axes[1].set_ylabel("Synaptic weight")
axes[1].set_title("Synaptic plasticity")
axes[1].set_ylim(0, 1)
axes[1].legend()

fig.tight_layout()

Path("figures").mkdir(exist_ok=True)

fig.savefig("figures/stdp_basics.png", dpi=200)

plt.show()
