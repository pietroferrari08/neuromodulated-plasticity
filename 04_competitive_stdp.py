
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BETA = 0.8
THRESHOLD = 1.0

A_PLUS = 0.025
A_MINUS = 0.030

TAU = 10
ALPHA = np.exp(-1 / TAU)

N_INPUTS = 8
N_OUTPUTS = 2

NUM_STEPS = 35
N_EPOCHS = 20

rng = np.random.default_rng(1)

# Piccole differenze iniziali per rompere la simmetria
weights = rng.uniform(
    0.35, 0.45,
    size=(N_OUTPUTS, N_INPUTS)
)

initial_weights = weights.copy()


# 1. GENERAZIONE DEI PATTERN

def generate_pattern(group, background=True):

    spikes = np.zeros((NUM_STEPS, N_INPUTS))

    first = 0 if group == 0 else 4
    second = 4 if group == 0 else 0

    for i in range(4):

        spikes[4 + i, first + i] = 1

        if background:
            spikes[12 + 4*i, second + i] = 1

    return spikes


# 2. SIMULAZIONE

def simulate(spikes, weights,
             learning=True, competition=True):

    weights = weights.copy()

    mem = np.zeros(N_OUTPUTS)

    pre_trace = np.zeros(N_INPUTS)
    post_trace = np.zeros(N_OUTPUTS)

    winner = -1

    # Ogni neurone può emettere al massimo
    # uno spike per presentazione
    fired = np.zeros(N_OUTPUTS, dtype=bool)

    spike_times = np.full(N_OUTPUTS, -1)

    for t in range(NUM_STEPS):

        spike_pre = spikes[t]
        spike_post = np.zeros(N_OUTPUTS)

        if not competition or winner == -1:

            current = np.dot(weights, spike_pre)

            mem = BETA * mem + current

            # Neuroni sopra soglia that non hanno
            # ancora generato uno spike
            candidates = np.flatnonzero(
                (mem >= THRESHOLD) & ~fired
            )

            if competition and candidates.size > 0:

                # Se entrambi raggiungono la soglia,
                # vince quello con potenziale maggiore
                winner = candidates[
                    np.argmax(mem[candidates])
                ]

                candidates = np.array([winner])

            spike_post[candidates] = 1

            spike_times[candidates] = t
            fired[candidates] = True

            mem[fired] = 0.0

        pre_trace = ALPHA * pre_trace 
        post_trace = ALPHA * post_trace 

        if learning:

            ltp = np.outer(
                spike_post, pre_trace   
            )

            ltd = -np.outer(
                post_trace, spike_pre
            )*A_MINUS

            weights += ltp + ltd

            weights = np.clip(weights, 0, 1)

        pre_trace += spike_pre
        post_trace += spike_post

    return weights, spike_times, winner


# 3. TRAINING

history = [weights.copy()]

# Righe: pattern A e B
# Colonne: neurone 1 e neurone 2
wins = np.zeros((2, 2), dtype=int)

for epoch in range(N_EPOCHS):

    for group in rng.permutation(2):

        pattern = generate_pattern(group)

        weights, times, winner = simulate(
            pattern,
            weights,
            learning=True,
            competition=True
        )

        if winner >= 0:
            wins[group, winner] += 1

    history.append(weights.copy())


# 4. TEST SENZA APPRENDIMENTO
# E SENZA COMPETIZIONE

for group in range(2):

    pattern = generate_pattern(
        group,
        background=False
    )

    _, before, _ = simulate(
        pattern,
        initial_weights,
        learning=False,
        competition=False
    )

    _, after, _ = simulate(
        pattern,
        weights,
        learning=False,
        competition=False
    )

    print(f"\nPattern {'AB'[group]}")
    print("Spike times before:", before)
    print("Spike times after:", after)


# 5. RISULTATI

print("\nInitial weights:")
print(initial_weights.round(4))

print("\nFinal weights:")
print(weights.round(4))

print("\nWinner matrix:")
print(wins)


# 6. VISUALIZZAZIONE DEI PESI

fig, axes = plt.subplots(
    1, 2, figsize=(12, 4),
    layout="constrained"
)

for ax, matrix, title in zip(
    axes,
    [initial_weights, weights],
    ["Before learning", "After learning"]
):

    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(8))

    ax.set_xticklabels([
        "A1", "A2", "A3", "A4",
        "B1", "B2", "B3", "B4"
    ])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Neuron 1", "Neuron 2"])

    ax.set_title(title)

    for i in range(N_OUTPUTS):
        for j in range(N_INPUTS):

            ax.text(
                j, i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < 0.5
                else "black"
            )

fig.colorbar(
    im, ax=axes,
    label="Synaptic weight"
)

Path("figures").mkdir(exist_ok=True)

fig.savefig(
    "figures/competitive_stdp.png",
    dpi=200
)

plt.show()
