
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
NUM_STEPS = 35
N_EPOCHS = 20

# Otto sinapsi con lo stesso peso iniziale
weights = np.full(N_INPUTS, 0.4)
initial_weights = weights.copy()


# 1. MODELLO LIF

def lif_step(mem, current):

    mem = BETA * mem + current

    spike = int(mem >= THRESHOLD)

    if spike:
        mem = 0.0

    return mem, spike


# 2. GENERAZIONE DEL PATTERN

def generate_pattern():

    spikes = np.zeros((NUM_STEPS, N_INPUTS))

    # Gruppo A: spike ravvicinati
    spikes[4, 0] = 1
    spikes[5, 1] = 1
    spikes[6, 2] = 1
    spikes[7, 3] = 1

    # Gruppo B: spike distanziati
    spikes[12, 4] = 1
    spikes[16, 5] = 1
    spikes[20, 6] = 1
    spikes[24, 7] = 1

    return spikes


# 3. SIMULAZIONE CON STDP

def simulate(spikes, weights, learning=True):

    mem = 0.0

    # Una traccia per ogni ingresso
    pre_trace = np.zeros(N_INPUTS)

    # Una traccia per il neurone di uscita
    post_trace = 0.0

    output_spikes = []

    for t in range(NUM_STEPS):

        spike_pre = spikes[t]

        current = np.dot(weights, spike_pre)

        mem, spike_post = lif_step(mem, current)

        pre_trace = ALPHA * pre_trace 
        post_trace = ALPHA * post_trace 

        if learning:

            ltp = A_PLUS * pre_trace * spike_post

            ltd = - A_MINUS * post_trace * spike_pre

            weights += ltp + ltd

            weights = np.clip(weights, 0, 1)

        pre_trace += spike_pre
        post_trace += spike_post
    
        output_spikes.append(spike_post)

    return weights, output_spikes


# 4. TRAINING

history = [weights.copy()]

for epoch in range(N_EPOCHS):

    pattern = generate_pattern()

    weights, output = simulate(
        pattern, weights, learning=True
    )

    history.append(weights.copy())

    print(
        f"Epoca {epoch + 1}: "
        f"Spike output = {sum(output)}"
    )


# 5. TEST SU PATTERN ISOLATI

def test_pattern(group):

    spikes = np.zeros((NUM_STEPS, N_INPUTS))

    start = 0 if group == "A" else 4

    for i in range(4):
        spikes[4 + i, start + i] = 1

    return spikes


for group in ["A", "B"]:

    pattern = test_pattern(group)

    _, before = simulate(
        pattern,
        initial_weights.copy(),
        learning=False
    )

    _, after = simulate(
        pattern,
        weights.copy(),
        learning=False
    )

    print(f"\nPattern {group}")
    print("Spike prima:", sum(before))
    print("Spike dopo:", sum(after))


# 6. RISULTATI E GRAFICO

print("\nPesi iniziali:", initial_weights)
print("Pesi finali:", weights.round(4))

history = np.array(history)

plt.figure(figsize=(10, 5))

for i in range(N_INPUTS):

    plt.plot(
        history[:, i],
        label=f"Synapse {i + 1}",
        color="tab:blue" if i < 4 else "tab:orange",
        linestyle=["-", "--", "-.", ":"][i % 4]
    )

plt.xlabel("Epoch")
plt.ylabel("Synaptic weight")
plt.title("Unsupervised STDP learning")
plt.legend()
plt.ylim(0, 1)
plt.tight_layout()

Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/stdp_pattern_learning.png", dpi=200)

plt.show()
