
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BETA = 0.8
THRESHOLD = 1.0

A_PLUS = 0.025
A_MINUS = 0.030
TAU = 10

INITIAL_WEIGHT = 0.5
N_PAIRS = 20
NUM_STEPS = 30


# 1. MODELLO LIF

def lif_step(mem, current):

    mem = mem * BETA + current

    spike = int(mem >= THRESHOLD)

    if spike:
        mem = 0.0
    return mem, spike


# 2. REGOLA STDP

def stdp_update(delta_t):

    if delta_t > 0:
        return A_PLUS*np.exp(-delta_t / TAU)

    elif delta_t < 0:
        return -A_MINUS*np.exp(delta_t / TAU)

    return 0.0


# 3. TRAINING DELLA SINAPSI

def train_synapse(pre_time, post_time):

    weight = INITIAL_WEIGHT
    history = [weight]

    for _ in range(N_PAIRS):

        mem_pre = 0.0
        mem_post = 0.0

        t_pre = None
        t_post = None

        for t in range(NUM_STEPS):

            # Stimoli esterni
            current_pre = 1.1 if t == pre_time else 0
            current_post = 1.1 if t == post_time else 0

            mem_pre, spike_pre = lif_step(mem_pre,current_pre)

            current_post +=  weight * spike_pre

            mem_post, spike_post = lif_step(mem_post,current_post)

            # Registrazione degli istanti di spike
            if spike_pre:
                t_pre = t

            if spike_post:
                t_post = t

        if t_pre is not None and t_post is not None:

            delta_t = t_post - t_pre

            weight += stdp_update(delta_t)

            weight = np.clip(weight, 0, 1)

        history.append(weight)

    return weight, history


# 4. TRAINING NELLE DUE CONDIZIONI

weight_ltp, history_ltp = train_synapse(5, 10)

weight_ltd, history_ltd = train_synapse(10, 5)


# 5. TEST SENZA PLASTICITÀ

def test_synapse(weight):

    mem_pre = 0.0
    mem_post = 0.0

    spikes_pre = []
    spikes_post = []

    for t in range(NUM_STEPS):

        # Corrente costante al neurone pre
        mem_pre, spike_pre = lif_step(mem_pre, 0.6)

        current_post =  weight * spike_pre

        mem_post, spike_post = lif_step(mem_post, current_post)

        spikes_pre.append(spike_pre)
        spikes_post.append(spike_post)

    return spikes_pre, spikes_post


pre, post_initial = test_synapse(INITIAL_WEIGHT)
_, post_ltp = test_synapse(weight_ltp)
_, post_ltd = test_synapse(weight_ltd)


# 6. RISULTATI

print(f"Peso iniziale: {INITIAL_WEIGHT:.4f}")
print(f"Peso LTP: {weight_ltp:.4f}")
print(f"Peso LTD: {weight_ltd:.4f}")

print("\nSpike durante il test:")

print("Presinaptico:", sum(pre))
print("Post iniziale:", sum(post_initial))
print("Post LTP:", sum(post_ltp))
print("Post LTD:", sum(post_ltd))


# 7. GRAFICI

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history_ltp, label="LTP", color="tab:blue")
axes[0].plot(history_ltd, label="LTD", color="tab:orange")

axes[0].set_xlabel("Spike pairs")
axes[0].set_ylabel("Synaptic weight")
axes[0].set_title("Synaptic learning")
axes[0].legend()

axes[1].eventplot([
    np.where(pre)[0],
    np.where(post_initial)[0],
    np.where(post_ltp)[0],
    np.where(post_ltd)[0]
])

axes[1].set_yticks(range(4))

axes[1].set_yticklabels([
    "Pre",
    "Post initial",
    "Post LTP",
    "Post LTD"
])

axes[1].set_xlabel("Time step")
axes[1].set_title("Response after learning")

fig.tight_layout()

Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/stdp_lif.png", dpi=200)

plt.show()
