
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

A_PLUS = 0.025
A_MINUS = 0.030

TAU = 10
TAU_E = 50

ALPHA = np.exp(-1 / TAU)
ALPHA_E = np.exp(-1 / TAU_E)

ETA = 1.0
INITIAL_WEIGHT = 0.5


def simulate(pre_time, post_time,
             reward_delay, reward_value):

    weight = INITIAL_WEIGHT

    pre_trace = 0.0
    post_trace = 0.0

    # Nuova variabile
    eligibility = 0.0

    eligibility_history = []
    weight_history = []

    reward_time = max(
        pre_time, post_time
    ) + reward_delay

    for t in range(reward_time + 30):

        spike_pre = int(t == pre_time)
        spike_post = int(t == post_time)

        reward = (
            reward_value if t == reward_time
            else 0.0
        )

        pre_trace = pre_trace * ALPHA  
        post_trace = post_trace * ALPHA 

        ltp = A_PLUS * spike_post * pre_trace   
        ltd = - A_MINUS * spike_pre * post_trace

        stdp = ltp + ltd

        eligibility = ALPHA_E*eligibility + stdp

        # aggiornamento del peso
        # Utilizza ETA, reward ed eligibility
        weight += ETA * reward * eligibility

        weight = np.clip(weight, 0, 1)

        pre_trace += spike_pre
        post_trace += spike_post

        eligibility_history.append(eligibility)
        weight_history.append(weight)

    return (
        weight,
        eligibility_history,
        weight_history
    )


# 1. EFFETTO DEL RITARDO

delays = [1, 10, 25, 50, 100]

final_weights = []

for delay in delays:

    weight, _, _ = simulate(
        pre_time=5,
        post_time=10,
        reward_delay=delay,
        reward_value=1
    )

    final_weights.append(weight)

    print(
        f"Delay={delay}: "
        f"Final weight={weight:.5f}"
    )


# 2. EFFETTO DEL SEGNO DELLA RICOMPENSA

reward_values = [-1, 0, 1]

histories = []

for reward in reward_values:

    weight, eligibility, history = simulate(
        pre_time=5,
        post_time=10,
        reward_delay=25,
        reward_value=reward
    )

    histories.append(history)

    print(
        f"Reward={reward}: "
        f"Final weight={weight:.5f}"
    )


# 3. GRAFICI

fig, axes = plt.subplots(
    1, 3, figsize=(15, 4)
)

axes[0].plot(eligibility)
axes[0].axvline(
    35, linestyle="--", color="gray"
)
axes[0].set_title("Eligibility trace")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Eligibility")

axes[1].plot(
    delays,
    final_weights,
    marker="o"
)
axes[1].axhline(
    INITIAL_WEIGHT,
    linestyle="--",
    color="gray"
)
axes[1].set_title("Effect of reward delay")
axes[1].set_xlabel("Reward delay")
axes[1].set_ylabel("Final weight")

for reward, history in zip(
    reward_values, histories
):

    axes[2].plot(
        history,
        label=f"Reward {reward}"
    )

axes[2].set_title("Reward-modulated learning")
axes[2].set_xlabel("Time step")
axes[2].set_ylabel("Synaptic weight")
axes[2].legend()

fig.tight_layout()

Path("figures").mkdir(exist_ok=True)

fig.savefig(
    "figures/eligibility_trace.png",
    dpi=200
)

plt.show()
