
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BETA = 0.8
THRESHOLD = 1.0

A_PLUS = 0.025
A_MINUS = 0.030

TAU = 10
TAU_E = 50

ALPHA = np.exp(-1 / TAU)
ALPHA_E = np.exp(-1 / TAU_E)

ETA = 1.0
EPSILON = 0.2

N_INPUTS = 8
N_OUTPUTS = 2

N_EPOCHS = 80
NUM_STEPS = 40

DECISION_TIME = 8
REWARD_DELAY = 25

INPUT_SCALE = 0.25

initial_weights = np.random.default_rng(1).uniform(
    0.35, 0.45,
    size=(N_OUTPUTS, N_INPUTS)
)


# 1. PATTERN

def generate_pattern(group, num_steps=NUM_STEPS):

    spikes = np.zeros((num_steps, N_INPUTS))

    start = 0 if group == 0 else 4

    for i in range(4):
        spikes[4 + i, start + i] = 1

    return spikes


# 2. AMBIENTE

def get_reward(group, action):

    # A -> azione 0
    # B -> azione 1

    return 1 if group == action else -1


# 3. SIMULAZIONE

def simulate(group, weights, rng,
             epsilon=0.0,
             reward_enabled=True,
             reward_delay=REWARD_DELAY):

    weights = weights.copy()

    reward_time = DECISION_TIME + reward_delay
    num_steps = reward_time + 1

    spikes = generate_pattern(group, num_steps)

    mem = np.zeros(N_OUTPUTS)

    pre_trace = np.zeros(N_INPUTS)
    post_trace = np.zeros(N_OUTPUTS)

    eligibility = np.zeros(
        (N_OUTPUTS, N_INPUTS)
    )

    action = -1
    outcome = 0

    for t in range(num_steps):

        spike_pre = spikes[t]
        spike_post = np.zeros(N_OUTPUTS)

        # DINAMICA LIF

        if t <= DECISION_TIME:

            # TODO 1: corrente sinaptica
            # Ricorda INPUT_SCALE
            current = INPUT_SCALE * np.dot(weights, spike_pre)

            if t == DECISION_TIME:

                # Corrente comune di decisione
                current += 1.0

                # Esplorazione
                if epsilon > 0 and rng.random() < epsilon:

                    explored_action = rng.integers(
                        N_OUTPUTS
                    )

                    current[explored_action] += 1.0

            mem = BETA * mem + current

            if t == DECISION_TIME:

                candidates = np.flatnonzero(
                    mem >= THRESHOLD
                )

                action = candidates[
                    np.argmax(mem[candidates])
                ]

                spike_post[action] = 1
                mem[:] = 0.0

                # L'ambiente valuta l'azione
                outcome = get_reward(group, action)

        # TRACCE DI ATTIVITÀ

        pre_trace = ALPHA * pre_trace 
        post_trace = ALPHA * post_trace

        # STDP

        # Utilizza np.outer()
        ltp = A_PLUS * np.outer(spike_post, pre_trace)
        ltd = -A_MINUS * np.outer(post_trace, spike_pre)

        stdp = ltp + ltd

        # ELIGIBILITY TRACE


        eligibility = ALPHA_E * eligibility + stdp

        # RICOMPENSA RITARDATA

        reward = 0.0

        if t == reward_time and reward_enabled:

            reward = outcome
        weights += ETA * reward * eligibility

        weights = np.clip(weights, 0, 1)

        pre_trace += spike_pre
        post_trace += spike_post

    return weights, action, outcome

if __name__ == "__main__":

    # 4. TRAINING

    def train(reward_enabled):

        rng = np.random.default_rng(42)

        weights = initial_weights.copy()

        history = []
        rewards = []

        for epoch in range(N_EPOCHS):

            for group in rng.permutation(2):

                weights, action, outcome = simulate(
                    group,
                    weights,
                    rng,
                    epsilon=EPSILON,
                    reward_enabled=reward_enabled
                )

                history.append(int(outcome == 1))
                rewards.append(outcome)

        return weights, history, rewards


    trained_weights, accuracy, rewards = train(
        reward_enabled=True
    )

    control_weights, control_accuracy, _ = train(
        reward_enabled=False
    )


    # 5. TEST SENZA ESPLORAZIONE E PLASTICITÀ

    def test(weights, title):

        rng = np.random.default_rng(100)

        correct = 0

        print(f"\n{title}")

        for group in range(2):

            _, action, outcome = simulate(
                group,
                weights,
                rng,
                epsilon=0.0,
                reward_enabled=False
            )

            correct += int(outcome == 1)

            print(
                f"Pattern {'AB'[group]}: "
                f"Action {action + 1}, "
                f"Reward {outcome}"
            )

        print(f"Accuracy: {correct / 2:.0%}")


    test(initial_weights, "Before training")

    test(trained_weights, "After R-STDP")

    test(control_weights, "Without reward")


    print("\nInitial weights:")
    print(initial_weights.round(3))

    print("\nTrained weights:")
    print(trained_weights.round(3))


    # 6. VISUALIZZAZIONE

    window = 20

    def moving_average(values):

        return np.convolve(
            values,
            np.ones(window) / window,
            mode="valid"
        )


    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4),
        layout="constrained"
    )

    for values, label in [
        (accuracy, "R-STDP"),
        (control_accuracy, "Without reward")
    ]:

        curve = moving_average(values)

        axes[0].plot(
            np.arange(window, len(values) + 1),
            curve,
            label=label
        )

    axes[0].set_title("Learning performance")
    axes[0].set_xlabel("Trial")
    axes[0].set_ylabel("Moving accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    for ax, matrix, title in zip(
        axes[1:],
        [initial_weights, trained_weights],
        ["Before training", "After R-STDP"]
    ):

        im = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto"
        )

        ax.set_title(title)

        ax.set_xticks(range(8))
        ax.set_xticklabels([
            "A1", "A2", "A3", "A4",
            "B1", "B2", "B3", "B4"
        ])

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["N1", "N2"])

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
        im, ax=axes[1:],
        label="Synaptic weight"
    )

    Path("figures").mkdir(exist_ok=True)

    fig.savefig(
        "figures/reward_modulated_stdp.png",
        dpi=200
    )

    plt.show()
