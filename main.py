
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt




ENV_NAME = "Taxi-v4"

TRAINING_EPISODES = 20000
MAX_STEPS = 200

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99

EPSILON = 1.0
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.9995

EVALUATION_EPISODES = 100



env = gym.make(ENV_NAME)

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

# Each row represents a state.
# Each column represents an action.
q_table = np.zeros(
    (number_of_states, number_of_actions)
)

print("Taxi environment created.")
print("States:", number_of_states)
print("Actions:", number_of_actions)
print("Q-table:", q_table.shape)



training_rewards = []
training_steps = []

for episode in range(TRAINING_EPISODES):

    # Start a new taxi journey.
    state, info = env.reset()

    total_reward = 0
    steps = 0

    for step in range(MAX_STEPS):

        # Choose whether to explore or exploit.
        if np.random.random() < EPSILON:

            # Explore: take a random action.
            action = env.action_space.sample()

        else:

            # Exploit: choose the best learned action.
            action = int(np.argmax(q_table[state]))

        # Perform the chosen action.
        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        # Find the estimated future reward.
        if terminated or truncated:
            future_reward = 0
        else:
            future_reward = np.max(q_table[next_state])

        # Q-learning formula.
        old_value = q_table[state, action]

        new_value = old_value + LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR * future_reward
            - old_value
        )

        # Update the Q-table.
        q_table[state, action] = new_value

        # Update the current state.
        state = next_state

        total_reward += reward
        steps += 1

        # Stop if the journey ends.
        if terminated or truncated:
            break

    training_rewards.append(total_reward)
    training_steps.append(steps)

    # Gradually reduce random exploration.
    EPSILON = max(
        MIN_EPSILON,
        EPSILON * EPSILON_DECAY
    )

    # Show training progress.
    if (episode + 1) % 1000 == 0:

        recent_rewards = training_rewards[-1000:]
        recent_steps = training_steps[-1000:]

        print(
            f"Episode {episode + 1} | "
            f"Average reward: {np.mean(recent_rewards):.2f} | "
            f"Average steps: {np.mean(recent_steps):.2f} | "
            f"Epsilon: {EPSILON:.3f}"
        )

env.close()




np.save("q_table.npy", q_table)

print("\nTraining finished!")
print("Q-table saved.")




evaluation_env = gym.make(ENV_NAME)

evaluation_rewards = []
evaluation_steps = []
successful_trips = 0

for episode in range(EVALUATION_EPISODES):

    state, info = evaluation_env.reset()

    total_reward = 0
    steps = 0

    for step in range(MAX_STEPS):

        # During evaluation, use the learned policy.
        action = int(np.argmax(q_table[state]))

        state, reward, terminated, truncated, info = (
            evaluation_env.step(action)
        )

        total_reward += reward
        steps += 1

        if terminated or truncated:

            if terminated:
                successful_trips += 1

            break

    evaluation_rewards.append(total_reward)
    evaluation_steps.append(steps)

evaluation_env.close()

average_reward = np.mean(evaluation_rewards)
average_steps = np.mean(evaluation_steps)

success_rate = (
    successful_trips / EVALUATION_EPISODES
) * 100

print("\n========== RESULTS ==========")
print("Average reward:", round(average_reward, 2))
print("Average steps:", round(average_steps, 2))
print("Successful trips:", successful_trips)
print("Success rate:", round(success_rate, 2), "%")



window = 500

rewards = np.array(training_rewards)

if len(rewards) >= window:

    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        training_rewards,
        alpha=0.2,
        label="Episode reward"
    )

    plt.plot(
        np.arange(window - 1, len(rewards)),
        moving_average,
        label="500-episode average",
        linewidth=2
    )

    plt.title("Taxi Q-learning Training Curve")
    plt.xlabel("Training episode")
    plt.ylabel("Total reward")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig("learning_curve.png")

    print("\nGraph saved as learning_curve.png")

    plt.show()


\

demo_env = gym.make(
    ENV_NAME,
    render_mode="ansi"
)

state, info = demo_env.reset()

print("\n========== TAXI DEMO ==========")

print(demo_env.render())

demo_reward = 0

for step in range(MAX_STEPS):

    # Choose the highest-value action.
    action = int(np.argmax(q_table[state]))

    state, reward, terminated, truncated, info = (
        demo_env.step(action)
    )

    demo_reward += reward

    print(
        f"\nStep {step + 1}, "
        f"Action: {action}, "
        f"Reward: {reward}"
    )

    print(demo_env.render())

    if terminated or truncated:
        break

print("\nDemo reward:", demo_reward)

demo_env.close()
