import numpy as np

class UCB:
    def __init__(self, n_arms, c=2):
        self.n_arms = n_arms
        self.c = c  # Exploration parameter
        self.counts = np.zeros(n_arms)  # Number of times each arm was chosen
        self.values = np.zeros(n_arms)  # Estimated value of each arm

    def select_arm(self):
        total_counts = np.sum(self.counts)

        # Explore each arm at least once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        # Apply UCB formula
        ucb_values = self.values + self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        # Increment the count for the chosen arm
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]

        # Update the estimated value using incremental mean
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n


def main():
    n_arms, n_trials = 10, 1000
    true_means = np.random.rand(n_arms)  # True reward probabilities (unknown to agent)
    ucb = UCB(n_arms)
    rewards = np.zeros(n_trials)

    print("True reward probabilities for each arm (hidden from UCB):")
    print(true_means)

    for t in range(n_trials):
        arm = ucb.select_arm()
        # Reward is drawn from a distribution centered at the true mean
        reward = np.random.randn() * 0.1 + true_means[arm]  # Added slight noise
        ucb.update(arm, reward)
        rewards[t] = reward

    print("\nTotal reward after", n_trials, "trials:", np.sum(rewards))
    print("Estimated values of each arm:", ucb.values)
    print("Number of times each arm was selected:", ucb.counts)
    print("Best estimated arm:", np.argmax(ucb.values))
    print("True best arm:", np.argmax(true_means))


if __name__ == "__main__":
    main()
