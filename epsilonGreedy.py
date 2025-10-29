import numpy as np

class BanditEnvironment:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.num_arms = len(probabilities)
    
    def pull(self, arm):
        # Return 1 with given arm probability, else 0
        return 1 if np.random.rand() < self.probabilities[arm] else 0


class EpsilonGreedy:
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms)  # Number of times each arm is selected
        self.values = np.zeros(num_arms)  # Estimated value of each arm
    
    def select_arm(self):
        # With probability epsilon, explore; otherwise exploit
        return np.random.randint(self.num_arms) if np.random.rand() < self.epsilon else return np.argmax(self.values)
        
    def update(self, arm, reward):
        # Increment the count for this arm
        self.counts[arm] += 1
        # Calculate running average
        n = self.counts[arm]
        value = self.values[arm]
        new_value = value + (1 / n) * (reward - value)
        self.values[arm] = new_value


def simulate(env, agent, num_pulls):
    rewards = np.zeros(num_pulls)
    for i in range(num_pulls):
        arm = agent.select_arm()
        reward = env.pull(arm)
        agent.update(arm, reward)
        rewards[i] = reward
    return rewards


# Example usage:
probabilities = [0.1, 0.5, 0.8]  # True reward probabilities for each arm
env = BanditEnvironment(probabilities)
epsilon = 0.1
agent = EpsilonGreedy(num_arms=len(probabilities), epsilon=epsilon)
num_pulls = 1000

rewards = simulate(env, agent, num_pulls)
print(f"Total reward after {num_pulls} pulls: {np.sum(rewards)}")
print(f"Estimated values of each arm: {agent.values}")
print(f"Number of times each arm was selected: {agent.counts}")
