import numpy as np

# True probabilities of each arm
actual_prob = [0.1, 0.7, 0.5]

# Each arm stores [successes, failures]
success_failures = [[0, 0], [0, 0], [0, 0]]

# Number of trials
for trial in range(101):
    # Draw samples from Beta distribution for each arm
    samples = [np.random.beta(s + 1, f + 1) for s, f in success_failures]

    # Choose the arm with the highest sampled value
    best_arm = np.argmax(samples)

    # Simulate pulling the best arm
    reward = np.random.uniform() < actual_prob[best_arm]
    
    if reward:
        success_failures[best_arm][0] += 1  # Success count
    else:
        success_failures[best_arm][1] += 1  # Failure count
    
    # Print every 10 trials
    if trial % 10 == 0:
        print(f"Trial: {trial}\tSuccess/Failures: {success_failures}")
