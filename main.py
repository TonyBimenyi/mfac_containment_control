import numpy as np
import matplotlib.pyplot as plt

# Define other parameters
rho = 0.3
eta = 0.5
mu = 1
epsilon = 0.0001
L = 500

# Function to simulate the system
def simulate(lambda_):
    # Initialize arrays
    phi1 = np.zeros((L, 1))
    phi2 = np.zeros((L, 1))
    phi3 = np.zeros((L, 1))
    phi4 = np.zeros((L, 1))

    u1 = np.zeros((L, 1))
    u2 = np.zeros((L, 1))
    u3 = np.zeros((L, 1))
    u4 = np.zeros((L, 1))

    y1 = np.zeros((L + 1, 1))
    y2 = np.zeros((L + 1, 1))
    y3 = np.zeros((L + 1, 1))
    y4 = np.zeros((L + 1, 1))

    w5 = np.zeros((L, 1))
    w6 = np.zeros((L, 1))

    si1 = np.zeros((L, 1))
    si2 = np.zeros((L, 1))
    si3 = np.zeros((L, 1))
    si4 = np.zeros((L, 1))

    # Set w5 and w6 according to conditions
    w5[:165] = 1.4
    w5[165:330] = 1.6
    w5[330:] = 1.3

    w6[:165] = 0.7
    w6[165:330] = 1.2
    w6[330:] = 1.1

    # Simulation loop
    for k in range(1, L-1):
        if k == 1:  
            phi1[1] = 0.5
            phi2[1] = 0.5
            phi3[1] = 0.5
            phi4[1] = 0.5       
        elif k == 2:
            phi1[k] = phi1[k - 1] + eta * u1[k - 1] / (mu + u1[k - 1]**2) * (y1[k] - y1[k - 1] - phi1[k - 1] * u1[k - 1])
            phi2[k] = phi2[k - 1] + eta * u2[k - 1] / (mu + u2[k - 1]**2) * (y2[k] - y2[k - 1] - phi2[k - 1] * u2[k - 1])
            phi3[k] = phi3[k - 1] + eta * u3[k - 1] / (mu + u3[k - 1]**2) * (y3[k] - y3[k - 1] - phi3[k - 1] * u3[k - 1])
            phi4[k] = phi4[k - 1] + eta * u4[k - 1] / (mu + u4[k - 1]**2) * (y4[k] - y4[k - 1] - phi4[k - 1] * u4[k - 1])
        else:
            phi1[k] = phi1[k - 1] + (eta * (u1[k - 1] - u1[k - 2]) / (mu + (abs(u1[k - 1] - u1[k - 2]))**2)) * (y1[k] - y1[k - 1] - phi1[k - 1] * (u1[k - 1] - u1[k - 2]))
            phi2[k] = phi2[k - 1] + (eta * (u2[k - 1] - u2[k - 2]) / (mu + (abs(u2[k - 1] - u2[k - 2]))**2)) * (y2[k] - y2[k - 1] - phi2[k - 1] * (u2[k - 1] - u2[k - 2]))
            phi3[k] = phi3[k - 1] + (eta * (u3[k - 1] - u3[k - 2]) / (mu + (abs(u3[k - 1] - u3[k - 2]))**2)) * (y3[k] - y3[k - 1] - phi3[k - 1] * (u3[k - 1] - u3[k - 2]))
            phi4[k] = phi4[k - 1] + (eta * (u4[k - 1] - u4[k - 2]) / (mu + (abs(u4[k - 1] - u4[k - 2]))**2)) * (y4[k] - y4[k - 1] - phi4[k - 1] * (u4[k - 1] - u4[k - 2]))

        si1[k] = y2[k] - 2 * y1[k] + w5[k]
        si2[k] = y3[k] - y2[k]
        si3[k] = y4[k] - 2 * y3[k] + y1[k]
        si4[k] = y2[k] - 2 * y4[k] + w6[k]

        if k == 1:
            u1[1] = 0
            u2[1] = 0
            u3[1] = 0
            u4[1] = 0
        else:
            u1[k] = u1[k - 1] + (rho * phi1[k]) / (lambda_ + abs(phi1[k])**2) * si1[k]
            u2[k] = u2[k - 1] + (rho * phi2[k]) / (lambda_ + abs(phi2[k])**2) * si2[k]
            u3[k] = u3[k - 1] + (rho * phi3[k]) / (lambda_ + abs(phi3[k])**2) * si3[k]
            u4[k] = u4[k - 1] + (rho * phi4[k]) / (lambda_ + abs(phi4[k])**2) * si4[k]

        y1[1] = 1.2
        y2[1] = 1.8
        y3[1] = 0.6
        y4[1] = 0.4

        y1[k + 1] = 0.7 * y1[k] + u1[k]
        y2[k + 1] = 0.6 * np.cos(y2[k]) + 0.3 * y2[k] + u2[k]
        y3[k + 1] = (y3[k]) / (1 + y3[k]**2) + u3[k]
        y4[k + 1] = (y4[k]) / (1 + y4[k]**2) + u4[k]**3

    return y1, y2, y3, y4, w5, w6, u1, u2, u3, u4

# Simulate for multiple lambda values for outputs and inputs
lambda_values = [5, 3, 2, 0.5]
outputs = {}
for lambda_value in lambda_values:
    y1, y2, y3, y4, w5, w6, u1, u2, u3, u4 = simulate(lambda_value)
    outputs[lambda_value] = (y1, y2, y3, y4, w5, w6, u1, u2, u3, u4)

# Plot y1 to y4 for each lambda value (Outputs)
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i, (lambda_value, (y1, y2, y3, y4, w5, w6, _, _, _, _)) in enumerate(outputs.items()):
    ax = axs[i // 2, i % 2]
    ax.plot(y1[:-1], label="agent1", color='orange', linestyle='solid')
    ax.plot(y2[:-1], label="agent2", color='green', linestyle='solid')
    ax.plot(y3[:-1], label="agent3", color='red', linestyle='solid')
    ax.plot(y4[:-1], label="agent4", color='purple', linestyle='solid')
    ax.plot(w5, label="agent5 (Leader 1)", color='blue', linestyle='dashed', drawstyle='steps-post')
    ax.plot(w6, label="agent6 (Leader 2)", color='blue', linestyle='dashed', drawstyle='steps-post')
    
    ax.set_title(f"Outputs for $\lambda$ = {lambda_value}")
    ax.grid(True)
    ax.set_xlabel("time step", fontsize=12)
    ax.set_ylabel("outputs", fontsize=12)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()

# Plot u1 to u4 for each lambda value (Inputs)
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
for i, (lambda_value, (_, _, _, _, _, _, u1, u2, u3, u4)) in enumerate(outputs.items()):
    ax = axs[i // 2, i % 2]
    ax.plot(u1, label="u1", color='orange', linestyle='solid')
    ax.plot(u2, label="u2", color='green', linestyle='solid')
    ax.plot(u3, label="u3", color='red', linestyle='solid')
    ax.plot(u4, label="u4", color='purple', linestyle='solid')

    ax.set_title(f"Inputs for $\lambda$ = {lambda_value}")
    ax.grid(True)
    ax.set_xlabel("time step", fontsize=12)
    ax.set_ylabel("inputs", fontsize=12)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.show()
