import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Setting the font to Poppins (Ensure it's installed on your system)
# rcParams['font.family'] = 'times new roman'

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

        # si1 to si4 updates
          # si1[k] = -y1[k] + w5[k] + y2[k] 
        # si2[k] = y1[k] - y2[k] + y3[k] + y4[k]
        # si3[k] = y1[k] + y2[k] - 2 * y3[k] + y4[k]
        # si4[k] = y2[k] + y3[k] - y4[k] + w6[k]

        # si1[k] = y1[k] - y3[k] - w5[k]
        # si2[k] = - y1[k] + 2 * y2[k] - y4[k]
        # si3[k] = - y2[k] + y3[k]
        # si4[k] = - y3[k] + y4[k] -w6[k] 

        # si1[k] = -y1[k] + y2[k] -w5[k]
        # si2[k] = -y2[k] + y3[k]
        # si3[k] = y1[k] - 2 * y3[k] + y4[k]
        # si4[k] = y2[k] - y4[k] - w6[k]

        si1[k] = y2[k] - 2 * y1[k] + w5[k]
        si2[k] = y3[k] - y2[k]
        si3[k] = y4[k] - 2 * y3[k] + y1[k]
        si4[k] = y2[k] - 2* y4[k] + w6[k]

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

        y1[k+1] = 0.7 * y1[k] + u1[k]
        y2[k+1] = 0.6 * np.cos(y2[k]) + 0.3 * y2[k] + u2[k]
        y3[k+1] = (y3[k]) / (1 + y3[k]**2) + u3[k]
        y4[k+1] = (y4[k]) / (1 + y4[k]**2) + u4[k]**3

    return y1, y2, y3, y4, w5, w6

# Different values of lambda_
lambda_values = [5, 3, 2, 0.5]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for i, lambda_ in enumerate(lambda_values):
    y1, y2, y3, y4, w5, w6 = simulate(lambda_)
    
    ax = axs[i//2, i%2]
    ax.plot(y1[:-1], label="agent1", color='orange', linestyle='solid')
    ax.plot(y2[:-1], label="agent2", color='green', linestyle='solid')
    ax.plot(y3[:-1], label="agent3", color='red', linestyle='solid')
    ax.plot(y4[:-1], label="agent4", color='purple', linestyle='solid')
    ax.plot(w5, label="agent5(Leader 1)", color='blue', linestyle='dashed', drawstyle='steps-post')
    ax.plot(w6, label="agent6(Leader 2)", color='blue', linestyle='dashed', drawstyle='steps-post')
    
    ax.set_title(f"$\lambda$ = {lambda_}")
    ax.grid(True)
    ax.set_xlabel("time step", fontsize=12)
    ax.set_ylabel("outputs", fontsize=12)
    ax.legend(fontsize=10)

plt.figure()

plt.tight_layout()
plt.show()
