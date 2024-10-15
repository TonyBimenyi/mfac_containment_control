import numpy as np
import matplotlib.pyplot as plt

# Define parameters
rho = 0.3
lambda_ = 0.5
eta = 0.5
mu = 1
epsilon = 0.0001
L = 500

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

e1 = np.zeros((L + 1, 1))
e2 = np.zeros((L + 1, 1))
e3 = np.zeros((L + 1, 1))
e4 = np.zeros((L + 1, 1))

si1 = np.zeros((L, 1))
si2 = np.zeros((L, 1))
si3 = np.zeros((L, 1))
si4 = np.zeros((L, 1))

# Initialize w5 with zeros for 400 time steps
w5 = np.zeros(L)

# Set w5 values according to the conditions
w5[:165] = 1.4      # for k < 165
w5[165:330] = 1.6   # for 165 ≤ k < 330
w5[330:] = 1.3      # for k ≥ 330

# Initialize w6 with zeros for 400 time steps
w6 = np.zeros(L)

# Set w6 values according to the conditions
w6[:165] = 0.7      # for k < 165
w6[165:330] = 1.2   # for 165 ≤ k < 330
w6[330:] = 1.1      # for k ≥ 330


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


    if k==1:
        si1[1] = 0
        si2[1] = 0
        si3[1] = 0
        si4[1] = 0

    else:

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

# Plot w5(k), w6(k), y1(k), y2(k), y3(k), and y4(k)
plt.figure(figsize=(8, 6))


plt.plot( y1[:-1], label="agent1", color='orange', linestyle='solid')
plt.plot( y2[:-1], label="agent2", color='green', linestyle='solid')
plt.plot( y3[:-1], label="agent3", color='red', linestyle='solid')
plt.plot( y4[:-1], label="agent4", color='purple', linestyle='solid')
plt.plot( w5, label="agent5(Leader 1)", color='blue', linestyle='dashed', drawstyle='steps-post')
plt.plot( w6, label="agent6(Leader 2)", color='blue', linestyle='dashed', drawstyle='steps-post')

plt.grid(True)
plt.xlabel("time step", fontsize=12)
plt.ylabel("outputs", fontsize=12)
plt.legend(fontsize=10)
plt.show()
