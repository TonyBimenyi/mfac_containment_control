import numpy as np
import matplotlib.pyplot as plt

#Define parameters

rho = 0.3
lambda_ = 0.5
eta = 0.5
mu = 1
epsilon = 10^-4

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

w5 = np.zeros((L + 1, 1))
w6 = np.zeros((L + 1, 1))

e1 = np.zeros((L + 1, 1))
e2 = np.zeros((L + 1, 1))
e3 = np.zeros((L + 1, 1))
e4 = np.zeros((L + 1, 1))

si1 = np.zeros((L, 1))
si2 = np.zeros((L, 1))
si3 = np.zeros((L, 1))
si4 = np.zeros((L, 1))

# Generate k values
k_values = np.arange(L + 1)

w5 = np.piecewise(k_values, 
                         [k_values < 165, 
                          (k_values >= 165) & (k_values < 330), 
                          k_values >= 330], 
                         [1.4, 1.6, 1.3])

#Simulation Loop

for k in range(1, L-1):
    if k == 0:
        phi1[0] = 1
        phi2[0] = 1
        phi3[0] = 1
        phi4[0] = 1
    elif k == 1:
        phi1[k] = phi1[k - 1] + eta * u1[k - 1] / (mu + u1[k - 1]**2) * (y1[k] - y1[k - 1] - phi1[k - 1] * u1[k - 1])
        phi2[k] = phi2[k - 1] + eta * u2[k - 1] / (mu + u2[k - 1]**2) * (y2[k] - y2[k - 1] - phi2[k - 1] * u2[k - 1])
        phi3[k] = phi3[k - 1] + eta * u3[k - 1] / (mu + u3[k - 1]**2) * (y3[k] - y3[k - 1] - phi3[k - 1] * u3[k - 1])
        phi4[k] = phi4[k - 1] + eta * u4[k - 1] / (mu + u4[k - 1]**2) * (y4[k] - y4[k - 1] - phi4[k - 1] * u4[k - 1])
    else:
        phi1[k] = phi1[k - 1] + (eta * (u1[k - 1] - u1[k - 2]) / (mu + (abs(u1[k - 1] - u1[k - 2]))**2)) * (y1[k] - y1[k - 1] - phi1[k - 1] * (u1[k - 1] - u1[k - 2]))
        phi2[k] = phi2[k - 1] + (eta * (u2[k - 1] - u2[k - 2]) / (mu + (abs(u2[k - 1] - u2[k - 2]))**2)) * (y2[k] - y2[k - 1] - phi2[k - 1] * (u2[k - 1] - u2[k - 2]))
        phi3[k] = phi3[k - 1] + (eta * (u3[k - 1] - u3[k - 2]) / (mu + (abs(u3[k - 1] - u3[k - 2]))**2)) * (y3[k] - y3[k - 1] - phi3[k - 1] * (u3[k - 1] - u3[k - 2]))
        phi4[k] = phi4[k - 1] + (eta * (u4[k - 1] - u4[k - 2]) / (mu + (abs(u4[k - 1] - u4[k - 2]))**2)) * (y4[k] - y4[k - 1] - phi4[k - 1] * (u4[k - 1] - u4[k - 2]))

    
    si1[k] = -y1[k] + w5[k] + y2[k] + y3[k]
    si2[k] = y1[k] - y2[k] + y3[k] + y4[k]
    si3[k] = y1[k] + y2[k] - 2 * y3[k] + y4[k]
    si4[k] = y2[k] + y3[k] -y4[k] + w6[k]

    if k == 1:
        u1[0] = 0
        u2[0] = 0
        u3[0] = 0
        u4[0] = 0
    else:
        u1[k] = u1[k - 1] + (rho * phi1[k]) / (lambda_ + abs(phi1[k])**2) * si1[k]
        u2[k] = u2[k - 1] + (rho * phi2[k]) / (lambda_ + abs(phi2[k])**2) * si2[k]
        u3[k] = u3[k - 1] + (rho * phi3[k]) / (lambda_ + abs(phi3[k])**2) * si3[k]
        u4[k] = u4[k - 1] + (rho * phi4[k]) / (lambda_ + abs(phi4[k])**2) * si4[k]

    y1[0] = 0
    y2[0] = 0
    y3[0] = 0
    y4[0] = 0

    y1[k+1] = 0.7 * y1[k] + u1[k]
    y2[k+1] = 0.6 * np.cos(y2[k]) + 0.3 * y2[k] + u2[k]
    y3[k+1] = (y3[k] / 1 + y3[k]^2) + u4[k]
    y4[k+1] = (y3[k] / 1 + y3[k]^2) + u4[k]^3



plt.plot(k_values, w5, label="w5(k)", color='b', drawstyle='steps-post')
plt.xlabel('k')
plt.ylabel('w5(k)')
plt.title('Piecewise Constant Function w5(k)')
plt.grid(True)
plt.legend()
plt.show()

