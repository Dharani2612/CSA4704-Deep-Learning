
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Actual weather distribution
P = np.array([0.5, 0.3, 0.2])

# Two prediction models
Q1 = np.array([0.6, 0.25, 0.15])
Q2 = np.array([0.3, 0.4, 0.3])

H = entropy(P, base=2)

CE1 = -np.sum(P*np.log2(Q1))
CE2 = -np.sum(P*np.log2(Q2))

KL1 = entropy(P, Q1, base=2)
KL2 = entropy(P, Q2, base=2)

print("Entropy:", H)
print("Cross Entropy Model 1:", CE1)
print("Cross Entropy Model 2:", CE2)
print("KL Model 1:", KL1)
print("KL Model 2:", KL2)

print("Verification 1:", H+KL1)
print("Verification 2:", H+KL2)

# KL divergence asymmetry
print("KL(P||Q1):", entropy(P, Q1))
print("KL(Q1||P):", entropy(Q1, P))

# Binary cross entropy
y_true = 1
p = np.linspace(0.01, 0.99, 100)

loss = -(y_true*np.log(p) +
         (1-y_true)*np.log(1-p))

plt.plot(p, loss)
plt.xlabel("Predicted Probability")
plt.ylabel("Binary Cross Entropy Loss")
plt.title("Binary Cross Entropy")
plt.grid()
plt.show()
