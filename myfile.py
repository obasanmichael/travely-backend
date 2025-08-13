import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

# Define the budget range (universe of discourse)
x = np.arange(0, 50001, 1)

# Define membership functions
low = fuzz.trapmf(x, [0, 0, 12000, 18000])
medium = fuzz.trimf(x, [15000, 25000, 35000])
high = fuzz.trapmf(x, [32000, 40000, 50000, 50000])

# Plot the membership functions
plt.figure(figsize=(10, 6))
plt.plot(x, low, 'b', linewidth=2, label='Low')
plt.plot(x, medium, 'g', linewidth=2, label='Medium')
plt.plot(x, high, 'r', linewidth=2, label='High')

plt.title('Fuzzy Membership Functions for Daily Budget Classification')
plt.xlabel('Daily Budget (₦)')
plt.ylabel('Degree of Membership')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig("myfile.png")
plt.show()

