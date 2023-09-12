# Import monkdata.py file as a module
import monkdata as m
import dtree

print("-----START-----")

# ---- ASSIGNEMENT 0 -----
print("\n### Assignement 0")

# ---- ASSIGNEMENT 1 -----
print("\n### Assignement 1")

# Calculate the entropy of each training datasets
entropy = [dtree.entropy(m.monk1), dtree.entropy(m.monk2), dtree.entropy(m.monk3)]
for i in range(len(entropy)):
    print(f"Entropy of monk{i+1}: {entropy[i]}")

# ---- ASSIGNEMENT 2 -----
print("\n### Assignement 2")
print("some remarks on values of entropies")

# ---- ASSIGNEMENT 3 -----
print("\n### Assignement 3")
information_gain = []
information_gain_monk1 = []
information_gain_monk2 = []
information_gain_monk3 = []

# Calculate the information gain for each attribute on each dataset
for attribute in m.attributes:
    information_gain_monk1.append(dtree.averageGain(m.monk1, attribute))
    information_gain_monk2.append(dtree.averageGain(m.monk2, attribute))
    information_gain_monk3.append(dtree.averageGain(m.monk3, attribute))
information_gain.append(information_gain_monk1)
information_gain.append(information_gain_monk2)
information_gain.append(information_gain_monk3)

# Display information gain for each attribute on each dataset
for i in range(len(information_gain)):
    print(f"--- dataset monk{i+1} Information gain ---")
    for j in range(len(information_gain[i])):
        print(f"A{j+1}: {information_gain[i][j]}")

# ---- ASSIGNEMENT 4 -----
print("\n### Assignement 4")
print("some analysis on values of information gain for datasets")

# ---- ASSIGNEMENT 5 -----
print("\n### Assignement 5")

# ---- ASSIGNEMENT 6 -----
print("\n### Assignement 6")

print("\n------END------")
