# Import monkdata.py file as a module
import monkdata as m
import dtree

print("-----START-----")

# ---- ASSIGNEMENT 1 -----
print("### Assignement 1")

# Calculate the entropy of each training datasets
print(f"Entropy of monk1: {dtree.entropy(m.monk1)}")
print(f"Entropy of monk2: {dtree.entropy(m.monk2)}")
print(f"Entropy of monk3: {dtree.entropy(m.monk3)}")


print("------END------")
