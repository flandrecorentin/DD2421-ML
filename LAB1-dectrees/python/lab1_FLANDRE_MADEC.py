# Import monkdata.py file as a module
import monkdata as m
import dtree
import random

# ---- FUNCTIONS USED IN LAB1 ----
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

# --------------------------
# --------- START ----------
# --------------------------

print("-----START-----")

# ---- ASSIGNEMENT 0 -----
print("\n### Assignement 0")
print("some explication about the monk2 dataset less adapted to decision trees algos")

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

# Personnal draw of the decision tree for monk1 on first two levels
# A5 is the attribute with the higher information gain -> {1, 2, 3, 4}
attribut_first_level = information_gain_monk1.index(max(information_gain_monk1))
tree_monk1 = []
for i in range(4):
    tree_monk1.append(dtree.select(m.monk1, m.attributes[4], i+1))

# Split (second level) using the attribute with the higher gain information
attributes_second_level = []
information_gain_second_level = []
for i in range(len(tree_monk1)):
    information_gain_subset = []
    for attribute in m.attributes:
        information_gain_subset.append(dtree.averageGain(tree_monk1[i], attribute))
    attribut_second_level = information_gain_subset.index(max(information_gain_subset))
    information_gain_second_level.append(max(information_gain_subset))
    attributes_second_level.append(attribut_second_level)
    second_subset = []
    for j in range(len(m.attributes[attribut_second_level].values)):
        second_subset.append(dtree.select(tree_monk1[i], m.attributes[attribut_second_level],j+1))
    tree_monk1[i] = second_subset

# Display personnal tree 
tree_monk1_str = "--Personnal monk1 tree: "
tree_monk1_str += "A" + str(attribut_first_level+1) + "("
index_second_level = 0
for level1 in tree_monk1:
    print(information_gain_second_level[index_second_level])
    if information_gain_second_level[index_second_level]==0:
        tree_monk1_str += "+" if dtree.mostCommon(level1[1]) else "-"
    else:
        tree_monk1_str += "A" + str(attributes_second_level[index_second_level]+1) + "("
        for level2 in level1:
            tree_monk1_str += "+" if dtree.mostCommon(level2) else "-"
        tree_monk1_str += ")"
    index_second_level +=1
tree_monk1_str += ")"
print(tree_monk1_str)

# Display the ID3 tree (depth = 2)
print(f"--ID3 monk1 tree: {dtree.buildTree(m.monk1, m.attributes, 2)}\n")


# Build all tree using dtree on training set
print("--Check validity on training set:")
t_monk1 = dtree.buildTree(m.monk1, m.attributes)
print(f"t_monk1: {dtree.check(t_monk1, m.monk1)}")
t_monk2 = dtree.buildTree(m.monk2, m.attributes)
print(f"t_monk2: {dtree.check(t_monk2, m.monk2)}")
t_monk3 = dtree.buildTree(m.monk3, m.attributes)
print(f"t_monk3: {dtree.check(t_monk3, m.monk3)}")

# Build all tree using dtree on testing set
print("--Check validity on testing set:")
t_monk1_test = dtree.buildTree(m.monk1test, m.attributes)
print(f"t_monk1_test: {dtree.check(t_monk1, m.monk1test)} | {1-dtree.check(t_monk1, m.monk1test)}")
t_monk2_test = dtree.buildTree(m.monk2test, m.attributes)
print(f"t_monk2_test: {dtree.check(t_monk2, m.monk2test)} | {1-dtree.check(t_monk2, m.monk2test)}")
t_monk3_test = dtree.buildTree(m.monk3test, m.attributes)
print(f"t_monk3_test: {dtree.check(t_monk3, m.monk3test)} | {1-dtree.check(t_monk3, m.monk3test)}")

# ---- ASSIGNEMENT 6 -----
print("\n### Assignement 6")
print("some explanation of pruning from a bias variance trade-off perspective")

# ---- ASSIGNEMENT 7 -----
print("\n### Assignement 7")

# monk1train, monk1val = partition(m.monk1, 0.6)
print(f"{dtree.buildTree(m.monk1test, m.attributes)}")
print(f"{len(dtree.allPruned(dtree.buildTree(m.monk1test, m.attributes)))}Pruned tree possibles:")
for pruned in dtree.allPruned(dtree.buildTree(m.monk1test, m.attributes)):
    print(pruned)


fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
monk1train, monk1val = partition(m.monk1, 0.3)
bestTree = None
scoreBestTree = 0
numberTry = 10
# for i in range(numberTry):


    
# for pruned_tree in dtree.allPruned(dtree.buildTree(monk1train, m.attributes)):
#     # print("\n-------------")
#     # print(f"pruned used: {pruned_tree}")
#     scoreTree = dtree.check(pruned_tree, monk1val)
#     # print(f"score: {scoreTree}")
#     if(scoreTree >= scoreBestTree):
#         scoreBestTree = scoreTree
#         bestTree = pruned_tree
#     # print("-------------")
# print(f"\n\nBetter pruned tree: {pruned_tree}")
# print(f"On testing set: {dtree.check(bestTree, m.monk1test)}")




for fraction in fractions:
    for i in range(numberTry):
        monk1train, monk1val = partition(m.monk1, 0.3)





print("\n------END------")
# --------------------------
# --------- END ----------
# --------------------------
