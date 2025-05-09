import numpy as np

# Union - max from both of the sets
def Union(set1, set2):
  return np.maximum(set1, set2)

# Intersection - min from both of the sets
def Intersection(set1, set2):
  return np.minimum(set1, set2)

# Complement = 1 - set
def Complement(fuzzy_set):
  return 1 - fuzzy_set

# Difference = min(A, 1-B)
def Difference(set1, set2):
  # return Intersection(set1, Complement(set2))
  return np.minimum(set1, 1-set2)

# np.outer - outter product of two sets. That is multily each element of set1 with each element of set2. ---> matrix
def CartesianProduct(set1, set2):
  return np.outer(set1, set2)

# Combining - 2 fuzzy relation
# First take the cartesian product of the two sets and then take the min value for each of the column
# then out of this column - extract the max value
def MaxMinComposition(set1, set2):
  return np.max(np.minimum.outer(set1, set2), axis=1)



if __name__ == "__main__":
  setA = np.array([0.8, 0.5, 0.6, 0.4, 0.7])
  setB = np.array([0.6, 0.7, 0.4, 0.5, 0.8])
  # Perform fuzzy set operations
  print("Fuzzy Union:", Union(setA, setB))
  print("Fuzzy Intersection:", Intersection(setA, setB))
  print("Fuzzy Complement (setA):", Complement(setA))
  print("Fuzzy Difference (setA - setB):", Difference(setA, setB))
  # Fuzzy relations
  relationA = np.array([0.2, 0.4, 0.6])
  relationB = np.array([0.3, 0.5, 0.7])
  # Perform fuzzy relation operations
  print("Cartesian Product:", CartesianProduct(relationA, relationB))
  print("Max-Min Composition:", MaxMinComposition(relationA, relationB))