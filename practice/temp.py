'''import os 
from multiprocessing import Pool

def create_sample_file(file_path):
  content="""MapReduce is a programming model designed to process and generate large datasets efficiently. It works by splitting the data into smaller chunks, which are thes proce"""
  with open(file_path, 'w') as file:
    file.write(content)


def split_file(file_name, num_chunks):
  with open(file_name, 'r') as file:
    content = file.read()
    chunk_size = len(content) // num_chunks
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    return chunks


def mapper(chunk):
  words = chunk.split()
  word_count = {}
  for word in words:
    if word in word_count:
      word_count[word] += 1
    else:
      word_count[word] = 1
  return word_count

def reducer(mapped_results):
  final_word_counts = {}
  for dictionary in mapped_results:
    for word,count in dictionary.items():
      if word in final_word_counts:
        final_word_counts[word] += count
      else:
        final_word_counts[word] = count
  return final_word_counts


file_path = 'temp.txt'
create_sample_file(file_path)
num_chunks = 4
chunks = split_file(file_path, num_chunks)
with Pool(processes=num_chunks) as pool:
  mapped_results = pool.map(mapper, chunks)

word_counts = reducer(mapped_results)
for word, count in word_counts.items(): 
  print(f"The word {word}' appears {count} times in the file.")


# get the content
# store into a file
# then take number of chunks and then read the file and split it into chunks, return these list of chunks
# now open a pool with same number of processes as the chunks, and call the mapping function with the chunks
# then we get the mapped_results, now we call the reducer function with the mapped_results
# and print the word and count 
'''
'''
import multiprocessing
from collections import defaultdict
import numpy as np

def mapper():
  pass

def reducer():
  pass

def matrix_multiplication():
  pass





# Taking user input for matrices
rows_a = int(input("Enter number of rows for Matrix A: "))
cols_a = int(input("Enter number of columns for Matrix A: "))

print("Enter elements for Matrix A:")
A = np.array([[int(input()) for _ in range(cols_a)] for _ in range(rows_a)])

rows_b = int(input("Enter number of rows for Matrix B: "))
cols_b = int(input("Enter number of columns for Matrix B: "))

if cols_a != rows_b:
  raise ValueError("Number of columns of A must be equal to number of rows of B")

print("Enter elements for Matrix B:")
B = np.array([[int(input()) for _ in range(cols_b)] for _ in range(rows_b)])

result = matrix_multiplication(A, B)
print("Matrix A:")
print(A)
print("Matrix B:")
print(B)
print("Resultant Matrix (A x B):")
print(result)
'''
'''
def get_letter_grade(a):
  if a >= 90: return "A"
  elif a >= 80: return "B"
  elif a >= 70: return "C"
  elif a >= 60: return "D"
  else: return "F"


def mapFunction(lines):
  mapping = []

  for line in lines:
    p = line.strip().split()
    if(len(p) < 3):
      continue
    try:
      mapping.append((p[0], int(p[2])))
    except Exception:
      pass

  return mapping


def reduceFunction(mapping):
  result = []
  mapping.sort(key=lambda x : x[0])

  current_student_id, total_score, subjects = None, 0, 0
  for student_id, score in mapping:
    if current_student_id is None:
      current_student_id = student_id
      total_score = score
      subjects = 1

    elif current_student_id == student_id:
      total_score += score
      subjects += 1

    else:
      avg = total_score / subjects
      grade = get_letter_grade(avg)
      result.append((current_student_id, avg, grade))
      current_student_id = student_id
      total_score = score
      subjects = 1

  if current_student_id and subjects:
    avg = total_score / subjects
    result.append((current_student_id, avg, get_letter_grade(avg)))

  return result
  

lines = []
print("Enter data (studentId subject score). Press ENTER on an empty line to finish:")
results = []


while True:
  # Take record as input
  # if we give empty line, then break
  # else keep on appending the line into the lines list
  line = input()

  if not line.strip():
    break

  lines.append(line)

# mapping the results
results = mapFunction(lines)
# reducing the results
results = reduceFunction(results)

# print the output
for sid, avg, grade in results:
  print(sid, f"{avg:.2f}", grade)
'''
'''
import pandas as pd

def map_reduce_with_pandas(input_file):
  df = pd.read_csv(input_file)
  deceased_males = df[(df['Survived'] == 0) & (df['Sex'] == 'male')]
  avg_age_male = deceased_males['Age'].mean()

  deceased_females = df[(df['Survived'] == 0) & (df['Sex'] == 'female')]
  count_female = deceased_females['Pclass'].value_counts()
  return avg_age_male, count_female



# Example usage
input_file = r'../CL4/Titanic-Dataset.csv' # Update this to the path of your Titanic dataset CSV file
average_age, female_class_count = map_reduce_with_pandas(input_file)
print(f"Average age of males who died: {average_age:.2f}")
print("Number of deceased females in each class:")
print(female_class_count)
'''
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


iris = load_iris()


X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

y_pred_train = logreg.predict(X_train_scaled)
y_pred_test = logreg.predict(X_test_scaled)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

print("Classification Report (Training):")
print(classification_report(y_train, y_pred_train))
'''
'''
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data
y = iris.target
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
'''




import multiprocessing
from collections import defaultdict
import numpy as np


def mapper(matrix_a, matrix_b, row_indices):
  """Emit one partial product per (i, j, k)."""
  results = []
  for i in row_indices:
    for k in range(matrix_a.shape[1]):
      # a_ik participates in all j outputs
      for j in range(matrix_b.shape[1]):
        results.append(((i, j), matrix_a[i, k] * matrix_b[k, j]))
  return results


def reducer(mapped_values):
  totals = defaultdict(int)
  for (i, j), partial in mapped_values:
    totals[(i, j)] += partial
  return totals


def matrix_multiplication_mapreduce(A, B):
  if A.shape[1] != B.shape[0]:
    raise ValueError("Incompatible shapes")

  # 1. Split rows across workers
  num_workers = multiprocessing.cpu_count()
  row_chunks = np.array_split(np.arange(A.shape[0]), num_workers)

  # 2. Map step (only row_chunks)
  with multiprocessing.Pool(num_workers) as pool:
    map_outputs = pool.starmap(mapper, [(A, B, rows) for rows in row_chunks])

  # 3. Flatten and Reduce
  all_partials = [p for chunk in map_outputs for p in chunk]
  reduced = reducer(all_partials)

  # 4. Build result matrix
  C = np.zeros((A.shape[0], B.shape[1]))
  for (i, j), total in reduced.items():
    C[i, j] = total
  return C


if __name__ == "__main__":

  # Taking user input for matrices
  rows_a = int(input("Enter number of rows for Matrix A: "))
  cols_a = int(input("Enter number of columns for Matrix A: "))

  print("Enter elements for Matrix A:")
  A = np.array([[int(input()) for _ in range(cols_a)] for _ in range(rows_a)])

  rows_b = int(input("Enter number of rows for Matrix B: "))
  cols_b = int(input("Enter number of columns for Matrix B: "))

  if cols_a != rows_b:
    raise ValueError(
        "Number of columns of A must be equal to number of rows of B")

  print("Enter elements for Matrix B:")
  B = np.array([[int(input()) for _ in range(cols_b)] for _ in range(rows_b)])

  # Compute matrix multiplication using MapReduce
  print("Matrix A:")
  print(A)
  print("Matrix B:")
  print(B)
  result = matrix_multiplication_mapreduce(A, B)
  print("Resultant Matrix (A x B):")
  print(result)
