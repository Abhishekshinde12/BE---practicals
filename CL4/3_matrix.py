import multiprocessing
from collections import defaultdict
import numpy as np


def mapper(matrix_a, matrix_b, row_indices):
  """Emit one partial product per (i, j, k)."""
  # to store the results
  # (i,k) and (k, j) hence k is common and it comes first before j --> 3 loops
  results = []
  # iterate on each row indice
  for i in row_indices:
    # now iterating on  each column in the row index
    for k in range(matrix_a.shape[1]):
          # iterate over every column j in B
          for j in range(matrix_b.shape[1]):
            # storing the index (i,j) and the partial product (a_ik * b_kj)
              results.append(((i, j), matrix_a[i, k] * matrix_b[k, j]))
  return results



def reducer(mapped_values):
  totals = defaultdict(int)
  for (i,j), partial in mapped_values:
      totals[(i,j)] += partial
  return totals



def matrix_multiplication_mapreduce(A, B):
  
  if A.shape[1] != B.shape[0]:
      raise ValueError("Incompatible shapes")

  # 1. Split rows across workers
  # get the no. of cpu cores
  # now get the no. of rows in A and then equally split the rows among the cpu cores
  num_workers = multiprocessing.cpu_count()
  # here we get an tuple representing the start and end index of each chunk
  row_chunks = np.array_split(np.arange(A.shape[0]), num_workers)

  
  # 2. Map step (only row_chunks)
  # call the mapper function for each chunk on each core
  with multiprocessing.Pool(num_workers) as pool:
      map_outputs = pool.starmap(mapper, [(A, B, rows) for rows in row_chunks])

  # 3. Flatten and Reduce
  # call the reducer function on the map outputs
  all_partials = [p for chunk in map_outputs for p in chunk]
  reduced = reducer(all_partials)

  # 4. Build result matrix
  # get the index and then store the result
  C = np.zeros((A.shape[0], B.shape[1]))
  for (i,j), total in reduced.items():
      C[i,j] = total
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
    raise ValueError("Number of columns of A must be equal to number of rows of B")
  
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