import os
from multiprocessing import Pool

# Create a sample file
def create_sample_file(file_path):
  content="""MapReduce is a programming model designed to process and generate large datasets efficiently. It works by splitting the data into smaller chunks, which are thes proce"""
  
  with open(file_path, 'w') as file:
    file.write(content)

# Mapper functions counts occurrences of each word in a chunk
def mapper(chunk):
  word_count = {}
  for line in chunk.splitlines():
    words = line.split()
    for word in words:
      if word in word_count:
        word_count[word] += 1
      else:
        word_count[word] = 1
  return word_count

# *Reducer function: combines dictionaries from all mappers
def reducer(mapped_results):
  final_counts = {}
  # iterating over all dictionaries
  for result in mapped_results:
    # iterate on the items of this dictionary
    for word, count in result.items():
      if word in final_counts:
        final_counts[word]+= count
      else:
        final_counts[word] = count
  return final_counts


# Function to split a file into chunks for parallel processing
def split_file(file_path, num_chunks):
  with open(file_path, 'r') as f:
    content = f.read()
  chunk_size = len(content) // num_chunks
  chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
  return chunks

#Main function
if __name__ == "__main__":
  file_path = "sample.txt"
  create_sample_file(file_path) # Create a sample file
  
  num_chunks = 4 # Number of chunks for parallel processing
  
  #Split the file into chunks
  chunks = split_file(file_path, num_chunks)
  
  #Use multiprocessing Pool to process chunks in parallel
  with Pool(processes=num_chunks) as pool:
    mapped_results = pool.map(mapper, chunks)
  
  # Reduce the results
  word_counts = reducer(mapped_results)
  
  #Print all word counts
  for word, count in word_counts.items(): 
    print(f"The word {word}' appears {count} times in the file.")