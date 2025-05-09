# Function to return the letter grade based on the average score for all subjects
def get_letter_grade(a):
  if a >= 90: return "A"
  elif a >= 80: return "B"
  elif a >= 70: return "C"
  elif a >= 60: return "D"
  else: return "F"



def mapFunction(lines):
  mapping = []
  
  # iterating over all record 
  for line in lines:
    
    # split the line into parts
    p = line.strip().split()
    
    # if the line has less than 3 parts, skip it
    if len(p) < 3:
      continue
    
    # try to convert the score to an int
    # in this mapping we store / append (studentId, score) in the list
    try:
      mapping.append((p[0], int(p[2])))
    except Exception:
      pass

  return mapping


def reduceFunction(mapping):
  # sort the mapping by studentId
  mapping.sort(key=lambda x: x[0])
  # results
  r = []
  # current student id, total score, subject count
  current_student_id, total_score, subject_count = None, 0, 0
  # iterating over the mapping
  for sid, score in mapping:
    # if the current student id is None, then we are at the first record
    if current_student_id is None:
      current_student_id, total_score, subject_count = sid, score, 1
      
    # if the current student id is the same as the previous record, then we are at the same student. increment the total score and subject count
    elif sid == current_student_id:
      total_score += score
      subject_count += 1

    # if the current student id is different from the previous record, then we are at a new student. calculate the average score and letter grade, and append it to the results
    # then reset the current student id, total score, and subject count
    else:
      avg = total_score / subject_count
      r.append((current_student_id, avg, get_letter_grade(avg)))
      current_student_id, total_score, subject_count = sid, score, 1

  # process the last record
  if current_student_id and subject_count:
    avg = total_score / subject_count
    r.append((current_student_id, avg, get_letter_grade(avg)))

  # return the results
  return r



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