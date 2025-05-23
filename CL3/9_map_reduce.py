import csv
from collections import defaultdict
from functools import reduce


# Define mapper function to emit (year, temperature) pairs
# Take one row from CSV, extract year from day column, convert temp to float and return (year, temp)
def mapper(row):
    year = row["day"].split("-")[0]  # Extract year from "Date/Time" column
    temperature = float(row["temperature"])  # Convert temperature to float
    return (year, temperature)


# Define reducer function to calculate sum and count of temperatures for each year
# accumulated = dict storing {year: (sum, count)}
def reducer(accumulated, current):
    accumulated[current[0]][0] += current[1]
    accumulated[current[0]][1] += 1
    return accumulated


# Read the weather dataset
weather_data = []
with open("practical_9_hadoop_map_reduce/weather_data.csv", "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        weather_data.append(row)


# Map phase
# apply mapper function to each row
mapped_data = map(mapper, weather_data)


# Reduce phase
# Accumulates temperature sums and counts per year.
# Uses defaultdict to initialize [0, 0] for each new year key automatically.
reduced_data = reduce(reducer, mapped_data, defaultdict(lambda: [0, 0]))


# Calculate average temperature for each year
# dictionary comprehension. create key-value pairs based on reduced_data
avg_temp_per_year = {
    year: total_temp / count
    for year, (total_temp, count) in reduced_data.items()
}


# Find coolest and hottest year
# consider the avg_temp in the dictionary
coolest_year = min(avg_temp_per_year.items(), key=lambda x: x[1])
hottest_year = max(avg_temp_per_year.items(), key=lambda x: x[1])


print("Coolest Year:", coolest_year[0], "Average Temperature:",
      coolest_year[1])
print("Hottest Year:", hottest_year[0], "Average Temperature:",
      hottest_year[1])