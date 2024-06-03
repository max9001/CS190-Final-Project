import csv
from operator import itemgetter

def load_csv(file_path):
    """Load CSV file into a list of dictionaries."""
    data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def sort_data(data):
    """Sort data based on a key combining all columns."""
    return sorted(data, key=itemgetter('layer_type', 'output_size', 'parameters', 'accuracy'))

def calculate_percentage_difference(csv1, csv2):
    """Calculate the percentage difference between two CSV files."""
    sorted_csv1 = sort_data(csv1)
    sorted_csv2 = sort_data(csv2)
    
    differences = []
    for row1, row2 in zip(sorted_csv1, sorted_csv2):
        diff = abs(float(row1['accuracy']) - float(row2['accuracy']))
        differences.append(diff)
    
    total_diff = sum(differences)
    average_diff = total_diff / len(differences)
    percentage_diff = (average_diff / max(differences)) * 100
    
    return percentage_diff



# Load CSV files
csv_file1 = 'zero_indices_14.csv'
csv_file2 = 'zero_indices.csv'
data1 = load_csv(csv_file1)
data2 = load_csv(csv_file2)

# Calculate and print the percentage difference
percentage_diff = calculate_percentage_difference(data1, data2)
print("The percentage difference between the two CSV files is %.2f%%" % percentage_diff)

