import pandas as pd
import random
import csv

def generate_dummy_csv(file_name, num_rows):
    layers = ['conv1', 'conv2', 'fc1', 'fc2']
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Layer', 'Index1', 'Index2', 'Index3'])  # Adjust based on your data structure
        for _ in range(num_rows):
            layer = random.choice(layers)
            index1 = random.randint(0, 99)
            index2 = random.randint(0, 99)
            index3 = random.randint(0, 99)
            writer.writerow([layer, index1, index2, index3])

# Generate two dummy CSV files
generate_dummy_csv('zero_indices_1.csv', 1000)
generate_dummy_csv('zero_indices_2.csv', 1000)

# The compare_pruning_schemes function from previous code
def compare_pruning_schemes(csv_file_1, csv_file_2):
    def load_pruned_indices(csv_file):
        pruned_indices = set()
        chunk_size = 10000  # Adjust chunk size as needed
        for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                layer_name = row['Layer']
                if row[1:].isnull().any():
                    continue
                index = tuple(map(int, row[1:]))
                pruned_indices.add((layer_name, index))
        return pruned_indices

    # Load the pruned indices from both CSV files
    pruned_indices_1 = load_pruned_indices(csv_file_1)
    pruned_indices_2 = load_pruned_indices(csv_file_2)
    
    # Debugging: print the number of loaded indices
    print(f"Total pruned indices in CSV 1: {len(pruned_indices_1)}")
    print(f"Total pruned indices in CSV 2: {len(pruned_indices_2)}")
    
    # Find the common pruned indices
    common_pruned_indices = pruned_indices_1.intersection(pruned_indices_2)
    
    # Calculate similarity percentage
    total_pruned_1 = len(pruned_indices_1)
    total_pruned_2 = len(pruned_indices_2)
    total_common = len(common_pruned_indices)
    
    similarity_percentage = (total_common / total_pruned_1) * 100 if total_pruned_1 > 0 else 0
    
    return similarity_percentage, total_pruned_1, total_pruned_2, total_common

# Example usage
csv_file_1 = 'zero_indices_14.csv'
csv_file_2 = 'zero_indices.csv'

similarity, total_1, total_2, common = compare_pruning_schemes(csv_file_1, csv_file_2)
print(f"Similarity: {similarity:.2f}%")
print(f"Total pruned in CSV 1: {total_1}")
print(f"Total pruned in CSV 2: {total_2}")
print(f"Total common pruned indices: {common}")
