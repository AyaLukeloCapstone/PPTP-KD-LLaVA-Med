import json

def extract_n_entries(input_file, output_file, n):
    # Step 1: Read the original JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Step 2: Extract the specified number of entries
    extracted_data = data[:n]

    # Step 3: Write the extracted entries to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(extracted_data, file, indent=4)

    print(f"Extracted {n} entries and saved to {output_file}")

# Example usage
input_file = '/scratch/ltl2113/LLaVA-Med/data/instruct/filtered_json_file.json'  # Replace with your original JSON file name
output_file = '/scratch/ltl2113/LLaVA-Med/data/instruct/KD_filtered_json_file.json'  # Replace with your desired new JSON file name
n = 30  # Replace with the number of entries you want to extract

extract_n_entries(input_file, output_file, n)