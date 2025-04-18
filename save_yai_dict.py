import json
import sys
import os

def save_dict_to_json(dictionary, filename):
    """
    Save a dictionary to a JSON file
    
    Args:
        dictionary: The dictionary to save
        filename: The filename to save to
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Convert any non-serializable objects to strings if necessary
    serializable_dict = {}
    for assessor, tasks in dictionary.items():
        serializable_dict[str(assessor)] = {}
        for task_id, orders in tasks.items():
            serializable_dict[str(assessor)][str(task_id)] = []
            for order in orders:
                if isinstance(order, list) or isinstance(order, tuple):
                    serializable_dict[str(assessor)][str(task_id)].append(list(order))
                else:
                    serializable_dict[str(assessor)][str(task_id)].append(order)
    
    # Save the dictionary to a JSON file
    with open(filename, 'w') as f:
        json.dump(serializable_dict, f, indent=2)
    
    print(f"Dictionary saved to {filename}")

# Example usage in notebook:
# save_dict_to_json(y_a_i_dict, 'data/y_a_i_dict.json') 