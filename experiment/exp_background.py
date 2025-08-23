import json
from collections import Counter

"""
This module contains the background split for the experiment.
It is used to separate data with a specific background from the rest of the dataset.
The background split is useful for training models that need to learn from specific background conditions.
"""

def get_color_stats(grid):
    """Calculate color statistics for a grid."""
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    total_cells = len(flat_grid)
    
    if len(color_counts) < 2:
        return 0, 0, 0, 0  # dominant_count, second_count, dominant_pct, second_pct
    
    sorted_counts = sorted(color_counts.values(), reverse=True)
    dominant_count = sorted_counts[0]
    second_count = sorted_counts[1]
    dominant_pct = (dominant_count / total_cells) * 100
    second_pct = (second_count / total_cells) * 100
    
    return dominant_count, second_count, dominant_pct, second_pct

def merge_challenge_with_solution(challenge_id, challenge_data, solution_data):
    """Merge challenge with its corresponding solution data."""
    merged_challenge = {
        'id': challenge_id,
        'train': [],
        'test': []
    }
    
    # Merge training examples
    for i, train_example in enumerate(challenge_data['train']):
        merged_example = {
            'input': train_example['input'],
            'output': train_example['output']
        }
        merged_challenge['train'].append(merged_example)
    
    # Merge test examples with solutions
    for i, test_example in enumerate(challenge_data['test']):
        merged_example = {
            'input': test_example['input'],
            'output': solution_data[i] if i < len(solution_data) else []
        }
        merged_challenge['test'].append(merged_example)
    
    return merged_challenge

def process_challenge(challenge_id, merged_challenge_data):
    """Process a single merged challenge and extract relevant features."""
    processed_items = []
    
    # Process training examples
    for i, example in enumerate(merged_challenge_data['train']):
        input_grid = example['input']
        output_grid = example['output']
        
        input_height, input_width = len(input_grid), len(input_grid[0])
        output_height = len(output_grid) if output_grid else 0
        output_width = len(output_grid[0]) if output_grid and len(output_grid) > 0 else 0
        
        # Get color stats for input grid
        dom_count, sec_count, dom_pct, sec_pct = get_color_stats(input_grid)
        
        processed_items.append({
            'id': f"{challenge_id}_train_{i}",
            'challenge_id': challenge_id,
            'type': 'train',
            'input_height': input_height,
            'input_width': input_width,
            'output_height': output_height,
            'output_width': output_width,
            'dominant_count': dom_count,
            'second_dominant_count': sec_count,
            'dominant_percentage': dom_pct,
            'second_dominant_percentage': sec_pct,
            'input_grid': input_grid,
            'output_grid': output_grid
        })
    
    # Process test examples
    for i, example in enumerate(merged_challenge_data['test']):
        input_grid = example['input']
        output_grid = example['output']
        
        input_height, input_width = len(input_grid), len(input_grid[0])
        output_height = len(output_grid) if output_grid else 0
        output_width = len(output_grid[0]) if output_grid and len(output_grid) > 0 else 0
        
        # Get color stats for input grid
        dom_count, sec_count, dom_pct, sec_pct = get_color_stats(input_grid)
        
        processed_items.append({
            'id': f"{challenge_id}_test_{i}",
            'challenge_id': challenge_id,
            'type': 'test',
            'input_height': input_height,
            'input_width': input_width,
            'output_height': output_height,
            'output_width': output_width,
            'dominant_count': dom_count,
            'second_dominant_count': sec_count,
            'dominant_percentage': dom_pct,
            'second_dominant_percentage': sec_pct,
            'input_grid': input_grid,
            'output_grid': output_grid
        })
    
    return processed_items

def background_split(dataset):
    """
    Splits the dataset into two lists based on dominant color condition:
    - background_no: dominant color is NOT 1.5 times more than second dominant color
    - background_yes: dominant color IS 1.5 times more than second dominant color
    """
    background_yes = []
    background_no = []

    for item in dataset:
        if item['dominant_count'] >= 1.5 * item['second_dominant_count']:
            background_yes.append(item)
        else:
            background_no.append(item)
    
    return background_yes, background_no

# Load the JSON files
with open('../arc-agi_training_challenges.json', 'r') as f:
    training_challenges = json.load(f)

with open('../arc-agi_training_solutions.json', 'r') as f:
    training_solutions = json.load(f)

# First, merge challenges with solutions
merged_data = {}
for challenge_id in training_challenges:
    challenge_data = training_challenges[challenge_id]
    solution_data = training_solutions.get(challenge_id, [])
    merged_data[challenge_id] = merge_challenge_with_solution(challenge_id, challenge_data, solution_data)

# Process all merged challenges
all_processed_data = []
for challenge_id in merged_data:
    merged_challenge_data = merged_data[challenge_id]
    processed_items = process_challenge(challenge_id, merged_challenge_data)
    all_processed_data.extend(processed_items)

# Split the dataset based on the dominant color condition
background_yes, background_no = background_split(all_processed_data)

# Extract relevant IDs for each group
background_yes_ids = [item['challenge_id'] for item in background_yes]
background_no_ids = [item['challenge_id'] for item in background_no]

# Get unique challenge IDs
unique_background_yes_ids = list(set(background_yes_ids))
unique_background_no_ids = list(set(background_no_ids))

print(f"Background Yes (dominant >= 1.5x second): {len(background_yes)} items")
print(f"Background No (dominant < 1.5x second): {len(background_no)} items")
print(f"Total processed items: {len(all_processed_data)}")
print(f"Unique challenge IDs in background_yes: {len(unique_background_yes_ids)}")
print(f"Unique challenge IDs in background_no: {len(unique_background_no_ids)}")

# Save the results with ID lists
results = {
    'background_yes': {
        'data': background_yes,
        'challenge_ids': unique_background_yes_ids,
        'count': len(background_yes)
    },
    'background_no': {
        'data': background_no,
        'challenge_ids': unique_background_no_ids,
        'count': len(background_no)
    }
}

with open('background_split_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Also save individual lists for convenience
with open('background_yes_ids.json', 'w') as f:
    json.dump(unique_background_yes_ids, f, indent=2)

with open('background_no_ids.json', 'w') as f:
    json.dump(unique_background_no_ids, f, indent=2)