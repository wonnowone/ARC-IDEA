import json
import csv
from collections import Counter, deque
import numpy as np

def merge_challenge_solution_data():
    """Merge training challenges and solutions into a single training.json file."""
    print("Loading challenge and solution files...")
    
    with open('../arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    
    with open('../arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    merged_training = {}
    
    for challenge_id in challenges:
        challenge_data = challenges[challenge_id]
        solution_data = solutions.get(challenge_id, [])
        
        # Merge train data (already has outputs)
        merged_training[challenge_id] = {
            'train': challenge_data['train'],
            'test': []
        }
        
        # Merge test data with solutions
        for i, test_example in enumerate(challenge_data['test']):
            test_with_solution = {
                'input': test_example['input'],
                'output': solution_data[i] if i < len(solution_data) else []
            }
            merged_training[challenge_id]['test'].append(test_with_solution)
    
    # Save merged training data
    with open('training.json', 'w') as f:
        json.dump(merged_training, f, indent=2)
    
    print(f"Merged training data saved to training.json ({len(merged_training)} challenges)")
    return merged_training

def get_color_stats(grid):
    """Get color statistics for background classification."""
    if not grid or not grid[0]:
        return 0, 0, 0, 0
    
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    total_cells = len(flat_grid)
    
    if len(color_counts) < 2:
        return color_counts.most_common(1)[0][1], 0, 100, 0
    
    sorted_counts = sorted(color_counts.values(), reverse=True)
    dominant_count = sorted_counts[0]
    second_count = sorted_counts[1]
    dominant_pct = (dominant_count / total_cells) * 100
    second_pct = (second_count / total_cells) * 100
    
    return dominant_count, second_count, dominant_pct, second_pct

def get_color_map(grid):
    """Get color mapping for the grid."""
    if not grid or not grid[0]:
        return {}
    
    flat_grid = [cell for row in grid for cell in row]
    color_counts = Counter(flat_grid)
    return dict(color_counts)

def classify_background(grid):
    """Classify if grid has background based on dominant color threshold."""
    dom_count, sec_count, _, _ = get_color_stats(grid)
    return "yes" if dom_count >= 1.5 * sec_count else "no"

def find_lines(grid, color):
    """Find horizontal, vertical, and diagonal lines of specific color."""
    if not grid or not grid[0]:
        return [], [], []
    
    height, width = len(grid), len(grid[0])
    horizontal_lines = []
    vertical_lines = []
    diagonal_lines = []
    
    # Find horizontal lines (full width)
    for row in range(height):
        if all(grid[row][col] == color for col in range(width)):
            horizontal_lines.append(f"row_{row}")
    
    # Find vertical lines (full height)
    for col in range(width):
        if all(grid[row][col] == color for row in range(height)):
            vertical_lines.append(f"col_{col}")
    
    # Find diagonal lines (top-left to bottom-right)
    if height == width:  # Only for square grids
        if all(grid[i][i] == color for i in range(height)):
            diagonal_lines.append("main_diagonal")
        if all(grid[i][height-1-i] == color for i in range(height)):
            diagonal_lines.append("anti_diagonal")
    
    return horizontal_lines, vertical_lines, diagonal_lines

def flood_fill_shape(grid, start_row, start_col, visited):
    """Extract shape using flood fill algorithm."""
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    target_color = grid[start_row][start_col]
    shape_pixels = []
    queue = deque([(start_row, start_col)])
    
    while queue:
        row, col = queue.popleft()
        
        if (row < 0 or row >= height or col < 0 or col >= width or 
            visited[row][col] or grid[row][col] != target_color):
            continue
        
        visited[row][col] = True
        shape_pixels.append((row, col))
        
        # Check 4-connected neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            queue.append((row + dr, col + dc))
    
    return shape_pixels

def extract_shapes(grid):
    """Extract all shapes from the grid."""
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    visited = [[False] * width for _ in range(height)]
    shapes = []
    
    for row in range(height):
        for col in range(width):
            if not visited[row][col]:
                shape_pixels = flood_fill_shape(grid, row, col, visited)
                if shape_pixels:
                    color = grid[row][col]
                    
                    # Calculate boundaries
                    rows = [p[0] for p in shape_pixels]
                    cols = [p[1] for p in shape_pixels]
                    min_row, max_row = min(rows), max(rows)
                    min_col, max_col = min(cols), max(cols)
                    
                    # Calculate relative positions within bounding box
                    relative_positions = []
                    for r, c in shape_pixels:
                        rel_row = r - min_row
                        rel_col = c - min_col
                        relative_positions.append(f"({rel_row},{rel_col})")
                    
                    # Calculate bounding box dimensions
                    bounding_width = max_col - min_col + 1
                    bounding_height = max_row - min_row + 1
                    
                    shapes.append({
                        'color': color,
                        'boundaries': f"({min_row},{min_col})-({max_row},{max_col})",
                        'relative_locations': ";".join(relative_positions),
                        'pixel_count': len(shape_pixels),
                        'bounding_width': bounding_width,
                        'bounding_height': bounding_height
                    })
    
    return shapes

def process_grid(prob_id, grid_type, grid_num, grid):
    """Process a single grid and extract all features."""
    if not grid or not grid[0]:
        return []
    
    height, width = len(grid), len(grid[0])
    grid_size = f"{height}x{width}"
    background = classify_background(grid)
    color_map = get_color_map(grid)
    
    features = []
    
    for color in color_map.keys():
        horizontal_lines, vertical_lines, diagonal_lines = find_lines(grid, color)
        
        # Get line locations as strings
        h_lines = ";".join(horizontal_lines) if horizontal_lines else ""
        v_lines = ";".join(vertical_lines) if vertical_lines else ""
        d_lines = ";".join(diagonal_lines) if diagonal_lines else ""
        
        features.append({
            'prob_id': prob_id,
            'grid_label': f"{grid_type}_{grid_num}",
            'grid_size': grid_size,
            'background': background,
            'color': color,
            'horizontal_lines': h_lines,
            'vertical_lines': v_lines,
            'diagonal_lines': d_lines,
            'shapes': ""  # Will be filled by shape extraction
        })
    
    # Extract shapes
    shapes = extract_shapes(grid)
    
    # Group shapes by color
    shape_by_color = {}
    for shape in shapes:
        color = shape['color']
        if color not in shape_by_color:
            shape_by_color[color] = []
        shape_by_color[color].append(shape)
    
    # Add shape information to features
    for feature in features:
        color = feature['color']
        if color in shape_by_color:
            shape_info = []
            for shape in shape_by_color[color]:
                shape_str = f"boundary:{shape['boundaries']}|location:{shape['relative_locations']}|pixels:{shape['pixel_count']}|bounds:{shape['bounding_width']}x{shape['bounding_height']}"
                shape_info.append(shape_str)
            feature['shapes'] = ";;".join(shape_info)
    
    return features

def extract_all_features():
    """Extract features from all training data."""
    print("Loading training data...")
    with open('training.json', 'r') as f:
        training_data = json.load(f)
    
    all_features = []
    
    for prob_id in training_data:
        challenge_data = training_data[prob_id]
        
        # Process training examples
        for i, example in enumerate(challenge_data['train']):
            input_features = process_grid(prob_id, 'input', i, example['input'])
            output_features = process_grid(prob_id, 'output', i, example['output'])
            all_features.extend(input_features)
            all_features.extend(output_features)
        
        # Process test examples
        for i, example in enumerate(challenge_data['test']):
            input_features = process_grid(prob_id, 'test_input', i, example['input'])
            output_features = process_grid(prob_id, 'test_output', i, example['output'])
            all_features.extend(input_features)
            all_features.extend(output_features)
    
    return all_features

def save_to_csv(features):
    """Save extracted features to CSV file."""
    if not features:
        print("No features to save")
        return
    
    fieldnames = [
        'prob_id', 'grid_label', 'grid_size', 'background', 'color',
        'horizontal_lines', 'vertical_lines', 'diagonal_lines', 'shapes'
    ]
    
    # Try different filenames if the file is locked
    import os
    import time
    
    base_filename = 'arc_features'
    # timestamp = int(time.time())
    filename = f'{base_filename}.csv'
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)
        
        print(f"Features saved to {filename} ({len(features)} rows)")
    except PermissionError:
        # If still locked, try a different approach
        filename = f'{base_filename}_backup_.csv'
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)
        
        print(f"Features saved to {filename} ({len(features)} rows)")

def main():
    """Main function to run feature extraction."""
    print("Starting feature extraction process...")
    
    # Step 1: Merge data
    merged_data = merge_challenge_solution_data()
    
    # Step 2-8: Extract all features
    print("Extracting features...")
    features = extract_all_features()
    
    # Save to CSV
    save_to_csv(features)
    
    print("Feature extraction completed!")
    print(f"Total features extracted: {len(features)}")

if __name__ == "__main__":
    main()