import numpy as np
import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class BaseSolver(ABC):
    """Base class for ARC solvers - experts of inference"""
    
    @abstractmethod
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Generate prediction for input grid"""
        pass
    
    @abstractmethod
    def get_thinking_flow(self) -> Dict[str, Any]:
        """Return solver's thinking process"""
        pass

class ColorPatternSolver(BaseSolver):
    """Solver specializing in color pattern transformations"""
    
    def __init__(self):
        self.thinking_flow = {
            'strategy': 'color_pattern_analysis',
            'steps': ['identify_colors', 'find_patterns', 'apply_transformation'],
            'confidence': 0.0
        }
    
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply color pattern transformations"""
        self.thinking_flow['confidence'] = 0.0
        
        # Step 1: Identify color distribution
        unique_colors, counts = np.unique(input_grid, return_counts=True)
        color_map = dict(zip(unique_colors, counts))
        
        # Step 2: Find dominant color patterns
        dominant_color = unique_colors[np.argmax(counts)]
        secondary_color = unique_colors[np.argmin(counts)] if len(unique_colors) > 1 else dominant_color
        
        # Step 3: Apply transformation based on color analysis
        output_grid = input_grid.copy()
        
        # Simple color swap transformation
        if len(unique_colors) >= 2:
            mask_dominant = (input_grid == dominant_color)
            mask_secondary = (input_grid == secondary_color)
            
            output_grid[mask_dominant] = secondary_color
            output_grid[mask_secondary] = dominant_color
            
            self.thinking_flow['confidence'] = 0.8
        else:
            # If single color, create checkerboard pattern
            for i in range(output_grid.shape[0]):
                for j in range(output_grid.shape[1]):
                    if (i + j) % 2 == 0:
                        output_grid[i, j] = dominant_color
                    else:
                        output_grid[i, j] = (dominant_color + 1) % 10
            
            self.thinking_flow['confidence'] = 0.6
        
        return output_grid
    
    def get_thinking_flow(self) -> Dict[str, Any]:
        return self.thinking_flow

class ShapeSymmetrySolver(BaseSolver):
    """Solver specializing in shape and symmetry transformations"""
    
    def __init__(self):
        self.thinking_flow = {
            'strategy': 'shape_symmetry_analysis',
            'steps': ['detect_shapes', 'analyze_symmetry', 'apply_transformation'],
            'confidence': 0.0
        }
    
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply shape and symmetry transformations"""
        self.thinking_flow['confidence'] = 0.0
        
        # Step 1: Detect basic shapes
        shapes = self._detect_shapes(input_grid)
        
        # Step 2: Analyze symmetry
        symmetry_type = self._analyze_symmetry(input_grid)
        
        # Step 3: Apply transformation
        output_grid = input_grid.copy()
        
        if symmetry_type == 'horizontal':
            output_grid = np.fliplr(input_grid)
            self.thinking_flow['confidence'] = 0.9
        elif symmetry_type == 'vertical':
            output_grid = np.flipud(input_grid)
            self.thinking_flow['confidence'] = 0.9
        elif symmetry_type == 'rotational':
            output_grid = np.rot90(input_grid)
            self.thinking_flow['confidence'] = 0.85
        else:
            # If no clear symmetry, try to create symmetry
            output_grid = self._create_symmetry(input_grid)
            self.thinking_flow['confidence'] = 0.6
        
        return output_grid
    
    def _detect_shapes(self, grid: np.ndarray) -> List[Dict]:
        """Detect basic shapes in the grid"""
        shapes = []
        
        # Find contiguous regions
        unique_colors = np.unique(grid)
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            
            mask = (grid == color)
            # Simple shape detection - count connected components
            components = self._find_connected_components(mask)
            
            for component in components:
                shape_info = {
                    'color': color,
                    'positions': component,
                    'size': len(component),
                    'bounding_box': self._get_bounding_box(component)
                }
                shapes.append(shape_info)
        
        return shapes
    
    def _find_connected_components(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find connected components in binary mask"""
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    component = []
                    self._dfs(mask, visited, i, j, component)
                    if component:
                        components.append(component)
        
        return components
    
    def _dfs(self, mask: np.ndarray, visited: np.ndarray, i: int, j: int, component: List):
        """Depth-first search for connected components"""
        if (i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1] or 
            visited[i, j] or not mask[i, j]):
            return
        
        visited[i, j] = True
        component.append((i, j))
        
        # Check 4-connected neighbors
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            self._dfs(mask, visited, i + di, j + dj, component)
    
    def _get_bounding_box(self, component: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
        """Get bounding box of component"""
        if not component:
            return (0, 0, 0, 0)
        
        positions = np.array(component)
        min_i, min_j = positions.min(axis=0)
        max_i, max_j = positions.max(axis=0)
        
        return (min_i, min_j, max_i, max_j)
    
    def _analyze_symmetry(self, grid: np.ndarray) -> str:
        """Analyze symmetry in the grid"""
        # Check horizontal symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            return 'horizontal'
        
        # Check vertical symmetry
        if np.array_equal(grid, np.flipud(grid)):
            return 'vertical'
        
        # Check rotational symmetry
        if np.array_equal(grid, np.rot90(grid, 2)):
            return 'rotational'
        
        return 'none'
    
    def _create_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """Create symmetry in the grid"""
        # Simple approach: mirror left half to right half
        output = grid.copy()
        mid = grid.shape[1] // 2
        
        if grid.shape[1] % 2 == 0:
            output[:, mid:] = np.fliplr(output[:, :mid])
        else:
            output[:, mid+1:] = np.fliplr(output[:, :mid])
        
        return output
    
    def get_thinking_flow(self) -> Dict[str, Any]:
        return self.thinking_flow

class GeometricTransformSolver(BaseSolver):
    """Solver specializing in geometric transformations"""
    
    def __init__(self):
        self.thinking_flow = {
            'strategy': 'geometric_transformation',
            'steps': ['analyze_geometry', 'identify_transform', 'apply_transform'],
            'confidence': 0.0
        }
    
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply geometric transformations"""
        self.thinking_flow['confidence'] = 0.0
        
        # Step 1: Analyze geometric properties
        center_of_mass = self._compute_center_of_mass(input_grid)
        
        # Step 2: Identify likely transformation
        transform_type = self._identify_transform_type(input_grid)
        
        # Step 3: Apply transformation
        if transform_type == 'rotation':
            output_grid = np.rot90(input_grid, k=1)
            self.thinking_flow['confidence'] = 0.8
        elif transform_type == 'scaling':
            output_grid = self._apply_scaling(input_grid)
            self.thinking_flow['confidence'] = 0.7
        elif transform_type == 'translation':
            output_grid = self._apply_translation(input_grid)
            self.thinking_flow['confidence'] = 0.75
        else:
            # Default: transpose
            output_grid = input_grid.T
            if output_grid.shape != input_grid.shape:
                # Pad or crop to maintain size
                output_grid = self._resize_to_match(output_grid, input_grid.shape)
            self.thinking_flow['confidence'] = 0.6
        
        return output_grid
    
    def _compute_center_of_mass(self, grid: np.ndarray) -> Tuple[float, float]:
        """Compute center of mass for non-zero elements"""
        non_zero_positions = np.argwhere(grid != 0)
        if len(non_zero_positions) == 0:
            return (grid.shape[0] / 2, grid.shape[1] / 2)
        
        center_i = np.mean(non_zero_positions[:, 0])
        center_j = np.mean(non_zero_positions[:, 1])
        return (center_i, center_j)
    
    def _identify_transform_type(self, grid: np.ndarray) -> str:
        """Identify most likely transformation type"""
        # Simple heuristics based on grid properties
        non_zero_count = np.sum(grid != 0)
        total_count = grid.size
        
        density = non_zero_count / total_count
        
        if density < 0.3:
            return 'translation'  # Sparse grids often need translation
        elif density > 0.7:
            return 'rotation'  # Dense grids often need rotation
        else:
            return 'scaling'  # Medium density might need scaling
    
    def _apply_scaling(self, grid: np.ndarray) -> np.ndarray:
        """Apply scaling transformation"""
        # Simple 2x scaling of pattern
        output = np.zeros_like(grid)
        
        # Find non-zero region
        non_zero_positions = np.argwhere(grid != 0)
        if len(non_zero_positions) == 0:
            return grid
        
        # Scale down by factor of 2
        for pos in non_zero_positions:
            i, j = pos
            new_i, new_j = i // 2, j // 2
            if new_i < output.shape[0] and new_j < output.shape[1]:
                output[new_i, new_j] = grid[i, j]
        
        return output
    
    def _apply_translation(self, grid: np.ndarray) -> np.ndarray:
        """Apply translation transformation"""
        # Shift pattern by 1 unit right and down
        output = np.zeros_like(grid)
        
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1] - 1):
                output[i + 1, j + 1] = grid[i, j]
        
        return output
    
    def _resize_to_match(self, grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize grid to match target shape"""
        output = np.zeros(target_shape)
        
        min_h = min(grid.shape[0], target_shape[0])
        min_w = min(grid.shape[1], target_shape[1])
        
        output[:min_h, :min_w] = grid[:min_h, :min_w]
        
        return output.astype(grid.dtype)
    
    def get_thinking_flow(self) -> Dict[str, Any]:
        return self.thinking_flow

class LogicalRuleSolver(BaseSolver):
    """Solver specializing in logical rule inference"""
    
    def __init__(self):
        self.thinking_flow = {
            'strategy': 'logical_rule_inference',
            'steps': ['extract_rules', 'validate_rules', 'apply_rules'],
            'confidence': 0.0
        }
    
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply logical rule transformations"""
        self.thinking_flow['confidence'] = 0.0
        
        # Step 1: Extract potential rules
        rules = self._extract_rules(input_grid)
        
        # Step 2: Validate and score rules
        scored_rules = self._score_rules(rules, input_grid)
        
        # Step 3: Apply best rule
        if scored_rules:
            best_rule = max(scored_rules, key=lambda x: x['score'])
            output_grid = self._apply_rule(best_rule, input_grid)
            self.thinking_flow['confidence'] = best_rule['score']
        else:
            output_grid = input_grid.copy()
            self.thinking_flow['confidence'] = 0.1
        
        return output_grid
    
    def _extract_rules(self, grid: np.ndarray) -> List[Dict]:
        """Extract potential logical rules from grid"""
        rules = []
        
        # Rule 1: If cell has value X, change to Y
        unique_values = np.unique(grid)
        for val in unique_values:
            if val != 0:  # Don't transform background
                new_val = (val + 1) % 10
                rules.append({
                    'type': 'value_mapping',
                    'condition': f'value == {val}',
                    'action': f'set_to_{new_val}',
                    'from_value': val,
                    'to_value': new_val
                })
        
        # Rule 2: If cell has neighbors with certain properties
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                neighbors = self._get_neighbors(grid, i, j)
                if len(neighbors) > 0:
                    neighbor_sum = sum(neighbors)
                    if neighbor_sum > 0:
                        rules.append({
                            'type': 'neighbor_based',
                            'condition': f'neighbor_sum > 0',
                            'action': 'increment',
                            'position': (i, j),
                            'neighbor_sum': neighbor_sum
                        })
        
        # Rule 3: Pattern-based rules
        if self._has_diagonal_pattern(grid):
            rules.append({
                'type': 'pattern_based',
                'condition': 'diagonal_pattern',
                'action': 'enhance_diagonal'
            })
        
        return rules
    
    def _get_neighbors(self, grid: np.ndarray, i: int, j: int) -> List[int]:
        """Get neighboring cell values"""
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                    neighbors.append(grid[ni, nj])
        return neighbors
    
    def _has_diagonal_pattern(self, grid: np.ndarray) -> bool:
        """Check if grid has diagonal pattern"""
        diag_sum = 0
        for i in range(min(grid.shape)):
            diag_sum += grid[i, i]
        
        return diag_sum > 0
    
    def _score_rules(self, rules: List[Dict], grid: np.ndarray) -> List[Dict]:
        """Score rules based on applicability and consistency"""
        scored_rules = []
        
        for rule in rules:
            score = 0.0
            
            if rule['type'] == 'value_mapping':
                # Score based on frequency of the value
                value_count = np.sum(grid == rule['from_value'])
                score = value_count / grid.size
                
            elif rule['type'] == 'neighbor_based':
                # Score based on neighbor consistency
                score = min(1.0, rule['neighbor_sum'] / 10.0)
                
            elif rule['type'] == 'pattern_based':
                # Fixed score for pattern rules
                score = 0.7
            
            rule['score'] = score
            if score > 0.1:  # Only keep rules with reasonable scores
                scored_rules.append(rule)
        
        return scored_rules
    
    def _apply_rule(self, rule: Dict, grid: np.ndarray) -> np.ndarray:
        """Apply a logical rule to the grid"""
        output = grid.copy()
        
        if rule['type'] == 'value_mapping':
            mask = (grid == rule['from_value'])
            output[mask] = rule['to_value']
            
        elif rule['type'] == 'neighbor_based':
            i, j = rule['position']
            if rule['action'] == 'increment':
                output[i, j] = min(9, output[i, j] + 1)
                
        elif rule['type'] == 'pattern_based':
            if rule['condition'] == 'diagonal_pattern':
                # Enhance diagonal
                for i in range(min(output.shape)):
                    if output[i, i] == 0:
                        output[i, i] = 1
        
        return output
    
    def get_thinking_flow(self) -> Dict[str, Any]:
        return self.thinking_flow

class SymbolicSolver(BaseSolver):
    """Symbolic reasoning solver for complex rule systems"""
    
    def __init__(self):
        self.thinking_flow = {
            'strategy': 'symbolic_reasoning',
            'steps': ['symbolic_analysis', 'rule_synthesis', 'symbolic_execution'],
            'confidence': 0.0
        }
    
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply symbolic reasoning"""
        self.thinking_flow['confidence'] = 0.0
        
        # Step 1: Convert grid to symbolic representation
        symbols = self._gridify_to_symbols(input_grid)
        
        # Step 2: Apply symbolic reasoning rules
        transformed_symbols = self._apply_symbolic_rules(symbols)
        
        # Step 3: Convert back to grid
        output_grid = self._symbols_to_grid(transformed_symbols, input_grid.shape)
        
        self.thinking_flow['confidence'] = 0.75
        
        return output_grid
    
    def _gridify_to_symbols(self, grid: np.ndarray) -> Dict[str, Any]:
        """Convert grid to symbolic representation"""
        symbols = {
            'objects': [],
            'relationships': [],
            'properties': {}
        }
        
        # Identify objects (connected components)
        unique_colors = np.unique(grid)
        object_id = 0
        
        for color in unique_colors:
            if color == 0:
                continue
            
            mask = (grid == color)
            components = self._find_connected_components_symbolic(mask)
            
            for component in components:
                obj = {
                    'id': object_id,
                    'color': color,
                    'positions': component,
                    'size': len(component),
                    'shape': self._classify_shape(component)
                }
                symbols['objects'].append(obj)
                object_id += 1
        
        # Find relationships between objects
        for i, obj1 in enumerate(symbols['objects']):
            for j, obj2 in enumerate(symbols['objects']):
                if i != j:
                    relationship = self._analyze_relationship(obj1, obj2)
                    if relationship:
                        symbols['relationships'].append(relationship)
        
        # Global properties
        symbols['properties']['total_objects'] = len(symbols['objects'])
        symbols['properties']['colors_used'] = len(unique_colors) - 1  # Exclude background
        
        return symbols
    
    def _find_connected_components_symbolic(self, mask: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Find connected components for symbolic analysis"""
        visited = np.zeros_like(mask, dtype=bool)
        components = []
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not visited[i, j]:
                    component = []
                    self._dfs_symbolic(mask, visited, i, j, component)
                    if component:
                        components.append(component)
        
        return components
    
    def _dfs_symbolic(self, mask: np.ndarray, visited: np.ndarray, i: int, j: int, component: List):
        """DFS for symbolic component finding"""
        if (i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1] or 
            visited[i, j] or not mask[i, j]):
            return
        
        visited[i, j] = True
        component.append((i, j))
        
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            self._dfs_symbolic(mask, visited, i + di, j + dj, component)
    
    def _classify_shape(self, positions: List[Tuple[int, int]]) -> str:
        """Classify shape of object"""
        if len(positions) == 1:
            return 'point'
        elif len(positions) == 2:
            return 'line_segment'
        elif len(positions) <= 4:
            return 'small_shape'
        else:
            # Check if it's roughly rectangular
            pos_array = np.array(positions)
            min_i, min_j = pos_array.min(axis=0)
            max_i, max_j = pos_array.max(axis=0)
            
            bounding_area = (max_i - min_i + 1) * (max_j - min_j + 1)
            actual_area = len(positions)
            
            if actual_area / bounding_area > 0.8:
                return 'rectangular'
            else:
                return 'complex'
    
    def _analyze_relationship(self, obj1: Dict, obj2: Dict) -> Dict:
        """Analyze spatial relationship between two objects"""
        pos1 = np.array(obj1['positions'])
        pos2 = np.array(obj2['positions'])
        
        # Compute centroids
        center1 = pos1.mean(axis=0)
        center2 = pos2.mean(axis=0)
        
        # Compute distance
        distance = np.linalg.norm(center2 - center1)
        
        # Determine spatial relationship
        if distance < 2.0:
            spatial_rel = 'adjacent'
        elif distance < 5.0:
            spatial_rel = 'nearby'
        else:
            spatial_rel = 'distant'
        
        # Determine directional relationship
        diff = center2 - center1
        if abs(diff[0]) > abs(diff[1]):
            direction = 'vertical' if diff[0] > 0 else 'vertical'
        else:
            direction = 'horizontal' if diff[1] > 0 else 'horizontal'
        
        return {
            'obj1_id': obj1['id'],
            'obj2_id': obj2['id'],
            'spatial': spatial_rel,
            'direction': direction,
            'distance': distance
        }
    
    def _apply_symbolic_rules(self, symbols: Dict) -> Dict:
        """Apply symbolic transformation rules"""
        transformed = copy.deepcopy(symbols)
        
        # Rule 1: If objects are adjacent, merge them
        for rel in symbols['relationships']:
            if rel['spatial'] == 'adjacent':
                obj1 = next(obj for obj in transformed['objects'] if obj['id'] == rel['obj1_id'])
                obj2 = next(obj for obj in transformed['objects'] if obj['id'] == rel['obj2_id'])
                
                # Create merged object
                merged_positions = obj1['positions'] + obj2['positions']
                merged_obj = {
                    'id': max(obj1['id'], obj2['id']),
                    'color': obj1['color'],  # Keep first object's color
                    'positions': merged_positions,
                    'size': len(merged_positions),
                    'shape': self._classify_shape(merged_positions)
                }
                
                # Replace objects
                transformed['objects'] = [obj for obj in transformed['objects'] 
                                        if obj['id'] not in [obj1['id'], obj2['id']]]
                transformed['objects'].append(merged_obj)
                break  # Apply one rule at a time
        
        # Rule 2: If single objects exist, duplicate them
        if len(transformed['objects']) == 1:
            obj = transformed['objects'][0]
            # Create duplicate with offset
            offset_positions = [(i+1, j+1) for i, j in obj['positions']]
            duplicate = {
                'id': obj['id'] + 100,
                'color': obj['color'],
                'positions': offset_positions,
                'size': len(offset_positions),
                'shape': obj['shape']
            }
            transformed['objects'].append(duplicate)
        
        return transformed
    
    def _symbols_to_grid(self, symbols: Dict, grid_shape: Tuple[int, int]) -> np.ndarray:
        """Convert symbolic representation back to grid"""
        output = np.zeros(grid_shape, dtype=int)
        
        for obj in symbols['objects']:
            for pos in obj['positions']:
                i, j = pos
                if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1]:
                    output[i, j] = obj['color']
        
        return output
    
    def get_thinking_flow(self) -> Dict[str, Any]:
        return self.thinking_flow