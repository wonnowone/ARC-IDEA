#!/usr/bin/env python3
"""
Comprehensive ARC Problem Analysis System

This module implements a sophisticated problem understanding pipeline:

1) Color Filtering: Extract each unique color into separate layers
2) 8-Connected Component Labeling: Find all connected components for shapes
3) Frame Storage: Store both color-based and shape-based representations
4) Change Analysis: Use subtraction to find:
   - Positional changes (meaningful blocks moved)
   - Color changes (absolute location transformations)
   - Object duplication (regardless of color)
   - Added/subtracted elements (not explained by simple transformations)

This provides deep problem understanding for the MoE system.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import cv2
import warnings
from enum import Enum

class ChangeType(Enum):
    """Types of changes detected between frames"""
    POSITIONAL_CHANGE = "positional_change"
    COLOR_CHANGE = "color_change"
    OBJECT_DUPLICATION = "object_duplication"
    OBJECT_ADDITION = "object_addition"
    OBJECT_REMOVAL = "object_removal"
    SHAPE_MODIFICATION = "shape_modification"
    NO_CHANGE = "no_change"

@dataclass
class ConnectedComponent:
    """Represents a connected component (shape)"""
    label: int
    color: int
    positions: List[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    area: int
    centroid: Tuple[float, float]
    shape_signature: str  # Simple shape descriptor
    
    def __post_init__(self):
        """Calculate derived properties"""
        if self.positions:
            positions_array = np.array(self.positions)
            self.centroid = (float(np.mean(positions_array[:, 0])), 
                           float(np.mean(positions_array[:, 1])))
            self.area = len(self.positions)
            
            # Simple shape signature based on bounding box ratio
            height = self.bounding_box[2] - self.bounding_box[0] + 1
            width = self.bounding_box[3] - self.bounding_box[1] + 1
            if height == 1 and width == 1:
                self.shape_signature = "point"
            elif height == 1 or width == 1:
                self.shape_signature = "line"
            elif abs(height - width) <= 1:
                self.shape_signature = "square"
            else:
                self.shape_signature = "rectangle"

@dataclass
class ColorFrame:
    """Represents a single color layer with its components"""
    color: int
    binary_mask: np.ndarray
    connected_components: List[ConnectedComponent]
    component_count: int
    total_pixels: int

@dataclass
class ShapeFrame:
    """Represents shape-based analysis regardless of color"""
    all_components: List[ConnectedComponent]
    component_count: int
    shape_distribution: Dict[str, int]
    size_distribution: Dict[int, int]

@dataclass
class ChangeAnalysis:
    """Analysis of changes between two frames"""
    change_type: ChangeType
    confidence: float
    details: Dict[str, Any]
    affected_components: List[int]
    transformation_hypothesis: Optional[str] = None

class ColorFilter:
    """Filters grid into separate color layers"""
    
    def __init__(self):
        self.color_layers = {}
    
    def extract_color_layers(self, grid: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Extract each unique color into separate binary layers
        
        Args:
            grid: Input grid with multiple colors
            
        Returns:
            Dict mapping color -> binary mask for that color
        """
        unique_colors = np.unique(grid)
        color_layers = {}
        
        for color in unique_colors:
            if color == 0:  # Skip background typically
                continue
            binary_mask = (grid == color).astype(np.uint8)
            color_layers[color] = binary_mask
            
        self.color_layers = color_layers
        return color_layers
    
    def visualize_color_layers(self, grid: np.ndarray) -> Dict[int, np.ndarray]:
        """Create visualization of each color layer"""
        color_layers = self.extract_color_layers(grid)
        visualizations = {}
        
        for color, mask in color_layers.items():
            # Create colored visualization
            vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
            # Simple color mapping (can be enhanced)
            if color == 1:
                vis[mask == 1] = [255, 0, 0]  # Red
            elif color == 2:
                vis[mask == 1] = [0, 255, 0]  # Green
            elif color == 3:
                vis[mask == 1] = [0, 0, 255]  # Blue
            else:
                # Generate color based on value
                vis[mask == 1] = [(color * 50) % 255, (color * 100) % 255, (color * 150) % 255]
            
            visualizations[color] = vis
            
        return visualizations

class ConnectedComponentLabeler:
    """Performs 8-connected component labeling"""
    
    def __init__(self):
        self.component_maps = {}
    
    def label_8_connected_components(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Perform 8-connected component labeling
        
        Args:
            binary_mask: Binary image (0s and 1s)
            
        Returns:
            (labeled_image, num_components)
        """
        try:
            # Use OpenCV for efficient CCL
            num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
            return labels, num_labels - 1  # Subtract 1 to exclude background
        except:
            # Fallback to custom implementation
            return self._custom_ccl_8_connected(binary_mask)
    
    def _custom_ccl_8_connected(self, binary_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Custom 8-connected component labeling implementation"""
        height, width = binary_mask.shape
        labels = np.zeros((height, width), dtype=np.int32)
        current_label = 1
        
        # 8-connectivity offsets
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(height):
            for j in range(width):
                if binary_mask[i, j] == 1 and labels[i, j] == 0:
                    # Start flood fill for new component
                    self._flood_fill_8_connected(binary_mask, labels, i, j, current_label, offsets)
                    current_label += 1
        
        return labels, current_label - 1
    
    def _flood_fill_8_connected(self, binary_mask: np.ndarray, labels: np.ndarray, 
                               start_i: int, start_j: int, label: int, offsets: List[Tuple[int, int]]):
        """Flood fill for 8-connected component labeling"""
        height, width = binary_mask.shape
        stack = [(start_i, start_j)]
        
        while stack:
            i, j = stack.pop()
            
            if (i < 0 or i >= height or j < 0 or j >= width or 
                labels[i, j] != 0 or binary_mask[i, j] != 1):
                continue
            
            labels[i, j] = label
            
            # Add all 8-connected neighbors
            for di, dj in offsets:
                stack.append((i + di, j + dj))
    
    def extract_components_from_labels(self, labels: np.ndarray, color: int) -> List[ConnectedComponent]:
        """Extract ConnectedComponent objects from labeled image"""
        components = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label == 0:  # Skip background
                continue
            
            # Find all positions for this component
            positions = list(zip(*np.where(labels == label)))
            
            if not positions:
                continue
            
            # Calculate bounding box
            positions_array = np.array(positions)
            min_row, min_col = positions_array.min(axis=0)
            max_row, max_col = positions_array.max(axis=0)
            bounding_box = (min_row, min_col, max_row, max_col)
            
            # Create component
            component = ConnectedComponent(
                label=int(label),
                color=color,
                positions=positions,
                bounding_box=bounding_box,
                area=len(positions),
                centroid=(0.0, 0.0),  # Will be calculated in __post_init__
                shape_signature=""     # Will be calculated in __post_init__
            )
            
            components.append(component)
        
        return components

class FrameStorage:
    """Stores both color-based and shape-based frame representations"""
    
    def __init__(self):
        self.color_frames = {}
        self.shape_frames = {}
        self.frame_history = []
    
    def process_and_store_frame(self, grid: np.ndarray, frame_id: str) -> Dict[str, Any]:
        """
        Process a grid and store both color and shape representations
        
        Args:
            grid: Input grid
            frame_id: Identifier for this frame (e.g., "input", "output", "step_1")
            
        Returns:
            Dict with processing results
        """
        # Step 1: Extract color layers
        color_filter = ColorFilter()
        color_layers = color_filter.extract_color_layers(grid)
        
        # Step 2: Process each color layer
        ccl = ConnectedComponentLabeler()
        color_frames = {}
        all_components = []
        
        for color, binary_mask in color_layers.items():
            # Perform CCL on this color layer
            labels, num_components = ccl.label_8_connected_components(binary_mask)
            
            # Extract components
            components = ccl.extract_components_from_labels(labels, color)
            all_components.extend(components)
            
            # Create color frame
            color_frame = ColorFrame(
                color=color,
                binary_mask=binary_mask,
                connected_components=components,
                component_count=len(components),
                total_pixels=int(np.sum(binary_mask))
            )
            
            color_frames[color] = color_frame
        
        # Step 3: Create shape frame (color-agnostic analysis)
        shape_distribution = defaultdict(int)
        size_distribution = defaultdict(int)
        
        for component in all_components:
            shape_distribution[component.shape_signature] += 1
            size_distribution[component.area] += 1
        
        shape_frame = ShapeFrame(
            all_components=all_components,
            component_count=len(all_components),
            shape_distribution=dict(shape_distribution),
            size_distribution=dict(size_distribution)
        )
        
        # Store frames
        self.color_frames[frame_id] = color_frames
        self.shape_frames[frame_id] = shape_frame
        self.frame_history.append(frame_id)
        
        return {
            'color_frames': color_frames,
            'shape_frame': shape_frame,
            'total_components': len(all_components),
            'colors_present': list(color_layers.keys()),
            'processing_stats': {
                'total_colors': len(color_layers),
                'total_components': len(all_components),
                'total_pixels': int(np.sum(grid != 0))
            }
        }
    
    def get_frame_summary(self, frame_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a stored frame"""
        if frame_id not in self.color_frames:
            return {'error': f'Frame {frame_id} not found'}
        
        color_frames = self.color_frames[frame_id]
        shape_frame = self.shape_frames[frame_id]
        
        summary = {
            'frame_id': frame_id,
            'colors_present': list(color_frames.keys()),
            'total_components': shape_frame.component_count,
            'shape_distribution': shape_frame.shape_distribution,
            'color_analysis': {}
        }
        
        for color, color_frame in color_frames.items():
            summary['color_analysis'][color] = {
                'component_count': color_frame.component_count,
                'total_pixels': color_frame.total_pixels,
                'components_summary': [
                    {
                        'label': comp.label,
                        'area': comp.area,
                        'shape': comp.shape_signature,
                        'centroid': comp.centroid
                    }
                    for comp in color_frame.connected_components
                ]
            }
        
        return summary

class ChangeAnalyzer:
    """Analyzes changes between frames to understand transformations"""
    
    def __init__(self):
        self.tolerance = 1e-6
    
    def analyze_changes(self, frame_storage: FrameStorage, 
                       from_frame: str, to_frame: str) -> List[ChangeAnalysis]:
        """
        Comprehensive change analysis between two frames
        
        Args:
            frame_storage: Storage containing both frames
            from_frame: Source frame ID
            to_frame: Target frame ID
            
        Returns:
            List of detected changes
        """
        if from_frame not in frame_storage.color_frames or to_frame not in frame_storage.color_frames:
            return [ChangeAnalysis(ChangeType.NO_CHANGE, 0.0, {}, [])]
        
        from_color_frames = frame_storage.color_frames[from_frame]
        to_color_frames = frame_storage.color_frames[to_frame]
        from_shape_frame = frame_storage.shape_frames[from_frame]
        to_shape_frame = frame_storage.shape_frames[to_frame]
        
        changes = []
        
        # 1. Analyze positional changes
        positional_changes = self._detect_positional_changes(
            from_shape_frame, to_shape_frame
        )
        changes.extend(positional_changes)
        
        # 2. Analyze color changes
        color_changes = self._detect_color_changes(
            from_color_frames, to_color_frames
        )
        changes.extend(color_changes)
        
        # 3. Analyze object duplication
        duplication_changes = self._detect_object_duplication(
            from_shape_frame, to_shape_frame
        )
        changes.extend(duplication_changes)
        
        # 4. Analyze additions/removals
        addition_removal_changes = self._detect_additions_removals(
            from_color_frames, to_color_frames
        )
        changes.extend(addition_removal_changes)
        
        # 5. Analyze shape modifications
        shape_changes = self._detect_shape_modifications(
            from_shape_frame, to_shape_frame
        )
        changes.extend(shape_changes)
        
        return changes if changes else [ChangeAnalysis(ChangeType.NO_CHANGE, 1.0, {}, [])]
    
    def _detect_positional_changes(self, from_frame: ShapeFrame, to_frame: ShapeFrame) -> List[ChangeAnalysis]:
        """Detect meaningful blocks that moved position"""
        changes = []
        
        from_components = from_frame.all_components
        to_components = to_frame.all_components
        
        # Match components by shape and size
        matches = self._match_components_by_shape_size(from_components, to_components)
        
        for from_comp, to_comp in matches:
            # Calculate centroid displacement
            from_centroid = from_comp.centroid
            to_centroid = to_comp.centroid
            
            displacement = np.sqrt((from_centroid[0] - to_centroid[0])**2 + 
                                 (from_centroid[1] - to_centroid[1])**2)
            
            if displacement > 0.5:  # Meaningful movement threshold
                changes.append(ChangeAnalysis(
                    change_type=ChangeType.POSITIONAL_CHANGE,
                    confidence=min(1.0, displacement / 5.0),  # Normalize confidence
                    details={
                        'from_component': from_comp.label,
                        'to_component': to_comp.label,
                        'displacement': displacement,
                        'from_centroid': from_centroid,
                        'to_centroid': to_centroid,
                        'shape': from_comp.shape_signature
                    },
                    affected_components=[from_comp.label, to_comp.label],
                    transformation_hypothesis="translation"
                ))
        
        return changes
    
    def _detect_color_changes(self, from_frames: Dict[int, ColorFrame], 
                            to_frames: Dict[int, ColorFrame]) -> List[ChangeAnalysis]:
        """Detect color changes at absolute locations"""
        changes = []
        
        all_colors = set(from_frames.keys()) | set(to_frames.keys())
        
        for color in all_colors:
            from_mask = from_frames.get(color, ColorFrame(color, np.array([]), [], 0, 0)).binary_mask
            to_mask = to_frames.get(color, ColorFrame(color, np.array([]), [], 0, 0)).binary_mask
            
            if from_mask.size == 0 or to_mask.size == 0:
                continue
            
            # Ensure same size
            if from_mask.shape != to_mask.shape:
                continue
            
            # Find differences
            added = (to_mask == 1) & (from_mask == 0)
            removed = (from_mask == 1) & (to_mask == 0)
            
            if np.any(added) or np.any(removed):
                changes.append(ChangeAnalysis(
                    change_type=ChangeType.COLOR_CHANGE,
                    confidence=0.8,
                    details={
                        'color': color,
                        'pixels_added': int(np.sum(added)),
                        'pixels_removed': int(np.sum(removed)),
                        'added_positions': list(zip(*np.where(added))) if np.any(added) else [],
                        'removed_positions': list(zip(*np.where(removed))) if np.any(removed) else []
                    },
                    affected_components=[],
                    transformation_hypothesis="color_transformation"
                ))
        
        return changes
    
    def _detect_object_duplication(self, from_frame: ShapeFrame, to_frame: ShapeFrame) -> List[ChangeAnalysis]:
        """Detect object duplication regardless of color"""
        changes = []
        
        from_shapes = defaultdict(list)
        to_shapes = defaultdict(list)
        
        # Group by shape signature and size
        for comp in from_frame.all_components:
            key = (comp.shape_signature, comp.area)
            from_shapes[key].append(comp)
        
        for comp in to_frame.all_components:
            key = (comp.shape_signature, comp.area)
            to_shapes[key].append(comp)
        
        # Find duplications
        for shape_key, from_comps in from_shapes.items():
            to_comps = to_shapes.get(shape_key, [])
            
            if len(to_comps) > len(from_comps):
                # Duplication detected
                duplications = len(to_comps) - len(from_comps)
                changes.append(ChangeAnalysis(
                    change_type=ChangeType.OBJECT_DUPLICATION,
                    confidence=0.9,
                    details={
                        'shape_signature': shape_key[0],
                        'area': shape_key[1],
                        'original_count': len(from_comps),
                        'new_count': len(to_comps),
                        'duplications': duplications
                    },
                    affected_components=[comp.label for comp in to_comps],
                    transformation_hypothesis="duplication"
                ))
        
        return changes
    
    def _detect_additions_removals(self, from_frames: Dict[int, ColorFrame], 
                                 to_frames: Dict[int, ColorFrame]) -> List[ChangeAnalysis]:
        """Detect elements added or removed (not explained by transformations)"""
        changes = []
        
        # Calculate total pixels per color
        from_totals = {color: frame.total_pixels for color, frame in from_frames.items()}
        to_totals = {color: frame.total_pixels for color, frame in to_frames.items()}
        
        all_colors = set(from_totals.keys()) | set(to_totals.keys())
        
        for color in all_colors:
            from_pixels = from_totals.get(color, 0)
            to_pixels = to_totals.get(color, 0)
            
            if from_pixels != to_pixels:
                if to_pixels > from_pixels:
                    change_type = ChangeType.OBJECT_ADDITION
                    pixel_change = to_pixels - from_pixels
                else:
                    change_type = ChangeType.OBJECT_REMOVAL
                    pixel_change = from_pixels - to_pixels
                
                changes.append(ChangeAnalysis(
                    change_type=change_type,
                    confidence=0.7,
                    details={
                        'color': color,
                        'pixel_change': pixel_change,
                        'from_pixels': from_pixels,
                        'to_pixels': to_pixels
                    },
                    affected_components=[],
                    transformation_hypothesis="addition_removal"
                ))
        
        return changes
    
    def _detect_shape_modifications(self, from_frame: ShapeFrame, to_frame: ShapeFrame) -> List[ChangeAnalysis]:
        """Detect shape modifications within objects"""
        changes = []
        
        # Compare shape distributions
        from_shapes = from_frame.shape_distribution
        to_shapes = to_frame.shape_distribution
        
        all_shape_types = set(from_shapes.keys()) | set(to_shapes.keys())
        
        for shape_type in all_shape_types:
            from_count = from_shapes.get(shape_type, 0)
            to_count = to_shapes.get(shape_type, 0)
            
            if from_count != to_count:
                changes.append(ChangeAnalysis(
                    change_type=ChangeType.SHAPE_MODIFICATION,
                    confidence=0.6,
                    details={
                        'shape_type': shape_type,
                        'from_count': from_count,
                        'to_count': to_count,
                        'change': to_count - from_count
                    },
                    affected_components=[],
                    transformation_hypothesis="shape_modification"
                ))
        
        return changes
    
    def _match_components_by_shape_size(self, from_components: List[ConnectedComponent], 
                                      to_components: List[ConnectedComponent]) -> List[Tuple[ConnectedComponent, ConnectedComponent]]:
        """Match components between frames by shape and size"""
        matches = []
        used_to = set()
        
        for from_comp in from_components:
            best_match = None
            best_score = float('inf')
            
            for to_comp in to_components:
                if to_comp.label in used_to:
                    continue
                
                # Match by shape signature and area
                if (from_comp.shape_signature == to_comp.shape_signature and 
                    from_comp.area == to_comp.area):
                    
                    # Calculate centroid distance as tie-breaker
                    distance = np.sqrt((from_comp.centroid[0] - to_comp.centroid[0])**2 + 
                                     (from_comp.centroid[1] - to_comp.centroid[1])**2)
                    
                    if distance < best_score:
                        best_score = distance
                        best_match = to_comp
            
            if best_match is not None:
                matches.append((from_comp, best_match))
                used_to.add(best_match.label)
        
        return matches

class ARCProblemAnalyzer:
    """Main class that orchestrates the complete problem analysis"""
    
    def __init__(self):
        self.frame_storage = FrameStorage()
        self.change_analyzer = ChangeAnalyzer()
        self.analysis_history = []
    
    def analyze_problem_sequence(self, grids: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a complete ARC problem sequence
        
        Args:
            grids: Dict with keys like 'input', 'output', 'step_1', etc.
            
        Returns:
            Comprehensive analysis results
        """
        print(f"🔍 Analyzing ARC problem sequence with {len(grids)} frames...")
        
        # Step 1: Process and store all frames
        frame_results = {}
        for frame_id, grid in grids.items():
            print(f"  Processing frame: {frame_id}")
            result = self.frame_storage.process_and_store_frame(grid, frame_id)
            frame_results[frame_id] = result
        
        # Step 2: Analyze changes between consecutive frames
        frame_ids = list(grids.keys())
        change_analyses = {}
        
        for i in range(len(frame_ids) - 1):
            from_frame = frame_ids[i]
            to_frame = frame_ids[i + 1]
            
            print(f"  Analyzing changes: {from_frame} → {to_frame}")
            changes = self.change_analyzer.analyze_changes(
                self.frame_storage, from_frame, to_frame
            )
            change_analyses[f"{from_frame}_to_{to_frame}"] = changes
        
        # Step 3: Synthesize overall understanding
        problem_understanding = self._synthesize_problem_understanding(
            frame_results, change_analyses
        )
        
        comprehensive_analysis = {
            'frame_results': frame_results,
            'change_analyses': change_analyses,
            'problem_understanding': problem_understanding,
            'processing_summary': {
                'total_frames': len(grids),
                'total_changes_detected': sum(len(changes) for changes in change_analyses.values()),
                'dominant_transformation': problem_understanding.get('dominant_transformation', 'unknown')
            }
        }
        
        self.analysis_history.append(comprehensive_analysis)
        return comprehensive_analysis
    
    def _synthesize_problem_understanding(self, frame_results: Dict, change_analyses: Dict) -> Dict[str, Any]:
        """Synthesize overall problem understanding from all analyses"""
        understanding = {
            'transformation_patterns': [],
            'dominant_transformation': None,
            'complexity_level': 'unknown',
            'key_insights': []
        }
        
        # Collect all change types
        all_changes = []
        for changes in change_analyses.values():
            all_changes.extend(changes)
        
        if not all_changes:
            understanding['dominant_transformation'] = 'identity'
            understanding['complexity_level'] = 'trivial'
            return understanding
        
        # Analyze change patterns
        change_types = [change.change_type for change in all_changes]
        change_counter = defaultdict(int)
        
        for change_type in change_types:
            change_counter[change_type] += 1
        
        # Determine dominant transformation
        if change_counter:
            dominant_change = max(change_counter.items(), key=lambda x: x[1])
            understanding['dominant_transformation'] = dominant_change[0].value
            
            # Classify complexity
            if len(change_counter) == 1:
                understanding['complexity_level'] = 'simple'
            elif len(change_counter) <= 3:
                understanding['complexity_level'] = 'moderate'
            else:
                understanding['complexity_level'] = 'complex'
        
        # Extract key insights
        insights = []
        
        if ChangeType.POSITIONAL_CHANGE in change_counter:
            insights.append(f"Objects move position ({change_counter[ChangeType.POSITIONAL_CHANGE]} instances)")
        
        if ChangeType.COLOR_CHANGE in change_counter:
            insights.append(f"Colors change location ({change_counter[ChangeType.COLOR_CHANGE]} instances)")
        
        if ChangeType.OBJECT_DUPLICATION in change_counter:
            insights.append(f"Objects are duplicated ({change_counter[ChangeType.OBJECT_DUPLICATION]} instances)")
        
        if ChangeType.OBJECT_ADDITION in change_counter:
            insights.append(f"New objects appear ({change_counter[ChangeType.OBJECT_ADDITION]} instances)")
        
        if ChangeType.OBJECT_REMOVAL in change_counter:
            insights.append(f"Objects disappear ({change_counter[ChangeType.OBJECT_REMOVAL]} instances)")
        
        understanding['key_insights'] = insights
        understanding['transformation_patterns'] = list(change_counter.keys())
        
        return understanding
    
    def get_problem_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Generate human-readable problem summary"""
        understanding = analysis_result['problem_understanding']
        
        summary_parts = [
            f"🎯 ARC Problem Analysis Summary",
            f"=" * 50,
            f"Dominant Transformation: {understanding.get('dominant_transformation', 'Unknown')}",
            f"Complexity Level: {understanding.get('complexity_level', 'Unknown')}",
            f"Total Frames Analyzed: {analysis_result['processing_summary']['total_frames']}",
            f"Total Changes Detected: {analysis_result['processing_summary']['total_changes_detected']}",
            "",
            f"🔍 Key Insights:"
        ]
        
        insights = understanding.get('key_insights', [])
        for insight in insights:
            summary_parts.append(f"  • {insight}")
        
        if not insights:
            summary_parts.append("  • No significant patterns detected")
        
        return "\\n".join(summary_parts)

# Example usage and testing functions
def create_test_problem() -> Dict[str, np.ndarray]:
    """Create a test ARC problem for demonstration"""
    # Input: Simple shape
    input_grid = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    # Output: Shape moved and duplicated
    output_grid = np.array([
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ])
    
    return {'input': input_grid, 'output': output_grid}

def demonstrate_analysis():
    """Demonstrate the complete analysis pipeline"""
    print("🚀 ARC Problem Analysis Pipeline Demonstration")
    print("=" * 60)
    
    # Create analyzer
    analyzer = ARCProblemAnalyzer()
    
    # Create test problem
    test_problem = create_test_problem()
    
    print("Test Problem:")
    print("Input:")
    print(test_problem['input'])
    print("\\nOutput:")
    print(test_problem['output'])
    
    # Run complete analysis
    analysis = analyzer.analyze_problem_sequence(test_problem)
    
    # Print summary
    summary = analyzer.get_problem_summary(analysis)
    print(f"\\n{summary}")
    
    return analysis

if __name__ == "__main__":
    demonstrate_analysis()