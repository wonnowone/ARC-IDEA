#!/usr/bin/env python3
"""
Movement Experts for ARC EFE System with Mixture of Experts

This module implements the core movement expert architecture that serves as atomic
transformation operations for the enhanced ARC solver system. Each expert specializes
in a specific type of grid transformation.

Architecture:
- BaseMovementExpert: Abstract interface for all movement operations
- Specialized experts: Flip, Rotation, Translation, ColorTransform, etc.
- MovementResult: Standardized result format with metadata
- MovementValidator: Safety and validation system
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings

@dataclass
class MovementResult:
    """Result of a movement expert operation"""
    output_grid: np.ndarray
    confidence: float
    operation_type: str
    parameters: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    intermediate_states: Optional[List[np.ndarray]] = None
    reversible: bool = True
    metadata: Optional[Dict[str, Any]] = None

class MovementType(Enum):
    """Enumeration of supported movement types"""
    FLIP = "flip"
    ROTATION = "rotation"  
    TRANSLATION = "translation"
    COLOR_TRANSFORM = "color_transform"
    SCALING = "scaling"
    PATTERN = "pattern"
    MORPHOLOGY = "morphology"
    LOGICAL = "logical"
    COMPOSITE = "composite"

class BaseMovementExpert(ABC):
    """Abstract base class for all movement experts"""
    
    def __init__(self, expert_name: str, movement_type: MovementType):
        self.expert_name = expert_name
        self.movement_type = movement_type
        self.execution_count = 0
        self.success_count = 0
        self.total_execution_time = 0.0
        self.confidence_history = []
        
    @abstractmethod
    def execute(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> MovementResult:
        """
        Execute the movement transformation
        
        Args:
            input_grid: Input grid to transform
            parameters: Transformation parameters specific to each expert
            
        Returns:
            MovementResult with transformation output and metadata
        """
        pass
    
    @abstractmethod
    def get_confidence(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> float:
        """
        Estimate confidence for applying this transformation
        
        Args:
            input_grid: Input grid to analyze
            parameters: Proposed transformation parameters
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass
    
    @abstractmethod
    def is_applicable(self, input_grid: np.ndarray, context: Dict[str, Any]) -> bool:
        """
        Check if this expert can be applied to the given grid
        
        Args:
            input_grid: Input grid to check
            context: Additional context about the problem
            
        Returns:
            True if expert is applicable, False otherwise
        """
        pass
    
    @abstractmethod
    def get_parameter_space(self) -> Dict[str, Any]:
        """
        Get the valid parameter space for this expert
        
        Returns:
            Dictionary describing valid parameters and their ranges
        """
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for this expert"""
        success_rate = self.success_count / max(1, self.execution_count)
        avg_time = self.total_execution_time / max(1, self.execution_count)
        avg_confidence = np.mean(self.confidence_history) if self.confidence_history else 0.0
        
        return {
            'expert_name': self.expert_name,
            'movement_type': self.movement_type.value,
            'execution_count': self.execution_count,
            'success_rate': success_rate,
            'average_execution_time': avg_time,
            'average_confidence': avg_confidence
        }
    
    def _validate_input(self, input_grid: np.ndarray) -> bool:
        """Validate input grid"""
        if input_grid is None or input_grid.size == 0:
            return False
        if not np.all(np.isfinite(input_grid)):
            return False
        return True
    
    def _update_statistics(self, result: MovementResult):
        """Update expert statistics"""
        self.execution_count += 1
        if result.success:
            self.success_count += 1
        self.total_execution_time += result.execution_time
        self.confidence_history.append(result.confidence)
        
        # Keep only recent history
        if len(self.confidence_history) > 100:
            self.confidence_history = self.confidence_history[-100:]

class FlipExpert(BaseMovementExpert):
    """Expert for flip transformations (horizontal, vertical, diagonal)"""
    
    def __init__(self):
        super().__init__("FlipExpert", MovementType.FLIP)
        self.supported_axes = ['horizontal', 'vertical', 'main_diagonal', 'anti_diagonal']
    
    def execute(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> MovementResult:
        """Execute flip transformation"""
        import time
        start_time = time.time()
        
        try:
            if not self._validate_input(input_grid):
                return MovementResult(
                    output_grid=input_grid.copy(),
                    confidence=0.0,
                    operation_type="flip",
                    parameters=parameters,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Invalid input grid"
                )
            
            axis = parameters.get('axis', 'horizontal')
            if axis not in self.supported_axes:
                return MovementResult(
                    output_grid=input_grid.copy(),
                    confidence=0.0,
                    operation_type="flip",
                    parameters=parameters,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message=f"Unsupported axis: {axis}"
                )
            
            # Perform flip operation
            if axis == 'horizontal':
                output_grid = np.fliplr(input_grid)
            elif axis == 'vertical':
                output_grid = np.flipud(input_grid)
            elif axis == 'main_diagonal':
                output_grid = np.transpose(input_grid)
            elif axis == 'anti_diagonal':
                output_grid = np.rot90(np.transpose(input_grid), 2)
            
            confidence = self.get_confidence(input_grid, parameters)
            execution_time = time.time() - start_time
            
            result = MovementResult(
                output_grid=output_grid,
                confidence=confidence,
                operation_type="flip",
                parameters=parameters,
                execution_time=execution_time,
                success=True,
                reversible=True,
                metadata={'axis': axis}
            )
            
            self._update_statistics(result)
            return result
            
        except Exception as e:
            return MovementResult(
                output_grid=input_grid.copy(),
                confidence=0.0,
                operation_type="flip",
                parameters=parameters,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def get_confidence(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> float:
        """Calculate confidence for flip operation"""
        try:
            axis = parameters.get('axis', 'horizontal')
            
            # Higher confidence for symmetric grids
            if axis == 'horizontal':
                flipped = np.fliplr(input_grid)
            elif axis == 'vertical':
                flipped = np.flipud(input_grid)
            elif axis == 'main_diagonal':
                if input_grid.shape[0] != input_grid.shape[1]:
                    return 0.3  # Low confidence for non-square grids
                flipped = np.transpose(input_grid)
            else:
                return 0.5  # Default confidence
            
            # Calculate symmetry score
            if flipped.shape == input_grid.shape:
                similarity = np.mean(input_grid == flipped)
                # If grid is already symmetric, flip operation is very reliable
                if similarity > 0.8:
                    return 0.95
                elif similarity > 0.5:
                    return 0.8
                else:
                    return 0.7
            
            return 0.6
            
        except:
            return 0.5
    
    def is_applicable(self, input_grid: np.ndarray, context: Dict[str, Any]) -> bool:
        """Check if flip can be applied"""
        if not self._validate_input(input_grid):
            return False
        
        # Flip is almost always applicable
        axis = context.get('axis', 'horizontal')
        if axis == 'main_diagonal' or axis == 'anti_diagonal':
            # Diagonal flips work best on square grids
            return input_grid.shape[0] == input_grid.shape[1]
        
        return True
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for flip operations"""
        return {
            'axis': {
                'type': 'categorical',
                'values': self.supported_axes,
                'default': 'horizontal'
            }
        }

class RotationExpert(BaseMovementExpert):
    """Expert for rotation transformations (90°, 180°, 270°)"""
    
    def __init__(self):
        super().__init__("RotationExpert", MovementType.ROTATION)
        self.supported_angles = [90, 180, 270]
    
    def execute(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> MovementResult:
        """Execute rotation transformation"""
        import time
        start_time = time.time()
        
        try:
            if not self._validate_input(input_grid):
                return MovementResult(
                    output_grid=input_grid.copy(),
                    confidence=0.0,
                    operation_type="rotation",
                    parameters=parameters,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Invalid input grid"
                )
            
            angle = parameters.get('angle', 90)
            if angle not in self.supported_angles:
                return MovementResult(
                    output_grid=input_grid.copy(),
                    confidence=0.0,
                    operation_type="rotation",
                    parameters=parameters,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message=f"Unsupported angle: {angle}"
                )
            
            # Perform rotation
            k = angle // 90  # Number of 90-degree rotations
            output_grid = np.rot90(input_grid, k)
            
            confidence = self.get_confidence(input_grid, parameters)
            execution_time = time.time() - start_time
            
            result = MovementResult(
                output_grid=output_grid,
                confidence=confidence,
                operation_type="rotation",
                parameters=parameters,
                execution_time=execution_time,
                success=True,
                reversible=True,
                metadata={'angle': angle, 'rotations': k}
            )
            
            self._update_statistics(result)
            return result
            
        except Exception as e:
            return MovementResult(
                output_grid=input_grid.copy(),
                confidence=0.0,
                operation_type="rotation",
                parameters=parameters,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def get_confidence(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> float:
        """Calculate confidence for rotation operation"""
        try:
            angle = parameters.get('angle', 90)
            
            # Higher confidence for square grids
            if input_grid.shape[0] == input_grid.shape[1]:
                base_confidence = 0.9
            else:
                # Non-square grids change shape with 90/270 degree rotations
                if angle in [90, 270]:
                    base_confidence = 0.6
                else:  # 180 degree rotation preserves shape
                    base_confidence = 0.85
            
            # Check for rotational symmetry
            k = angle // 90
            rotated = np.rot90(input_grid, k)
            if rotated.shape == input_grid.shape:
                similarity = np.mean(input_grid == rotated)
                if similarity > 0.9:
                    base_confidence = 0.95
            
            return base_confidence
            
        except:
            return 0.7
    
    def is_applicable(self, input_grid: np.ndarray, context: Dict[str, Any]) -> bool:
        """Check if rotation can be applied"""
        if not self._validate_input(input_grid):
            return False
        
        # Rotation is generally applicable
        # For 90/270 degree rotations on non-square grids, shape will change
        angle = context.get('angle', 90)
        if angle in [90, 270] and input_grid.shape[0] != input_grid.shape[1]:
            # Check if context allows shape changes
            return context.get('allow_shape_change', True)
        
        return True
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for rotation operations"""
        return {
            'angle': {
                'type': 'categorical',
                'values': self.supported_angles,
                'default': 90
            }
        }

class TranslationExpert(BaseMovementExpert):
    """Expert for translation transformations (shift with boundary handling)"""
    
    def __init__(self):
        super().__init__("TranslationExpert", MovementType.TRANSLATION)
        self.boundary_modes = ['wrap', 'constant', 'edge', 'reflect']
    
    def execute(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> MovementResult:
        """Execute translation transformation"""
        import time
        start_time = time.time()
        
        try:
            if not self._validate_input(input_grid):
                return MovementResult(
                    output_grid=input_grid.copy(),
                    confidence=0.0,
                    operation_type="translation",
                    parameters=parameters,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Invalid input grid"
                )
            
            shift_x = parameters.get('shift_x', 0)
            shift_y = parameters.get('shift_y', 0)
            mode = parameters.get('mode', 'wrap')
            fill_value = parameters.get('fill_value', 0)
            
            if mode not in self.boundary_modes:
                mode = 'wrap'
            
            # Perform translation
            output_grid = self._shift_grid(input_grid, shift_x, shift_y, mode, fill_value)
            
            confidence = self.get_confidence(input_grid, parameters)
            execution_time = time.time() - start_time
            
            result = MovementResult(
                output_grid=output_grid,
                confidence=confidence,
                operation_type="translation",
                parameters=parameters,
                execution_time=execution_time,
                success=True,
                reversible=True,
                metadata={'shift_x': shift_x, 'shift_y': shift_y, 'mode': mode}
            )
            
            self._update_statistics(result)
            return result
            
        except Exception as e:
            return MovementResult(
                output_grid=input_grid.copy(),
                confidence=0.0,
                operation_type="translation",
                parameters=parameters,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _shift_grid(self, grid: np.ndarray, shift_x: int, shift_y: int, mode: str, fill_value: int) -> np.ndarray:
        """Shift grid with specified boundary handling"""
        if shift_x == 0 and shift_y == 0:
            return grid.copy()
        
        output = np.zeros_like(grid)
        h, w = grid.shape
        
        if mode == 'wrap':
            # Circular shift
            for i in range(h):
                for j in range(w):
                    new_i = (i + shift_x) % h
                    new_j = (j + shift_y) % w
                    output[new_i, new_j] = grid[i, j]
        elif mode == 'constant':
            # Fill with constant value
            for i in range(h):
                for j in range(w):
                    new_i = i + shift_x
                    new_j = j + shift_y
                    if 0 <= new_i < h and 0 <= new_j < w:
                        output[new_i, new_j] = grid[i, j]
                    else:
                        if new_i >= 0 and new_i < h and new_j >= 0 and new_j < w:
                            output[new_i, new_j] = fill_value
            # Fill empty areas
            mask = (output == 0) & (grid != 0)  # Areas that should be filled
            output[mask] = fill_value
        elif mode == 'edge':
            # Extend edge values
            for i in range(h):
                for j in range(w):
                    new_i = i + shift_x
                    new_j = j + shift_y
                    if 0 <= new_i < h and 0 <= new_j < w:
                        output[new_i, new_j] = grid[i, j]
            # Fill edges
            self._fill_edges(output, grid, shift_x, shift_y)
        elif mode == 'reflect':
            # Reflect at boundaries
            for i in range(h):
                for j in range(w):
                    new_i = (i + shift_x) 
                    new_j = (j + shift_y)
                    
                    # Reflect coordinates
                    if new_i < 0:
                        new_i = -new_i - 1
                    elif new_i >= h:
                        new_i = 2 * h - new_i - 1
                    
                    if new_j < 0:
                        new_j = -new_j - 1
                    elif new_j >= w:
                        new_j = 2 * w - new_j - 1
                    
                    # Ensure coordinates are valid
                    new_i = max(0, min(new_i, h - 1))
                    new_j = max(0, min(new_j, w - 1))
                    
                    output[new_i, new_j] = grid[i, j]
        
        return output
    
    def _fill_edges(self, output: np.ndarray, original: np.ndarray, shift_x: int, shift_y: int):
        """Fill edges for edge mode"""
        h, w = output.shape
        
        # Simple edge filling - extend nearest values
        for i in range(h):
            for j in range(w):
                if output[i, j] == 0:  # Empty cell
                    # Find nearest non-zero value
                    min_dist = float('inf')
                    nearest_val = 0
                    
                    for ii in range(h):
                        for jj in range(w):
                            if output[ii, jj] != 0:
                                dist = abs(i - ii) + abs(j - jj)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_val = output[ii, jj]
                    
                    output[i, j] = nearest_val if nearest_val != 0 else original[min(i, h-1), min(j, w-1)]
    
    def get_confidence(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> float:
        """Calculate confidence for translation operation"""
        try:
            shift_x = abs(parameters.get('shift_x', 0))
            shift_y = abs(parameters.get('shift_y', 0))
            mode = parameters.get('mode', 'wrap')
            
            # Base confidence depends on shift magnitude
            max_shift = max(shift_x, shift_y)
            grid_size = min(input_grid.shape)
            
            if max_shift == 0:
                return 0.95  # Identity transformation
            elif max_shift >= grid_size:
                return 0.3   # Large shift, may lose information
            else:
                base_confidence = 0.8 - (max_shift / grid_size) * 0.3
            
            # Mode affects confidence
            mode_confidence = {
                'wrap': 0.9,
                'constant': 0.7,
                'edge': 0.75,
                'reflect': 0.8
            }
            
            return base_confidence * mode_confidence.get(mode, 0.7)
            
        except:
            return 0.6
    
    def is_applicable(self, input_grid: np.ndarray, context: Dict[str, Any]) -> bool:
        """Check if translation can be applied"""
        if not self._validate_input(input_grid):
            return False
        
        # Translation is generally applicable
        return True
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for translation operations"""
        return {
            'shift_x': {
                'type': 'integer',
                'range': [-10, 10],
                'default': 0
            },
            'shift_y': {
                'type': 'integer', 
                'range': [-10, 10],
                'default': 0
            },
            'mode': {
                'type': 'categorical',
                'values': self.boundary_modes,
                'default': 'wrap'
            },
            'fill_value': {
                'type': 'integer',
                'range': [0, 9],
                'default': 0
            }
        }

class ColorTransformExpert(BaseMovementExpert):
    """Expert for color transformation operations"""
    
    def __init__(self):
        super().__init__("ColorTransformExpert", MovementType.COLOR_TRANSFORM)
        self.transform_types = ['swap', 'map', 'increment', 'replace', 'gradient']
    
    def execute(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> MovementResult:
        """Execute color transformation"""
        import time
        start_time = time.time()
        
        try:
            if not self._validate_input(input_grid):
                return MovementResult(
                    output_grid=input_grid.copy(),
                    confidence=0.0,
                    operation_type="color_transform",
                    parameters=parameters,
                    execution_time=time.time() - start_time,
                    success=False,
                    error_message="Invalid input grid"
                )
            
            transform_type = parameters.get('type', 'swap')
            
            if transform_type == 'swap':
                output_grid = self._color_swap(input_grid, parameters)
            elif transform_type == 'map':
                output_grid = self._color_map(input_grid, parameters)
            elif transform_type == 'increment':
                output_grid = self._color_increment(input_grid, parameters)
            elif transform_type == 'replace':
                output_grid = self._color_replace(input_grid, parameters)
            elif transform_type == 'gradient':
                output_grid = self._color_gradient(input_grid, parameters)
            else:
                output_grid = input_grid.copy()
            
            confidence = self.get_confidence(input_grid, parameters)
            execution_time = time.time() - start_time
            
            result = MovementResult(
                output_grid=output_grid,
                confidence=confidence,
                operation_type="color_transform",
                parameters=parameters,
                execution_time=execution_time,
                success=True,
                reversible=(transform_type in ['swap', 'map']),
                metadata={'transform_type': transform_type}
            )
            
            self._update_statistics(result)
            return result
            
        except Exception as e:
            return MovementResult(
                output_grid=input_grid.copy(),
                confidence=0.0,
                operation_type="color_transform",
                parameters=parameters,
                execution_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _color_swap(self, grid: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Swap two colors"""
        color1 = parameters.get('color1', 1)
        color2 = parameters.get('color2', 2)
        
        output = grid.copy()
        mask1 = (grid == color1)
        mask2 = (grid == color2)
        
        output[mask1] = color2
        output[mask2] = color1
        
        return output
    
    def _color_map(self, grid: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply color mapping"""
        color_map = parameters.get('mapping', {})
        
        output = grid.copy()
        for old_color, new_color in color_map.items():
            mask = (grid == old_color)
            output[mask] = new_color
        
        return output
    
    def _color_increment(self, grid: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Increment colors with wrapping"""
        increment = parameters.get('increment', 1)
        modulo = parameters.get('modulo', 10)
        
        output = (grid + increment) % modulo
        return output
    
    def _color_replace(self, grid: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Replace specific color with another"""
        old_color = parameters.get('old_color', 0)
        new_color = parameters.get('new_color', 1)
        
        output = grid.copy()
        mask = (grid == old_color)
        output[mask] = new_color
        
        return output
    
    def _color_gradient(self, grid: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply color gradient"""
        direction = parameters.get('direction', 'horizontal')
        start_color = parameters.get('start_color', 0)
        end_color = parameters.get('end_color', 9)
        
        output = np.zeros_like(grid)
        h, w = grid.shape
        
        if direction == 'horizontal':
            for j in range(w):
                ratio = j / max(1, w - 1)
                color = int(start_color + ratio * (end_color - start_color))
                output[:, j] = color
        elif direction == 'vertical':
            for i in range(h):
                ratio = i / max(1, h - 1)
                color = int(start_color + ratio * (end_color - start_color))
                output[i, :] = color
        
        return output
    
    def get_confidence(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> float:
        """Calculate confidence for color transformation"""
        try:
            transform_type = parameters.get('type', 'swap')
            
            # Analyze color distribution
            unique_colors = np.unique(input_grid)
            
            if transform_type == 'swap':
                color1 = parameters.get('color1', 1)
                color2 = parameters.get('color2', 2)
                if color1 in unique_colors and color2 in unique_colors:
                    return 0.9
                elif color1 in unique_colors or color2 in unique_colors:
                    return 0.7
                else:
                    return 0.3
                    
            elif transform_type == 'map':
                mapping = parameters.get('mapping', {})
                relevant_colors = [c for c in mapping.keys() if c in unique_colors]
                return min(0.9, 0.5 + len(relevant_colors) * 0.1)
                
            else:
                return 0.7
                
        except:
            return 0.6
    
    def is_applicable(self, input_grid: np.ndarray, context: Dict[str, Any]) -> bool:
        """Check if color transform can be applied"""
        if not self._validate_input(input_grid):
            return False
        
        # Color transforms are generally applicable if grid has colors
        unique_colors = np.unique(input_grid)
        return len(unique_colors) > 0
    
    def get_parameter_space(self) -> Dict[str, Any]:
        """Get parameter space for color transformations"""
        return {
            'type': {
                'type': 'categorical',
                'values': self.transform_types,
                'default': 'swap'
            },
            'color1': {
                'type': 'integer',
                'range': [0, 9],
                'default': 1
            },
            'color2': {
                'type': 'integer',
                'range': [0, 9],
                'default': 2
            },
            'mapping': {
                'type': 'dict',
                'key_range': [0, 9],
                'value_range': [0, 9],
                'default': {}
            },
            'increment': {
                'type': 'integer',
                'range': [1, 9],
                'default': 1
            }
        }

class MovementValidator:
    """Validation system for movement operations"""
    
    @staticmethod
    def validate_grid(grid: np.ndarray) -> Tuple[bool, str]:
        """Validate grid format and content"""
        if grid is None:
            return False, "Grid is None"
        
        if grid.size == 0:
            return False, "Grid is empty"
        
        if len(grid.shape) != 2:
            return False, "Grid must be 2D"
        
        if not np.all(np.isfinite(grid)):
            return False, "Grid contains non-finite values"
        
        if grid.dtype not in [np.int32, np.int64, int]:
            if not np.all(grid == grid.astype(int)):
                return False, "Grid must contain integer values"
        
        if np.any(grid < 0) or np.any(grid > 9):
            return False, "Grid values must be between 0 and 9"
        
        return True, "Valid"
    
    @staticmethod
    def validate_parameters(parameters: Dict[str, Any], expert: BaseMovementExpert) -> Tuple[bool, str]:
        """Validate parameters for specific expert"""
        param_space = expert.get_parameter_space()
        
        for param_name, param_config in param_space.items():
            if param_name in parameters:
                value = parameters[param_name]
                param_type = param_config.get('type')
                
                if param_type == 'integer':
                    if not isinstance(value, (int, np.integer)):
                        return False, f"Parameter {param_name} must be integer"
                    param_range = param_config.get('range', [0, 9])
                    if value < param_range[0] or value > param_range[1]:
                        return False, f"Parameter {param_name} out of range {param_range}"
                
                elif param_type == 'categorical':
                    valid_values = param_config.get('values', [])
                    if value not in valid_values:
                        return False, f"Parameter {param_name} must be one of {valid_values}"
        
        return True, "Valid"