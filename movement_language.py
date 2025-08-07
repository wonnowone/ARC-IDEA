#!/usr/bin/env python3
"""
Declarative Movement Language for ARC MoE System

This module provides a high-level declarative language for composing movement
operations. Solvers can express their intentions using this language, which
gets compiled into expert calls and executed by the MoE system.

Components:
- MovementScript: High-level movement program representation
- MovementCompiler: Compiles scripts into expert calls
- MovementPrimitives: Basic movement operations
- MovementCompositions: Complex movement sequences
- ConditionalMovements: Logic-based movement selection
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from abc import ABC, abstractmethod

from movement_experts import MovementType, MovementResult
from moe_router import MovementMoERouter, ExpertCall, RoutingStrategy

class MovementOperator(Enum):
    """Basic movement operators"""
    FLIP = "flip"
    ROTATE = "rotate" 
    TRANSLATE = "translate"
    COLOR_SWAP = "color_swap"
    COLOR_MAP = "color_map"
    SCALE = "scale"
    PATTERN = "pattern"
    
    # Compound operators
    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    REPEAT = "repeat"
    WHILE = "while"

class MovementCondition(Enum):
    """Conditions for conditional movements"""
    HAS_SYMMETRY = "has_symmetry"
    HAS_COLOR = "has_color"
    HAS_PATTERN = "has_pattern"
    GRID_SIZE = "grid_size"
    COLOR_COUNT = "color_count"
    COMPLEXITY = "complexity"

@dataclass
class MovementInstruction:
    """Single movement instruction"""
    operator: MovementOperator
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[Tuple[MovementCondition, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate instruction after creation"""
        self._validate_instruction()
    
    def _validate_instruction(self):
        """Validate instruction parameters"""
        if self.operator == MovementOperator.FLIP:
            if 'axis' not in self.parameters:
                self.parameters['axis'] = 'horizontal'
            if self.parameters['axis'] not in ['horizontal', 'vertical', 'main_diagonal', 'anti_diagonal']:
                raise ValueError(f"Invalid flip axis: {self.parameters['axis']}")
        
        elif self.operator == MovementOperator.ROTATE:
            if 'angle' not in self.parameters:
                self.parameters['angle'] = 90
            if self.parameters['angle'] not in [90, 180, 270]:
                raise ValueError(f"Invalid rotation angle: {self.parameters['angle']}")
        
        elif self.operator == MovementOperator.TRANSLATE:
            if 'shift_x' not in self.parameters:
                self.parameters['shift_x'] = 0
            if 'shift_y' not in self.parameters:
                self.parameters['shift_y'] = 0

@dataclass 
class MovementScript:
    """Complete movement script with multiple instructions"""
    instructions: List[MovementInstruction] = field(default_factory=list)
    name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_instruction(self, instruction: MovementInstruction):
        """Add instruction to script"""
        self.instructions.append(instruction)
    
    def add_flip(self, axis: str = 'horizontal', conditions: List = None):
        """Add flip instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.FLIP,
            parameters={'axis': axis},
            conditions=conditions or []
        )
        self.add_instruction(instruction)
    
    def add_rotation(self, angle: int = 90, conditions: List = None):
        """Add rotation instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.ROTATE,
            parameters={'angle': angle},
            conditions=conditions or []
        )
        self.add_instruction(instruction)
    
    def add_translation(self, shift_x: int = 0, shift_y: int = 0, mode: str = 'wrap', conditions: List = None):
        """Add translation instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.TRANSLATE,
            parameters={'shift_x': shift_x, 'shift_y': shift_y, 'mode': mode},
            conditions=conditions or []
        )
        self.add_instruction(instruction)
    
    def add_color_swap(self, color1: int = 1, color2: int = 2, conditions: List = None):
        """Add color swap instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.COLOR_SWAP,
            parameters={'type': 'swap', 'color1': color1, 'color2': color2},
            conditions=conditions or []
        )
        self.add_instruction(instruction)
    
    def add_color_mapping(self, mapping: Dict[int, int], conditions: List = None):
        """Add color mapping instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.COLOR_MAP,
            parameters={'type': 'map', 'mapping': mapping},
            conditions=conditions or []
        )
        self.add_instruction(instruction)
    
    def add_sequence(self, sub_instructions: List[MovementInstruction]):
        """Add sequential execution of sub-instructions"""
        instruction = MovementInstruction(
            operator=MovementOperator.SEQUENCE,
            parameters={'sub_instructions': sub_instructions}
        )
        self.add_instruction(instruction)
    
    def add_parallel(self, sub_instructions: List[MovementInstruction]):
        """Add parallel execution of sub-instructions"""
        instruction = MovementInstruction(
            operator=MovementOperator.PARALLEL,
            parameters={'sub_instructions': sub_instructions}
        )
        self.add_instruction(instruction)
    
    def add_conditional(self, condition: Tuple[MovementCondition, Any], 
                       true_instruction: MovementInstruction,
                       false_instruction: Optional[MovementInstruction] = None):
        """Add conditional instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.CONDITIONAL,
            parameters={
                'condition': condition,
                'true_instruction': true_instruction,
                'false_instruction': false_instruction
            }
        )
        self.add_instruction(instruction)
    
    def add_repeat(self, sub_instruction: MovementInstruction, count: int):
        """Add repeat instruction"""
        instruction = MovementInstruction(
            operator=MovementOperator.REPEAT,
            parameters={'sub_instruction': sub_instruction, 'count': count}
        )
        self.add_instruction(instruction)

class MovementConditionEvaluator:
    """Evaluates movement conditions against grid state"""
    
    def __init__(self):
        self.evaluation_cache = {}
    
    def evaluate_condition(self, 
                          condition: Tuple[MovementCondition, Any], 
                          grid: np.ndarray,
                          grid_analysis: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        condition_type, expected_value = condition
        
        try:
            if condition_type == MovementCondition.HAS_SYMMETRY:
                symmetries = grid_analysis.get('has_symmetry', {})
                return symmetries.get(expected_value, False)
            
            elif condition_type == MovementCondition.HAS_COLOR:
                unique_colors = grid_analysis.get('unique_colors', [])
                return expected_value in unique_colors
            
            elif condition_type == MovementCondition.HAS_PATTERN:
                patterns = grid_analysis.get('dominant_patterns', [])
                return expected_value in patterns
            
            elif condition_type == MovementCondition.GRID_SIZE:
                grid_size = grid.size
                if isinstance(expected_value, tuple):
                    min_size, max_size = expected_value
                    return min_size <= grid_size <= max_size
                else:
                    return grid_size == expected_value
            
            elif condition_type == MovementCondition.COLOR_COUNT:
                color_count = grid_analysis.get('num_colors', 0)
                if isinstance(expected_value, tuple):
                    min_count, max_count = expected_value
                    return min_count <= color_count <= max_count
                else:
                    return color_count == expected_value
            
            elif condition_type == MovementCondition.COMPLEXITY:
                complexity = grid_analysis.get('complexity', 0.5)
                if isinstance(expected_value, tuple):
                    min_complexity, max_complexity = expected_value
                    return min_complexity <= complexity <= max_complexity
                else:
                    return abs(complexity - expected_value) < 0.1
            
            return False
            
        except Exception as e:
            warnings.warn(f"Condition evaluation failed: {e}")
            return False
    
    def evaluate_conditions(self, 
                           conditions: List[Tuple[MovementCondition, Any]], 
                           grid: np.ndarray,
                           grid_analysis: Dict[str, Any]) -> bool:
        """Evaluate all conditions (AND logic)"""
        if not conditions:
            return True
        
        return all(self.evaluate_condition(condition, grid, grid_analysis) 
                  for condition in conditions)

class MovementCompiler:
    """Compiles movement scripts into executable expert calls"""
    
    def __init__(self, moe_router: MovementMoERouter):
        self.moe_router = moe_router
        self.condition_evaluator = MovementConditionEvaluator()
        self.compilation_cache = {}
    
    def compile_script(self, 
                      script: MovementScript, 
                      grid: np.ndarray,
                      context: Dict[str, Any] = None) -> List[ExpertCall]:
        """Compile movement script into expert calls"""
        if context is None:
            context = {}
        
        # Get grid analysis for condition evaluation
        grid_analysis = self.moe_router.grid_analyzer.analyze_grid(grid)
        context.update(grid_analysis)
        
        compiled_calls = []
        
        for instruction in script.instructions:
            try:
                # Evaluate conditions
                if not self.condition_evaluator.evaluate_conditions(
                    instruction.conditions, grid, grid_analysis):
                    continue  # Skip instruction if conditions not met
                
                # Compile instruction
                expert_calls = self._compile_instruction(instruction, grid, context)
                compiled_calls.extend(expert_calls)
                
            except Exception as e:
                warnings.warn(f"Failed to compile instruction {instruction.operator}: {e}")
                continue
        
        return compiled_calls
    
    def _compile_instruction(self, 
                           instruction: MovementInstruction, 
                           grid: np.ndarray,
                           context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile a single instruction"""
        if instruction.operator == MovementOperator.FLIP:
            return self._compile_flip(instruction, context)
        
        elif instruction.operator == MovementOperator.ROTATE:
            return self._compile_rotate(instruction, context)
        
        elif instruction.operator == MovementOperator.TRANSLATE:
            return self._compile_translate(instruction, context)
        
        elif instruction.operator == MovementOperator.COLOR_SWAP:
            return self._compile_color_swap(instruction, context)
        
        elif instruction.operator == MovementOperator.COLOR_MAP:
            return self._compile_color_map(instruction, context)
        
        elif instruction.operator == MovementOperator.SEQUENCE:
            return self._compile_sequence(instruction, grid, context)
        
        elif instruction.operator == MovementOperator.PARALLEL:
            return self._compile_parallel(instruction, grid, context)
        
        elif instruction.operator == MovementOperator.CONDITIONAL:
            return self._compile_conditional(instruction, grid, context)
        
        elif instruction.operator == MovementOperator.REPEAT:
            return self._compile_repeat(instruction, grid, context)
        
        else:
            warnings.warn(f"Unknown operator: {instruction.operator}")
            return []
    
    def _compile_flip(self, instruction: MovementInstruction, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile flip instruction"""
        expert_call = ExpertCall(
            expert_name="FlipExpert",
            movement_type=MovementType.FLIP,
            parameters=instruction.parameters.copy(),
            expected_confidence=0.8,
            context=context
        )
        return [expert_call]
    
    def _compile_rotate(self, instruction: MovementInstruction, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile rotation instruction"""
        expert_call = ExpertCall(
            expert_name="RotationExpert",
            movement_type=MovementType.ROTATION,
            parameters=instruction.parameters.copy(),
            expected_confidence=0.8,
            context=context
        )
        return [expert_call]
    
    def _compile_translate(self, instruction: MovementInstruction, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile translation instruction"""
        expert_call = ExpertCall(
            expert_name="TranslationExpert",
            movement_type=MovementType.TRANSLATION,
            parameters=instruction.parameters.copy(),
            expected_confidence=0.7,
            context=context
        )
        return [expert_call]
    
    def _compile_color_swap(self, instruction: MovementInstruction, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile color swap instruction"""
        expert_call = ExpertCall(
            expert_name="ColorTransformExpert",
            movement_type=MovementType.COLOR_TRANSFORM,
            parameters=instruction.parameters.copy(),
            expected_confidence=0.8,
            context=context
        )
        return [expert_call]
    
    def _compile_color_map(self, instruction: MovementInstruction, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile color mapping instruction"""
        expert_call = ExpertCall(
            expert_name="ColorTransformExpert",
            movement_type=MovementType.COLOR_TRANSFORM,
            parameters=instruction.parameters.copy(),
            expected_confidence=0.8,
            context=context
        )
        return [expert_call]
    
    def _compile_sequence(self, instruction: MovementInstruction, grid: np.ndarray, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile sequence instruction"""
        sub_instructions = instruction.parameters.get('sub_instructions', [])
        compiled_calls = []
        
        for sub_instruction in sub_instructions:
            sub_calls = self._compile_instruction(sub_instruction, grid, context)
            compiled_calls.extend(sub_calls)
        
        return compiled_calls
    
    def _compile_parallel(self, instruction: MovementInstruction, grid: np.ndarray, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile parallel instruction"""
        sub_instructions = instruction.parameters.get('sub_instructions', [])
        compiled_calls = []
        
        # For parallel execution, all sub-instructions should be compiled
        # The MoE router will handle parallel execution
        for sub_instruction in sub_instructions:
            sub_calls = self._compile_instruction(sub_instruction, grid, context)
            compiled_calls.extend(sub_calls)
        
        # Mark calls for parallel execution
        for call in compiled_calls:
            call.context['execution_mode'] = 'parallel'
        
        return compiled_calls
    
    def _compile_conditional(self, instruction: MovementInstruction, grid: np.ndarray, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile conditional instruction"""
        condition = instruction.parameters.get('condition')
        true_instruction = instruction.parameters.get('true_instruction')
        false_instruction = instruction.parameters.get('false_instruction')
        
        grid_analysis = context
        
        # Evaluate condition
        condition_result = self.condition_evaluator.evaluate_condition(
            condition, grid, grid_analysis
        )
        
        # Execute appropriate branch
        if condition_result and true_instruction:
            return self._compile_instruction(true_instruction, grid, context)
        elif not condition_result and false_instruction:
            return self._compile_instruction(false_instruction, grid, context)
        else:
            return []
    
    def _compile_repeat(self, instruction: MovementInstruction, grid: np.ndarray, context: Dict[str, Any]) -> List[ExpertCall]:
        """Compile repeat instruction"""
        sub_instruction = instruction.parameters.get('sub_instruction')
        count = instruction.parameters.get('count', 1)
        
        compiled_calls = []
        
        # Repeat the sub-instruction 'count' times
        for i in range(count):
            sub_calls = self._compile_instruction(sub_instruction, grid, context)
            # Mark with repeat iteration
            for call in sub_calls:
                call.context['repeat_iteration'] = i
            compiled_calls.extend(sub_calls)
        
        return compiled_calls

class MovementScriptBuilder:
    """Builder for creating common movement scripts"""
    
    @staticmethod
    def create_symmetry_script() -> MovementScript:
        """Create script for symmetry transformations"""
        script = MovementScript(
            name="symmetry_transformation",
            description="Apply symmetry-based transformations based on grid properties"
        )
        
        # If grid has horizontal symmetry, try vertical flip
        script.add_flip(
            axis='vertical',
            conditions=[(MovementCondition.HAS_SYMMETRY, 'horizontal')]
        )
        
        # If grid has vertical symmetry, try horizontal flip
        script.add_flip(
            axis='horizontal', 
            conditions=[(MovementCondition.HAS_SYMMETRY, 'vertical')]
        )
        
        # If square grid with rotational symmetry, try rotation
        script.add_rotation(
            angle=90,
            conditions=[(MovementCondition.HAS_SYMMETRY, 'rotation_90')]
        )
        
        return script
    
    @staticmethod
    def create_color_pattern_script() -> MovementScript:
        """Create script for color pattern transformations"""
        script = MovementScript(
            name="color_pattern_transformation",
            description="Transform color patterns based on analysis"
        )
        
        # If grid has exactly 2 colors, try color swap
        script.add_color_swap(
            color1=1, color2=2,
            conditions=[(MovementCondition.COLOR_COUNT, 2)]
        )
        
        # For simple grids, try adding complexity
        script.add_color_mapping(
            mapping={0: 1, 1: 2, 2: 0},
            conditions=[(MovementCondition.COMPLEXITY, (0.0, 0.3))]
        )
        
        return script
    
    @staticmethod
    def create_geometric_script() -> MovementScript:
        """Create script for geometric transformations"""
        script = MovementScript(
            name="geometric_transformation",
            description="Apply geometric transformations"
        )
        
        # Try rotation for square grids
        script.add_rotation(angle=90)
        
        # Try translation for patterns
        script.add_translation(
            shift_x=1, shift_y=0, mode='wrap',
            conditions=[(MovementCondition.HAS_PATTERN, 'stripes')]
        )
        
        return script
    
    @staticmethod
    def create_complex_composition_script() -> MovementScript:
        """Create script with complex compositions"""
        script = MovementScript(
            name="complex_composition",
            description="Complex movement composition with conditionals"
        )
        
        # Conditional: if has symmetry, do flip; otherwise, do rotation
        flip_instruction = MovementInstruction(
            operator=MovementOperator.FLIP,
            parameters={'axis': 'horizontal'}
        )
        
        rotate_instruction = MovementInstruction(
            operator=MovementOperator.ROTATE,
            parameters={'angle': 90}
        )
        
        script.add_conditional(
            condition=(MovementCondition.HAS_SYMMETRY, 'horizontal'),
            true_instruction=flip_instruction,
            false_instruction=rotate_instruction
        )
        
        # Sequence: color swap followed by translation
        color_swap = MovementInstruction(
            operator=MovementOperator.COLOR_SWAP,
            parameters={'color1': 1, 'color2': 2}
        )
        
        translation = MovementInstruction(
            operator=MovementOperator.TRANSLATE,
            parameters={'shift_x': 1, 'shift_y': 1, 'mode': 'wrap'}
        )
        
        script.add_sequence([color_swap, translation])
        
        return script
    
    @staticmethod
    def create_adaptive_script() -> MovementScript:
        """Create adaptive script that responds to grid properties"""
        script = MovementScript(
            name="adaptive_transformation",
            description="Adaptive transformation based on comprehensive analysis"
        )
        
        # For small grids, use simple operations
        script.add_flip(
            axis='horizontal',
            conditions=[(MovementCondition.GRID_SIZE, (1, 25))]
        )
        
        # For medium grids with patterns, use translation
        script.add_translation(
            shift_x=2, shift_y=0, mode='wrap',
            conditions=[
                (MovementCondition.GRID_SIZE, (26, 100)),
                (MovementCondition.HAS_PATTERN, 'stripes')
            ]
        )
        
        # For complex grids, use color mapping
        script.add_color_mapping(
            mapping={0: 2, 1: 0, 2: 1},
            conditions=[
                (MovementCondition.COMPLEXITY, (0.5, 1.0)),
                (MovementCondition.COLOR_COUNT, (3, 5))
            ]
        )
        
        return script

# Factory functions for common movement scripts
def create_flip_script(axis: str = 'horizontal') -> MovementScript:
    """Create simple flip script"""
    script = MovementScript(name=f"flip_{axis}")
    script.add_flip(axis=axis)
    return script

def create_rotation_script(angle: int = 90) -> MovementScript:
    """Create simple rotation script"""
    script = MovementScript(name=f"rotate_{angle}")
    script.add_rotation(angle=angle)
    return script

def create_color_swap_script(color1: int = 1, color2: int = 2) -> MovementScript:
    """Create simple color swap script"""
    script = MovementScript(name=f"swap_{color1}_{color2}")
    script.add_color_swap(color1=color1, color2=color2)
    return script