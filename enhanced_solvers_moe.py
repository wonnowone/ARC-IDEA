#!/usr/bin/env python3
"""
Enhanced ARC Solvers with MoE Integration

This module refactors the existing solver architecture to use the new Mixture of Experts
system. Each solver now acts as a high-level strategist that composes movement sequences
using the declarative movement language and MoE router.

Enhanced Solver Architecture:
- Solvers focus on strategy and decision-making
- Movement execution delegated to MoE experts
- Declarative movement language for expressing transformations
- Multi-step reasoning with intermediate verification
- Expert preference learning integration
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
import time
import warnings

from movement_experts import MovementResult, MovementType
from moe_router import MovementMoERouter, create_default_moe_router, RoutingStrategy
from movement_language import (
    MovementScript, MovementCompiler, MovementScriptBuilder, 
    MovementOperator, MovementCondition, create_flip_script,
    create_rotation_script, create_color_swap_script
)

class EnhancedBaseSolver(ABC):
    """Enhanced base solver with MoE integration"""
    
    def __init__(self, solver_name: str):
        self.solver_name = solver_name
        self.moe_router = create_default_moe_router()
        self.movement_compiler = MovementCompiler(self.moe_router)
        
        # Performance tracking
        self.execution_history = []
        self.success_count = 0
        self.total_executions = 0
        
        # Strategy preferences (learned over time)
        self.strategy_preferences = {}
        self.movement_success_rates = {}
        
        # Solver-specific metadata
        self.thinking_flow = {
            'strategy': '',
            'steps': [],
            'confidence': 0.0,
            'movement_sequence': [],
            'expert_usage': {}
        }
    
    @abstractmethod
    def analyze_problem(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze the problem and identify strategy"""
        pass
    
    @abstractmethod
    def create_movement_script(self, input_grid: np.ndarray, analysis: Dict[str, Any]) -> MovementScript:
        """Create movement script based on analysis"""
        pass
    
    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        """Enhanced prediction using MoE system"""
        start_time = time.time()
        
        try:
            # Reset thinking flow
            self.thinking_flow = {
                'strategy': '',
                'steps': [],
                'confidence': 0.0,
                'movement_sequence': [],
                'expert_usage': {}
            }
            
            # Step 1: Analyze problem
            self.thinking_flow['steps'].append('analyze_problem')
            analysis = self.analyze_problem(input_grid)
            self.thinking_flow['strategy'] = analysis.get('strategy_name', 'unknown')
            
            # Step 2: Create movement script
            self.thinking_flow['steps'].append('create_movement_script')
            movement_script = self.create_movement_script(input_grid, analysis)
            
            # Step 3: Compile script to expert calls
            self.thinking_flow['steps'].append('compile_script')
            expert_calls = self.movement_compiler.compile_script(
                movement_script, input_grid, analysis
            )
            
            if not expert_calls:
                warnings.warn(f"{self.solver_name}: No expert calls generated")
                self.thinking_flow['confidence'] = 0.1
                return input_grid.copy()
            
            # Step 4: Execute movement sequence
            self.thinking_flow['steps'].append('execute_movements')
            self.thinking_flow['movement_sequence'] = [call.expert_name for call in expert_calls]
            
            # Execute with appropriate strategy
            execution_strategy = analysis.get('execution_strategy', 'confidence_based')
            final_result = self.moe_router.execute_sequence(
                input_grid, expert_calls, execution_strategy
            )
            
            # Step 5: Update statistics and learning
            self.thinking_flow['steps'].append('update_learning')
            execution_time = time.time() - start_time
            
            self._update_performance_stats(final_result, execution_time, expert_calls)
            self._update_strategy_learning(analysis, final_result, expert_calls)
            
            # Set final confidence
            self.thinking_flow['confidence'] = final_result.confidence
            
            # Update expert usage statistics
            expert_usage = {}
            for call in expert_calls:
                expert_usage[call.expert_name] = expert_usage.get(call.expert_name, 0) + 1
            self.thinking_flow['expert_usage'] = expert_usage
            
            return final_result.output_grid
            
        except Exception as e:
            warnings.warn(f"{self.solver_name} prediction failed: {e}")
            self.thinking_flow['confidence'] = 0.0
            self.thinking_flow['steps'].append('error')
            return input_grid.copy()
    
    def get_thinking_flow(self) -> Dict[str, Any]:
        """Get enhanced thinking flow with MoE details"""
        # Add MoE router statistics
        router_stats = self.moe_router.get_routing_statistics()
        
        enhanced_flow = self.thinking_flow.copy()
        enhanced_flow.update({
            'solver_name': self.solver_name,
            'total_executions': self.total_executions,
            'success_rate': self.success_count / max(1, self.total_executions),
            'router_statistics': router_stats,
            'strategy_preferences': self.strategy_preferences.copy(),
            'movement_success_rates': self.movement_success_rates.copy()
        })
        
        return enhanced_flow
    
    def _update_performance_stats(self, result: MovementResult, execution_time: float, expert_calls: List):
        """Update performance statistics"""
        self.total_executions += 1
        if result.success and result.confidence > 0.5:
            self.success_count += 1
        
        execution_record = {
            'timestamp': time.time(),
            'success': result.success,
            'confidence': result.confidence,
            'execution_time': execution_time,
            'experts_used': [call.expert_name for call in expert_calls],
            'operation_type': result.operation_type
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only recent history
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def _update_strategy_learning(self, analysis: Dict[str, Any], result: MovementResult, expert_calls: List):
        """Update strategy learning based on results"""
        strategy_name = analysis.get('strategy_name', 'default')
        
        # Update strategy preferences
        if strategy_name not in self.strategy_preferences:
            self.strategy_preferences[strategy_name] = {'successes': 0, 'attempts': 0, 'avg_confidence': 0.0}
        
        strategy_stats = self.strategy_preferences[strategy_name]
        strategy_stats['attempts'] += 1
        
        if result.success and result.confidence > 0.5:
            strategy_stats['successes'] += 1
        
        # Update average confidence
        old_avg = strategy_stats['avg_confidence']
        new_confidence = result.confidence
        strategy_stats['avg_confidence'] = old_avg + (new_confidence - old_avg) / strategy_stats['attempts']
        
        # Update movement success rates
        for call in expert_calls:
            expert_name = call.expert_name
            if expert_name not in self.movement_success_rates:
                self.movement_success_rates[expert_name] = {'successes': 0, 'attempts': 0}
            
            self.movement_success_rates[expert_name]['attempts'] += 1
            if result.success:
                self.movement_success_rates[expert_name]['successes'] += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if self.total_executions == 0:
            return {'message': 'No executions recorded'}
        
        success_rate = self.success_count / self.total_executions
        avg_confidence = np.mean([record['confidence'] for record in self.execution_history])
        avg_execution_time = np.mean([record['execution_time'] for record in self.execution_history])
        
        # Most used experts
        expert_usage = {}
        for record in self.execution_history:
            for expert in record['experts_used']:
                expert_usage[expert] = expert_usage.get(expert, 0) + 1
        
        most_used_expert = max(expert_usage.items(), key=lambda x: x[1]) if expert_usage else ('none', 0)
        
        return {
            'solver_name': self.solver_name,
            'total_executions': self.total_executions,
            'success_rate': success_rate,
            'average_confidence': avg_confidence,
            'average_execution_time': avg_execution_time,
            'most_used_expert': most_used_expert[0],
            'expert_usage_distribution': expert_usage,
            'strategy_preferences': self.strategy_preferences
        }

class EnhancedColorPatternSolver(EnhancedBaseSolver):
    """Enhanced color pattern solver with MoE integration"""
    
    def __init__(self):
        super().__init__("EnhancedColorPatternSolver")
    
    def analyze_problem(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze color patterns and determine strategy"""
        analysis = {
            'strategy_name': 'color_pattern_analysis',
            'execution_strategy': 'confidence_based'
        }
        
        # Basic color analysis
        unique_colors, counts = np.unique(input_grid, return_counts=True)
        color_distribution = dict(zip(unique_colors, counts))
        
        analysis.update({
            'unique_colors': unique_colors.tolist(),
            'num_colors': len(unique_colors),
            'color_distribution': color_distribution,
            'dominant_color': unique_colors[np.argmax(counts)],
            'rare_color': unique_colors[np.argmin(counts)] if len(unique_colors) > 1 else None
        })
        
        # Pattern detection
        if len(unique_colors) == 2:
            analysis['pattern_type'] = 'binary'
            analysis['strategy_name'] = 'binary_color_swap'
        elif len(unique_colors) <= 3:
            analysis['pattern_type'] = 'simple'
            analysis['strategy_name'] = 'simple_color_mapping'
        else:
            analysis['pattern_type'] = 'complex'
            analysis['strategy_name'] = 'complex_color_reduction'
        
        # Check for checkerboard pattern
        if self._is_checkerboard_pattern(input_grid):
            analysis['has_checkerboard'] = True
            analysis['strategy_name'] = 'checkerboard_transformation'
        
        return analysis
    
    def create_movement_script(self, input_grid: np.ndarray, analysis: Dict[str, Any]) -> MovementScript:
        """Create color-focused movement script"""
        strategy_name = analysis.get('strategy_name', 'default')
        
        if strategy_name == 'binary_color_swap':
            return self._create_binary_swap_script(analysis)
        elif strategy_name == 'simple_color_mapping':
            return self._create_simple_mapping_script(analysis)
        elif strategy_name == 'checkerboard_transformation':
            return self._create_checkerboard_script(analysis)
        elif strategy_name == 'complex_color_reduction':
            return self._create_color_reduction_script(analysis)
        else:
            return self._create_default_color_script(analysis)
    
    def _create_binary_swap_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for binary color swap"""
        script = MovementScript(
            name="binary_color_swap",
            description="Swap two colors in binary pattern"
        )
        
        unique_colors = analysis.get('unique_colors', [0, 1])
        if len(unique_colors) >= 2:
            script.add_color_swap(
                color1=unique_colors[0],
                color2=unique_colors[1]
            )
        
        return script
    
    def _create_simple_mapping_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for simple color mapping"""
        script = MovementScript(
            name="simple_color_mapping",
            description="Apply simple color transformations"
        )
        
        # Try color increment for simple patterns
        script.add_instruction({
            'operator': MovementOperator.COLOR_MAP,
            'parameters': {
                'type': 'increment',
                'increment': 1,
                'modulo': 10
            }
        })
        
        return script
    
    def _create_checkerboard_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for checkerboard patterns"""
        script = MovementScript(
            name="checkerboard_transformation",
            description="Transform checkerboard patterns"
        )
        
        # For checkerboard, try flipping and color swapping
        script.add_flip(axis='horizontal')
        
        unique_colors = analysis.get('unique_colors', [0, 1])
        if len(unique_colors) >= 2:
            script.add_color_swap(
                color1=unique_colors[0],
                color2=unique_colors[1]
            )
        
        return script
    
    def _create_color_reduction_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for complex color reduction"""
        script = MovementScript(
            name="color_reduction",
            description="Reduce color complexity"
        )
        
        # Map multiple colors to fewer colors
        unique_colors = analysis.get('unique_colors', [])
        if len(unique_colors) > 3:
            # Create mapping to reduce colors
            color_mapping = {}
            target_colors = [0, 1, 2]  # Reduce to 3 colors
            
            for i, color in enumerate(unique_colors):
                color_mapping[color] = target_colors[i % len(target_colors)]
            
            script.add_color_mapping(mapping=color_mapping)
        
        return script
    
    def _create_default_color_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create default color transformation script"""
        return MovementScriptBuilder.create_color_pattern_script()
    
    def _is_checkerboard_pattern(self, grid: np.ndarray) -> bool:
        """Check if grid has checkerboard pattern"""
        if grid.shape[0] < 2 or grid.shape[1] < 2:
            return False
        
        # Sample a few positions to check alternating pattern
        try:
            pattern1 = grid[0, 0]
            pattern2 = grid[0, 1] if grid.shape[1] > 1 else grid[1, 0]
            
            if pattern1 == pattern2:
                return False
            
            # Check a few positions for alternating pattern
            for i in range(min(3, grid.shape[0])):
                for j in range(min(3, grid.shape[1])):
                    expected = pattern1 if (i + j) % 2 == 0 else pattern2
                    if grid[i, j] != expected:
                        return False
            
            return True
        except:
            return False

class EnhancedShapeSymmetrySolver(EnhancedBaseSolver):
    """Enhanced shape and symmetry solver with MoE integration"""
    
    def __init__(self):
        super().__init__("EnhancedShapeSymmetrySolver")
    
    def analyze_problem(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze shape and symmetry patterns"""
        analysis = {
            'strategy_name': 'shape_symmetry_analysis',
            'execution_strategy': 'confidence_based'
        }
        
        # Check symmetries
        symmetries = self._check_all_symmetries(input_grid)
        analysis['symmetries'] = symmetries
        analysis['has_symmetry'] = any(symmetries.values())
        
        # Determine strategy based on symmetries
        if symmetries.get('horizontal', False):
            analysis['strategy_name'] = 'horizontal_symmetry'
        elif symmetries.get('vertical', False):
            analysis['strategy_name'] = 'vertical_symmetry'
        elif symmetries.get('rotation_180', False):
            analysis['strategy_name'] = 'rotational_symmetry'
        elif input_grid.shape[0] == input_grid.shape[1]:
            analysis['strategy_name'] = 'square_geometric'
        else:
            analysis['strategy_name'] = 'general_geometric'
        
        return analysis
    
    def create_movement_script(self, input_grid: np.ndarray, analysis: Dict[str, Any]) -> MovementScript:
        """Create symmetry-focused movement script"""
        strategy_name = analysis.get('strategy_name', 'general_geometric')
        
        if strategy_name == 'horizontal_symmetry':
            return self._create_horizontal_symmetry_script()
        elif strategy_name == 'vertical_symmetry':
            return self._create_vertical_symmetry_script()
        elif strategy_name == 'rotational_symmetry':
            return self._create_rotational_symmetry_script()
        elif strategy_name == 'square_geometric':
            return self._create_square_geometric_script()
        else:
            return self._create_general_geometric_script()
    
    def _create_horizontal_symmetry_script(self) -> MovementScript:
        """Create script for horizontal symmetry"""
        script = MovementScript(
            name="horizontal_symmetry_transform",
            description="Transform horizontally symmetric patterns"
        )
        
        # Try vertical flip for horizontal symmetry
        script.add_flip(axis='vertical')
        
        return script
    
    def _create_vertical_symmetry_script(self) -> MovementScript:
        """Create script for vertical symmetry"""
        script = MovementScript(
            name="vertical_symmetry_transform",
            description="Transform vertically symmetric patterns"
        )
        
        # Try horizontal flip for vertical symmetry
        script.add_flip(axis='horizontal')
        
        return script
    
    def _create_rotational_symmetry_script(self) -> MovementScript:
        """Create script for rotational symmetry"""
        script = MovementScript(
            name="rotational_symmetry_transform",
            description="Transform rotationally symmetric patterns"
        )
        
        # Try 90-degree rotation
        script.add_rotation(angle=90)
        
        return script
    
    def _create_square_geometric_script(self) -> MovementScript:
        """Create script for square geometric transformations"""
        script = MovementScript(
            name="square_geometric_transform",
            description="Geometric transformations for square grids"
        )
        
        # Try various rotations
        script.add_rotation(angle=90)
        script.add_rotation(angle=180)
        
        # Try diagonal flip
        script.add_flip(axis='main_diagonal')
        
        return script
    
    def _create_general_geometric_script(self) -> MovementScript:
        """Create general geometric transformation script"""
        return MovementScriptBuilder.create_geometric_script()
    
    def _check_all_symmetries(self, grid: np.ndarray) -> Dict[str, bool]:
        """Check all types of symmetry"""
        symmetries = {}
        
        try:
            symmetries['horizontal'] = np.array_equal(grid, np.fliplr(grid))
            symmetries['vertical'] = np.array_equal(grid, np.flipud(grid))
            symmetries['rotation_180'] = np.array_equal(grid, np.rot90(grid, 2))
            
            if grid.shape[0] == grid.shape[1]:  # Square grids only
                symmetries['rotation_90'] = np.array_equal(grid, np.rot90(grid, 1))
                symmetries['main_diagonal'] = np.array_equal(grid, np.transpose(grid))
            else:
                symmetries['rotation_90'] = False
                symmetries['main_diagonal'] = False
                
        except Exception:
            symmetries = {key: False for key in ['horizontal', 'vertical', 'rotation_90', 'rotation_180', 'main_diagonal']}
        
        return symmetries

class EnhancedGeometricTransformSolver(EnhancedBaseSolver):
    """Enhanced geometric transformation solver with MoE integration"""
    
    def __init__(self):
        super().__init__("EnhancedGeometricTransformSolver")
    
    def analyze_problem(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze geometric properties"""
        analysis = {
            'strategy_name': 'geometric_analysis',
            'execution_strategy': 'multi_expert'
        }
        
        # Grid properties
        analysis['is_square'] = input_grid.shape[0] == input_grid.shape[1]
        analysis['size'] = input_grid.size
        analysis['aspect_ratio'] = input_grid.shape[0] / input_grid.shape[1]
        
        # Determine strategy
        if analysis['is_square']:
            analysis['strategy_name'] = 'square_transform'
        elif analysis['aspect_ratio'] > 1.5 or analysis['aspect_ratio'] < 0.67:
            analysis['strategy_name'] = 'rectangular_transform'
        else:
            analysis['strategy_name'] = 'general_transform'
        
        return analysis
    
    def create_movement_script(self, input_grid: np.ndarray, analysis: Dict[str, Any]) -> MovementScript:
        """Create geometric transformation script"""
        strategy_name = analysis.get('strategy_name', 'general_transform')
        
        if strategy_name == 'square_transform':
            return self._create_square_transform_script()
        elif strategy_name == 'rectangular_transform':
            return self._create_rectangular_transform_script()
        else:
            return self._create_general_transform_script()
    
    def _create_square_transform_script(self) -> MovementScript:
        """Create script for square transformations"""
        script = MovementScript(
            name="square_geometric_transform",
            description="Geometric transformations for square grids"
        )
        
        # Try rotation sequence
        script.add_rotation(angle=90)
        
        # Try translation
        script.add_translation(shift_x=1, shift_y=1, mode='wrap')
        
        return script
    
    def _create_rectangular_transform_script(self) -> MovementScript:
        """Create script for rectangular transformations"""
        script = MovementScript(
            name="rectangular_geometric_transform",
            description="Geometric transformations for rectangular grids"
        )
        
        # 180-degree rotation preserves shape
        script.add_rotation(angle=180)
        
        # Translation works well for rectangles
        script.add_translation(shift_x=2, shift_y=0, mode='wrap')
        
        return script
    
    def _create_general_transform_script(self) -> MovementScript:
        """Create general transformation script"""
        return MovementScriptBuilder.create_geometric_script()

class EnhancedLogicalRuleSolver(EnhancedBaseSolver):
    """Enhanced logical rule solver with MoE integration"""
    
    def __init__(self):
        super().__init__("EnhancedLogicalRuleSolver")
    
    def analyze_problem(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze logical patterns and rules"""
        analysis = {
            'strategy_name': 'logical_rule_analysis',
            'execution_strategy': 'sequential'
        }
        
        # Pattern complexity analysis
        unique_colors = np.unique(input_grid)
        color_transitions = self._count_color_transitions(input_grid)
        
        analysis.update({
            'num_colors': len(unique_colors),
            'color_transitions': color_transitions,
            'complexity_score': self._calculate_logical_complexity(input_grid)
        })
        
        # Determine logical strategy
        if len(unique_colors) == 2:
            analysis['strategy_name'] = 'binary_logic'
        elif color_transitions > input_grid.size * 0.5:
            analysis['strategy_name'] = 'high_complexity_logic'
        else:
            analysis['strategy_name'] = 'pattern_logic'
        
        return analysis
    
    def create_movement_script(self, input_grid: np.ndarray, analysis: Dict[str, Any]) -> MovementScript:
        """Create logical rule-based movement script"""
        strategy_name = analysis.get('strategy_name', 'pattern_logic')
        
        if strategy_name == 'binary_logic':
            return self._create_binary_logic_script(analysis)
        elif strategy_name == 'high_complexity_logic':
            return self._create_complexity_reduction_script(analysis)
        else:
            return self._create_pattern_logic_script(analysis)
    
    def _create_binary_logic_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for binary logical patterns"""
        script = MovementScript(
            name="binary_logic_transform",
            description="Transform binary logical patterns"
        )
        
        # Binary patterns often benefit from flipping
        script.add_flip(axis='horizontal')
        script.add_flip(axis='vertical')
        
        return script
    
    def _create_complexity_reduction_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script to reduce pattern complexity"""
        script = MovementScript(
            name="complexity_reduction",
            description="Reduce pattern complexity through transformations"
        )
        
        # Use translation to potentially align patterns
        script.add_translation(shift_x=1, shift_y=0, mode='wrap')
        
        # Try color simplification
        script.add_color_mapping(mapping={2: 1, 3: 1, 4: 2})
        
        return script
    
    def _create_pattern_logic_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for pattern-based logic"""
        return MovementScriptBuilder.create_adaptive_script()
    
    def _count_color_transitions(self, grid: np.ndarray) -> int:
        """Count color transitions in grid"""
        transitions = 0
        
        # Horizontal transitions
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1] - 1):
                if grid[i, j] != grid[i, j + 1]:
                    transitions += 1
        
        # Vertical transitions
        for i in range(grid.shape[0] - 1):
            for j in range(grid.shape[1]):
                if grid[i, j] != grid[i + 1, j]:
                    transitions += 1
        
        return transitions
    
    def _calculate_logical_complexity(self, grid: np.ndarray) -> float:
        """Calculate complexity score for logical patterns"""
        unique_colors = len(np.unique(grid))
        transitions = self._count_color_transitions(grid)
        
        # Normalize by grid size
        transition_density = transitions / (grid.size * 2)
        color_density = unique_colors / 10.0  # Assume max 10 colors
        
        return (transition_density + color_density) / 2

class EnhancedSymbolicSolver(EnhancedBaseSolver):
    """Enhanced symbolic reasoning solver with MoE integration"""
    
    def __init__(self):
        super().__init__("EnhancedSymbolicSolver")
    
    def analyze_problem(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze symbolic patterns and relationships"""
        analysis = {
            'strategy_name': 'symbolic_analysis',
            'execution_strategy': 'ensemble'
        }
        
        # Object detection
        objects = self._detect_objects(input_grid)
        analysis['objects'] = objects
        analysis['num_objects'] = len(objects)
        
        # Relationship analysis
        if len(objects) > 1:
            relationships = self._analyze_object_relationships(objects)
            analysis['relationships'] = relationships
            analysis['strategy_name'] = 'multi_object_symbolic'
        else:
            analysis['strategy_name'] = 'single_object_symbolic'
        
        return analysis
    
    def create_movement_script(self, input_grid: np.ndarray, analysis: Dict[str, Any]) -> MovementScript:
        """Create symbolic reasoning movement script"""
        strategy_name = analysis.get('strategy_name', 'single_object_symbolic')
        
        if strategy_name == 'multi_object_symbolic':
            return self._create_multi_object_script(analysis)
        else:
            return self._create_single_object_script(analysis)
    
    def _create_multi_object_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for multi-object symbolic reasoning"""
        script = MovementScript(
            name="multi_object_symbolic",
            description="Transform multiple symbolic objects"
        )
        
        # Try various transformations to see interactions
        script.add_rotation(angle=90)
        script.add_flip(axis='horizontal')
        script.add_translation(shift_x=1, shift_y=0, mode='wrap')
        
        return script
    
    def _create_single_object_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for single object symbolic reasoning"""
        script = MovementScript(
            name="single_object_symbolic",
            description="Transform single symbolic object"
        )
        
        # Focus on object transformation
        script.add_rotation(angle=180)
        script.add_color_mapping(mapping={1: 2, 2: 3, 3: 1})
        
        return script
    
    def _detect_objects(self, grid: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in the grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:  # Non-background
                    obj_positions = self._flood_fill(grid, i, j, visited)
                    if len(obj_positions) > 1:  # Only consider multi-cell objects
                        objects.append({
                            'color': grid[i, j],
                            'positions': obj_positions,
                            'size': len(obj_positions)
                        })
        
        return objects
    
    def _flood_fill(self, grid: np.ndarray, start_i: int, start_j: int, visited: np.ndarray) -> List[Tuple[int, int]]:
        """Flood fill to find connected component"""
        color = grid[start_i, start_j]
        stack = [(start_i, start_j)]
        positions = []
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or 
                visited[i, j] or grid[i, j] != color):
                continue
            
            visited[i, j] = True
            positions.append((i, j))
            
            # Add neighbors
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((i + di, j + dj))
        
        return positions
    
    def _analyze_object_relationships(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze relationships between objects"""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    # Calculate distance
                    pos1 = np.array(obj1['positions'])
                    pos2 = np.array(obj2['positions'])
                    
                    center1 = pos1.mean(axis=0)
                    center2 = pos2.mean(axis=0)
                    
                    distance = np.linalg.norm(center2 - center1)
                    
                    relationships.append({
                        'obj1_idx': i,
                        'obj2_idx': j,
                        'distance': distance,
                        'size_ratio': obj1['size'] / max(1, obj2['size'])
                    })
        
        return relationships