#!/usr/bin/env python3
"""
Mixture of Experts Router for ARC Movement System

This module implements the intelligent routing system that selects and combines
movement experts based on grid analysis, context, and learned preferences.

Components:
- MovementMoERouter: Main routing intelligence
- ExpertSelector: Expert selection algorithms  
- ExpertCombiner: Multi-expert result combination
- RoutingStrategy: Different routing strategies
- GatingNetwork: Neural network for expert selection
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import defaultdict
import time

from movement_experts import (
    BaseMovementExpert, MovementResult, MovementType, MovementValidator,
    FlipExpert, RotationExpert, TranslationExpert, ColorTransformExpert
)

@dataclass
class ExpertCall:
    """Represents a call to a specific expert with parameters"""
    expert_name: str
    movement_type: MovementType
    parameters: Dict[str, Any]
    expected_confidence: float
    priority: int = 0
    context: Optional[Dict[str, Any]] = None

@dataclass
class RoutingResult:
    """Result of routing operation"""
    selected_experts: List[ExpertCall]
    routing_confidence: float
    routing_time: float
    routing_strategy: str
    grid_analysis: Dict[str, Any]
    fallback_used: bool = False

class RoutingStrategy(Enum):
    """Different routing strategies"""
    CONFIDENCE_BASED = "confidence_based"
    GATING_NETWORK = "gating_network"
    MULTI_EXPERT = "multi_expert"
    SEQUENTIAL = "sequential"
    ENSEMBLE = "ensemble"

class GridAnalyzer:
    """Analyzes grid properties to inform routing decisions"""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_grid(self, grid: np.ndarray) -> Dict[str, Any]:
        """Comprehensive grid analysis for routing"""
        # Use grid hash for caching
        grid_hash = hash(grid.tobytes())
        if grid_hash in self.analysis_cache:
            return self.analysis_cache[grid_hash]
        
        analysis = {}
        
        # Basic properties
        analysis['shape'] = grid.shape
        analysis['size'] = grid.size
        analysis['is_square'] = grid.shape[0] == grid.shape[1]
        
        # Color analysis
        unique_colors = np.unique(grid)
        analysis['unique_colors'] = unique_colors.tolist()
        analysis['num_colors'] = len(unique_colors)
        analysis['color_distribution'] = {}
        for color in unique_colors:
            analysis['color_distribution'][int(color)] = int(np.sum(grid == color))
        
        # Pattern analysis
        analysis['has_symmetry'] = self._check_symmetries(grid)
        analysis['complexity'] = self._calculate_complexity(grid)
        analysis['dominant_patterns'] = self._find_patterns(grid)
        
        # Movement hints
        analysis['movement_hints'] = self._generate_movement_hints(grid, analysis)
        
        # Cache result
        if len(self.analysis_cache) > 100:  # Limit cache size
            self.analysis_cache.clear()
        self.analysis_cache[grid_hash] = analysis
        
        return analysis
    
    def _check_symmetries(self, grid: np.ndarray) -> Dict[str, bool]:
        """Check various types of symmetry"""
        symmetries = {}
        
        try:
            # Horizontal symmetry
            symmetries['horizontal'] = np.array_equal(grid, np.fliplr(grid))
            
            # Vertical symmetry
            symmetries['vertical'] = np.array_equal(grid, np.flipud(grid))
            
            # Rotational symmetries
            if grid.shape[0] == grid.shape[1]:  # Square grids only
                symmetries['rotation_90'] = np.array_equal(grid, np.rot90(grid, 1))
                symmetries['rotation_180'] = np.array_equal(grid, np.rot90(grid, 2))
                symmetries['rotation_270'] = np.array_equal(grid, np.rot90(grid, 3))
                
                # Diagonal symmetry
                symmetries['main_diagonal'] = np.array_equal(grid, np.transpose(grid))
            else:
                symmetries['rotation_90'] = False
                symmetries['rotation_180'] = np.array_equal(grid, np.rot90(grid, 2)) if grid.shape[0] == grid.shape[1] else False
                symmetries['rotation_270'] = False
                symmetries['main_diagonal'] = False
                
        except Exception:
            # Default to no symmetry if analysis fails
            symmetries = {key: False for key in ['horizontal', 'vertical', 'rotation_90', 'rotation_180', 'rotation_270', 'main_diagonal']}
        
        return symmetries
    
    def _calculate_complexity(self, grid: np.ndarray) -> float:
        """Calculate grid complexity score"""
        try:
            # Color entropy
            unique_colors, counts = np.unique(grid, return_counts=True)
            color_probs = counts / np.sum(counts)
            color_entropy = -np.sum(color_probs * np.log2(color_probs + 1e-12))
            
            # Spatial complexity (neighboring differences)
            diff_h = np.sum(grid[:, :-1] != grid[:, 1:])
            diff_v = np.sum(grid[:-1, :] != grid[1:, :])
            spatial_complexity = (diff_h + diff_v) / (grid.size * 2)
            
            # Combined complexity
            complexity = (color_entropy + spatial_complexity) / 2
            return float(np.clip(complexity, 0.0, 1.0))
            
        except:
            return 0.5
    
    def _find_patterns(self, grid: np.ndarray) -> List[str]:
        """Identify dominant patterns in the grid"""
        patterns = []
        
        try:
            # Check for uniform regions
            if len(np.unique(grid)) == 1:
                patterns.append('uniform')
            
            # Check for checkerboard pattern
            if self._is_checkerboard(grid):
                patterns.append('checkerboard')
            
            # Check for stripes
            if self._has_stripes(grid):
                patterns.append('stripes')
            
            # Check for border pattern
            if self._has_border(grid):
                patterns.append('border')
            
            # Check for cross pattern
            if self._has_cross(grid):
                patterns.append('cross')
                
        except:
            pass
        
        return patterns
    
    def _is_checkerboard(self, grid: np.ndarray) -> bool:
        """Check if grid has checkerboard pattern"""
        try:
            if grid.shape[0] < 2 or grid.shape[1] < 2:
                return False
            
            # Sample a few positions
            pattern1 = grid[0, 0]
            pattern2 = grid[0, 1] if grid.shape[1] > 1 else grid[1, 0]
            
            if pattern1 == pattern2:
                return False
            
            # Check alternating pattern
            for i in range(min(4, grid.shape[0])):
                for j in range(min(4, grid.shape[1])):
                    expected = pattern1 if (i + j) % 2 == 0 else pattern2
                    if grid[i, j] != expected:
                        return False
            
            return True
        except:
            return False
    
    def _has_stripes(self, grid: np.ndarray) -> bool:
        """Check for stripe patterns"""
        try:
            # Horizontal stripes
            for i in range(grid.shape[0]):
                row = grid[i, :]
                if len(np.unique(row)) == 1 and i > 0:
                    if grid[i, 0] != grid[i-1, 0]:
                        return True
            
            # Vertical stripes
            for j in range(grid.shape[1]):
                col = grid[:, j]
                if len(np.unique(col)) == 1 and j > 0:
                    if grid[0, j] != grid[0, j-1]:
                        return True
            
            return False
        except:
            return False
    
    def _has_border(self, grid: np.ndarray) -> bool:
        """Check if grid has border pattern"""
        try:
            if grid.shape[0] < 3 or grid.shape[1] < 3:
                return False
            
            # Check if border elements are different from center
            border_color = grid[0, 0]
            center_color = grid[1, 1]
            
            if border_color == center_color:
                return False
            
            # Check border consistency
            border_positions = [(0, j) for j in range(grid.shape[1])] + \
                              [(grid.shape[0]-1, j) for j in range(grid.shape[1])] + \
                              [(i, 0) for i in range(1, grid.shape[0]-1)] + \
                              [(i, grid.shape[1]-1) for i in range(1, grid.shape[0]-1)]
            
            for i, j in border_positions[:min(10, len(border_positions))]:
                if grid[i, j] != border_color:
                    return False
            
            return True
        except:
            return False
    
    def _has_cross(self, grid: np.ndarray) -> bool:
        """Check for cross pattern"""
        try:
            if grid.shape[0] < 3 or grid.shape[1] < 3:
                return False
            
            center_i, center_j = grid.shape[0] // 2, grid.shape[1] // 2
            cross_color = grid[center_i, center_j]
            
            # Check horizontal line
            h_line = grid[center_i, :]
            if not np.all(h_line == cross_color):
                return False
            
            # Check vertical line
            v_line = grid[:, center_j]
            if not np.all(v_line == cross_color):
                return False
            
            return True
        except:
            return False
    
    def _generate_movement_hints(self, grid: np.ndarray, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Generate movement hints based on analysis"""
        hints = {}
        
        # Symmetry-based hints
        symmetries = analysis.get('has_symmetry', {})
        if symmetries.get('horizontal', False):
            hints['flip_horizontal'] = 0.9
        if symmetries.get('vertical', False):
            hints['flip_vertical'] = 0.9
        if symmetries.get('rotation_180', False):
            hints['rotate_180'] = 0.9
        
        # Pattern-based hints
        patterns = analysis.get('dominant_patterns', [])
        if 'checkerboard' in patterns:
            hints['color_transform'] = 0.8
        if 'stripes' in patterns:
            hints['translation'] = 0.7
        if 'uniform' in patterns:
            hints['color_gradient'] = 0.6
        
        # Color-based hints
        num_colors = analysis.get('num_colors', 1)
        if num_colors == 2:
            hints['color_swap'] = 0.8
        elif num_colors > 5:
            hints['color_simplification'] = 0.7
        
        # Complexity-based hints
        complexity = analysis.get('complexity', 0.5)
        if complexity < 0.3:
            hints['add_complexity'] = 0.6
        elif complexity > 0.7:
            hints['reduce_complexity'] = 0.6
        
        return hints

class GatingNetwork(nn.Module):
    """Neural network for expert selection"""
    
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Initialize weights
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute expert weights"""
        logits = self.layers(grid_features)
        weights = F.softmax(logits, dim=-1)
        return weights

class ExpertSelector:
    """Intelligent expert selection algorithms"""
    
    def __init__(self):
        self.expert_performance_history = defaultdict(list)
        self.selection_history = []
    
    def select_by_confidence(self, 
                           experts: List[BaseMovementExpert], 
                           grid: np.ndarray, 
                           context: Dict[str, Any], 
                           top_k: int = 3) -> List[ExpertCall]:
        """Select experts based on confidence scores"""
        expert_scores = []
        
        for expert in experts:
            if expert.is_applicable(grid, context):
                # Get default parameters for confidence estimation
                param_space = expert.get_parameter_space()
                default_params = {}
                for param_name, param_config in param_space.items():
                    default_params[param_name] = param_config.get('default')
                
                confidence = expert.get_confidence(grid, default_params)
                expert_scores.append((expert, confidence, default_params))
        
        # Sort by confidence and take top_k
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_experts = []
        for i, (expert, confidence, params) in enumerate(expert_scores[:top_k]):
            expert_call = ExpertCall(
                expert_name=expert.expert_name,
                movement_type=expert.movement_type,
                parameters=params,
                expected_confidence=confidence,
                priority=i,
                context=context
            )
            selected_experts.append(expert_call)
        
        return selected_experts
    
    def select_by_hints(self, 
                       experts: List[BaseMovementExpert], 
                       grid: np.ndarray, 
                       movement_hints: Dict[str, float], 
                       context: Dict[str, Any]) -> List[ExpertCall]:
        """Select experts based on movement hints from grid analysis"""
        selected_experts = []
        
        # Map hints to expert types
        hint_to_expert = {
            'flip_horizontal': ('FlipExpert', {'axis': 'horizontal'}),
            'flip_vertical': ('FlipExpert', {'axis': 'vertical'}),
            'rotate_180': ('RotationExpert', {'angle': 180}),
            'color_swap': ('ColorTransformExpert', {'type': 'swap', 'color1': 1, 'color2': 2}),
            'translation': ('TranslationExpert', {'shift_x': 1, 'shift_y': 0, 'mode': 'wrap'}),
            'color_gradient': ('ColorTransformExpert', {'type': 'gradient', 'direction': 'horizontal'})
        }
        
        # Sort hints by confidence
        sorted_hints = sorted(movement_hints.items(), key=lambda x: x[1], reverse=True)
        
        for hint_name, hint_confidence in sorted_hints[:3]:  # Top 3 hints
            if hint_name in hint_to_expert:
                expert_name, parameters = hint_to_expert[hint_name]
                
                # Find matching expert
                matching_expert = None
                for expert in experts:
                    if expert.expert_name == expert_name:
                        matching_expert = expert
                        break
                
                if matching_expert and matching_expert.is_applicable(grid, context):
                    expert_call = ExpertCall(
                        expert_name=expert_name,
                        movement_type=matching_expert.movement_type,
                        parameters=parameters,
                        expected_confidence=hint_confidence,
                        priority=len(selected_experts),
                        context=context
                    )
                    selected_experts.append(expert_call)
        
        return selected_experts
    
    def select_multi_expert(self, 
                          experts: List[BaseMovementExpert], 
                          grid: np.ndarray, 
                          context: Dict[str, Any]) -> List[ExpertCall]:
        """Select multiple experts for ensemble execution"""
        selected_experts = []
        
        # Try to select one expert from each type
        expert_types_used = set()
        
        for expert in experts:
            if (expert.movement_type not in expert_types_used and 
                expert.is_applicable(grid, context)):
                
                # Get default parameters
                param_space = expert.get_parameter_space()
                default_params = {}
                for param_name, param_config in param_space.items():
                    default_params[param_name] = param_config.get('default')
                
                confidence = expert.get_confidence(grid, default_params)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    expert_call = ExpertCall(
                        expert_name=expert.expert_name,
                        movement_type=expert.movement_type,
                        parameters=default_params,
                        expected_confidence=confidence,
                        priority=len(selected_experts),
                        context=context
                    )
                    selected_experts.append(expert_call)
                    expert_types_used.add(expert.movement_type)
        
        return selected_experts

class ExpertCombiner:
    """Combines results from multiple experts"""
    
    def __init__(self):
        self.combination_strategies = [
            'voting', 'weighted_average', 'confidence_based', 'sequential'
        ]
    
    def combine_results(self, 
                       expert_results: List[MovementResult], 
                       strategy: str = 'confidence_based') -> MovementResult:
        """Combine multiple expert results"""
        if not expert_results:
            raise ValueError("No expert results to combine")
        
        if len(expert_results) == 1:
            return expert_results[0]
        
        # Filter successful results
        successful_results = [r for r in expert_results if r.success]
        if not successful_results:
            # Return first result if all failed
            return expert_results[0]
        
        if strategy == 'confidence_based':
            return self._combine_by_confidence(successful_results)
        elif strategy == 'voting':
            return self._combine_by_voting(successful_results)
        elif strategy == 'weighted_average':
            return self._combine_by_weighted_average(successful_results)
        elif strategy == 'sequential':
            return self._combine_sequentially(successful_results)
        else:
            # Default: return highest confidence result
            return max(successful_results, key=lambda r: r.confidence)
    
    def _combine_by_confidence(self, results: List[MovementResult]) -> MovementResult:
        """Select result with highest confidence"""
        return max(results, key=lambda r: r.confidence)
    
    def _combine_by_voting(self, results: List[MovementResult]) -> MovementResult:
        """Use majority voting among results"""
        if not results:
            raise ValueError("No results to vote on")
        
        # For grid outputs, this is complex - use confidence for now
        return self._combine_by_confidence(results)
    
    def _combine_by_weighted_average(self, results: List[MovementResult]) -> MovementResult:
        """Weighted average based on confidence"""
        # For discrete grids, this is challenging - use highest confidence
        return self._combine_by_confidence(results)
    
    def _combine_sequentially(self, results: List[MovementResult]) -> MovementResult:
        """Apply results sequentially (composition)"""
        if len(results) == 1:
            return results[0]
        
        # Apply first result, then apply second result to the output, etc.
        current_grid = results[0].output_grid
        combined_confidence = results[0].confidence
        combined_time = results[0].execution_time
        
        for i in range(1, len(results)):
            # This would require re-executing experts on intermediate results
            # For now, just return the last result
            current_grid = results[i].output_grid
            combined_confidence *= results[i].confidence
            combined_time += results[i].execution_time
        
        # Create combined result
        combined_result = MovementResult(
            output_grid=current_grid,
            confidence=combined_confidence,
            operation_type="sequential_combination",
            parameters={'num_operations': len(results)},
            execution_time=combined_time,
            success=True,
            metadata={'combination_strategy': 'sequential', 'num_experts': len(results)}
        )
        
        return combined_result

class MovementMoERouter:
    """Main MoE router for intelligent expert selection and coordination"""
    
    def __init__(self, experts: List[BaseMovementExpert]):
        self.experts = {expert.expert_name: expert for expert in experts}
        self.expert_list = experts
        
        self.grid_analyzer = GridAnalyzer()
        self.expert_selector = ExpertSelector()
        self.expert_combiner = ExpertCombiner()
        
        # Routing statistics
        self.routing_history = []
        self.expert_usage_stats = defaultdict(int)
        
        # Initialize gating network (optional)
        self.use_gating_network = False
        self.gating_network = None
        
    def route(self, 
             grid: np.ndarray, 
             solver_intent: str = "general", 
             strategy: RoutingStrategy = RoutingStrategy.CONFIDENCE_BASED,
             context: Optional[Dict[str, Any]] = None) -> RoutingResult:
        """Main routing function"""
        start_time = time.time()
        
        if context is None:
            context = {}
        
        try:
            # Validate input
            is_valid, error_msg = MovementValidator.validate_grid(grid)
            if not is_valid:
                return self._create_fallback_routing(grid, f"Invalid grid: {error_msg}")
            
            # Analyze grid
            grid_analysis = self.grid_analyzer.analyze_grid(grid)
            context.update(grid_analysis)
            
            # Select routing strategy and execute
            if strategy == RoutingStrategy.CONFIDENCE_BASED:
                selected_experts = self.expert_selector.select_by_confidence(
                    self.expert_list, grid, context
                )
            elif strategy == RoutingStrategy.GATING_NETWORK and self.gating_network:
                selected_experts = self._route_with_gating_network(grid, context)
            elif strategy == RoutingStrategy.MULTI_EXPERT:
                selected_experts = self.expert_selector.select_multi_expert(
                    self.expert_list, grid, context
                )
            else:
                # Default to confidence-based
                selected_experts = self.expert_selector.select_by_confidence(
                    self.expert_list, grid, context
                )
            
            # Add hint-based selection
            movement_hints = grid_analysis.get('movement_hints', {})
            if movement_hints:
                hint_experts = self.expert_selector.select_by_hints(
                    self.expert_list, grid, movement_hints, context
                )
                # Merge with existing selections (avoid duplicates)
                existing_names = {e.expert_name for e in selected_experts}
                for hint_expert in hint_experts:
                    if hint_expert.expert_name not in existing_names:
                        selected_experts.append(hint_expert)
            
            # Update statistics
            for expert_call in selected_experts:
                self.expert_usage_stats[expert_call.expert_name] += 1
            
            routing_time = time.time() - start_time
            routing_confidence = np.mean([e.expected_confidence for e in selected_experts]) if selected_experts else 0.0
            
            result = RoutingResult(
                selected_experts=selected_experts,
                routing_confidence=routing_confidence,
                routing_time=routing_time,
                routing_strategy=strategy.value,
                grid_analysis=grid_analysis,
                fallback_used=False
            )
            
            self.routing_history.append(result)
            return result
            
        except Exception as e:
            warnings.warn(f"Routing failed: {e}")
            return self._create_fallback_routing(grid, str(e))
    
    def execute_sequence(self, 
                        grid: np.ndarray, 
                        expert_calls: List[ExpertCall],
                        combination_strategy: str = 'confidence_based') -> MovementResult:
        """Execute a sequence of expert calls"""
        results = []
        
        for expert_call in expert_calls:
            expert = self.experts.get(expert_call.expert_name)
            if expert is None:
                warnings.warn(f"Expert {expert_call.expert_name} not found")
                continue
            
            # Execute expert
            result = expert.execute(grid, expert_call.parameters)
            results.append(result)
            
            # For sequential execution, use output as input for next step
            if result.success and combination_strategy == 'sequential':
                grid = result.output_grid
        
        # Combine results
        if results:
            return self.expert_combiner.combine_results(results, combination_strategy)
        else:
            # Fallback result
            return MovementResult(
                output_grid=grid,
                confidence=0.0,
                operation_type="fallback",
                parameters={},
                execution_time=0.0,
                success=False,
                error_message="No experts executed successfully"
            )
    
    def _route_with_gating_network(self, 
                                  grid: np.ndarray, 
                                  context: Dict[str, Any]) -> List[ExpertCall]:
        """Route using neural gating network"""
        # This would use the gating network for expert selection
        # For now, fallback to confidence-based selection
        return self.expert_selector.select_by_confidence(self.expert_list, grid, context)
    
    def _create_fallback_routing(self, grid: np.ndarray, error_msg: str) -> RoutingResult:
        """Create fallback routing result"""
        # Select a safe default expert (FlipExpert with horizontal flip)
        fallback_call = ExpertCall(
            expert_name="FlipExpert",
            movement_type=MovementType.FLIP,
            parameters={'axis': 'horizontal'},
            expected_confidence=0.3,
            priority=0,
            context={'fallback': True}
        )
        
        return RoutingResult(
            selected_experts=[fallback_call],
            routing_confidence=0.3,
            routing_time=0.0,
            routing_strategy="fallback",
            grid_analysis={'error': error_msg},
            fallback_used=True
        )
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing and expert usage statistics"""
        if not self.routing_history:
            return {'message': 'No routing history available'}
        
        total_routings = len(self.routing_history)
        avg_routing_time = np.mean([r.routing_time for r in self.routing_history])
        avg_confidence = np.mean([r.routing_confidence for r in self.routing_history])
        fallback_rate = sum(1 for r in self.routing_history if r.fallback_used) / total_routings
        
        # Expert usage statistics
        total_usages = sum(self.expert_usage_stats.values())
        expert_usage_percentages = {}
        for expert_name, usage_count in self.expert_usage_stats.items():
            expert_usage_percentages[expert_name] = usage_count / total_usages if total_usages > 0 else 0.0
        
        return {
            'total_routings': total_routings,
            'average_routing_time': avg_routing_time,
            'average_confidence': avg_confidence,
            'fallback_rate': fallback_rate,
            'expert_usage_stats': dict(self.expert_usage_stats),
            'expert_usage_percentages': expert_usage_percentages,
            'strategies_used': list(set([r.routing_strategy for r in self.routing_history]))
        }

# Factory function for creating a default MoE router
def create_default_moe_router() -> MovementMoERouter:
    """Create a MoE router with default experts"""
    experts = [
        FlipExpert(),
        RotationExpert(),
        TranslationExpert(),
        ColorTransformExpert()
    ]
    
    return MovementMoERouter(experts)