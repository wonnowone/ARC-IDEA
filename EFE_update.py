#!/usr/bin/env python3
"""
EFE Update - Consolidated ARC Expected Free Energy System
=======================================================

This file consolidates all recent updates to the ARC EFE system, including:

1. Core EFE Solver (arc_efe_solver.py)
2. Robust Majority Voting (arc_efe_robust.py)  
3. Ensemble Solver (arc_efe_ensemble_solver.py)
4. Enhanced MoE Integration (enhanced_arc_ensemble_moe.py)
5. Problem Analyzer (arc_problem_analyzer.py)
6. Movement Experts (movement_experts.py)
7. Movement Language (movement_language.py)
8. Enhanced Solvers with MoE (enhanced_solvers_moe.py)
9. MoE Router (moe_router.py)
10. Enhanced MoE with Analysis (enhanced_moe_with_analysis.py)

Version: 2.0 - Full Integration Update
Author: ARC-IDEA Team
Date: 2025-08-09
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from enum import Enum
import copy
import warnings
import time
import cv2
from abc import ABC, abstractmethod

# =====================================================================================
# SECTION 1: CORE DATA STRUCTURES AND ENUMS
# =====================================================================================

@dataclass
class ARCState:
    """State representation for ARC problems"""
    grid: np.ndarray
    constraints: Dict[str, Any]
    step: int
    solver_history: List[str]
    confidence: float

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

@dataclass
class EnsembleOutput:
    """Output from ensemble of solvers with metadata"""
    output: np.ndarray
    solver_outputs: Dict[str, np.ndarray]
    consensus_reached: bool
    majority_count: int
    total_solvers: int
    ambiguity_score: float
    confidence: float

@dataclass
class EnhancedEnsembleOutput:
    """Enhanced ensemble output with MoE metadata"""
    output: np.ndarray
    solver_outputs: Dict[str, np.ndarray]
    movement_traces: Dict[str, List[MovementResult]]
    expert_consensus: Dict[str, Any]
    consensus_reached: bool
    majority_count: int
    total_solvers: int
    ambiguity_score: float
    confidence: float
    moe_statistics: Dict[str, Any]

@dataclass
class ConnectedComponent:
    """Represents a connected component (shape)"""
    label: int
    color: int
    positions: List[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]
    area: int
    centroid: Tuple[float, float]
    shape_signature: str
    
    def __post_init__(self):
        """Calculate derived properties"""
        if self.positions:
            positions_array = np.array(self.positions)
            self.centroid = (float(np.mean(positions_array[:, 0])), 
                           float(np.mean(positions_array[:, 1])))
            self.area = len(self.positions)
            
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
    change_type: 'ChangeType'
    confidence: float
    details: Dict[str, Any]
    affected_components: List[int]
    transformation_hypothesis: Optional[str] = None

class ChangeType(Enum):
    """Types of changes detected between frames"""
    POSITIONAL_CHANGE = "positional_change"
    COLOR_CHANGE = "color_change"
    OBJECT_DUPLICATION = "object_duplication"
    OBJECT_ADDITION = "object_addition"
    OBJECT_REMOVAL = "object_removal"
    SHAPE_MODIFICATION = "shape_modification"
    NO_CHANGE = "no_change"

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

class MovementOperator(Enum):
    """Basic movement operators"""
    FLIP = "flip"
    ROTATE = "rotate"
    TRANSLATE = "translate"
    COLOR_MAP = "color_map"
    SCALE = "scale"
    PATTERN_FILL = "pattern_fill"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    CONDITIONAL = "conditional"

class RoutingStrategy(Enum):
    """Different routing strategies for MoE"""
    EFE_BASED = "efe_based"
    CONFIDENCE_BASED = "confidence_based"
    CONSENSUS_BASED = "consensus_based"
    HYBRID = "hybrid"

# =====================================================================================
# SECTION 2: CORE EFE SOLVER
# =====================================================================================

class EFESolver:
    """Expected Free Energy based solver selection for ARC problems"""
    
    def __init__(self, solvers: List[Any], planning_horizon: int = 3):
        self.solvers = solvers
        self.solver_names = [s.__class__.__name__ for s in solvers]
        self.N = planning_horizon
        self.solver_preferences = {}
        self.eps = 1e-8
        
        for name in self.solver_names:
            self.solver_preferences[name] = 1.0 / len(self.solver_names)
    
    def compute_efe(self, solver_idx: int, state_t: ARCState) -> float:
        """Compute EFE for a solver given current state"""
        solver = self.solvers[solver_idx]
        solver_name = self.solver_names[solver_idx]
        
        predicted_output = solver.predict(state_t.grid)
        Q_o_t = self.get_output_distribution(predicted_output)
        C_t = self.get_constraint_distribution(state_t)
        
        risk = self.kl_divergence(Q_o_t, C_t)
        constraint_prob = self.evaluate_constraints(predicted_output, state_t.constraints)
        ambiguity = -np.log(constraint_prob + self.eps)
        
        efe = risk + ambiguity
        return efe
    
    def kl_divergence(self, Q: np.ndarray, P: np.ndarray) -> float:
        """Compute KL divergence D_KL(Q||P)"""
        Q = np.clip(Q, self.eps, 1.0)
        P = np.clip(P, self.eps, 1.0)
        return np.sum(Q * np.log(Q / P))
    
    def get_output_distribution(self, output: np.ndarray) -> np.ndarray:
        """Convert solver output to probability distribution"""
        flat_output = output.flatten()
        exp_out = np.exp(flat_output - np.max(flat_output))
        return exp_out / np.sum(exp_out)
    
    def get_constraint_distribution(self, state: ARCState) -> np.ndarray:
        """Generate constraint-based target distribution C_t(o_t)"""
        constraints = state.constraints
        grid_size = state.grid.size
        
        C_t = np.ones(grid_size) / grid_size
        
        if 'color_constraints' in constraints:
            color_weights = constraints['color_constraints']
            for i, weight in enumerate(color_weights):
                C_t[i] *= weight
        
        if 'pattern_constraints' in constraints:
            pattern_mask = constraints['pattern_constraints']
            C_t *= pattern_mask.flatten()
        
        C_t = C_t / (np.sum(C_t) + self.eps)
        return C_t
    
    def evaluate_constraints(self, output: np.ndarray, constraints: Dict) -> float:
        """Evaluate how well output satisfies constraints"""
        satisfaction = 1.0
        
        if 'color_constraints' in constraints:
            unique_colors = np.unique(output)
            expected_colors = constraints['color_constraints']
            color_match = len(set(unique_colors) & set(expected_colors)) / len(expected_colors)
            satisfaction *= color_match
        
        if 'pattern_constraints' in constraints:
            pattern_score = self.check_pattern_match(output, constraints['pattern_constraints'])
            satisfaction *= pattern_score
        
        if 'symmetry' in constraints:
            symmetry_score = self.check_symmetry(output, constraints['symmetry'])
            satisfaction *= symmetry_score
        
        return max(satisfaction, self.eps)
    
    def check_pattern_match(self, output: np.ndarray, pattern: np.ndarray) -> float:
        """Check pattern matching score"""
        if output.shape != pattern.shape:
            return 0.0
        matches = np.sum(output == pattern)
        total = output.size
        return matches / total
    
    def check_symmetry(self, output: np.ndarray, symmetry_type: str) -> float:
        """Check symmetry constraints"""
        if symmetry_type == 'horizontal':
            flipped = np.fliplr(output)
            matches = np.sum(output == flipped)
        elif symmetry_type == 'vertical':
            flipped = np.flipud(output)
            matches = np.sum(output == flipped)
        elif symmetry_type == 'rotational':
            rotated = np.rot90(output, 2)
            matches = np.sum(output == rotated)
        else:
            return 1.0
        
        return matches / output.size

# =====================================================================================
# SECTION 3: REVTHINK VERIFIER
# =====================================================================================

class RevThinkVerifier:
    """Reverse thinking verification system with actual prompting"""
    
    def __init__(self, llm_wrappers=None):
        self.verification_threshold = 0.7
        self.llm_wrappers = llm_wrappers or []
        
        # ARC-specific RevThink prompts based on RevThink methodology
        self.arc_revthink_prompts = {
            'backward_question_generation': self._get_backward_question_prompt(),
            'forward_reasoning': self._get_forward_reasoning_prompt(),
            'consistency_check': self._get_consistency_check_prompt()
        }
    
    def verify_solution(self, solution: np.ndarray, original_state: ARCState) -> Dict[str, float]:
        """Apply RevThink verification with actual LLM prompting"""
        results = {}
        
        if self.llm_wrappers:
            # Use actual prompting with LLMs for RevThink
            results['prompt_forward_score'] = self.prompt_forward_verification(solution, original_state)
            results['prompt_backward_score'] = self.prompt_backward_verification(solution, original_state)
            results['prompt_consistency_score'] = self.prompt_consistency_verification(solution, original_state)
            
            # Combined prompt-based score
            results['prompt_combined_score'] = (
                results['prompt_forward_score'] * 0.4 + 
                results['prompt_backward_score'] * 0.4 + 
                results['prompt_consistency_score'] * 0.2
            )
        else:
            # Fallback to computational verification
            results['prompt_forward_score'] = 0.5
            results['prompt_backward_score'] = 0.5
            results['prompt_consistency_score'] = 0.5
            results['prompt_combined_score'] = 0.5
        
        # Also include original computational methods for comparison
        results['forward_score'] = self.forward_verification(solution, original_state)
        results['backward_score'] = self.backward_verification(solution, original_state)
        results['process_score'] = self.process_verification(solution, original_state)
        
        # Combined score using prompt-based if available, otherwise computational
        if self.llm_wrappers:
            results['combined_score'] = results['prompt_combined_score']
        else:
            results['combined_score'] = (
                results['forward_score'] * 0.4 + 
                results['backward_score'] * 0.4 + 
                results['process_score'] * 0.2
            )
        
        return results
    
    def forward_verification(self, solution: np.ndarray, state: ARCState) -> float:
        """Verify if solution logically follows from input"""
        input_patterns = self.extract_patterns(state.grid)
        output_patterns = self.extract_patterns(solution)
        
        consistency_score = self.measure_transformation_consistency(input_patterns, output_patterns)
        return consistency_score
    
    def backward_verification(self, solution: np.ndarray, state: ARCState) -> float:
        """Verify if input could reasonably lead to this solution"""
        possible_inputs = self.generate_reverse_inputs(solution)
        similarity_scores = [self.grid_similarity(state.grid, inp) for inp in possible_inputs]
        return max(similarity_scores) if similarity_scores else 0.0
    
    def process_verification(self, solution: np.ndarray, state: ARCState) -> float:
        """Verify the reasoning process used"""
        solver_sequence = state.solver_history
        sequence_validity = self.validate_solver_sequence(solver_sequence)
        return sequence_validity
    
    def extract_patterns(self, grid: np.ndarray) -> List[Dict]:
        """Extract visual patterns from grid"""
        patterns = []
        
        unique, counts = np.unique(grid, return_counts=True)
        patterns.append({'type': 'color_dist', 'data': dict(zip(unique, counts))})
        
        contours = self.find_contiguous_regions(grid)
        patterns.append({'type': 'shapes', 'data': contours})
        
        return patterns
    
    def find_contiguous_regions(self, grid: np.ndarray) -> List[Dict]:
        """Find contiguous regions of same color"""
        regions = []
        visited = np.zeros_like(grid, dtype=bool)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j]:
                    region = self.flood_fill(grid, i, j, visited)
                    if len(region) > 1:
                        regions.append({
                            'color': grid[i, j],
                            'size': len(region),
                            'positions': region
                        })
        
        return regions
    
    def flood_fill(self, grid: np.ndarray, start_i: int, start_j: int, visited: np.ndarray) -> List[Tuple[int, int]]:
        """Flood fill to find connected component"""
        if visited[start_i, start_j]:
            return []
        
        color = grid[start_i, start_j]
        stack = [(start_i, start_j)]
        region = []
        
        while stack:
            i, j = stack.pop()
            if (i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or 
                visited[i, j] or grid[i, j] != color):
                continue
            
            visited[i, j] = True
            region.append((i, j))
            
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((i + di, j + dj))
        
        return region
    
    def measure_transformation_consistency(self, input_patterns: List[Dict], output_patterns: List[Dict]) -> float:
        """Measure consistency of transformation between input and output patterns"""
        if not input_patterns or not output_patterns:
            return 0.5
        
        consistency_scores = []
        
        for inp_pattern in input_patterns:
            for out_pattern in output_patterns:
                if inp_pattern['type'] == out_pattern['type']:
                    score = self.pattern_similarity(inp_pattern['data'], out_pattern['data'])
                    consistency_scores.append(score)
        
        return np.mean(consistency_scores) if consistency_scores else 0.3
    
    def pattern_similarity(self, pattern1: Any, pattern2: Any) -> float:
        """Compute similarity between two patterns"""
        if isinstance(pattern1, dict) and isinstance(pattern2, dict):
            common_keys = set(pattern1.keys()) & set(pattern2.keys())
            if not common_keys:
                return 0.0
            
            similarity = 0.0
            for key in common_keys:
                similarity += min(pattern1[key], pattern2[key]) / max(pattern1[key], pattern2[key])
            
            return similarity / len(common_keys)
        
        return 0.5
    
    def generate_reverse_inputs(self, solution: np.ndarray) -> List[np.ndarray]:
        """Generate possible inputs that could lead to the solution"""
        possible_inputs = []
        
        possible_inputs.append(solution)
        possible_inputs.append(np.fliplr(solution))
        possible_inputs.append(np.flipud(solution))
        possible_inputs.append(np.rot90(solution, -1))
        
        return possible_inputs
    
    def grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute similarity between two grids"""
        if grid1.shape != grid2.shape:
            return 0.0
        
        matches = np.sum(grid1 == grid2)
        total = grid1.size
        return matches / total
    
    def validate_solver_sequence(self, solver_sequence: List[str]) -> float:
        """Validate the sequence of solvers used"""
        if not solver_sequence:
            return 0.5
        
        switches = 0
        for i in range(1, len(solver_sequence)):
            if solver_sequence[i] != solver_sequence[i-1]:
                switches += 1
        
        switch_ratio = switches / len(solver_sequence)
        return max(0.1, 1.0 - switch_ratio)
    
    # ===== NEW PROMPT-BASED REVTHINK METHODS =====
    
    def prompt_forward_verification(self, solution: np.ndarray, original_state: ARCState) -> float:
        """Forward verification using LLM prompts"""
        input_grid_str = self._grid_to_string(original_state.grid)
        output_grid_str = self._grid_to_string(solution)
        
        forward_prompt = self.arc_revthink_prompts['forward_reasoning'].format(
            input_grid=input_grid_str,
            output_grid=output_grid_str
        )
        
        scores = []
        for llm_wrapper in self.llm_wrappers:
            try:
                response = llm_wrapper.generate(forward_prompt, max_tokens=200)
                score = self._parse_reasoning_quality(response)
                scores.append(score)
            except Exception as e:
                warnings.warn(f"Forward verification failed for {llm_wrapper.__class__.__name__}: {e}")
                scores.append(0.3)
        
        return np.mean(scores) if scores else 0.3
    
    def prompt_backward_verification(self, solution: np.ndarray, original_state: ARCState) -> float:
        """Backward verification using LLM prompts"""
        input_grid_str = self._grid_to_string(original_state.grid)
        output_grid_str = self._grid_to_string(solution)
        
        # Generate backward question
        backward_question = self._generate_backward_question(input_grid_str, output_grid_str)
        
        scores = []
        for llm_wrapper in self.llm_wrappers:
            try:
                # Ask the backward question
                response = llm_wrapper.generate(backward_question, max_tokens=200)
                score = self._parse_reasoning_quality(response)
                scores.append(score)
            except Exception as e:
                warnings.warn(f"Backward verification failed for {llm_wrapper.__class__.__name__}: {e}")
                scores.append(0.3)
        
        return np.mean(scores) if scores else 0.3
    
    def prompt_consistency_verification(self, solution: np.ndarray, original_state: ARCState) -> float:
        """Consistency verification using LLM prompts"""
        input_grid_str = self._grid_to_string(original_state.grid)
        output_grid_str = self._grid_to_string(solution)
        
        consistency_prompt = self.arc_revthink_prompts['consistency_check'].format(
            input_grid=input_grid_str,
            output_grid=output_grid_str
        )
        
        scores = []
        for llm_wrapper in self.llm_wrappers:
            try:
                response = llm_wrapper.generate(consistency_prompt, max_tokens=150)
                score = self._parse_consistency_score(response)
                scores.append(score)
            except Exception as e:
                warnings.warn(f"Consistency verification failed for {llm_wrapper.__class__.__name__}: {e}")
                scores.append(0.3)
        
        return np.mean(scores) if scores else 0.3
    
    def _generate_backward_question(self, input_grid_str: str, output_grid_str: str) -> str:
        """Generate backward question based on RevThink methodology"""
        return f"""<INSTRUCTIONS>Your task is to generate an inverse question based on the ARC transformation.
Follow these rules:
1. Use the output grid to create a question about what input could lead to this output.
2. Consider the visual patterns, colors, and transformations.
3. Make sure the question tests understanding of the transformation logic.
</INSTRUCTIONS>

<EXAMPLE>
INPUT GRID:
1 0 1
0 1 0  
1 0 1

OUTPUT GRID:
0 1 0
1 0 1
0 1 0

BACKWARD QUESTION: If the output grid shows a checkerboard pattern with colors flipped from the original, what transformation rule was applied?
</EXAMPLE>

INPUT GRID:
{input_grid_str}

OUTPUT GRID:
{output_grid_str}

Generate a backward question that tests understanding of this transformation:"""
    
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert grid to readable string representation"""
        if grid.size == 0:
            return "Empty grid"
        
        rows = []
        for row in grid:
            rows.append(' '.join(map(str, row)))
        return '\n'.join(rows)
    
    def _parse_reasoning_quality(self, response: str) -> float:
        """Parse reasoning quality from LLM response"""
        if not response or len(response.strip()) < 10:
            return 0.1
        
        # Simple heuristic scoring based on response characteristics
        score = 0.3  # Base score
        
        # Check for logical reasoning indicators
        reasoning_indicators = [
            'because', 'therefore', 'since', 'so', 'thus', 'hence',
            'pattern', 'transformation', 'rule', 'logic', 'reason'
        ]
        
        response_lower = response.lower()
        for indicator in reasoning_indicators:
            if indicator in response_lower:
                score += 0.1
        
        # Check response length (more detailed = better)
        if len(response.strip()) > 50:
            score += 0.1
        if len(response.strip()) > 100:
            score += 0.1
        
        # Check for uncertainty expressions (lower score)
        uncertainty_indicators = [
            'maybe', 'perhaps', 'might', 'could be', 'not sure', 'unclear'
        ]
        
        for indicator in uncertainty_indicators:
            if indicator in response_lower:
                score -= 0.1
        
        return np.clip(score, 0.0, 1.0)
    
    def _parse_consistency_score(self, response: str) -> float:
        """Parse consistency score from LLM response"""
        if not response:
            return 0.3
        
        response_lower = response.lower().strip()
        
        # Look for explicit consistency indicators
        if any(word in response_lower for word in ['consistent', 'true', 'correct', 'valid']):
            return 0.8
        elif any(word in response_lower for word in ['inconsistent', 'false', 'incorrect', 'invalid']):
            return 0.2
        else:
            # Parse based on reasoning quality
            return self._parse_reasoning_quality(response) * 0.7 + 0.3
    
    def _get_backward_question_prompt(self) -> str:
        """ARC-specific backward question generation prompt"""
        return """<INSTRUCTIONS>Your task is to generate an inverse question with the same visual reasoning challenge, based on the input ARC transformation and its correct output. Follow these rules:
1. Use the correct output from the input transformation to create a new, related but inverse question.
2. Ensure the new question tests the same visual reasoning pattern in reverse.
3. Make sure only one logical answer exists for your generated question.
4. The generated question should test understanding of the transformation rule.
5. Focus on visual patterns, color changes, shape transformations, and spatial relationships.
</INSTRUCTIONS>

<EXAMPLE>
INPUT TRANSFORMATION: Grid with scattered red squares becomes grid with red squares connected in lines.
CORRECT OUTPUT: Connected red lines pattern.

INVERSE QUESTION: If you see a grid with connected red lines, what was the most likely original pattern before the transformation? 
(A) Scattered red squares (B) Blue connected lines (C) Red filled rectangle (D) Random colored dots.
The correct answer is (A).
</EXAMPLE>

{transformation_description}
"""
    
    def _get_forward_reasoning_prompt(self) -> str:
        """ARC-specific forward reasoning prompt"""
        return """Provide your step-by-step visual reasoning for this ARC transformation. 

INPUT GRID:
{input_grid}

OUTPUT GRID:
{output_grid}

Analyze:
1. What visual patterns do you see in the input?
2. What changes occurred to create the output?
3. What transformation rule explains this change?
4. Does this transformation make logical sense?

Provide your reasoning and conclude with: "The transformation is [VALID/INVALID] because [reason]"."""
    
    def _get_consistency_check_prompt(self) -> str:
        """ARC-specific consistency check prompt"""
        return """<INSTRUCTIONS>You will be given an ARC transformation pair (input grid, output grid).
Your task is to check if the output is consistent with the input.
If the output logically follows from the input via a clear transformation rule, output `True`.
Otherwise, if the output doesn't make sense given the input, output `False`.</INSTRUCTIONS>

<EXAMPLE>
INPUT GRID:
1 0 1
0 1 0
1 0 1

OUTPUT GRID:
0 1 0  
1 0 1
0 1 0

ANALYSIS:
The output shows each cell's color flipped (1→0, 0→1).
This is a consistent "invert colors" transformation rule.
True

INPUT GRID:
2 2 2
2 2 2  
2 2 2

OUTPUT GRID:
1 3 5
7 2 9
4 6 8

ANALYSIS:
The output shows random different colors with no clear pattern.
There's no consistent transformation rule apparent.
False
</EXAMPLE>

INPUT GRID:
{input_grid}

OUTPUT GRID:
{output_grid}

ANALYSIS:"""

# =====================================================================================
# SECTION 4: Z-LEARNING UPDATER
# =====================================================================================

class ZLearningUpdater:
    """Z-learning based preference updates"""
    
    def __init__(self, learning_rate: float = 0.1, temperature: float = 1.0):
        self.alpha = learning_rate
        self.temperature = temperature
    
    def update_preferences(self, 
                         solver_preferences: Dict[str, float],
                         solver_name: str,
                         efe_score: float,
                         verification_results: Dict[str, float]) -> Dict[str, float]:
        """Update solver preferences using Z-learning approach"""
        efe_reward = 1.0 / (1.0 + efe_score)
        verification_reward = verification_results.get('combined_score', 0.5)
        total_reward = 0.7 * efe_reward + 0.3 * verification_reward
        
        current_pref = solver_preferences[solver_name]
        prediction_error = total_reward - current_pref
        updated_pref = current_pref + self.alpha * prediction_error
        
        new_preferences = solver_preferences.copy()
        new_preferences[solver_name] = updated_pref
        
        pref_values = np.array(list(new_preferences.values()))
        softmax_prefs = np.exp(pref_values / self.temperature)
        softmax_prefs = softmax_prefs / np.sum(softmax_prefs)
        
        for i, name in enumerate(new_preferences.keys()):
            new_preferences[name] = softmax_prefs[i]
        
        return new_preferences

# =====================================================================================
# SECTION 5: MAJORITY VOTING CONSENSUS
# =====================================================================================

class MajorityVotingConsensus:
    """Implements majority voting consensus mechanism for solver ensemble"""
    
    def __init__(self, min_consensus_threshold: float = 0.5):
        self.min_consensus_threshold = min_consensus_threshold
        self.eps = 1e-8
    
    def compute_consensus(self, solver_outputs: Dict[str, np.ndarray]) -> EnsembleOutput:
        """Compute majority voting consensus from multiple solver outputs"""
        if not solver_outputs:
            raise ValueError("No solver outputs provided")
        
        first_output = next(iter(solver_outputs.values()))
        grid_shape = first_output.shape
        
        consensus_grid = np.zeros(grid_shape, dtype=int)
        vote_matrix = np.zeros(grid_shape + (len(solver_outputs),), dtype=int)
        
        solver_names = list(solver_outputs.keys())
        
        for pos_i in range(grid_shape[0]):
            for pos_j in range(grid_shape[1]):
                votes = []
                for solver_idx, solver_name in enumerate(solver_names):
                    value = solver_outputs[solver_name][pos_i, pos_j]
                    votes.append(value)
                    vote_matrix[pos_i, pos_j, solver_idx] = value
                
                vote_counts = Counter(votes)
                majority_value, majority_count = vote_counts.most_common(1)[0]
                consensus_grid[pos_i, pos_j] = majority_value
        
        total_positions = grid_shape[0] * grid_shape[1]
        consensus_positions = 0
        ambiguity_scores = []
        
        for pos_i in range(grid_shape[0]):
            for pos_j in range(grid_shape[1]):
                position_votes = vote_matrix[pos_i, pos_j, :]
                vote_counts = Counter(position_votes)
                
                if len(vote_counts) == 1:
                    consensus_positions += 1
                    ambiguity_scores.append(0.0)
                else:
                    most_common_count = vote_counts.most_common(1)[0][1]
                    consensus_ratio = most_common_count / len(solver_names)
                    
                    if consensus_ratio >= self.min_consensus_threshold:
                        consensus_positions += 1
                    
                    vote_probs = np.array(list(vote_counts.values())) / len(solver_names)
                    entropy = -np.sum(vote_probs * np.log(vote_probs + self.eps))
                    ambiguity_scores.append(entropy)
        
        consensus_reached = consensus_positions >= (total_positions * self.min_consensus_threshold)
        overall_ambiguity = np.mean(ambiguity_scores)
        confidence = consensus_positions / total_positions
        
        majority_count = sum(1 for solver_name in solver_names 
                           if np.array_equal(solver_outputs[solver_name], consensus_grid))
        
        return EnsembleOutput(
            output=consensus_grid,
            solver_outputs=solver_outputs,
            consensus_reached=consensus_reached,
            majority_count=majority_count,
            total_solvers=len(solver_names),
            ambiguity_score=overall_ambiguity,
            confidence=confidence
        )

# =====================================================================================
# SECTION 6: ROBUST MAJORITY VOTING
# =====================================================================================

class RobustMajorityVoting:
    """Robust majority voting with fallback mechanisms"""
    
    def __init__(self, min_consensus_threshold: float = 0.5):
        self.min_consensus_threshold = max(0.1, min(0.9, min_consensus_threshold))
        self.eps = 1e-12
        self.max_retries = 3
        
    def compute_consensus_safe(self, solver_outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Safe consensus computation with comprehensive error handling"""
        
        if not solver_outputs:
            return self._create_fallback_consensus()
            
        shapes = [output.shape for output in solver_outputs.values()]
        if len(set(shapes)) > 1:
            warnings.warn("Inconsistent grid shapes detected, normalizing...")
            solver_outputs = self._normalize_grid_shapes(solver_outputs)
        
        try:
            return self._compute_consensus_core(solver_outputs)
        except Exception as e:
            warnings.warn(f"Consensus computation failed: {e}. Using fallback.")
            return self._create_fallback_consensus()
    
    def _compute_consensus_core(self, solver_outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Core consensus computation with safety checks"""
        
        first_output = next(iter(solver_outputs.values()))
        grid_shape = first_output.shape
        consensus_grid = np.zeros(grid_shape, dtype=int)
        
        for name, output in solver_outputs.items():
            if not np.all(np.isfinite(output)):
                warnings.warn(f"Solver {name} produced non-finite values, excluding...")
                solver_outputs = {k: v for k, v in solver_outputs.items() if k != name}
        
        if not solver_outputs:
            return self._create_fallback_consensus()
        
        ambiguity_scores = []
        consensus_positions = 0
        total_positions = grid_shape[0] * grid_shape[1]
        
        for pos_i in range(grid_shape[0]):
            for pos_j in range(grid_shape[1]):
                votes = [solver_outputs[name][pos_i, pos_j] for name in solver_outputs.keys()]
                
                if len(set(votes)) <= 1:
                    consensus_grid[pos_i, pos_j] = votes[0] if votes else 0
                    consensus_positions += 1
                    ambiguity_scores.append(0.0)
                    continue
                
                vote_counts = Counter(votes)
                majority_value, majority_count = vote_counts.most_common(1)[0]
                
                top_votes = vote_counts.most_common(2)
                if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
                    tied_values = [val for val, count in top_votes if count == top_votes[0][1]]
                    majority_value = min(tied_values)
                
                consensus_grid[pos_i, pos_j] = majority_value
                
                consensus_ratio = majority_count / max(1, len(votes))
                if consensus_ratio >= self.min_consensus_threshold:
                    consensus_positions += 1
                
                vote_probs = np.array(list(vote_counts.values()), dtype=np.float64)
                vote_probs = vote_probs / np.sum(vote_probs)
                vote_probs = np.maximum(vote_probs, self.eps)
                
                entropy = -np.sum(vote_probs * np.log(vote_probs))
                ambiguity_scores.append(min(entropy, 10.0))
        
        consensus_reached = consensus_positions >= max(1, int(total_positions * self.min_consensus_threshold))
        confidence = consensus_positions / max(1, total_positions)
        overall_ambiguity = np.clip(np.mean(ambiguity_scores), 0.0, 10.0)
        
        majority_count = 0
        for name in solver_outputs.keys():
            try:
                if np.array_equal(solver_outputs[name], consensus_grid):
                    majority_count += 1
            except:
                continue
        
        return {
            'output': consensus_grid,
            'solver_outputs': solver_outputs,
            'consensus_reached': consensus_reached,
            'majority_count': majority_count,
            'total_solvers': len(solver_outputs),
            'ambiguity_score': overall_ambiguity,
            'confidence': confidence,
            'valid': True
        }
    
    def _normalize_grid_shapes(self, solver_outputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Normalize all grids to consistent shape"""
        shapes = [output.shape for output in solver_outputs.values()]
        target_shape = max(shapes, key=lambda x: x[0] * x[1])
        
        normalized = {}
        for name, output in solver_outputs.items():
            if output.shape != target_shape:
                normalized_grid = np.zeros(target_shape, dtype=output.dtype)
                min_h = min(output.shape[0], target_shape[0])
                min_w = min(output.shape[1], target_shape[1])
                normalized_grid[:min_h, :min_w] = output[:min_h, :min_w]
                normalized[name] = normalized_grid
            else:
                normalized[name] = output
        
        return normalized
    
    def _create_fallback_consensus(self) -> Dict[str, Any]:
        """Create fallback consensus when main computation fails"""
        fallback_grid = np.zeros((5, 5), dtype=int)
        
        return {
            'output': fallback_grid,
            'solver_outputs': {'fallback': fallback_grid},
            'consensus_reached': False,
            'majority_count': 0,
            'total_solvers': 1,
            'ambiguity_score': 5.0,
            'confidence': 0.1,
            'valid': False
        }

# =====================================================================================
# SECTION 7: MOVEMENT EXPERTS
# =====================================================================================

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
        """Execute the movement transformation"""
        pass
    
    @abstractmethod
    def get_confidence(self, input_grid: np.ndarray, parameters: Dict[str, Any]) -> float:
        """Estimate confidence for applying this transformation"""
        pass
    
    @abstractmethod
    def is_applicable(self, input_grid: np.ndarray, context: Dict[str, Any]) -> bool:
        """Check if this expert can be applied to the given grid"""
        pass

# =====================================================================================
# SECTION 8: PROBLEM ANALYZER
# =====================================================================================

class ColorFilter:
    """Filters grid into separate color layers"""
    
    def __init__(self):
        self.color_layers = {}
    
    def extract_color_layers(self, grid: np.ndarray) -> Dict[int, np.ndarray]:
        """Extract each unique color into separate binary layers"""
        unique_colors = np.unique(grid)
        color_layers = {}
        
        for color in unique_colors:
            if color == 0:  # Skip background
                continue
            binary_mask = (grid == color).astype(int)
            color_layers[color] = binary_mask
        
        return color_layers

class ComponentLabeler:
    """8-connected component labeling for shape analysis"""
    
    def __init__(self):
        pass
    
    def find_connected_components(self, binary_mask: np.ndarray, color: int) -> List[ConnectedComponent]:
        """Find all connected components in binary mask"""
        components = []
        labeled_mask, num_labels = cv2.connectedComponents(binary_mask.astype(np.uint8), connectivity=8)
        
        for label in range(1, num_labels + 1):
            positions = np.where(labeled_mask == label)
            positions = list(zip(positions[0], positions[1]))
            
            if positions:
                min_row, min_col = np.min(positions, axis=0)
                max_row, max_col = np.max(positions, axis=0)
                
                component = ConnectedComponent(
                    label=label,
                    color=color,
                    positions=positions,
                    bounding_box=(min_row, min_col, max_row, max_col),
                    area=len(positions),
                    centroid=(0, 0),  # Will be calculated in __post_init__
                    shape_signature=""  # Will be calculated in __post_init__
                )
                components.append(component)
        
        return components

class ARCProblemAnalyzer:
    """Comprehensive ARC problem analysis system"""
    
    def __init__(self):
        self.color_filter = ColorFilter()
        self.component_labeler = ComponentLabeler()
        
    def analyze_problem(self, input_grid: np.ndarray, output_grid: np.ndarray = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of ARC problem"""
        
        # Color-based analysis
        input_color_layers = self.color_filter.extract_color_layers(input_grid)
        input_color_frames = {}
        
        for color, binary_mask in input_color_layers.items():
            components = self.component_labeler.find_connected_components(binary_mask, color)
            color_frame = ColorFrame(
                color=color,
                binary_mask=binary_mask,
                connected_components=components,
                component_count=len(components),
                total_pixels=np.sum(binary_mask)
            )
            input_color_frames[color] = color_frame
        
        # Shape-based analysis
        all_components = []
        for frame in input_color_frames.values():
            all_components.extend(frame.connected_components)
        
        shape_distribution = {}
        size_distribution = {}
        
        for component in all_components:
            shape_distribution[component.shape_signature] = shape_distribution.get(component.shape_signature, 0) + 1
            size_distribution[component.area] = size_distribution.get(component.area, 0) + 1
        
        input_shape_frame = ShapeFrame(
            all_components=all_components,
            component_count=len(all_components),
            shape_distribution=shape_distribution,
            size_distribution=size_distribution
        )
        
        analysis_result = {
            'input_analysis': {
                'color_frames': input_color_frames,
                'shape_frame': input_shape_frame,
                'unique_colors': list(input_color_layers.keys()),
                'total_components': len(all_components)
            }
        }
        
        # Output analysis if provided
        if output_grid is not None:
            output_color_layers = self.color_filter.extract_color_layers(output_grid)
            output_color_frames = {}
            
            for color, binary_mask in output_color_layers.items():
                components = self.component_labeler.find_connected_components(binary_mask, color)
                color_frame = ColorFrame(
                    color=color,
                    binary_mask=binary_mask,
                    connected_components=components,
                    component_count=len(components),
                    total_pixels=np.sum(binary_mask)
                )
                output_color_frames[color] = color_frame
            
            # Change analysis
            change_analysis = self._analyze_changes(input_color_frames, output_color_frames)
            analysis_result['output_analysis'] = {
                'color_frames': output_color_frames,
                'change_analysis': change_analysis
            }
        
        return analysis_result
    
    def _analyze_changes(self, input_frames: Dict[int, ColorFrame], 
                        output_frames: Dict[int, ColorFrame]) -> List[ChangeAnalysis]:
        """Analyze changes between input and output frames"""
        changes = []
        
        # Simple change detection - can be expanded
        input_colors = set(input_frames.keys())
        output_colors = set(output_frames.keys())
        
        if input_colors != output_colors:
            added_colors = output_colors - input_colors
            removed_colors = input_colors - output_colors
            
            if added_colors:
                changes.append(ChangeAnalysis(
                    change_type=ChangeType.OBJECT_ADDITION,
                    confidence=0.8,
                    details={'added_colors': list(added_colors)},
                    affected_components=[]
                ))
            
            if removed_colors:
                changes.append(ChangeAnalysis(
                    change_type=ChangeType.OBJECT_REMOVAL,
                    confidence=0.8,
                    details={'removed_colors': list(removed_colors)},
                    affected_components=[]
                ))
        
        return changes

# =====================================================================================
# SECTION 9: CONTRASTIVE LEARNING MODULE
# =====================================================================================

class ContrastiveLearningModule:
    """Contrastive learning for distinguishing correct vs incorrect solver outputs"""
    
    def __init__(self, embedding_dim: int = 128, temperature: float = 0.1):
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.eps = 1e-8
        
        self.embedding_net = self._create_embedding_network()
        
    def _create_embedding_network(self):
        """Create embedding network for grid representations"""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(25, 64),  # Assuming 5x5 grids
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
    
    def compute_contrastive_loss(self, 
                                consensus_output: np.ndarray,
                                solver_outputs: Dict[str, np.ndarray],
                                correct_solvers: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute contrastive loss between consensus (positive) and failed outputs (negative)"""
        
        consensus_tensor = torch.FloatTensor(consensus_output).unsqueeze(0)
        consensus_embedding = self.embedding_net(consensus_tensor)
        
        positive_embeddings = []
        negative_embeddings = []
        similarity_scores = {}
        
        for solver_name, output in solver_outputs.items():
            output_tensor = torch.FloatTensor(output).unsqueeze(0)
            output_embedding = self.embedding_net(output_tensor)
            
            similarity = F.cosine_similarity(consensus_embedding, output_embedding).item()
            similarity_scores[solver_name] = similarity
            
            if solver_name in correct_solvers:
                positive_embeddings.append(output_embedding)
            else:
                negative_embeddings.append(output_embedding)
        
        if positive_embeddings and negative_embeddings:
            loss = self._compute_infonce_loss(
                consensus_embedding, 
                positive_embeddings, 
                negative_embeddings
            )
        else:
            loss = torch.tensor(0.0)
        
        return loss, similarity_scores
    
    def _compute_infonce_loss(self, 
                             anchor: torch.Tensor,
                             positives: List[torch.Tensor], 
                             negatives: List[torch.Tensor]) -> torch.Tensor:
        """Compute InfoNCE contrastive loss"""
        all_embeddings = torch.cat(positives + negatives, dim=0)
        similarities = F.cosine_similarity(anchor, all_embeddings) / self.temperature
        
        num_positives = len(positives)
        labels = torch.cat([
            torch.ones(num_positives),
            torch.zeros(len(negatives))
        ])
        
        loss = F.cross_entropy(similarities.unsqueeze(0), labels.long().unsqueeze(0))
        return loss

# =====================================================================================
# SECTION 10: COMPREHENSIVE LOSS CALCULATOR
# =====================================================================================

class ComprehensiveLossCalculator:
    """Calculates the comprehensive loss function with all components"""
    
    def __init__(self, 
                 lambda_contrast: float = 1.0,
                 lambda_ambiguity: float = 0.5,
                 lambda_chaos: float = 0.3):
        self.lambda_contrast = lambda_contrast
        self.lambda_ambiguity = lambda_ambiguity
        self.lambda_chaos = lambda_chaos
        
    def compute_total_loss(self,
                          efe_loss: float,
                          contrastive_loss: torch.Tensor,
                          ambiguity_score: float,
                          diversity_score: float,
                          solver_outputs: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute total loss combining all components"""
        
        contrastive_value = contrastive_loss.item() if torch.is_tensor(contrastive_loss) else contrastive_loss
        ambiguity_penalty = ambiguity_score
        chaos_loss = self._compute_chaos_loss(diversity_score, solver_outputs)
        
        total_loss = (efe_loss + 
                     self.lambda_contrast * contrastive_value + 
                     self.lambda_ambiguity * ambiguity_penalty + 
                     self.lambda_chaos * chaos_loss)
        
        return {
            'total_loss': total_loss,
            'efe_loss': efe_loss,
            'contrastive_loss': contrastive_value,
            'ambiguity_penalty': ambiguity_penalty,
            'chaos_loss': chaos_loss,
            'diversity_score': diversity_score
        }
    
    def _compute_chaos_loss(self, diversity_score: float, solver_outputs: Dict[str, np.ndarray]) -> float:
        """Compute chaos loss to maintain controlled diversity"""
        outputs = list(solver_outputs.values())
        if len(outputs) < 2:
            return 0.0
        
        pairwise_differences = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff = np.mean(outputs[i] != outputs[j])
                pairwise_differences.append(diff)
        
        avg_diversity = np.mean(pairwise_differences)
        optimal_diversity = 0.3
        chaos_loss = abs(avg_diversity - optimal_diversity)
        
        return chaos_loss

# =====================================================================================
# SECTION 11: MAIN ARC EFE SYSTEM
# =====================================================================================

class ARCEFESystem:
    """Main system integrating EFE-based solver selection with RevThink verification"""
    
    def __init__(self, solvers: List[Any], planning_horizon: int = 3):
        self.efe_solver = EFESolver(solvers, planning_horizon)
        self.revthink_verifier = RevThinkVerifier()
        self.z_learner = ZLearningUpdater()
        self.max_iterations = 10
    
    def solve_arc_problem(self, initial_grid: np.ndarray, constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Main solving loop following EFE approach"""
        
        current_state = ARCState(
            grid=initial_grid.copy(),
            constraints=constraints,
            step=0,
            solver_history=[],
            confidence=0.0
        )
        
        solution_history = []
        
        for iteration in range(self.max_iterations):
            # Evaluate all solvers with EFE
            efe_scores = {}
            for i, solver_name in enumerate(self.efe_solver.solver_names):
                efe_score = self.efe_solver.compute_efe(i, current_state)
                efe_scores[solver_name] = efe_score
            
            # Select solver with minimum EFE
            weighted_scores = {}
            for name, efe in efe_scores.items():
                preference = self.efe_solver.solver_preferences[name]
                weighted_scores[name] = efe / (preference + 1e-8)
            
            best_solver_name = min(weighted_scores.keys(), key=lambda x: weighted_scores[x])
            best_solver_idx = self.efe_solver.solver_names.index(best_solver_name)
            best_solver = self.efe_solver.solvers[best_solver_idx]
            
            # Get solver's intermediate output
            intermediate_output = best_solver.predict(current_state.grid)
            
            # RevThink verification
            verification_results = self.revthink_verifier.verify_solution(
                intermediate_output, current_state
            )
            
            # Z-learning preference update
            self.efe_solver.solver_preferences = self.z_learner.update_preferences(
                self.efe_solver.solver_preferences,
                best_solver_name,
                efe_scores[best_solver_name],
                verification_results
            )
            
            # Update state for next iteration
            current_state.grid = intermediate_output.copy()
            current_state.step = iteration + 1
            current_state.solver_history.append(best_solver_name)
            current_state.confidence = verification_results['combined_score']
            
            solution_history.append({
                'iteration': iteration,
                'solver': best_solver_name,
                'efe_score': efe_scores[best_solver_name],
                'verification': verification_results,
                'solution': intermediate_output.copy()
            })
            
            # Check convergence
            if verification_results['combined_score'] > 0.9:
                break
        
        final_results = {
            'solution': current_state.grid,
            'confidence': current_state.confidence,
            'solver_history': current_state.solver_history,
            'final_preferences': self.efe_solver.solver_preferences,
            'iteration_history': solution_history
        }
        
        return current_state.grid, final_results

# =====================================================================================
# SECTION 12: ENHANCED ARC EFE SYSTEM WITH ENSEMBLE
# =====================================================================================

class EnhancedARCEFESystem:
    """Enhanced ARC system with multi-solver ensemble and contrastive learning"""
    
    def __init__(self, 
                 solvers: List[Any], 
                 planning_horizon: int = 3,
                 consensus_threshold: float = 0.5):
        
        # Core components
        self.efe_solver = EFESolver(solvers, planning_horizon)
        self.revthink_verifier = RevThinkVerifier()
        self.z_learner = ZLearningUpdater()
        
        # New ensemble components
        self.consensus_module = MajorityVotingConsensus(consensus_threshold)
        self.contrastive_module = ContrastiveLearningModule()
        self.loss_calculator = ComprehensiveLossCalculator()
        
        # Training parameters
        self.max_iterations = 10
        self.training_history = []
        
    def solve_with_ensemble(self, 
                           initial_grid: np.ndarray, 
                           constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Enhanced solving loop with multi-solver ensemble approach"""
        
        current_state = ARCState(
            grid=initial_grid.copy(),
            constraints=constraints,
            step=0,
            solver_history=[],
            confidence=0.0
        )
        
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            print(f"\\n--- Iteration {iteration + 1} ---")
            
            # Step 1: Execute all solvers simultaneously
            solver_outputs = {}
            efe_scores = {}
            
            for i, solver_name in enumerate(self.efe_solver.solver_names):
                solver = self.efe_solver.solvers[i]
                
                output = solver.predict(current_state.grid)
                solver_outputs[solver_name] = output
                
                efe_score = self.efe_solver.compute_efe(i, current_state)
                efe_scores[solver_name] = efe_score
            
            # Step 2: Majority voting consensus
            ensemble_result = self.consensus_module.compute_consensus(solver_outputs)
            
            print(f"Consensus reached: {ensemble_result.consensus_reached}")
            print(f"Majority agreements: {ensemble_result.majority_count}/{ensemble_result.total_solvers}")
            
            # Step 3: Identify correct solvers
            correct_solvers = []
            for solver_name, output in solver_outputs.items():
                if np.array_equal(output, ensemble_result.output):
                    correct_solvers.append(solver_name)
            
            # Step 4: Contrastive learning
            contrastive_loss, similarity_scores = self.contrastive_module.compute_contrastive_loss(
                ensemble_result.output,
                solver_outputs,
                correct_solvers
            )
            
            # Step 5: RevThink verification
            verification_results = self.revthink_verifier.verify_solution(
                ensemble_result.output, current_state
            )
            
            # Step 6: Compute comprehensive loss
            avg_efe = np.mean(list(efe_scores.values()))
            diversity_score = self._compute_diversity_score(solver_outputs)
            
            loss_components = self.loss_calculator.compute_total_loss(
                efe_loss=avg_efe,
                contrastive_loss=contrastive_loss,
                ambiguity_score=ensemble_result.ambiguity_score,
                diversity_score=diversity_score,
                solver_outputs=solver_outputs
            )
            
            # Step 7: Update solver preferences
            for solver_name in self.efe_solver.solver_names:
                bonus_reward = 1.0 if solver_name in correct_solvers else 0.0
                enhanced_verification = verification_results.copy()
                enhanced_verification['consensus_bonus'] = bonus_reward
                
                self.efe_solver.solver_preferences = self.z_learner.update_preferences(
                    self.efe_solver.solver_preferences,
                    solver_name,
                    efe_scores[solver_name],
                    enhanced_verification
                )
            
            # Step 8: Update state
            current_state.grid = ensemble_result.output.copy()
            current_state.step = iteration + 1
            current_state.solver_history.extend(correct_solvers)
            current_state.confidence = verification_results['combined_score']
            
            iteration_data = {
                'iteration': iteration,
                'ensemble_result': ensemble_result,
                'efe_scores': efe_scores,
                'verification': verification_results,
                'loss_components': loss_components,
                'similarity_scores': similarity_scores,
                'correct_solvers': correct_solvers,
                'solver_preferences': self.efe_solver.solver_preferences.copy()
            }
            iteration_history.append(iteration_data)
            
            print(f"Total loss: {loss_components['total_loss']:.3f}")
            print(f"Verification score: {verification_results['combined_score']:.3f}")
            
            # Check convergence
            if (verification_results['combined_score'] > 0.9 and 
                ensemble_result.consensus_reached and
                loss_components['total_loss'] < 1.0):
                print("🎯 Convergence reached!")
                break
        
        final_results = {
            'solution': current_state.grid,
            'confidence': current_state.confidence,
            'solver_history': current_state.solver_history,
            'final_preferences': self.efe_solver.solver_preferences,
            'iteration_history': iteration_history,
            'ensemble_metrics': {
                'final_ambiguity': ensemble_result.ambiguity_score,
                'final_consensus': ensemble_result.consensus_reached,
                'total_iterations': len(iteration_history)
            }
        }
        
        return current_state.grid, final_results
    
    def _compute_diversity_score(self, solver_outputs: Dict[str, np.ndarray]) -> float:
        """Compute diversity score among solver outputs"""
        outputs = list(solver_outputs.values())
        if len(outputs) < 2:
            return 0.0
        
        differences = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff_ratio = np.mean(outputs[i] != outputs[j])
                differences.append(diff_ratio)
        
        return np.mean(differences)

# =====================================================================================
# SECTION 13: HELPER FUNCTIONS AND UTILITIES
# =====================================================================================

def create_enhanced_arc_system(solver_classes: List[type], 
                              planning_horizon: int = 3,
                              consensus_threshold: float = 0.5) -> EnhancedARCEFESystem:
    """Factory function to create enhanced ARC system with solver instances"""
    
    # Instantiate solvers (assuming they have default constructors)
    solvers = [solver_class() for solver_class in solver_classes]
    
    return EnhancedARCEFESystem(
        solvers=solvers,
        planning_horizon=planning_horizon,
        consensus_threshold=consensus_threshold
    )

def analyze_arc_problem_comprehensive(input_grid: np.ndarray, 
                                    output_grid: np.ndarray = None) -> Dict[str, Any]:
    """Comprehensive analysis of ARC problem using the analyzer"""
    
    analyzer = ARCProblemAnalyzer()
    return analyzer.analyze_problem(input_grid, output_grid)

def create_sample_arc_problem() -> Dict[str, Any]:
    """Create a sample ARC problem for testing"""
    
    # Simple 5x5 pattern problem
    input_grid = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ])
    
    output_grid = np.array([
        [0, 2, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0]
    ])
    
    constraints = {
        'color_constraints': [1, 2],
        'pattern_constraints': input_grid,
        'symmetry': None
    }
    
    return {
        'input': input_grid,
        'output': output_grid,
        'constraints': constraints,
        'description': 'Color transformation: blue (1) -> red (2)'
    }

# =====================================================================================
# SECTION 14: INTEGRATION AND TESTING
# =====================================================================================

if __name__ == "__main__":
    print("EFE Update - Consolidated ARC Expected Free Energy System")
    print("=" * 60)
    print("This is a consolidated file containing all recent updates.")
    print("Individual modules are now integrated into a single system.")
    print("\nAvailable classes:")
    print("- ARCEFESystem: Basic EFE system")
    print("- EnhancedARCEFESystem: Enhanced ensemble system")
    print("- ARCProblemAnalyzer: Comprehensive problem analysis")
    print("- RobustMajorityVoting: Robust consensus mechanism")
    print("\nFor usage examples, see the individual demo files.")