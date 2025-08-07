import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import copy

@dataclass
class ARCState:
    """State representation for ARC problems"""
    grid: np.ndarray
    constraints: Dict[str, Any]
    step: int
    solver_history: List[str]
    confidence: float

class EFESolver:
    """Expected Free Energy based solver selection for ARC problems"""
    
    def __init__(self, solvers: List[Any], planning_horizon: int = 3):
        self.solvers = solvers
        self.solver_names = [s.__class__.__name__ for s in solvers]
        self.N = planning_horizon
        self.solver_preferences = {}  # Z-learning preferences
        self.eps = 1e-8
        
        # Initialize solver preferences uniformly
        for name in self.solver_names:
            self.solver_preferences[name] = 1.0 / len(self.solver_names)
    
    def compute_efe(self, solver_idx: int, state_t: ARCState) -> float:
        """
        Compute EFE for a solver given current state
        EFE_t(solver) = Risk: D_KL(Q(o_t)||C_t(o_t)) + Ambiguity: -logP(constraint_t|o_t)
        """
        solver = self.solvers[solver_idx]
        solver_name = self.solver_names[solver_idx]
        
        # Get solver's predicted output distribution Q(o_t)
        predicted_output = solver.predict(state_t.grid)
        Q_o_t = self.get_output_distribution(predicted_output)
        
        # Get constraint-based target distribution C_t(o_t)
        C_t = self.get_constraint_distribution(state_t)
        
        # Risk term: KL divergence between prediction and constraint
        risk = self.kl_divergence(Q_o_t, C_t)
        
        # Ambiguity term: negative log probability of constraints given output
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
        # Flatten and normalize output to create distribution
        flat_output = output.flatten()
        # Apply softmax to create valid probability distribution
        exp_out = np.exp(flat_output - np.max(flat_output))
        return exp_out / np.sum(exp_out)
    
    def get_constraint_distribution(self, state: ARCState) -> np.ndarray:
        """Generate constraint-based target distribution C_t(o_t)"""
        # Extract constraints from ARC problem structure
        constraints = state.constraints
        grid_size = state.grid.size
        
        # Initialize uniform distribution
        C_t = np.ones(grid_size) / grid_size
        
        # Modify distribution based on constraints
        if 'color_constraints' in constraints:
            color_weights = constraints['color_constraints']
            for i, weight in enumerate(color_weights):
                C_t[i] *= weight
        
        if 'pattern_constraints' in constraints:
            pattern_mask = constraints['pattern_constraints']
            C_t *= pattern_mask.flatten()
        
        # Renormalize
        C_t = C_t / (np.sum(C_t) + self.eps)
        return C_t
    
    def evaluate_constraints(self, output: np.ndarray, constraints: Dict) -> float:
        """Evaluate how well output satisfies constraints"""
        satisfaction = 1.0
        
        # Check color constraints
        if 'color_constraints' in constraints:
            unique_colors = np.unique(output)
            expected_colors = constraints['color_constraints']
            color_match = len(set(unique_colors) & set(expected_colors)) / len(expected_colors)
            satisfaction *= color_match
        
        # Check pattern constraints
        if 'pattern_constraints' in constraints:
            pattern_score = self.check_pattern_match(output, constraints['pattern_constraints'])
            satisfaction *= pattern_score
        
        # Check symmetry constraints
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

class RevThinkVerifier:
    """Reverse thinking verification system"""
    
    def __init__(self):
        self.verification_threshold = 0.7
    
    def verify_solution(self, solution: np.ndarray, original_state: ARCState) -> Dict[str, float]:
        """
        Apply RevThink verification:
        1. Forward reasoning: does solution follow from input?
        2. Backward reasoning: does input lead to this solution?
        3. Process examination: is the reasoning path valid?
        """
        results = {}
        
        # Forward reasoning verification
        results['forward_score'] = self.forward_verification(solution, original_state)
        
        # Backward reasoning verification  
        results['backward_score'] = self.backward_verification(solution, original_state)
        
        # Process examination
        results['process_score'] = self.process_verification(solution, original_state)
        
        # Combined verification score
        results['combined_score'] = (
            results['forward_score'] * 0.4 + 
            results['backward_score'] * 0.4 + 
            results['process_score'] * 0.2
        )
        
        return results
    
    def forward_verification(self, solution: np.ndarray, state: ARCState) -> float:
        """Verify if solution logically follows from input"""
        # Check if transformation rules are consistently applied
        input_patterns = self.extract_patterns(state.grid)
        output_patterns = self.extract_patterns(solution)
        
        consistency_score = self.measure_transformation_consistency(input_patterns, output_patterns)
        return consistency_score
    
    def backward_verification(self, solution: np.ndarray, state: ARCState) -> float:
        """Verify if input could reasonably lead to this solution"""
        # Generate possible inputs that could lead to this output
        possible_inputs = self.generate_reverse_inputs(solution)
        
        # Compare with actual input
        similarity_scores = [self.grid_similarity(state.grid, inp) for inp in possible_inputs]
        return max(similarity_scores) if similarity_scores else 0.0
    
    def process_verification(self, solution: np.ndarray, state: ARCState) -> float:
        """Verify the reasoning process used"""
        # Check if the solver sequence makes logical sense
        solver_sequence = state.solver_history
        sequence_validity = self.validate_solver_sequence(solver_sequence)
        
        return sequence_validity
    
    def extract_patterns(self, grid: np.ndarray) -> List[Dict]:
        """Extract visual patterns from grid"""
        patterns = []
        
        # Color distribution pattern
        unique, counts = np.unique(grid, return_counts=True)
        patterns.append({'type': 'color_dist', 'data': dict(zip(unique, counts))})
        
        # Shape patterns (simplified)
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
                    if len(region) > 1:  # Only include multi-cell regions
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
            
            # Add neighbors
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                stack.append((i + di, j + dj))
        
        return region
    
    def measure_transformation_consistency(self, input_patterns: List[Dict], output_patterns: List[Dict]) -> float:
        """Measure consistency of transformation between input and output patterns"""
        if not input_patterns or not output_patterns:
            return 0.5
        
        # Simple consistency measure based on pattern preservation
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
            # Color distribution similarity
            common_keys = set(pattern1.keys()) & set(pattern2.keys())
            if not common_keys:
                return 0.0
            
            similarity = 0.0
            for key in common_keys:
                similarity += min(pattern1[key], pattern2[key]) / max(pattern1[key], pattern2[key])
            
            return similarity / len(common_keys)
        
        return 0.5  # Default similarity for unknown pattern types
    
    def generate_reverse_inputs(self, solution: np.ndarray) -> List[np.ndarray]:
        """Generate possible inputs that could lead to the solution"""
        # Simplified reverse generation - in practice would be more sophisticated
        possible_inputs = []
        
        # Generate variations by applying inverse transformations
        possible_inputs.append(solution)  # Identity
        possible_inputs.append(np.fliplr(solution))  # Horizontal flip
        possible_inputs.append(np.flipud(solution))  # Vertical flip
        possible_inputs.append(np.rot90(solution, -1))  # Counter-clockwise rotation
        
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
        
        # Check for reasonable solver progression
        # Simple heuristic: avoid excessive switching between solvers
        switches = 0
        for i in range(1, len(solver_sequence)):
            if solver_sequence[i] != solver_sequence[i-1]:
                switches += 1
        
        # Penalize excessive switching
        switch_ratio = switches / len(solver_sequence)
        return max(0.1, 1.0 - switch_ratio)

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
        """
        Update solver preferences using Z-learning approach
        Based on EFE performance and RevThink verification
        """
        # Compute reward signal from EFE (lower EFE is better)
        efe_reward = 1.0 / (1.0 + efe_score)
        
        # Incorporate verification score
        verification_reward = verification_results.get('combined_score', 0.5)
        
        # Combined reward
        total_reward = 0.7 * efe_reward + 0.3 * verification_reward
        
        # Update preference for the used solver
        current_pref = solver_preferences[solver_name]
        
        # Z-learning update rule (simplified)
        prediction_error = total_reward - current_pref
        updated_pref = current_pref + self.alpha * prediction_error
        
        # Update preferences dictionary
        new_preferences = solver_preferences.copy()
        new_preferences[solver_name] = updated_pref
        
        # Softmax normalization across all solvers
        pref_values = np.array(list(new_preferences.values()))
        softmax_prefs = np.exp(pref_values / self.temperature)
        softmax_prefs = softmax_prefs / np.sum(softmax_prefs)
        
        # Update dictionary with normalized preferences
        for i, name in enumerate(new_preferences.keys()):
            new_preferences[name] = softmax_prefs[i]
        
        return new_preferences

class ARCEFESystem:
    """Main system integrating EFE-based solver selection with RevThink verification"""
    
    def __init__(self, solvers: List[Any], planning_horizon: int = 3):
        self.efe_solver = EFESolver(solvers, planning_horizon)
        self.revthink_verifier = RevThinkVerifier()
        self.z_learner = ZLearningUpdater()
        self.max_iterations = 10
    
    def solve_arc_problem(self, initial_grid: np.ndarray, constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """
        Main solving loop following the user's specified flow:
        문제 입력 → 현재 상태 → [multiple solvers] → EFE 평가 → 최적 solver 결정 → 
        중간 output → (RevThink 검증, Z-learning 업데이트) → state 업데이트 → 반복
        """
        # Initialize state
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
            
            # Select solver with minimum EFE (incorporating preferences)
            weighted_scores = {}
            for name, efe in efe_scores.items():
                preference = self.efe_solver.solver_preferences[name]
                # Lower EFE is better, higher preference is better
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