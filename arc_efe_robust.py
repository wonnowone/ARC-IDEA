import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import copy
import warnings
from collections import Counter
from arc_efe_solver import ARCState, EFESolver, RevThinkVerifier, ZLearningUpdater
from example_solvers import BaseSolver

class RobustMajorityVoting:
    """Robust majority voting with fallback mechanisms"""
    
    def __init__(self, min_consensus_threshold: float = 0.5):
        self.min_consensus_threshold = max(0.1, min(0.9, min_consensus_threshold))  # Bounds check
        self.eps = 1e-12
        self.max_retries = 3
        
    def compute_consensus_safe(self, solver_outputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Safe consensus computation with comprehensive error handling"""
        
        # Input validation
        if not solver_outputs:
            return self._create_fallback_consensus()
            
        # Shape validation
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
        
        # Validate all outputs are finite
        for name, output in solver_outputs.items():
            if not np.all(np.isfinite(output)):
                warnings.warn(f"Solver {name} produced non-finite values, excluding...")
                solver_outputs = {k: v for k, v in solver_outputs.items() if k != name}
        
        if not solver_outputs:
            return self._create_fallback_consensus()
        
        # Robust voting process
        ambiguity_scores = []
        consensus_positions = 0
        total_positions = grid_shape[0] * grid_shape[1]
        
        for pos_i in range(grid_shape[0]):
            for pos_j in range(grid_shape[1]):
                votes = [solver_outputs[name][pos_i, pos_j] for name in solver_outputs.keys()]
                
                # Handle edge case: all votes are the same
                if len(set(votes)) <= 1:
                    consensus_grid[pos_i, pos_j] = votes[0] if votes else 0
                    consensus_positions += 1
                    ambiguity_scores.append(0.0)
                    continue
                
                # Majority vote with tie-breaking
                vote_counts = Counter(votes)
                majority_value, majority_count = vote_counts.most_common(1)[0]
                
                # Check for ties
                top_votes = vote_counts.most_common(2)
                if len(top_votes) > 1 and top_votes[0][1] == top_votes[1][1]:
                    # Tie-breaking: use solver with highest current preference
                    tied_values = [val for val, count in top_votes if count == top_votes[0][1]]
                    majority_value = min(tied_values)  # Deterministic tie-breaking
                
                consensus_grid[pos_i, pos_j] = majority_value
                
                # Safe consensus ratio calculation
                consensus_ratio = majority_count / max(1, len(votes))
                if consensus_ratio >= self.min_consensus_threshold:
                    consensus_positions += 1
                
                # Safe entropy calculation
                vote_probs = np.array(list(vote_counts.values()), dtype=np.float64)
                vote_probs = vote_probs / np.sum(vote_probs)
                vote_probs = np.maximum(vote_probs, self.eps)  # Ensure no zeros
                
                entropy = -np.sum(vote_probs * np.log(vote_probs))
                ambiguity_scores.append(min(entropy, 10.0))  # Cap extreme values
        
        # Calculate final metrics with bounds
        consensus_reached = consensus_positions >= max(1, int(total_positions * self.min_consensus_threshold))
        confidence = consensus_positions / max(1, total_positions)
        overall_ambiguity = np.clip(np.mean(ambiguity_scores), 0.0, 10.0)
        
        # Count majority agreements safely
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
        target_shape = max(shapes, key=lambda x: x[0] * x[1])  # Use largest grid
        
        normalized = {}
        for name, output in solver_outputs.items():
            if output.shape != target_shape:
                # Pad or crop to target shape
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
        fallback_grid = np.zeros((5, 5), dtype=int)  # Default 5x5 grid
        
        return {
            'output': fallback_grid,
            'solver_outputs': {'fallback': fallback_grid},
            'consensus_reached': False,
            'majority_count': 0,
            'total_solvers': 1,
            'ambiguity_score': 5.0,  # High ambiguity for fallback
            'confidence': 0.1,
            'valid': False
        }

class RobustLossCalculator:
    """Robust loss calculation with numerical stability"""
    
    def __init__(self, 
                 lambda_contrast: float = 1.0,
                 lambda_ambiguity: float = 0.5,
                 lambda_chaos: float = 0.3):
        self.lambda_contrast = np.clip(lambda_contrast, 0.0, 10.0)
        self.lambda_ambiguity = np.clip(lambda_ambiguity, 0.0, 10.0)
        self.lambda_chaos = np.clip(lambda_chaos, 0.0, 10.0)
        self.eps = 1e-12
        
        # Adaptive lambda scheduling
        self.lambda_decay = 0.99
        self.min_lambda = 0.01
    
    def compute_total_loss_safe(self,
                               efe_loss: float,
                               contrastive_loss: torch.Tensor,
                               ambiguity_score: float,
                               diversity_score: float,
                               solver_outputs: Dict[str, np.ndarray],
                               iteration: int = 0) -> Dict[str, float]:
        """Safe total loss computation with bounds checking"""
        
        try:
            # Validate inputs
            efe_loss = self._validate_loss_component(efe_loss, "EFE")
            contrastive_value = self._validate_tensor_loss(contrastive_loss, "Contrastive")
            ambiguity_score = np.clip(ambiguity_score, 0.0, 10.0)
            diversity_score = np.clip(diversity_score, 0.0, 1.0)
            
            # Adaptive lambda scheduling
            adaptive_lambdas = self._compute_adaptive_lambdas(iteration)
            
            # Safe loss component calculations
            ambiguity_penalty = self._compute_safe_ambiguity_penalty(ambiguity_score)
            chaos_loss = self._compute_safe_chaos_loss(diversity_score, solver_outputs)
            
            # Total loss with overflow protection
            total_loss = (efe_loss + 
                         adaptive_lambdas['contrast'] * contrastive_value + 
                         adaptive_lambdas['ambiguity'] * ambiguity_penalty + 
                         adaptive_lambdas['chaos'] * chaos_loss)
            
            # Sanity check
            if not np.isfinite(total_loss) or total_loss > 1000.0:
                warnings.warn(f"Loss explosion detected: {total_loss}, using fallback")
                total_loss = 10.0  # Fallback value
            
            return {
                'total_loss': float(total_loss),
                'efe_loss': float(efe_loss),
                'contrastive_loss': float(contrastive_value),
                'ambiguity_penalty': float(ambiguity_penalty),
                'chaos_loss': float(chaos_loss),
                'diversity_score': float(diversity_score),
                'adaptive_lambdas': adaptive_lambdas,
                'valid': True
            }
            
        except Exception as e:
            warnings.warn(f"Loss calculation failed: {e}")
            return self._create_fallback_loss()
    
    def _validate_loss_component(self, value: float, name: str) -> float:
        """Validate and bound loss component"""
        if not np.isfinite(value):
            warnings.warn(f"{name} loss is not finite: {value}")
            return 1.0
        return np.clip(value, 0.0, 100.0)
    
    def _validate_tensor_loss(self, tensor_loss: torch.Tensor, name: str) -> float:
        """Safely convert tensor loss to float"""
        try:
            if torch.is_tensor(tensor_loss):
                value = tensor_loss.item()
            else:
                value = float(tensor_loss)
                
            if not np.isfinite(value):
                warnings.warn(f"{name} tensor loss is not finite")
                return 1.0
            return np.clip(value, 0.0, 100.0)
        except:
            warnings.warn(f"Failed to extract {name} loss value")
            return 1.0
    
    def _compute_adaptive_lambdas(self, iteration: int) -> Dict[str, float]:
        """Compute adaptive lambda values"""
        decay_factor = self.lambda_decay ** iteration
        
        return {
            'contrast': max(self.min_lambda, self.lambda_contrast * decay_factor),
            'ambiguity': max(self.min_lambda, self.lambda_ambiguity * decay_factor),
            'chaos': max(self.min_lambda, self.lambda_chaos * decay_factor)
        }
    
    def _compute_safe_ambiguity_penalty(self, ambiguity_score: float) -> float:
        """Safe ambiguity penalty calculation"""
        return np.clip(ambiguity_score, 0.0, 5.0)
    
    def _compute_safe_chaos_loss(self, diversity_score: float, solver_outputs: Dict[str, np.ndarray]) -> float:
        """Safe chaos loss with sampling for large ensembles"""
        try:
            outputs = list(solver_outputs.values())
            if len(outputs) < 2:
                return 0.0
            
            # Use sampling for large ensembles to avoid O(n²) complexity
            if len(outputs) > 10:
                sample_size = min(10, len(outputs))
                sampled_outputs = np.random.choice(len(outputs), sample_size, replace=False)
                outputs = [outputs[i] for i in sampled_outputs]
            
            # Safe pairwise diversity computation
            pairwise_differences = []
            for i in range(min(5, len(outputs))):  # Limit to prevent explosion
                for j in range(i + 1, min(5, len(outputs))):
                    try:
                        diff = np.mean(outputs[i] != outputs[j])
                        if np.isfinite(diff):
                            pairwise_differences.append(diff)
                    except:
                        continue
            
            if not pairwise_differences:
                return 0.0
            
            avg_diversity = np.mean(pairwise_differences)
            optimal_diversity = 0.3
            chaos_loss = abs(avg_diversity - optimal_diversity)
            
            return np.clip(chaos_loss, 0.0, 1.0)
            
        except Exception as e:
            warnings.warn(f"Chaos loss computation failed: {e}")
            return 0.0
    
    def _create_fallback_loss(self) -> Dict[str, float]:
        """Fallback loss when computation fails"""
        return {
            'total_loss': 5.0,
            'efe_loss': 2.0,
            'contrastive_loss': 1.0,
            'ambiguity_penalty': 1.0,
            'chaos_loss': 1.0,
            'diversity_score': 0.5,
            'adaptive_lambdas': {'contrast': 0.5, 'ambiguity': 0.5, 'chaos': 0.5},
            'valid': False
        }

class RobustPreferenceUpdater:
    """Robust Z-learning with preference collapse prevention"""
    
    def __init__(self, learning_rate: float = 0.1, temperature: float = 1.0):
        self.alpha = np.clip(learning_rate, 0.001, 0.5)  # Prevent aggressive updates
        self.temperature = max(0.1, temperature)  # Prevent division by zero
        self.min_preference = 0.01  # Prevent complete collapse
        self.max_preference = 0.99  # Prevent dominance
        
    def update_preferences_safe(self, 
                               solver_preferences: Dict[str, float],
                               solver_name: str,
                               efe_score: float,
                               verification_results: Dict[str, float],
                               iteration: int = 0) -> Dict[str, float]:
        """Safe preference update with collapse prevention"""
        
        try:
            # Validate inputs
            if not solver_preferences or solver_name not in solver_preferences:
                return self._initialize_uniform_preferences(list(solver_preferences.keys()))
            
            # Safe reward calculation
            efe_reward = self._compute_safe_efe_reward(efe_score)
            verification_reward = np.clip(verification_results.get('combined_score', 0.5), 0.0, 1.0)
            consensus_bonus = np.clip(verification_results.get('consensus_bonus', 0.0), 0.0, 1.0)
            
            # Combined reward with bounds
            total_reward = np.clip(
                0.5 * efe_reward + 0.3 * verification_reward + 0.2 * consensus_bonus,
                0.0, 1.0
            )
            
            # Adaptive learning rate
            adaptive_alpha = self.alpha * (0.99 ** iteration)  # Decay learning rate
            
            # Safe preference update
            current_pref = solver_preferences[solver_name]
            prediction_error = total_reward - current_pref
            updated_pref = current_pref + adaptive_alpha * prediction_error
            
            # Update preferences
            new_preferences = solver_preferences.copy()
            new_preferences[solver_name] = updated_pref
            
            # Prevent preference collapse
            new_preferences = self._prevent_preference_collapse(new_preferences)
            
            # Safe normalization
            new_preferences = self._safe_softmax_normalization(new_preferences)
            
            return new_preferences
            
        except Exception as e:
            warnings.warn(f"Preference update failed: {e}")
            return self._initialize_uniform_preferences(list(solver_preferences.keys()))
    
    def _compute_safe_efe_reward(self, efe_score: float) -> float:
        """Safe EFE reward computation"""
        if not np.isfinite(efe_score):
            return 0.1
        
        # Lower EFE is better, so invert and bound
        safe_efe = np.clip(efe_score, 0.1, 100.0)
        return np.clip(1.0 / (1.0 + safe_efe), 0.0, 1.0)
    
    def _prevent_preference_collapse(self, preferences: Dict[str, float]) -> Dict[str, float]:
        """Prevent preference collapse by enforcing minimum values"""
        total_solvers = len(preferences)
        min_pref_per_solver = self.min_preference
        max_pref_per_solver = 1.0 - (total_solvers - 1) * min_pref_per_solver
        
        # Enforce bounds
        bounded_prefs = {}
        for name, pref in preferences.items():
            bounded_prefs[name] = np.clip(pref, min_pref_per_solver, max_pref_per_solver)
        
        return bounded_prefs
    
    def _safe_softmax_normalization(self, preferences: Dict[str, float]) -> Dict[str, float]:
        """Safe softmax normalization with numerical stability"""
        try:
            pref_values = np.array(list(preferences.values()), dtype=np.float64)
            
            # Numerical stability: subtract max
            pref_values = pref_values - np.max(pref_values)
            
            # Safe softmax
            exp_prefs = np.exp(pref_values / self.temperature)
            softmax_prefs = exp_prefs / (np.sum(exp_prefs) + 1e-12)
            
            # Ensure no NaN or inf
            if not np.all(np.isfinite(softmax_prefs)):
                return self._initialize_uniform_preferences(list(preferences.keys()))
            
            # Enforce minimum preferences
            softmax_prefs = np.maximum(softmax_prefs, self.min_preference)
            softmax_prefs = softmax_prefs / np.sum(softmax_prefs)  # Renormalize
            
            # Create result dictionary
            result = {}
            for i, name in enumerate(preferences.keys()):
                result[name] = float(softmax_prefs[i])
            
            return result
            
        except Exception as e:
            warnings.warn(f"Softmax normalization failed: {e}")
            return self._initialize_uniform_preferences(list(preferences.keys()))
    
    def _initialize_uniform_preferences(self, solver_names: List[str]) -> Dict[str, float]:
        """Initialize uniform preferences as fallback"""
        if not solver_names:
            return {}
        
        uniform_pref = 1.0 / len(solver_names)
        return {name: uniform_pref for name in solver_names}

class RobustARCEFESystem:
    """Ultra-robust ARC system with comprehensive error handling"""
    
    def __init__(self, 
                 solvers: List[BaseSolver], 
                 planning_horizon: int = 3,
                 consensus_threshold: float = 0.5,
                 max_iterations: int = 10):
        
        # Validate inputs
        if not solvers:
            raise ValueError("At least one solver must be provided")
        
        self.solvers = solvers
        self.max_iterations = max(1, min(100, max_iterations))  # Reasonable bounds
        
        # Initialize robust components
        self.consensus_module = RobustMajorityVoting(consensus_threshold)
        self.loss_calculator = RobustLossCalculator()
        self.preference_updater = RobustPreferenceUpdater()
        
        # Initialize solver preferences safely
        self.solver_preferences = self._initialize_solver_preferences()
        
        # Convergence tracking
        self.convergence_history = []
        self.early_stopping_patience = 3
        
    def _initialize_solver_preferences(self) -> Dict[str, float]:
        """Initialize solver preferences uniformly"""
        solver_names = [s.__class__.__name__ for s in self.solvers]
        uniform_pref = 1.0 / len(solver_names)
        return {name: uniform_pref for name in solver_names}
    
    def solve_with_robust_ensemble(self, 
                                  initial_grid: np.ndarray, 
                                  constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Ultra-robust ensemble solving with comprehensive error handling"""
        
        # Input validation
        if initial_grid is None or initial_grid.size == 0:
            return np.zeros((5, 5)), {'error': 'Invalid input grid'}
        
        # Ensure finite values
        if not np.all(np.isfinite(initial_grid)):
            warnings.warn("Input grid contains non-finite values")
            initial_grid = np.nan_to_num(initial_grid, nan=0.0, posinf=9.0, neginf=0.0)
        
        current_grid = initial_grid.copy()
        iteration_history = []
        stagnation_counter = 0
        
        for iteration in range(self.max_iterations):
            print(f"\\n🛡️  ROBUST Iteration {iteration + 1}")
            
            try:
                # Safe solver execution
                solver_outputs = self._execute_solvers_safely(current_grid)
                
                if not solver_outputs:
                    print("❌ All solvers failed, using fallback")
                    break
                
                # Robust consensus
                consensus_result = self.consensus_module.compute_consensus_safe(solver_outputs)
                
                if not consensus_result['valid']:
                    print("⚠️  Consensus computation failed, using fallback")
                    stagnation_counter += 1
                else:
                    print(f"✅ Consensus: {consensus_result['consensus_reached']}, "
                          f"Agreement: {consensus_result['majority_count']}/{consensus_result['total_solvers']}")
                
                # Safe loss computation
                loss_components = self.loss_calculator.compute_total_loss_safe(
                    efe_loss=1.0,  # Placeholder
                    contrastive_loss=torch.tensor(0.5),  # Placeholder
                    ambiguity_score=consensus_result['ambiguity_score'],
                    diversity_score=self._compute_diversity_safe(solver_outputs),
                    solver_outputs=solver_outputs,
                    iteration=iteration
                )
                
                # Safe preference updates
                for solver_name in solver_outputs.keys():
                    verification_results = {'combined_score': consensus_result['confidence']}
                    self.solver_preferences = self.preference_updater.update_preferences_safe(
                        self.solver_preferences,
                        solver_name,
                        loss_components['efe_loss'],
                        verification_results,
                        iteration
                    )
                
                # Store iteration data
                iteration_data = {
                    'iteration': iteration,
                    'consensus_result': consensus_result,
                    'loss_components': loss_components,
                    'solver_preferences': self.solver_preferences.copy(),
                    'valid': consensus_result['valid'] and loss_components['valid']
                }
                iteration_history.append(iteration_data)
                
                # Update current grid
                if consensus_result['valid']:
                    new_grid = consensus_result['output']
                    if not np.array_equal(current_grid, new_grid):
                        current_grid = new_grid
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                
                # Convergence checks
                if (consensus_result['consensus_reached'] and 
                    consensus_result['confidence'] > 0.8 and
                    loss_components['total_loss'] < 2.0):
                    print("🎯 Robust convergence achieved!")
                    break
                
                # Early stopping
                if stagnation_counter >= self.early_stopping_patience:
                    print("⏰ Early stopping due to stagnation")
                    break
                    
                print(f"📊 Loss: {loss_components['total_loss']:.3f}, "
                      f"Confidence: {consensus_result['confidence']:.3f}")
                
            except Exception as e:
                warnings.warn(f"Iteration {iteration} failed: {e}")
                stagnation_counter += 1
                if stagnation_counter >= self.early_stopping_patience:
                    break
        
        # Prepare results
        final_results = {
            'solution': current_grid,
            'confidence': consensus_result.get('confidence', 0.1) if 'consensus_result' in locals() else 0.1,
            'solver_preferences': self.solver_preferences,
            'iteration_history': iteration_history,
            'total_iterations': len(iteration_history),
            'converged': stagnation_counter < self.early_stopping_patience,
            'robust_execution': True
        }
        
        return current_grid, final_results
    
    def _execute_solvers_safely(self, grid: np.ndarray) -> Dict[str, np.ndarray]:
        """Execute all solvers with individual error handling"""
        solver_outputs = {}
        
        for solver in self.solvers:
            try:
                output = solver.predict(grid)
                
                # Validate output
                if output is None or output.size == 0:
                    continue
                
                if not np.all(np.isfinite(output)):
                    warnings.warn(f"Solver {solver.__class__.__name__} produced non-finite output")
                    output = np.nan_to_num(output, nan=0.0)
                
                # Ensure integer values in valid range
                output = np.clip(np.round(output).astype(int), 0, 9)
                
                solver_outputs[solver.__class__.__name__] = output
                
            except Exception as e:
                warnings.warn(f"Solver {solver.__class__.__name__} failed: {e}")
                continue
        
        return solver_outputs
    
    def _compute_diversity_safe(self, solver_outputs: Dict[str, np.ndarray]) -> float:
        """Safe diversity computation"""
        try:
            outputs = list(solver_outputs.values())
            if len(outputs) < 2:
                return 0.0
            
            # Sample for efficiency
            if len(outputs) > 5:
                sample_indices = np.random.choice(len(outputs), 5, replace=False)
                outputs = [outputs[i] for i in sample_indices]
            
            differences = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    diff = np.mean(outputs[i] != outputs[j])
                    if np.isfinite(diff):
                        differences.append(diff)
            
            return np.mean(differences) if differences else 0.0
            
        except:
            return 0.0