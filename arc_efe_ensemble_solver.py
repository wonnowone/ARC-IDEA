import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
import copy
from collections import Counter
from arc_efe_solver import ARCState, EFESolver, RevThinkVerifier, ZLearningUpdater
from example_solvers import BaseSolver

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

class MajorityVotingConsensus:
    """Implements majority voting consensus mechanism for solver ensemble"""
    
    def __init__(self, min_consensus_threshold: float = 0.5):
        self.min_consensus_threshold = min_consensus_threshold
        self.eps = 1e-8
    
    def compute_consensus(self, solver_outputs: Dict[str, np.ndarray]) -> EnsembleOutput:
        """
        Compute majority voting consensus from multiple solver outputs
        
        Args:
            solver_outputs: Dict mapping solver names to their output grids
            
        Returns:
            EnsembleOutput with consensus result and metadata
        """
        if not solver_outputs:
            raise ValueError("No solver outputs provided")
        
        # Get dimensions from first output
        first_output = next(iter(solver_outputs.values()))
        grid_shape = first_output.shape
        
        # Initialize consensus grid and voting matrix
        consensus_grid = np.zeros(grid_shape, dtype=int)
        vote_matrix = np.zeros(grid_shape + (len(solver_outputs),), dtype=int)
        
        solver_names = list(solver_outputs.keys())
        
        # Collect votes for each position
        for pos_i in range(grid_shape[0]):
            for pos_j in range(grid_shape[1]):
                votes = []
                for solver_idx, solver_name in enumerate(solver_names):
                    value = solver_outputs[solver_name][pos_i, pos_j]
                    votes.append(value)
                    vote_matrix[pos_i, pos_j, solver_idx] = value
                
                # Find majority vote
                vote_counts = Counter(votes)
                majority_value, majority_count = vote_counts.most_common(1)[0]
                consensus_grid[pos_i, pos_j] = majority_value
        
        # Calculate consensus metrics
        total_positions = grid_shape[0] * grid_shape[1]
        consensus_positions = 0
        ambiguity_scores = []
        
        for pos_i in range(grid_shape[0]):
            for pos_j in range(grid_shape[1]):
                position_votes = vote_matrix[pos_i, pos_j, :]
                vote_counts = Counter(position_votes)
                
                if len(vote_counts) == 1:
                    # Perfect consensus
                    consensus_positions += 1
                    ambiguity_scores.append(0.0)
                else:
                    # Calculate position ambiguity
                    most_common_count = vote_counts.most_common(1)[0][1]
                    consensus_ratio = most_common_count / len(solver_names)
                    
                    if consensus_ratio >= self.min_consensus_threshold:
                        consensus_positions += 1
                    
                    # Ambiguity = entropy of vote distribution
                    vote_probs = np.array(list(vote_counts.values())) / len(solver_names)
                    entropy = -np.sum(vote_probs * np.log(vote_probs + self.eps))
                    ambiguity_scores.append(entropy)
        
        # Overall metrics
        consensus_reached = consensus_positions >= (total_positions * self.min_consensus_threshold)
        overall_ambiguity = np.mean(ambiguity_scores)
        confidence = consensus_positions / total_positions
        
        # Count majority agreements
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

class ContrastiveLearningModule:
    """Contrastive learning for distinguishing correct vs incorrect solver outputs"""
    
    def __init__(self, embedding_dim: int = 128, temperature: float = 0.1):
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.eps = 1e-8
        
        # Initialize learnable embedding network
        self.embedding_net = self._create_embedding_network()
        
    def _create_embedding_network(self):
        """Create embedding network for grid representations"""
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(25, 64),  # Assuming 5x5 grids, adjust as needed
            nn.ReLU(),
            nn.Linear(64, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim)
        )
    
    def compute_contrastive_loss(self, 
                                consensus_output: np.ndarray,
                                solver_outputs: Dict[str, np.ndarray],
                                correct_solvers: List[str]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss between consensus (positive) and failed outputs (negative)
        
        Args:
            consensus_output: The majority voting result (positive example)
            solver_outputs: All solver outputs
            correct_solvers: List of solvers that agreed with consensus
            
        Returns:
            Contrastive loss and similarity scores
        """
        # Convert to tensors
        consensus_tensor = torch.FloatTensor(consensus_output).unsqueeze(0)
        consensus_embedding = self.embedding_net(consensus_tensor)
        
        positive_embeddings = []
        negative_embeddings = []
        similarity_scores = {}
        
        for solver_name, output in solver_outputs.items():
            output_tensor = torch.FloatTensor(output).unsqueeze(0)
            output_embedding = self.embedding_net(output_tensor)
            
            # Compute similarity to consensus
            similarity = F.cosine_similarity(consensus_embedding, output_embedding).item()
            similarity_scores[solver_name] = similarity
            
            if solver_name in correct_solvers:
                positive_embeddings.append(output_embedding)
            else:
                negative_embeddings.append(output_embedding)
        
        # Compute contrastive loss if we have both positive and negative examples
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
        # Stack embeddings
        all_embeddings = torch.cat(positives + negatives, dim=0)
        
        # Compute similarities
        similarities = F.cosine_similarity(anchor, all_embeddings) / self.temperature
        
        # Create labels (positives are labeled as 1, negatives as 0)
        num_positives = len(positives)
        labels = torch.cat([
            torch.ones(num_positives),
            torch.zeros(len(negatives))
        ])
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(similarities.unsqueeze(0), labels.long().unsqueeze(0))
        return loss

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
        """
        Compute total loss: L_total = L_EFE + λ_contrast*L_contrastive + λ_ambiguity*ambiguity_penalty + λ_chaos*L_chaos
        
        Args:
            efe_loss: EFE loss (consistency term)
            contrastive_loss: Contrastive learning loss (clarity term)
            ambiguity_score: Ambiguity between solvers (agreement penalty)
            diversity_score: Output diversity score (flexibility term)
            solver_outputs: All solver outputs for analysis
            
        Returns:
            Dictionary with all loss components
        """
        # Convert contrastive loss to float if it's a tensor
        contrastive_value = contrastive_loss.item() if torch.is_tensor(contrastive_loss) else contrastive_loss
        
        # Ambiguity penalty (discourage disagreement)
        ambiguity_penalty = ambiguity_score
        
        # Chaos loss (encourage diversity but controlled)
        chaos_loss = self._compute_chaos_loss(diversity_score, solver_outputs)
        
        # Total loss computation
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
        """
        Compute chaos loss to maintain controlled diversity
        
        Chaos loss encourages exploration and creative paths while preventing complete randomness
        """
        # Measure pairwise diversity between solver outputs
        outputs = list(solver_outputs.values())
        if len(outputs) < 2:
            return 0.0
        
        pairwise_differences = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                diff = np.mean(outputs[i] != outputs[j])
                pairwise_differences.append(diff)
        
        avg_diversity = np.mean(pairwise_differences)
        
        # Encourage moderate diversity (not too similar, not too different)
        optimal_diversity = 0.3  # Target diversity level
        chaos_loss = abs(avg_diversity - optimal_diversity)
        
        return chaos_loss

class EnhancedARCEFESystem:
    """Enhanced ARC system with multi-solver ensemble and contrastive learning"""
    
    def __init__(self, 
                 solvers: List[BaseSolver], 
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
        """
        Enhanced solving loop with multi-solver ensemble approach
        
        Flow: 문제입력 → 현재상태 → [모든 solvers 동시실행] → majority voting → consensus 도출 
        → contrastive learning → loss 계산 → preference 업데이트 → 다음 iteration
        """
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
                
                # Get solver output
                output = solver.predict(current_state.grid)
                solver_outputs[solver_name] = output
                
                # Compute EFE for this solver
                efe_score = self.efe_solver.compute_efe(i, current_state)
                efe_scores[solver_name] = efe_score
            
            # Step 2: Majority voting consensus
            ensemble_result = self.consensus_module.compute_consensus(solver_outputs)
            
            print(f"Consensus reached: {ensemble_result.consensus_reached}")
            print(f"Majority agreements: {ensemble_result.majority_count}/{ensemble_result.total_solvers}")
            print(f"Ambiguity score: {ensemble_result.ambiguity_score:.3f}")
            
            # Step 3: Identify correct solvers (those agreeing with consensus)
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
            
            # Step 5: RevThink verification on consensus output
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
            
            # Step 7: Update solver preferences using Z-learning
            for solver_name in self.efe_solver.solver_names:
                # Reward solvers that agreed with consensus
                bonus_reward = 1.0 if solver_name in correct_solvers else 0.0
                enhanced_verification = verification_results.copy()
                enhanced_verification['consensus_bonus'] = bonus_reward
                
                self.efe_solver.solver_preferences = self.z_learner.update_preferences(
                    self.efe_solver.solver_preferences,
                    solver_name,
                    efe_scores[solver_name],
                    enhanced_verification
                )
            
            # Step 8: Update state for next iteration
            current_state.grid = ensemble_result.output.copy()
            current_state.step = iteration + 1
            current_state.solver_history.extend(correct_solvers)
            current_state.confidence = verification_results['combined_score']
            
            # Store iteration results
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
            
            # Print iteration summary
            print(f"Total loss: {loss_components['total_loss']:.3f}")
            print(f"  - EFE: {loss_components['efe_loss']:.3f}")
            print(f"  - Contrastive: {loss_components['contrastive_loss']:.3f}")
            print(f"  - Ambiguity penalty: {loss_components['ambiguity_penalty']:.3f}")
            print(f"  - Chaos: {loss_components['chaos_loss']:.3f}")
            print(f"Verification score: {verification_results['combined_score']:.3f}")
            
            # Check convergence
            if (verification_results['combined_score'] > 0.9 and 
                ensemble_result.consensus_reached and
                loss_components['total_loss'] < 1.0):
                print("🎯 Convergence reached!")
                break
        
        # Prepare final results
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
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.training_history:
            return {'message': 'No training history available'}
        
        return {
            'total_iterations': len(self.training_history),
            'final_loss': self.training_history[-1]['loss_components']['total_loss'],
            'loss_evolution': [iter_data['loss_components']['total_loss'] 
                              for iter_data in self.training_history],
            'consensus_evolution': [iter_data['ensemble_result'].consensus_reached 
                                   for iter_data in self.training_history],
            'solver_preference_evolution': [iter_data['solver_preferences'] 
                                           for iter_data in self.training_history]
        }