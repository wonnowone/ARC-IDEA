#!/usr/bin/env python3
"""
Enhanced ARC EFE Ensemble System with MoE Integration

This module integrates the new Mixture of Experts system with the existing
ensemble architecture, creating a hierarchical system:

Level 1: High-level solvers (strategy and decision-making)
Level 2: MoE routing (expert selection and coordination)  
Level 3: Movement experts (atomic transformations)

The integration maintains backward compatibility while adding:
- Movement-level EFE calculation
- Expert consensus voting at the movement level
- Enhanced RevThink verification with movement reasoning traces
- Multi-level preference learning (solver + expert)
"""

import numpy as np
import torch
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import existing ensemble components
from arc_efe_ensemble_solver import (
    MajorityVotingConsensus, ContrastiveLearningModule, 
    ComprehensiveLossCalculator, EnsembleOutput
)
from arc_efe_solver import ARCState, RevThinkVerifier, ZLearningUpdater

# Import new MoE components
from enhanced_solvers_moe import (
    EnhancedBaseSolver, EnhancedColorPatternSolver, EnhancedShapeSymmetrySolver,
    EnhancedGeometricTransformSolver, EnhancedLogicalRuleSolver, EnhancedSymbolicSolver
)
from movement_experts import MovementResult, MovementType
from moe_router import MovementMoERouter, RoutingStrategy

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

class MovementLevelConsensus:
    """Consensus mechanism at the movement expert level"""
    
    def __init__(self, consensus_threshold: float = 0.5):
        self.consensus_threshold = consensus_threshold
        
    def compute_movement_consensus(self, 
                                 solver_movement_traces: Dict[str, List[MovementResult]]) -> Dict[str, Any]:
        """Compute consensus at movement level across solvers"""
        movement_consensus = {}
        
        # Group movement results by operation type
        movement_by_type = defaultdict(list)
        for solver_name, trace in solver_movement_traces.items():
            for movement_result in trace:
                movement_by_type[movement_result.operation_type].append({
                    'solver': solver_name,
                    'result': movement_result
                })
        
        # Find consensus for each movement type
        for movement_type, movements in movement_by_type.items():
            if len(movements) >= 2:  # Need at least 2 for consensus
                consensus_output = self._find_movement_consensus(movements)
                movement_consensus[movement_type] = consensus_output
        
        return movement_consensus
    
    def _find_movement_consensus(self, movements: List[Dict]) -> Dict[str, Any]:
        """Find consensus among movement results of same type"""
        # Group by output grid
        output_groups = defaultdict(list)
        for movement in movements:
            output_key = hash(movement['result'].output_grid.tobytes())
            output_groups[output_key].append(movement)
        
        # Find majority group
        majority_group = max(output_groups.values(), key=len)
        majority_size = len(majority_group)
        total_movements = len(movements)
        
        consensus_reached = majority_size >= max(2, int(total_movements * self.consensus_threshold))
        
        if consensus_reached:
            representative_movement = majority_group[0]['result']
            return {
                'consensus_reached': True,
                'consensus_output': representative_movement.output_grid,
                'consensus_confidence': np.mean([m['result'].confidence for m in majority_group]),
                'supporting_solvers': [m['solver'] for m in majority_group],
                'majority_count': majority_size,
                'total_count': total_movements
            }
        else:
            return {
                'consensus_reached': False,
                'majority_count': majority_size,
                'total_count': total_movements
            }

class EnhancedEFECalculator:
    """Enhanced EFE calculation with movement-level optimization"""
    
    def __init__(self):
        self.movement_efe_cache = {}
        
    def compute_hierarchical_efe(self, 
                                solver_name: str,
                                solver_output: np.ndarray,
                                movement_trace: List[MovementResult],
                                state: ARCState,
                                movement_consensus: Dict[str, Any]) -> Dict[str, float]:
        """Compute hierarchical EFE at both solver and movement levels"""
        
        # Level 1: Solver-level EFE (existing)
        solver_efe = self._compute_solver_efe(solver_output, state)
        
        # Level 2: Movement-level EFE
        movement_efe = self._compute_movement_level_efe(movement_trace, movement_consensus, state)
        
        # Level 3: Expert-level EFE
        expert_efe = self._compute_expert_level_efe(movement_trace)
        
        # Combined hierarchical EFE
        hierarchical_efe = self._combine_hierarchical_efe(solver_efe, movement_efe, expert_efe)
        
        return {
            'solver_efe': solver_efe,
            'movement_efe': movement_efe,
            'expert_efe': expert_efe,
            'hierarchical_efe': hierarchical_efe,
            'breakdown': {
                'solver_weight': 0.5,
                'movement_weight': 0.3,
                'expert_weight': 0.2
            }
        }
    
    def _compute_solver_efe(self, output: np.ndarray, state: ARCState) -> float:
        """Compute traditional solver-level EFE"""
        try:
            # Simple EFE approximation based on constraint satisfaction
            constraints = state.constraints
            
            # Risk: deviation from expected pattern
            risk = self._calculate_pattern_risk(output, constraints)
            
            # Ambiguity: uncertainty in constraint satisfaction
            ambiguity = self._calculate_constraint_ambiguity(output, constraints)
            
            return risk + ambiguity
        except:
            return 1.0  # Default EFE
    
    def _compute_movement_level_efe(self, 
                                   movement_trace: List[MovementResult],
                                   movement_consensus: Dict[str, Any],
                                   state: ARCState) -> float:
        """Compute EFE at movement level"""
        if not movement_trace:
            return 0.5
        
        try:
            # Movement coherence: how well movements work together
            coherence_score = self._calculate_movement_coherence(movement_trace)
            
            # Consensus alignment: how well movements align with consensus
            consensus_score = self._calculate_consensus_alignment(movement_trace, movement_consensus)
            
            # Movement efficiency: ratio of successful vs total movements
            efficiency_score = self._calculate_movement_efficiency(movement_trace)
            
            # Combined movement EFE
            movement_efe = 1.0 - (coherence_score * 0.4 + consensus_score * 0.3 + efficiency_score * 0.3)
            
            return np.clip(movement_efe, 0.0, 2.0)
        except:
            return 0.5
    
    def _compute_expert_level_efe(self, movement_trace: List[MovementResult]) -> float:
        """Compute EFE at expert level"""
        if not movement_trace:
            return 0.5
        
        try:
            # Expert reliability: average confidence of expert executions
            expert_confidences = [result.confidence for result in movement_trace if result.success]
            avg_confidence = np.mean(expert_confidences) if expert_confidences else 0.3
            
            # Expert diversity: variety of experts used
            expert_types = set([result.operation_type for result in movement_trace])
            diversity_score = min(1.0, len(expert_types) / 4.0)  # Normalize by expected max types
            
            # Expert success rate
            success_rate = sum(1 for result in movement_trace if result.success) / len(movement_trace)
            
            expert_efe = 1.0 - (avg_confidence * 0.5 + diversity_score * 0.2 + success_rate * 0.3)
            
            return np.clip(expert_efe, 0.0, 2.0)
        except:
            return 0.5
    
    def _combine_hierarchical_efe(self, solver_efe: float, movement_efe: float, expert_efe: float) -> float:
        """Combine EFE scores from different levels"""
        weights = [0.5, 0.3, 0.2]  # Solver, Movement, Expert
        scores = [solver_efe, movement_efe, expert_efe]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def _calculate_pattern_risk(self, output: np.ndarray, constraints: Dict[str, Any]) -> float:
        """Calculate pattern risk component"""
        # Simple approximation
        if 'pattern_constraints' in constraints:
            expected_pattern = constraints['pattern_constraints']
            if hasattr(expected_pattern, 'shape') and output.shape == expected_pattern.shape:
                match_ratio = np.mean(output == expected_pattern)
                return 1.0 - match_ratio
        
        return 0.5  # Default risk
    
    def _calculate_constraint_ambiguity(self, output: np.ndarray, constraints: Dict[str, Any]) -> float:
        """Calculate constraint ambiguity component"""
        # Simple approximation
        if 'color_constraints' in constraints:
            expected_colors = set(constraints['color_constraints'])
            actual_colors = set(np.unique(output))
            
            if expected_colors:
                color_match = len(expected_colors & actual_colors) / len(expected_colors)
                return 1.0 - color_match
        
        return 0.3  # Default ambiguity
    
    def _calculate_movement_coherence(self, movement_trace: List[MovementResult]) -> float:
        """Calculate how coherently movements work together"""
        if len(movement_trace) < 2:
            return 1.0
        
        # Check if movements are complementary (e.g., flip followed by rotation)
        complementary_pairs = [
            ('flip', 'rotation'),
            ('color_transform', 'translation'),
            ('rotation', 'translation')
        ]
        
        coherence_score = 0.0
        for i in range(len(movement_trace) - 1):
            current_type = movement_trace[i].operation_type
            next_type = movement_trace[i + 1].operation_type
            
            for pair in complementary_pairs:
                if (current_type, next_type) == pair or (next_type, current_type) == pair:
                    coherence_score += 1.0
                    break
        
        return coherence_score / max(1, len(movement_trace) - 1)
    
    def _calculate_consensus_alignment(self, 
                                     movement_trace: List[MovementResult],
                                     movement_consensus: Dict[str, Any]) -> float:
        """Calculate alignment with movement consensus"""
        if not movement_consensus:
            return 0.5
        
        alignment_scores = []
        for result in movement_trace:
            movement_type = result.operation_type
            if movement_type in movement_consensus:
                consensus_info = movement_consensus[movement_type]
                if consensus_info.get('consensus_reached', False):
                    # Compare with consensus output
                    consensus_output = consensus_info.get('consensus_output')
                    if consensus_output is not None and hasattr(result, 'output_grid'):
                        if result.output_grid.shape == consensus_output.shape:
                            alignment = np.mean(result.output_grid == consensus_output)
                            alignment_scores.append(alignment)
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    def _calculate_movement_efficiency(self, movement_trace: List[MovementResult]) -> float:
        """Calculate movement execution efficiency"""
        if not movement_trace:
            return 0.5
        
        success_count = sum(1 for result in movement_trace if result.success)
        return success_count / len(movement_trace)

class EnhancedRevThinkVerifier:
    """Enhanced RevThink verifier with movement reasoning traces"""
    
    def __init__(self):
        self.base_verifier = RevThinkVerifier()
        
    def verify_with_movement_traces(self, 
                                   solution: np.ndarray,
                                   original_state: ARCState,
                                   movement_traces: Dict[str, List[MovementResult]]) -> Dict[str, float]:
        """Enhanced verification with movement reasoning analysis"""
        
        # Base verification
        base_results = self.base_verifier.verify_solution(solution, original_state)
        
        # Movement-specific verification
        movement_results = self._verify_movement_reasoning(movement_traces, original_state)
        
        # Trace consistency verification
        trace_consistency = self._verify_trace_consistency(movement_traces)
        
        # Combined enhanced verification
        enhanced_results = base_results.copy()
        enhanced_results.update({
            'movement_reasoning_score': movement_results['reasoning_quality'],
            'trace_consistency_score': trace_consistency,
            'movement_diversity_score': movement_results['diversity_score'],
            'expert_reliability_score': movement_results['reliability_score']
        })
        
        # Updated combined score
        enhanced_results['enhanced_combined_score'] = (
            base_results['combined_score'] * 0.6 +
            movement_results['reasoning_quality'] * 0.2 +
            trace_consistency * 0.1 +
            movement_results['reliability_score'] * 0.1
        )
        
        return enhanced_results
    
    def _verify_movement_reasoning(self, 
                                  movement_traces: Dict[str, List[MovementResult]], 
                                  original_state: ARCState) -> Dict[str, float]:
        """Verify quality of movement reasoning"""
        if not movement_traces:
            return {'reasoning_quality': 0.3, 'diversity_score': 0.0, 'reliability_score': 0.3}
        
        all_movements = []
        for trace in movement_traces.values():
            all_movements.extend(trace)
        
        if not all_movements:
            return {'reasoning_quality': 0.3, 'diversity_score': 0.0, 'reliability_score': 0.3}
        
        # Reasoning quality: average confidence of movements
        confidences = [m.confidence for m in all_movements if m.success]
        reasoning_quality = np.mean(confidences) if confidences else 0.3
        
        # Diversity: variety of movement types used
        movement_types = set([m.operation_type for m in all_movements])
        diversity_score = min(1.0, len(movement_types) / 5.0)  # Normalize by expected variety
        
        # Reliability: success rate across all movements
        success_rate = sum(1 for m in all_movements if m.success) / len(all_movements)
        
        return {
            'reasoning_quality': reasoning_quality,
            'diversity_score': diversity_score,
            'reliability_score': success_rate
        }
    
    def _verify_trace_consistency(self, movement_traces: Dict[str, List[MovementResult]]) -> float:
        """Verify consistency across solver movement traces"""
        if len(movement_traces) < 2:
            return 1.0  # Perfect consistency for single trace
        
        # Check for similar movement patterns across solvers
        trace_patterns = []
        for solver_name, trace in movement_traces.items():
            pattern = [result.operation_type for result in trace if result.success]
            trace_patterns.append(pattern)
        
        # Calculate pattern similarity
        consistency_scores = []
        for i in range(len(trace_patterns)):
            for j in range(i + 1, len(trace_patterns)):
                similarity = self._calculate_pattern_similarity(trace_patterns[i], trace_patterns[j])
                consistency_scores.append(similarity)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_pattern_similarity(self, pattern1: List[str], pattern2: List[str]) -> float:
        """Calculate similarity between movement patterns"""
        if not pattern1 and not pattern2:
            return 1.0
        if not pattern1 or not pattern2:
            return 0.0
        
        # Use longest common subsequence approach
        common_elements = set(pattern1) & set(pattern2)
        total_elements = set(pattern1) | set(pattern2)
        
        return len(common_elements) / len(total_elements) if total_elements else 0.0

class EnhancedARCEFESystem:
    """Enhanced ARC EFE system with full MoE integration"""
    
    def __init__(self, solvers: List[EnhancedBaseSolver]):
        self.solvers = {solver.solver_name: solver for solver in solvers}
        self.solver_list = solvers
        
        # Enhanced components
        self.consensus_module = MajorityVotingConsensus()
        self.movement_consensus = MovementLevelConsensus()
        self.enhanced_efe_calculator = EnhancedEFECalculator()
        self.enhanced_verifier = EnhancedRevThinkVerifier()
        self.contrastive_module = ContrastiveLearningModule()
        self.loss_calculator = ComprehensiveLossCalculator()
        self.z_learner = ZLearningUpdater()
        
        # System state
        self.solver_preferences = self._initialize_solver_preferences()
        self.execution_history = []
        self.max_iterations = 10
        
    def solve_with_enhanced_ensemble(self, 
                                   initial_grid: np.ndarray,
                                   constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Enhanced ensemble solving with full MoE integration"""
        
        current_state = ARCState(
            grid=initial_grid.copy(),
            constraints=constraints,
            step=0,
            solver_history=[],
            confidence=0.0
        )
        
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            print(f"\\n🔄 Enhanced Iteration {iteration + 1}")
            
            # Step 1: Execute all enhanced solvers with MoE
            solver_outputs = {}
            movement_traces = {}
            
            for solver in self.solver_list:
                try:
                    # Get solver output
                    output = solver.predict(current_state.grid)
                    solver_outputs[solver.solver_name] = output
                    
                    # Get movement trace from solver's thinking flow
                    thinking_flow = solver.get_thinking_flow()
                    movement_sequence = thinking_flow.get('movement_sequence', [])
                    
                    # Create mock movement results for now (in practice, would come from solver)
                    movement_trace = []
                    for i, expert_name in enumerate(movement_sequence):
                        movement_result = MovementResult(
                            output_grid=output,
                            confidence=thinking_flow.get('confidence', 0.5),
                            operation_type=expert_name.lower().replace('expert', ''),
                            parameters={},
                            execution_time=0.1,
                            success=True
                        )
                        movement_trace.append(movement_result)
                    
                    movement_traces[solver.solver_name] = movement_trace
                    
                except Exception as e:
                    warnings.warn(f"Solver {solver.solver_name} failed: {e}")
                    continue
            
            if not solver_outputs:
                print("❌ All solvers failed")
                break
            
            print(f"✅ {len(solver_outputs)} solvers executed successfully")
            
            # Step 2: Solver-level consensus
            solver_consensus = self.consensus_module.compute_consensus(solver_outputs)
            
            # Step 3: Movement-level consensus
            movement_consensus = self.movement_consensus.compute_movement_consensus(movement_traces)
            
            print(f"📊 Movement consensus: {len(movement_consensus)} movement types agreed")
            
            # Step 4: Enhanced EFE calculation
            enhanced_efe_scores = {}
            for solver_name, output in solver_outputs.items():
                movement_trace = movement_traces.get(solver_name, [])
                efe_breakdown = self.enhanced_efe_calculator.compute_hierarchical_efe(
                    solver_name, output, movement_trace, current_state, movement_consensus
                )
                enhanced_efe_scores[solver_name] = efe_breakdown
            
            # Step 5: Enhanced RevThink verification
            verification_results = self.enhanced_verifier.verify_with_movement_traces(
                solver_consensus.output,
                current_state,
                movement_traces
            )
            
            print(f"🔍 Enhanced verification score: {verification_results['enhanced_combined_score']:.3f}")
            
            # Step 6: Contrastive learning with movement traces
            correct_solvers = []
            for solver_name, output in solver_outputs.items():
                if np.array_equal(output, solver_consensus.output):
                    correct_solvers.append(solver_name)
            
            contrastive_loss, similarity_scores = self.contrastive_module.compute_contrastive_loss(
                solver_consensus.output,
                solver_outputs,
                correct_solvers
            )
            
            # Step 7: Enhanced loss calculation
            avg_hierarchical_efe = np.mean([
                scores['hierarchical_efe'] for scores in enhanced_efe_scores.values()
            ])
            
            loss_components = self.loss_calculator.compute_total_loss(
                efe_loss=avg_hierarchical_efe,
                contrastive_loss=contrastive_loss,
                ambiguity_score=solver_consensus.ambiguity_score,
                diversity_score=verification_results.get('movement_diversity_score', 0.5),
                solver_outputs=solver_outputs
            )
            
            # Step 8: Multi-level preference updates
            for solver_name in self.solvers.keys():
                enhanced_verification = verification_results.copy()
                enhanced_verification['consensus_bonus'] = 1.0 if solver_name in correct_solvers else 0.0
                
                # Include movement-level feedback
                if solver_name in enhanced_efe_scores:
                    efe_breakdown = enhanced_efe_scores[solver_name]
                    enhanced_verification['movement_efe'] = efe_breakdown['movement_efe']
                    enhanced_verification['expert_efe'] = efe_breakdown['expert_efe']
                
                self.solver_preferences = self.z_learner.update_preferences(
                    self.solver_preferences,
                    solver_name,
                    enhanced_efe_scores.get(solver_name, {}).get('hierarchical_efe', 1.0),
                    enhanced_verification
                )
            
            # Step 9: Update state
            current_state.grid = solver_consensus.output.copy()
            current_state.step = iteration + 1
            current_state.solver_history.extend(correct_solvers)
            current_state.confidence = verification_results['enhanced_combined_score']
            
            # Store enhanced iteration data
            iteration_data = {
                'iteration': iteration,
                'solver_outputs': solver_outputs,
                'movement_traces': movement_traces,
                'solver_consensus': solver_consensus,
                'movement_consensus': movement_consensus,
                'enhanced_efe_scores': enhanced_efe_scores,
                'verification_results': verification_results,
                'loss_components': loss_components,
                'correct_solvers': correct_solvers,
                'solver_preferences': self.solver_preferences.copy()
            }
            iteration_history.append(iteration_data)
            
            print(f"📈 Total loss: {loss_components['total_loss']:.3f}")
            print(f"   - Hierarchical EFE: {avg_hierarchical_efe:.3f}")
            print(f"   - Movement diversity: {verification_results.get('movement_diversity_score', 0):.3f}")
            
            # Enhanced convergence check
            if (verification_results['enhanced_combined_score'] > 0.85 and
                solver_consensus.consensus_reached and
                loss_components['total_loss'] < 1.5):
                print("🎯 Enhanced convergence achieved!")
                break
        
        # Compile MoE statistics
        moe_statistics = self._compile_moe_statistics(iteration_history)
        
        final_results = {
            'solution': current_state.grid,
            'confidence': current_state.confidence,
            'solver_history': current_state.solver_history,
            'final_preferences': self.solver_preferences,
            'iteration_history': iteration_history,
            'moe_statistics': moe_statistics,
            'enhanced_metrics': {
                'movement_consensus_rate': self._calculate_movement_consensus_rate(iteration_history),
                'expert_diversity_score': self._calculate_expert_diversity(iteration_history),
                'hierarchical_efe_evolution': self._extract_efe_evolution(iteration_history)
            }
        }
        
        return current_state.grid, final_results
    
    def _initialize_solver_preferences(self) -> Dict[str, float]:
        """Initialize solver preferences uniformly"""
        uniform_pref = 1.0 / len(self.solver_list)
        return {solver.solver_name: uniform_pref for solver in self.solver_list}
    
    def _compile_moe_statistics(self, iteration_history: List[Dict]) -> Dict[str, Any]:
        """Compile comprehensive MoE statistics"""
        if not iteration_history:
            return {}
        
        # Expert usage statistics
        expert_usage = defaultdict(int)
        movement_type_usage = defaultdict(int)
        
        for iteration_data in iteration_history:
            movement_traces = iteration_data.get('movement_traces', {})
            for solver_name, trace in movement_traces.items():
                for movement_result in trace:
                    expert_usage[movement_result.operation_type] += 1
                    movement_type_usage[movement_result.operation_type] += 1
        
        # Success rates by expert
        expert_success_rates = {}
        for iteration_data in iteration_history:
            movement_traces = iteration_data.get('movement_traces', {})
            for solver_name, trace in movement_traces.items():
                for movement_result in trace:
                    expert_name = movement_result.operation_type
                    if expert_name not in expert_success_rates:
                        expert_success_rates[expert_name] = {'successes': 0, 'total': 0}
                    
                    expert_success_rates[expert_name]['total'] += 1
                    if movement_result.success:
                        expert_success_rates[expert_name]['successes'] += 1
        
        # Calculate final success rates
        final_success_rates = {}
        for expert_name, stats in expert_success_rates.items():
            final_success_rates[expert_name] = stats['successes'] / max(1, stats['total'])
        
        return {
            'expert_usage_counts': dict(expert_usage),
            'movement_type_distribution': dict(movement_type_usage),
            'expert_success_rates': final_success_rates,
            'total_movements_executed': sum(expert_usage.values()),
            'unique_experts_used': len(expert_usage),
            'most_used_expert': max(expert_usage.items(), key=lambda x: x[1])[0] if expert_usage else None
        }
    
    def _calculate_movement_consensus_rate(self, iteration_history: List[Dict]) -> float:
        """Calculate rate of movement consensus achievement"""
        if not iteration_history:
            return 0.0
        
        consensus_count = 0
        total_movement_types = 0
        
        for iteration_data in iteration_history:
            movement_consensus = iteration_data.get('movement_consensus', {})
            for movement_type, consensus_info in movement_consensus.items():
                total_movement_types += 1
                if consensus_info.get('consensus_reached', False):
                    consensus_count += 1
        
        return consensus_count / max(1, total_movement_types)
    
    def _calculate_expert_diversity(self, iteration_history: List[Dict]) -> float:
        """Calculate diversity of expert usage across iterations"""
        all_experts_used = set()
        
        for iteration_data in iteration_history:
            movement_traces = iteration_data.get('movement_traces', {})
            for solver_name, trace in movement_traces.items():
                for movement_result in trace:
                    all_experts_used.add(movement_result.operation_type)
        
        # Normalize by expected maximum diversity (assuming 5 main expert types)
        return min(1.0, len(all_experts_used) / 5.0)
    
    def _extract_efe_evolution(self, iteration_history: List[Dict]) -> List[float]:
        """Extract evolution of hierarchical EFE scores"""
        efe_evolution = []
        
        for iteration_data in iteration_history:
            enhanced_efe_scores = iteration_data.get('enhanced_efe_scores', {})
            if enhanced_efe_scores:
                avg_hierarchical_efe = np.mean([
                    scores['hierarchical_efe'] for scores in enhanced_efe_scores.values()
                ])
                efe_evolution.append(avg_hierarchical_efe)
        
        return efe_evolution

# Factory function for creating enhanced system
def create_enhanced_arc_system() -> EnhancedARCEFESystem:
    """Create enhanced ARC system with MoE solvers"""
    enhanced_solvers = [
        EnhancedColorPatternSolver(),
        EnhancedShapeSymmetrySolver(),
        EnhancedGeometricTransformSolver(),
        EnhancedLogicalRuleSolver(),
        EnhancedSymbolicSolver()
    ]
    
    return EnhancedARCEFESystem(enhanced_solvers)