#!/usr/bin/env python3
"""
Multi-LLM ARC-IDEA System Integration

This module integrates multiple LLM models (GPT-OSS-20B + Kanana-1.5-15.7B-A3B) 
into the ARC EFE system, creating a sophisticated multi-model ensemble with:

- Cross-architecture consensus mechanisms
- Model synergy analysis and optimization
- Adaptive weighting based on problem types
- Comprehensive multi-model performance tracking
"""

import numpy as np
import time
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

# Import core systems
from EFE_update import (
    EnhancedARCEFESystem, ARCState, EnsembleOutput,
    MajorityVotingConsensus, ContrastiveLearningModule, 
    ComprehensiveLossCalculator, RevThinkVerifier, ZLearningUpdater
)

# Import LLM integrations
from gpt_oss_model_wrapper import GPTOSSSolver, ModelConfig, ModelResponse
from multi_llm_wrapper import (
    KananaSolver, MultiLLMEnsemble, LLMModelType,
    create_gpt_oss_solver, create_kanana_solver, create_multi_llm_ensemble
)
from arc_gpt_oss_integration import HybridSolverEnsemble, LLMEnhancedConsensus

@dataclass
class MultiLLMResult:
    """Result from multi-LLM ensemble processing"""
    consensus_output: np.ndarray
    gpt_oss_outputs: List[np.ndarray]
    kanana_outputs: List[np.ndarray]
    cross_model_agreement: float
    synergy_score: float
    model_contributions: Dict[str, float]
    ensemble_confidence: float
    processing_times: Dict[str, float]
    reasoning_traces: Dict[str, str]

class MultiLLMConsensusModule:
    """Advanced consensus mechanism for multiple LLM architectures"""
    
    def __init__(self, consensus_threshold: float = 0.6):
        self.consensus_threshold = consensus_threshold
        self.traditional_consensus = MajorityVotingConsensus(consensus_threshold)
        
        # Model-specific weights that adapt over time
        self.adaptive_weights = {
            'gpt_oss': 0.55,
            'kanana': 0.45,
            'traditional': 0.7
        }
        
        # Performance tracking for adaptive weighting
        self.model_performance_history = defaultdict(list)
        self.consensus_quality_history = []
        
    def compute_multi_llm_consensus(self, 
                                   traditional_outputs: Dict[str, np.ndarray],
                                   gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                   kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> Dict[str, Any]:
        """Compute consensus across traditional solvers and multiple LLM architectures"""
        
        try:
            # Separate LLM outputs and metadata
            gpt_oss_outputs = {f"GPT_OSS_{i}": result[0] for i, result in enumerate(gpt_oss_results)}
            kanana_outputs = {f"Kanana_{i}": result[0] for i, result in enumerate(kanana_results)}
            
            # Combine all outputs
            all_outputs = {}
            all_outputs.update(traditional_outputs)
            all_outputs.update(gpt_oss_outputs) 
            all_outputs.update(kanana_outputs)
            
            if not all_outputs:
                return self._create_fallback_consensus()
            
            # Multi-level consensus approach
            consensus_results = {}
            
            # Level 1: Traditional solver consensus
            if traditional_outputs:
                trad_consensus = self.traditional_consensus.compute_consensus(traditional_outputs)
                consensus_results['traditional'] = {
                    'output': trad_consensus.output,
                    'confidence': trad_consensus.confidence,
                    'weight': self.adaptive_weights['traditional']
                }
            
            # Level 2: GPT-OSS consensus
            if gpt_oss_outputs:
                gpt_consensus = self.traditional_consensus.compute_consensus(gpt_oss_outputs)
                # Weight by average confidence
                avg_confidence = np.mean([r[1].confidence for r in gpt_oss_results])
                consensus_results['gpt_oss'] = {
                    'output': gpt_consensus.output,
                    'confidence': avg_confidence,
                    'weight': self.adaptive_weights['gpt_oss'] * avg_confidence
                }
            
            # Level 3: Kanana consensus
            if kanana_outputs:
                kanana_consensus = self.traditional_consensus.compute_consensus(kanana_outputs)
                # Weight by average confidence
                avg_confidence = np.mean([r[1].confidence for r in kanana_results])
                consensus_results['kanana'] = {
                    'output': kanana_consensus.output,
                    'confidence': avg_confidence,
                    'weight': self.adaptive_weights['kanana'] * avg_confidence
                }
            
            # Level 4: Cross-architecture final consensus
            final_consensus = self._compute_cross_architecture_consensus(consensus_results)
            
            # Calculate cross-model metrics
            cross_metrics = self._calculate_cross_model_metrics(
                traditional_outputs, gpt_oss_results, kanana_results, final_consensus['output']
            )
            
            # Update adaptive weights based on performance
            self._update_adaptive_weights(consensus_results, cross_metrics)
            
            result = {
                'output': final_consensus['output'],
                'consensus_reached': final_consensus['consensus_reached'],
                'multi_level_results': consensus_results,
                'cross_model_agreement': cross_metrics['cross_model_agreement'],
                'synergy_score': cross_metrics['synergy_score'],
                'model_contributions': cross_metrics['model_contributions'],
                'ensemble_confidence': final_consensus['confidence'],
                'adaptive_weights': self.adaptive_weights.copy(),
                'consensus_quality': cross_metrics['consensus_quality']
            }
            
            self.consensus_quality_history.append(cross_metrics['consensus_quality'])
            return result
            
        except Exception as e:
            warnings.warn(f"Multi-LLM consensus computation failed: {e}")
            return self._create_fallback_consensus()
    
    def _compute_cross_architecture_consensus(self, consensus_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compute final consensus across different architecture types"""
        
        if not consensus_results:
            return {'output': np.zeros((5, 5), dtype=int), 'confidence': 0.1, 'consensus_reached': False}
        
        # Weighted voting across architecture consensus results
        all_outputs = [(result['output'], result['weight']) for result in consensus_results.values()]
        
        if not all_outputs:
            return {'output': np.zeros((5, 5), dtype=int), 'confidence': 0.1, 'consensus_reached': False}
        
        # Get target shape
        target_shape = all_outputs[0][0].shape
        final_grid = np.zeros(target_shape, dtype=int)
        
        # Position-wise weighted voting
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                position_votes = defaultdict(float)
                
                for output, weight in all_outputs:
                    if output.shape == target_shape:
                        value = output[i, j]
                        position_votes[value] += weight
                
                # Choose value with highest weight
                if position_votes:
                    best_value = max(position_votes.keys(), key=lambda x: position_votes[x])
                    final_grid[i, j] = best_value
        
        # Calculate overall confidence
        total_weight = sum(result['weight'] for result in consensus_results.values())
        weighted_confidence = sum(
            result['confidence'] * result['weight'] 
            for result in consensus_results.values()
        ) / max(total_weight, 1.0)
        
        # Determine if consensus reached
        consensus_reached = len(consensus_results) >= 2 and weighted_confidence > self.consensus_threshold
        
        return {
            'output': final_grid,
            'confidence': weighted_confidence,
            'consensus_reached': consensus_reached
        }
    
    def _calculate_cross_model_metrics(self, 
                                     traditional_outputs: Dict[str, np.ndarray],
                                     gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                     kanana_results: List[Tuple[np.ndarray, ModelResponse]],
                                     final_consensus: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive cross-model performance metrics"""
        
        metrics = {}
        
        # Cross-model agreement calculation
        all_outputs = list(traditional_outputs.values())
        all_outputs.extend([r[0] for r in gpt_oss_results])
        all_outputs.extend([r[0] for r in kanana_results])
        
        if len(all_outputs) >= 2:
            agreements = []
            for i in range(len(all_outputs)):
                for j in range(i + 1, len(all_outputs)):
                    if all_outputs[i].shape == all_outputs[j].shape:
                        agreement = np.mean(all_outputs[i] == all_outputs[j])
                        agreements.append(agreement)
            
            metrics['cross_model_agreement'] = np.mean(agreements) if agreements else 0.5
        else:
            metrics['cross_model_agreement'] = 1.0
        
        # Model contribution analysis
        contributions = {}
        
        # Traditional contribution
        if traditional_outputs:
            trad_similarities = []
            for output in traditional_outputs.values():
                if output.shape == final_consensus.shape:
                    similarity = np.mean(output == final_consensus)
                    trad_similarities.append(similarity)
            contributions['traditional'] = np.mean(trad_similarities) if trad_similarities else 0.0
        
        # GPT-OSS contribution
        if gpt_oss_results:
            gpt_similarities = []
            for output, response in gpt_oss_results:
                if output.shape == final_consensus.shape:
                    similarity = np.mean(output == final_consensus) * response.confidence
                    gpt_similarities.append(similarity)
            contributions['gpt_oss'] = np.mean(gpt_similarities) if gpt_similarities else 0.0
        
        # Kanana contribution
        if kanana_results:
            kanana_similarities = []
            for output, response in kanana_results:
                if output.shape == final_consensus.shape:
                    similarity = np.mean(output == final_consensus) * response.confidence
                    kanana_similarities.append(similarity)
            contributions['kanana'] = np.mean(kanana_similarities) if kanana_similarities else 0.0
        
        metrics['model_contributions'] = contributions
        
        # Synergy score (how well different model types complement each other)
        synergy_factors = []
        
        # Confidence complementarity
        if gpt_oss_results and kanana_results:
            gpt_conf = np.mean([r[1].confidence for r in gpt_oss_results])
            kanana_conf = np.mean([r[1].confidence for r in kanana_results])
            confidence_synergy = min(gpt_conf, kanana_conf) * 1.5  # Boost when both confident
            synergy_factors.append(confidence_synergy)
        
        # Output diversity (healthy disagreement)
        if len(all_outputs) >= 2:
            diversity = 1.0 - metrics['cross_model_agreement']
            optimal_diversity = 0.25  # Some disagreement is good for robustness
            diversity_synergy = 1.0 - abs(diversity - optimal_diversity) / 0.75
            synergy_factors.append(diversity_synergy)
        
        metrics['synergy_score'] = np.clip(np.mean(synergy_factors), 0.0, 1.0) if synergy_factors else 0.5
        
        # Overall consensus quality
        quality_factors = [
            metrics['cross_model_agreement'],
            metrics['synergy_score'],
            max(contributions.values()) if contributions else 0.5
        ]
        
        metrics['consensus_quality'] = np.mean(quality_factors)
        
        return metrics
    
    def _update_adaptive_weights(self, consensus_results: Dict[str, Dict], cross_metrics: Dict[str, Any]):
        """Update adaptive weights based on performance feedback"""
        
        # Track performance for each model type
        for model_type, result in consensus_results.items():
            if model_type in cross_metrics['model_contributions']:
                contribution = cross_metrics['model_contributions'][model_type]
                confidence = result['confidence']
                performance_score = (contribution + confidence) / 2.0
                
                self.model_performance_history[model_type].append(performance_score)
                
                # Keep only recent history
                if len(self.model_performance_history[model_type]) > 10:
                    self.model_performance_history[model_type] = \
                        self.model_performance_history[model_type][-10:]
        
        # Adapt weights based on recent performance
        if len(self.model_performance_history) >= 2:
            for model_type in self.adaptive_weights.keys():
                if model_type in self.model_performance_history:
                    recent_performance = self.model_performance_history[model_type]
                    if len(recent_performance) >= 3:
                        avg_performance = np.mean(recent_performance[-3:])
                        
                        # Gradual weight adjustment
                        if avg_performance > 0.7:
                            self.adaptive_weights[model_type] = min(0.8, 
                                self.adaptive_weights[model_type] * 1.05)
                        elif avg_performance < 0.4:
                            self.adaptive_weights[model_type] = max(0.2, 
                                self.adaptive_weights[model_type] * 0.95)
        
        # Normalize weights
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total_weight
    
    def _create_fallback_consensus(self) -> Dict[str, Any]:
        """Create fallback consensus when main computation fails"""
        
        return {
            'output': np.zeros((5, 5), dtype=int),
            'consensus_reached': False,
            'multi_level_results': {},
            'cross_model_agreement': 0.0,
            'synergy_score': 0.0,
            'model_contributions': {},
            'ensemble_confidence': 0.1,
            'adaptive_weights': self.adaptive_weights.copy(),
            'consensus_quality': 0.0
        }

class MultiLLMEnhancedEFESystem(EnhancedARCEFESystem):
    """Enhanced ARC EFE System with Multi-LLM integration"""
    
    def __init__(self,
                 traditional_solvers: List[Any],
                 gpt_oss_config: ModelConfig,
                 kanana_config: ModelConfig,
                 planning_horizon: int = 3,
                 consensus_threshold: float = 0.6):
        
        # Create multi-LLM ensemble
        self.multi_llm_ensemble = MultiLLMEnsemble(gpt_oss_config, kanana_config)
        
        # Combine all solvers (traditional + multi-LLM)
        all_solvers = traditional_solvers + self.multi_llm_ensemble.get_all_llm_solvers()
        
        # Initialize parent system
        super().__init__(
            solvers=all_solvers,
            planning_horizon=planning_horizon,
            consensus_threshold=consensus_threshold
        )
        
        # Override RevThinkVerifier with LLM-enabled version
        llm_wrappers = [
            self.multi_llm_ensemble.gpt_oss_solver,
            self.multi_llm_ensemble.kanana_solver
        ]
        self.revthink_verifier = RevThinkVerifier(llm_wrappers=llm_wrappers)
        
        # Multi-LLM specific components
        self.multi_llm_consensus = MultiLLMConsensusModule(consensus_threshold)
        self.cross_model_verifier = CrossModelVerifier()
        
        # Enhanced tracking
        self.multi_llm_interaction_history = []
        self.cross_architecture_performance = {}
        self.synergy_evolution = []
        
    def solve_with_multi_llm_ensemble(self,
                                     initial_grid: np.ndarray,
                                     constraints: Dict[str, Any]) -> Tuple[np.ndarray, Dict]:
        """Enhanced solving with multi-LLM ensemble (GPT-OSS-20B + Kanana-1.5-15.7B-A3B)"""
        
        current_state = ARCState(
            grid=initial_grid.copy(),
            constraints=constraints,
            step=0,
            solver_history=[],
            confidence=0.0
        )
        
        iteration_history = []
        
        for iteration in range(self.max_iterations):
            print(f"\\n🧠 Multi-LLM Iteration {iteration + 1} (GPT-OSS-20B + Kanana-1.5-15.7B-A3B)")
            
            # Categorize solvers
            traditional_solvers = {}
            gpt_oss_results = []
            kanana_results = []
            
            # Execute all solvers with categorization
            for i, solver_name in enumerate(self.efe_solver.solver_names):
                solver = self.efe_solver.solvers[i]
                
                try:
                    if isinstance(solver, GPTOSSSolver):
                        # GPT-OSS solver with metadata
                        output, response = solver.predict_with_metadata(current_state.grid, constraints)
                        gpt_oss_results.append((output, response))
                        print(f"  🤖 {solver_name}: conf={response.confidence:.3f}, "
                              f"time={response.processing_time:.2f}s")
                        
                    elif isinstance(solver, KananaSolver):
                        # Kanana solver with metadata
                        output, response = solver.predict_with_metadata(current_state.grid, constraints)
                        kanana_results.append((output, response))
                        print(f"  🧮 {solver_name}: conf={response.confidence:.3f}, "
                              f"time={response.processing_time:.2f}s")
                        
                    else:
                        # Traditional solver
                        output = solver.predict(current_state.grid)
                        traditional_solvers[solver_name] = output
                        print(f"  🔧 {solver_name}: completed")
                    
                except Exception as e:
                    warnings.warn(f"Solver {solver_name} failed: {e}")
                    continue
            
            if not traditional_solvers and not gpt_oss_results and not kanana_results:
                print("❌ All solvers failed!")
                break
            
            # Multi-LLM consensus computation
            consensus_result = self.multi_llm_consensus.compute_multi_llm_consensus(
                traditional_solvers, gpt_oss_results, kanana_results
            )
            
            print(f"📊 Multi-LLM Consensus:")
            print(f"  - Cross-model agreement: {consensus_result['cross_model_agreement']:.3f}")
            print(f"  - Synergy score: {consensus_result['synergy_score']:.3f}")
            print(f"  - GPT-OSS contribution: {consensus_result['model_contributions'].get('gpt_oss', 0):.3f}")
            print(f"  - Kanana contribution: {consensus_result['model_contributions'].get('kanana', 0):.3f}")
            
            # Cross-model verification
            verification_results = self.cross_model_verifier.verify_across_models(
                initial_grid, 
                consensus_result['output'],
                gpt_oss_results,
                kanana_results,
                current_state.solver_history
            )
            
            # Enhanced loss calculation with multi-LLM factors
            loss_components = self._compute_multi_llm_loss(
                traditional_solvers, gpt_oss_results, kanana_results,
                consensus_result, verification_results
            )
            
            # Update solver preferences with multi-LLM awareness
            self._update_multi_llm_preferences(
                traditional_solvers, gpt_oss_results, kanana_results,
                consensus_result, verification_results, iteration
            )
            
            # Track multi-LLM specific metrics
            multi_llm_data = {
                'iteration': iteration,
                'traditional_count': len(traditional_solvers),
                'gpt_oss_count': len(gpt_oss_results),
                'kanana_count': len(kanana_results),
                'consensus_result': consensus_result,
                'verification_results': verification_results,
                'loss_components': loss_components,
                'cross_architecture_performance': self._analyze_cross_architecture_performance(
                    gpt_oss_results, kanana_results
                )
            }
            
            self.multi_llm_interaction_history.append(multi_llm_data)
            self.synergy_evolution.append(consensus_result['synergy_score'])
            
            # Update state
            current_state.grid = consensus_result['output'].copy()
            current_state.step = iteration + 1
            current_state.confidence = verification_results['combined_score']
            
            # Enhanced convergence criteria
            if (verification_results['combined_score'] > 0.88 and 
                consensus_result['consensus_reached'] and
                consensus_result['synergy_score'] > 0.7 and
                loss_components['total_loss'] < 1.1):
                print("🏆 Multi-LLM convergence achieved!")
                break
                
            print(f"🎯 Loss: {loss_components['total_loss']:.3f}, "
                  f"Verification: {verification_results['combined_score']:.3f}")
        
        # Comprehensive results with multi-LLM analytics
        final_results = {
            'solution': current_state.grid,
            'confidence': current_state.confidence,
            'solver_history': current_state.solver_history,
            'final_preferences': self.efe_solver.solver_preferences,
            'iteration_history': iteration_history,
            'multi_llm_history': self.multi_llm_interaction_history,
            'multi_llm_ensemble_stats': self.multi_llm_ensemble.get_ensemble_stats(),
            'synergy_evolution': self.synergy_evolution,
            'cross_architecture_analysis': self._final_cross_architecture_analysis(),
            'adaptive_weights_evolution': self._get_adaptive_weights_history(),
            'ensemble_metrics': {
                'final_synergy': self.synergy_evolution[-1] if self.synergy_evolution else 0.0,
                'average_synergy': np.mean(self.synergy_evolution) if self.synergy_evolution else 0.0,
                'synergy_trend': self._calculate_synergy_trend(),
                'total_iterations': len(self.multi_llm_interaction_history),
                'gpt_oss_utilization': self._calculate_model_utilization('gpt_oss'),
                'kanana_utilization': self._calculate_model_utilization('kanana')
            }
        }
        
        return current_state.grid, final_results
    
    def _compute_multi_llm_loss(self, 
                               traditional_outputs: Dict[str, np.ndarray],
                               gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                               kanana_results: List[Tuple[np.ndarray, ModelResponse]],
                               consensus_result: Dict[str, Any],
                               verification_results: Dict[str, Any]) -> Dict[str, float]:
        """Compute enhanced loss with multi-LLM specific factors"""
        
        # Base loss components
        base_efe = 1.0  # Placeholder
        diversity_score = self._compute_multi_model_diversity(traditional_outputs, gpt_oss_results, kanana_results)
        
        # Multi-LLM specific loss terms
        synergy_penalty = max(0, 0.5 - consensus_result['synergy_score'])  # Penalty for low synergy
        agreement_bonus = consensus_result['cross_model_agreement'] * 0.3  # Bonus for agreement
        
        # Cross-architecture consistency term
        cross_consistency = verification_results.get('cross_model_consistency', 0.5)
        consistency_term = (1.0 - cross_consistency) * 0.4
        
        total_loss = (base_efe + 
                     synergy_penalty + 
                     consistency_term - 
                     agreement_bonus +
                     abs(diversity_score - 0.3) * 0.2)  # Optimal diversity around 0.3
        
        return {
            'total_loss': total_loss,
            'base_efe': base_efe,
            'synergy_penalty': synergy_penalty,
            'consistency_term': consistency_term,
            'agreement_bonus': agreement_bonus,
            'diversity_score': diversity_score,
            'multi_llm_enhanced': True
        }
    
    def _compute_multi_model_diversity(self, 
                                     traditional_outputs: Dict[str, np.ndarray],
                                     gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                     kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> float:
        """Compute diversity across all model types"""
        
        all_outputs = list(traditional_outputs.values())
        all_outputs.extend([r[0] for r in gpt_oss_results])
        all_outputs.extend([r[0] for r in kanana_results])
        
        if len(all_outputs) < 2:
            return 0.0
        
        diversities = []
        for i in range(len(all_outputs)):
            for j in range(i + 1, len(all_outputs)):
                if all_outputs[i].shape == all_outputs[j].shape:
                    diversity = 1.0 - np.mean(all_outputs[i] == all_outputs[j])
                    diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _update_multi_llm_preferences(self, 
                                    traditional_outputs: Dict[str, np.ndarray],
                                    gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                    kanana_results: List[Tuple[np.ndarray, ModelResponse]],
                                    consensus_result: Dict[str, Any],
                                    verification_results: Dict[str, Any],
                                    iteration: int):
        """Update preferences with multi-LLM awareness"""
        
        consensus_grid = consensus_result['output']
        
        # Update traditional solver preferences
        for solver_name, output in traditional_outputs.items():
            if solver_name in self.efe_solver.solver_preferences:
                similarity_bonus = np.mean(output == consensus_grid) if output.shape == consensus_grid.shape else 0.0
                enhanced_verification = verification_results.copy()
                enhanced_verification['consensus_bonus'] = similarity_bonus
                
                # Update with Z-learning
                self.efe_solver.solver_preferences = self.z_learner.update_preferences(
                    self.efe_solver.solver_preferences,
                    solver_name,
                    1.0,  # Placeholder EFE
                    enhanced_verification
                )
        
        # Update LLM solver preferences with model-specific factors
        all_llm_results = [(f"GPT_OSS_{i}", r) for i, r in enumerate(gpt_oss_results)]
        all_llm_results.extend([(f"Kanana_{i}", r) for i, r in enumerate(kanana_results)])
        
        for solver_id, (output, response) in all_llm_results:
            # Find corresponding solver in preferences
            matching_solver = None
            for solver_name in self.efe_solver.solver_preferences.keys():
                if solver_id.split('_')[0].lower() in solver_name.lower():
                    matching_solver = solver_name
                    break
            
            if matching_solver:
                # Multi-factor bonus calculation
                similarity_bonus = np.mean(output == consensus_grid) if output.shape == consensus_grid.shape else 0.0
                confidence_bonus = response.confidence * 0.3
                synergy_bonus = consensus_result['synergy_score'] * 0.2
                
                total_bonus = similarity_bonus + confidence_bonus + synergy_bonus
                
                enhanced_verification = verification_results.copy()
                enhanced_verification['consensus_bonus'] = total_bonus
                enhanced_verification['llm_confidence'] = response.confidence
                
                self.efe_solver.solver_preferences = self.z_learner.update_preferences(
                    self.efe_solver.solver_preferences,
                    matching_solver,
                    max(0.1, 2.0 - response.confidence),  # Lower EFE for higher confidence
                    enhanced_verification
                )
    
    def _analyze_cross_architecture_performance(self, 
                                              gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                              kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> Dict[str, Any]:
        """Analyze performance across different LLM architectures"""
        
        analysis = {}
        
        # GPT-OSS analysis
        if gpt_oss_results:
            gpt_confidences = [r[1].confidence for r in gpt_oss_results]
            gpt_times = [r[1].processing_time for r in gpt_oss_results]
            
            analysis['gpt_oss'] = {
                'average_confidence': np.mean(gpt_confidences),
                'confidence_std': np.std(gpt_confidences),
                'average_time': np.mean(gpt_times),
                'consistency': self._calculate_output_consistency([r[0] for r in gpt_oss_results])
            }
        
        # Kanana analysis
        if kanana_results:
            kanana_confidences = [r[1].confidence for r in kanana_results]
            kanana_times = [r[1].processing_time for r in kanana_results]
            
            analysis['kanana'] = {
                'average_confidence': np.mean(kanana_confidences),
                'confidence_std': np.std(kanana_confidences),
                'average_time': np.mean(kanana_times),
                'consistency': self._calculate_output_consistency([r[0] for r in kanana_results])
            }
        
        # Cross-architecture comparison
        if gpt_oss_results and kanana_results:
            analysis['cross_comparison'] = {
                'confidence_difference': abs(analysis['gpt_oss']['average_confidence'] - 
                                           analysis['kanana']['average_confidence']),
                'time_ratio': analysis['gpt_oss']['average_time'] / max(analysis['kanana']['average_time'], 0.1),
                'complementarity': self._calculate_model_complementarity(gpt_oss_results, kanana_results)
            }
        
        return analysis
    
    def _calculate_output_consistency(self, outputs: List[np.ndarray]) -> float:
        """Calculate consistency among outputs from same model type"""
        
        if len(outputs) < 2:
            return 1.0
        
        consistencies = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if outputs[i].shape == outputs[j].shape:
                    consistency = np.mean(outputs[i] == outputs[j])
                    consistencies.append(consistency)
        
        return np.mean(consistencies) if consistencies else 1.0
    
    def _calculate_model_complementarity(self, 
                                       gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                       kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> float:
        """Calculate how well different model types complement each other"""
        
        complementarity_scores = []
        
        for gpt_output, gpt_response in gpt_oss_results:
            for kanana_output, kanana_response in kanana_results:
                if gpt_output.shape == kanana_output.shape:
                    # Measure difference (diversity) weighted by confidence
                    diversity = 1.0 - np.mean(gpt_output == kanana_output)
                    confidence_product = gpt_response.confidence * kanana_response.confidence
                    
                    # Optimal complementarity: some diversity with high confidence
                    complementarity = diversity * confidence_product * 2.0  # Scale factor
                    complementarity_scores.append(min(complementarity, 1.0))
        
        return np.mean(complementarity_scores) if complementarity_scores else 0.5
    
    def _final_cross_architecture_analysis(self) -> Dict[str, Any]:
        """Final analysis of cross-architecture performance"""
        
        if not self.multi_llm_interaction_history:
            return {}
        
        # Aggregate cross-architecture data
        synergy_scores = [data['consensus_result']['synergy_score'] 
                         for data in self.multi_llm_interaction_history]
        
        agreement_scores = [data['consensus_result']['cross_model_agreement'] 
                           for data in self.multi_llm_interaction_history]
        
        gpt_contributions = [data['consensus_result']['model_contributions'].get('gpt_oss', 0)
                            for data in self.multi_llm_interaction_history]
        
        kanana_contributions = [data['consensus_result']['model_contributions'].get('kanana', 0)
                               for data in self.multi_llm_interaction_history]
        
        return {
            'average_synergy': np.mean(synergy_scores),
            'synergy_improvement': synergy_scores[-1] - synergy_scores[0] if len(synergy_scores) > 1 else 0,
            'average_agreement': np.mean(agreement_scores),
            'gpt_oss_average_contribution': np.mean(gpt_contributions),
            'kanana_average_contribution': np.mean(kanana_contributions),
            'model_balance': abs(np.mean(gpt_contributions) - np.mean(kanana_contributions)),
            'performance_stability': 1.0 - np.std(synergy_scores) if synergy_scores else 1.0
        }
    
    def _get_adaptive_weights_history(self) -> List[Dict[str, float]]:
        """Get history of adaptive weight changes"""
        
        return [data['consensus_result']['adaptive_weights'] 
                for data in self.multi_llm_interaction_history]
    
    def _calculate_synergy_trend(self) -> str:
        """Calculate overall synergy trend"""
        
        if len(self.synergy_evolution) < 3:
            return 'insufficient_data'
        
        recent_avg = np.mean(self.synergy_evolution[-3:])
        early_avg = np.mean(self.synergy_evolution[:3])
        
        if recent_avg > early_avg + 0.1:
            return 'improving'
        elif recent_avg < early_avg - 0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _calculate_model_utilization(self, model_type: str) -> float:
        """Calculate utilization rate for specific model type"""
        
        if not self.multi_llm_interaction_history:
            return 0.0
        
        total_utilization = 0.0
        for data in self.multi_llm_interaction_history:
            contribution = data['consensus_result']['model_contributions'].get(model_type, 0)
            total_utilization += contribution
        
        return total_utilization / len(self.multi_llm_interaction_history)

class CrossModelVerifier:
    """Cross-model verification system for multi-LLM results"""
    
    def __init__(self):
        self.verification_threshold = 0.75
        
    def verify_across_models(self, 
                           input_grid: np.ndarray,
                           consensus_output: np.ndarray,
                           gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                           kanana_results: List[Tuple[np.ndarray, ModelResponse]],
                           solver_history: List[str]) -> Dict[str, float]:
        """Comprehensive verification across multiple LLM models"""
        
        verification_results = {}
        
        # Individual model verification
        gpt_verification = self._verify_model_outputs(gpt_oss_results, consensus_output)
        kanana_verification = self._verify_model_outputs(kanana_results, consensus_output)
        
        # Cross-model consistency
        cross_consistency = self._verify_cross_model_consistency(gpt_oss_results, kanana_results)
        
        # Consensus quality
        consensus_quality = self._verify_consensus_quality(input_grid, consensus_output, solver_history)
        
        # Combined scores
        verification_results = {
            'gpt_oss_verification': gpt_verification,
            'kanana_verification': kanana_verification,
            'cross_model_consistency': cross_consistency,
            'consensus_quality': consensus_quality,
            'forward_score': consensus_quality * 0.8,
            'backward_score': cross_consistency * 0.9,
            'process_score': (gpt_verification + kanana_verification) / 2.0,
            'combined_score': (gpt_verification * 0.3 + 
                             kanana_verification * 0.3 + 
                             cross_consistency * 0.2 + 
                             consensus_quality * 0.2)
        }
        
        return verification_results
    
    def _verify_model_outputs(self, 
                            model_results: List[Tuple[np.ndarray, ModelResponse]],
                            consensus_output: np.ndarray) -> float:
        """Verify outputs from a specific model type"""
        
        if not model_results:
            return 0.5
        
        similarities = []
        confidence_weights = []
        
        for output, response in model_results:
            if output.shape == consensus_output.shape:
                similarity = np.mean(output == consensus_output)
                similarities.append(similarity)
                confidence_weights.append(response.confidence)
        
        if similarities:
            # Weighted average by confidence
            weighted_similarity = np.average(similarities, weights=confidence_weights)
            return weighted_similarity
        else:
            return 0.5
    
    def _verify_cross_model_consistency(self, 
                                      gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                      kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> float:
        """Verify consistency between different model architectures"""
        
        if not gpt_oss_results or not kanana_results:
            return 0.5
        
        consistency_scores = []
        
        for gpt_output, gpt_response in gpt_oss_results:
            for kanana_output, kanana_response in kanana_results:
                if gpt_output.shape == kanana_output.shape:
                    consistency = np.mean(gpt_output == kanana_output)
                    # Weight by confidence product
                    weight = gpt_response.confidence * kanana_response.confidence
                    consistency_scores.append(consistency * weight)
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _verify_consensus_quality(self, 
                                input_grid: np.ndarray,
                                consensus_output: np.ndarray,
                                solver_history: List[str]) -> float:
        """Verify overall quality of consensus solution"""
        
        quality_factors = []
        
        # Basic transformation validity
        if input_grid.shape == consensus_output.shape:
            # Check if output is reasonable transformation of input
            difference_ratio = np.mean(input_grid != consensus_output)
            # Good transformations change 20-80% of grid
            if 0.2 <= difference_ratio <= 0.8:
                quality_factors.append(0.8)
            else:
                quality_factors.append(0.4)
        else:
            quality_factors.append(0.6)  # Different shapes can still be valid
        
        # Solver sequence quality
        if solver_history:
            unique_solvers = len(set(solver_history))
            total_solvers = len(solver_history)
            # Diversity in solver usage is generally good
            diversity_score = min(1.0, unique_solvers / max(total_solvers, 1) * 2)
            quality_factors.append(diversity_score)
        else:
            quality_factors.append(0.5)
        
        # Grid structure quality (non-trivial patterns)
        output_variety = len(np.unique(consensus_output))
        if output_variety >= 2:  # At least background + 1 color
            quality_factors.append(0.7)
        else:
            quality_factors.append(0.3)
        
        return np.mean(quality_factors)

# Factory function
def create_multi_llm_arc_system(traditional_solver_classes: List[type],
                               gpt_oss_endpoint: str,
                               kanana_endpoint: str,
                               gpt_oss_key: str = None,
                               kanana_key: str = None,
                               gpt_oss_temp: float = 0.3,
                               kanana_temp: float = 0.2) -> MultiLLMEnhancedEFESystem:
    """Factory function to create multi-LLM enhanced ARC system"""
    
    # Create traditional solvers
    traditional_solvers = [solver_class() for solver_class in traditional_solver_classes]
    
    # Configure models
    gpt_oss_config = ModelConfig(
        model_name="gpt-oss-20b",
        api_endpoint=gpt_oss_endpoint,
        api_key=gpt_oss_key,
        temperature=gpt_oss_temp
    )
    
    kanana_config = ModelConfig(
        model_name="kanana-1.5-15.7b-a3b",
        api_endpoint=kanana_endpoint,
        api_key=kanana_key,
        temperature=kanana_temp
    )
    
    return MultiLLMEnhancedEFESystem(
        traditional_solvers=traditional_solvers,
        gpt_oss_config=gpt_oss_config,
        kanana_config=kanana_config
    )

if __name__ == "__main__":
    print("Multi-LLM ARC-IDEA Integration")
    print("=" * 40)
    print("Enhanced system with:")
    print("- GPT-OSS-20B: Large-scale reasoning")
    print("- Kanana-1.5-15.7B-A3B: Analytical precision")
    print("- Cross-architecture consensus")
    print("- Adaptive model weighting")
    print("- Comprehensive synergy analysis")
    print("\\nUsage:")
    print("system = create_multi_llm_arc_system(solvers, gpt_endpoint, kanana_endpoint)")
    print("result = system.solve_with_multi_llm_ensemble(grid, constraints)")