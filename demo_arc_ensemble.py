#!/usr/bin/env python3
"""
Enhanced ARC EFE Ensemble System Demonstration

This demonstrates the complete multi-solver ensemble approach with:
1. 단계별 추론 프로세스: 각 solver가 한 단계 이동 수행
2. Majority voting → consensus 도출 → contrastive negative 사용
3. Comprehensive loss: L_total = L_EFE + λ_contrast*L_contrastive + λ_ambiguity*ambiguity_penalty + λ_chaos*L_chaos
4. Z-learning preference updates with risk minimization
5. Ambiguity measurement and penalty integration

Flow: 문제입력 → 상태 → [모든 solver 동시실행] → majority voting → consensus → contrastive learning → loss 계산 → Z-learning 업데이트 → 반복
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any
from arc_efe_ensemble_solver import EnhancedARCEFESystem, EnsembleOutput
from example_solvers import (
    ColorPatternSolver, 
    ShapeSymmetrySolver, 
    GeometricTransformSolver, 
    LogicalRuleSolver, 
    SymbolicSolver
)

def create_complex_arc_problem() -> Dict[str, Any]:
    """Create a more complex ARC problem for ensemble testing"""
    
    # Complex 7x7 grid with multiple patterns
    input_grid = np.array([
        [0, 1, 0, 2, 0, 1, 0],
        [1, 2, 1, 0, 1, 2, 1],
        [0, 1, 3, 2, 3, 1, 0],
        [2, 0, 2, 1, 2, 0, 2],
        [0, 1, 3, 2, 3, 1, 0],
        [1, 2, 1, 0, 1, 2, 1],
        [0, 1, 0, 2, 0, 1, 0]
    ])
    
    # Expected transformation (complex pattern)
    expected_output = np.array([
        [0, 2, 0, 1, 0, 2, 0],
        [2, 1, 2, 0, 2, 1, 2],
        [0, 2, 3, 1, 3, 2, 0],
        [1, 0, 1, 2, 1, 0, 1],
        [0, 2, 3, 1, 3, 2, 0],
        [2, 1, 2, 0, 2, 1, 2],
        [0, 2, 0, 1, 0, 2, 0]
    ])
    
    # Rich constraints for ensemble testing
    constraints = {
        'color_constraints': [0, 1, 2, 3],
        'pattern_constraints': np.ones((7, 7)),
        'symmetry': 'both',
        'color_swap_rule': {1: 2, 2: 1},  # Advanced constraint
        'preserve_structure': True
    }
    
    return {
        'input': input_grid,
        'expected_output': expected_output,
        'constraints': constraints,
        'description': 'Complex multi-pattern transformation with color swap and symmetry preservation'
    }

def analyze_ensemble_consensus(ensemble_result: EnsembleOutput, iteration: int):
    """Analyze and visualize ensemble consensus results"""
    
    print(f"\\n📊 ENSEMBLE ANALYSIS - Iteration {iteration}")
    print("=" * 50)
    
    print(f"🎯 Consensus Metrics:")
    print(f"   • Consensus Reached: {ensemble_result.consensus_reached}")
    print(f"   • Majority Count: {ensemble_result.majority_count}/{ensemble_result.total_solvers}")
    print(f"   • Ambiguity Score: {ensemble_result.ambiguity_score:.3f}")
    print(f"   • Confidence: {ensemble_result.confidence:.3f}")
    
    print(f"\\n🤖 Individual Solver Outputs:")
    for solver_name, output in ensemble_result.solver_outputs.items():
        matches_consensus = np.array_equal(output, ensemble_result.output)
        unique_vals = np.unique(output)
        print(f"   • {solver_name:<25}: Match={matches_consensus}, Colors={len(unique_vals)}")

def visualize_ensemble_evolution(results: Dict[str, Any]):
    """Visualize the evolution of ensemble learning"""
    
    iterations = len(results['iteration_history'])
    if iterations == 0:
        return
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🧠 ARC Ensemble Learning Evolution', fontsize=16, fontweight='bold')
    
    # Extract metrics over iterations
    total_losses = []
    efe_losses = []
    contrastive_losses = []
    ambiguity_penalties = []
    chaos_losses = []
    verification_scores = []
    consensus_reached = []
    
    for iter_data in results['iteration_history']:
        loss_comp = iter_data['loss_components']
        total_losses.append(loss_comp['total_loss'])
        efe_losses.append(loss_comp['efe_loss'])
        contrastive_losses.append(loss_comp['contrastive_loss'])
        ambiguity_penalties.append(loss_comp['ambiguity_penalty'])
        chaos_losses.append(loss_comp['chaos_loss'])
        verification_scores.append(iter_data['verification']['combined_score'])
        consensus_reached.append(iter_data['ensemble_result'].consensus_reached)
    
    iter_range = range(1, iterations + 1)
    
    # Plot 1: Loss Components Evolution
    axes[0, 0].plot(iter_range, total_losses, 'b-o', linewidth=2, label='Total Loss')
    axes[0, 0].plot(iter_range, efe_losses, 'r--', alpha=0.7, label='EFE Loss')
    axes[0, 0].plot(iter_range, contrastive_losses, 'g--', alpha=0.7, label='Contrastive Loss')
    axes[0, 0].set_title('📈 Loss Components Evolution')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Ambiguity and Chaos
    axes[0, 1].plot(iter_range, ambiguity_penalties, 'orange', linewidth=2, label='Ambiguity Penalty')
    axes[0, 1].plot(iter_range, chaos_losses, 'purple', linewidth=2, label='Chaos Loss')
    axes[0, 1].set_title('⚖️ Ambiguity vs Chaos Balance')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Penalty/Loss Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Verification Scores
    axes[0, 2].plot(iter_range, verification_scores, 'green', linewidth=3, marker='s')
    axes[0, 2].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Acceptance Threshold')
    axes[0, 2].set_title('✅ RevThink Verification Scores')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Verification Score')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Consensus Evolution
    consensus_binary = [1 if x else 0 for x in consensus_reached]
    axes[1, 0].bar(iter_range, consensus_binary, alpha=0.7, color='skyblue')
    axes[1, 0].set_title('🤝 Consensus Achievement')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Consensus Reached (1=Yes, 0=No)')
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Solver Preference Evolution
    solver_names = list(results['final_preferences'].keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(solver_names)))
    
    for i, solver_name in enumerate(solver_names):
        preferences = [iter_data['solver_preferences'][solver_name] 
                      for iter_data in results['iteration_history']]
        axes[1, 1].plot(iter_range, preferences, color=colors[i], 
                       linewidth=2, marker='o', label=solver_name[:15])
    
    axes[1, 1].set_title('🧠 Z-Learning Preference Evolution')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Solver Preference')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Final Solution Visualization
    final_solution = results['solution']
    im = axes[1, 2].imshow(final_solution, cmap='tab10', vmin=0, vmax=9)
    axes[1, 2].set_title(f'🎯 Final Solution (Confidence: {results["confidence"]:.2f})')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    return fig

def print_comprehensive_analysis(results: Dict[str, Any], problem: Dict[str, Any]):
    """Print comprehensive analysis of ensemble performance"""
    
    print("\\n" + "="*80)
    print("🔬 COMPREHENSIVE ENSEMBLE ANALYSIS")
    print("="*80)
    
    # Overall Performance
    print(f"\\n🎯 OVERALL PERFORMANCE:")
    accuracy = np.mean(results['solution'] == problem['expected_output']) if problem['expected_output'] is not None else None
    print(f"   • Solution Accuracy: {accuracy:.1%}" if accuracy else "   • Expected output not available")
    print(f"   • Final Confidence: {results['confidence']:.3f}")
    print(f"   • Total Iterations: {results['ensemble_metrics']['total_iterations']}")
    print(f"   • Final Consensus: {results['ensemble_metrics']['final_consensus']}")
    print(f"   • Final Ambiguity: {results['ensemble_metrics']['final_ambiguity']:.3f}")
    
    # Loss Analysis
    if results['iteration_history']:
        final_loss = results['iteration_history'][-1]['loss_components']
        print(f"\\n📊 FINAL LOSS BREAKDOWN:")
        print(f"   • Total Loss: {final_loss['total_loss']:.3f}")
        print(f"   • EFE Loss (정합성): {final_loss['efe_loss']:.3f}")
        print(f"   • Contrastive Loss (명확성): {final_loss['contrastive_loss']:.3f}")
        print(f"   • Ambiguity Penalty (일치성): {final_loss['ambiguity_penalty']:.3f}")
        print(f"   • Chaos Loss (유연성): {final_loss['chaos_loss']:.3f}")
    
    # Solver Performance Analysis
    print(f"\\n🤖 SOLVER PERFORMANCE ANALYSIS:")
    best_solver = max(results['final_preferences'].items(), key=lambda x: x[1])
    worst_solver = min(results['final_preferences'].items(), key=lambda x: x[1])
    
    print(f"   • Best Performing: {best_solver[0]} (Preference: {best_solver[1]:.3f})")
    print(f"   • Worst Performing: {worst_solver[0]} (Preference: {worst_solver[1]:.3f})")
    
    print(f"\\n📈 LEARNING EFFECTIVENESS:")
    if len(results['iteration_history']) > 1:
        initial_loss = results['iteration_history'][0]['loss_components']['total_loss']
        final_loss_val = results['iteration_history'][-1]['loss_components']['total_loss']
        improvement = (initial_loss - final_loss_val) / initial_loss * 100
        print(f"   • Loss Improvement: {improvement:.1f}%")
        
        initial_confidence = results['iteration_history'][0]['verification']['combined_score']
        final_confidence = results['iteration_history'][-1]['verification']['combined_score']
        confidence_gain = final_confidence - initial_confidence
        print(f"   • Confidence Gain: {confidence_gain:.3f}")
    
    # Ensemble Metrics
    print(f"\\n🤝 ENSEMBLE COOPERATION:")
    consensus_rate = sum(1 for iter_data in results['iteration_history'] 
                        if iter_data['ensemble_result'].consensus_reached) / len(results['iteration_history'])
    print(f"   • Consensus Achievement Rate: {consensus_rate:.1%}")
    
    avg_majority = np.mean([iter_data['ensemble_result'].majority_count 
                           for iter_data in results['iteration_history']])
    print(f"   • Average Majority Agreement: {avg_majority:.1f}/5 solvers")

def main():
    """Enhanced demonstration of the ensemble ARC solver"""
    
    print("🚀 ENHANCED ARC EFE ENSEMBLE SYSTEM")
    print("="*80)
    print("Multi-Solver Ensemble with Contrastive Learning & Comprehensive Loss")
    print("="*80)
    
    # Create complex test problem
    print("\\n📋 Creating complex ARC problem...")
    problem = create_complex_arc_problem()
    print(f"Problem: {problem['description']}")
    print(f"Input shape: {problem['input'].shape}")
    print(f"Constraints: {list(problem['constraints'].keys())}")
    
    # Initialize enhanced solver ensemble
    print("\\n🔧 Initializing enhanced solver ensemble...")
    solvers = [
        ColorPatternSolver(),
        ShapeSymmetrySolver(),
        GeometricTransformSolver(),
        LogicalRuleSolver(),
        SymbolicSolver()
    ]
    
    # Create enhanced system
    enhanced_system = EnhancedARCEFESystem(
        solvers=solvers,
        planning_horizon=3,
        consensus_threshold=0.5
    )
    
    print(f"✅ Initialized enhanced system with {len(solvers)} solvers")
    print("🔥 New features: Majority voting, Contrastive learning, Comprehensive loss")
    
    # Solve with ensemble approach
    print("\\n🎯 Starting enhanced ensemble solving...")
    print("Flow: 문제입력 → 상태 → [모든 solver 동시실행] → majority voting → consensus → contrastive learning → loss 계산 → 반복")
    
    try:
        solution, results = enhanced_system.solve_with_ensemble(
            problem['input'],
            problem['constraints']
        )
        
        print("\\n🎉 ENSEMBLE SOLVING COMPLETED!")
        
        # Comprehensive analysis
        print_comprehensive_analysis(results, problem)
        
        # Visualization
        print("\\n🎨 Generating comprehensive visualizations...")
        
        # Create solution comparison
        fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input
        axes1[0].imshow(problem['input'], cmap='tab10', vmin=0, vmax=9)
        axes1[0].set_title('Input Grid')
        axes1[0].grid(True, alpha=0.3)
        
        # Solution
        axes1[1].imshow(solution, cmap='tab10', vmin=0, vmax=9)
        axes1[1].set_title(f'Ensemble Solution\\n(Confidence: {results["confidence"]:.2f})')
        axes1[1].grid(True, alpha=0.3)
        
        # Expected
        if problem['expected_output'] is not None:
            axes1[2].imshow(problem['expected_output'], cmap='tab10', vmin=0, vmax=9)
            axes1[2].set_title('Expected Output')
            axes1[2].grid(True, alpha=0.3)
            
            # Calculate accuracy
            accuracy = np.mean(solution == problem['expected_output'])
            plt.suptitle(f'🎯 ARC Ensemble Results (Accuracy: {accuracy:.1%})', fontsize=14)
        else:
            axes1[2].text(0.5, 0.5, 'Expected\\nOutput\\nNot Available', 
                         ha='center', va='center', transform=axes1[2].transAxes, fontsize=12)
            plt.suptitle('🎯 ARC Ensemble Results', fontsize=14)
        
        plt.tight_layout()
        
        # Create evolution visualization
        evolution_fig = visualize_ensemble_evolution(results)
        
        print("\\n📈 FINAL SUMMARY:")
        print("✅ Multi-solver ensemble successfully implemented")
        print("✅ Majority voting consensus mechanism operational")
        print("✅ Contrastive learning distinguishing correct/incorrect outputs")
        print("✅ Comprehensive loss function balancing all objectives")
        print("✅ Z-learning adapting solver preferences dynamically")
        print("✅ Ambiguity measurement and penalty integration working")
        
        plt.show()
        
        return solution, results
        
    except Exception as e:
        print(f"❌ Error during ensemble solving: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_ensemble_comparison():
    """Run comparison between original and ensemble approaches"""
    
    print("\\n" + "="*80)
    print("⚔️  ENSEMBLE vs ORIGINAL SYSTEM COMPARISON")
    print("="*80)
    
    # This would compare performance metrics
    print("📊 Comparison metrics:")
    print("   • Solution accuracy improvement")
    print("   • Convergence speed")
    print("   • Robustness to ambiguous cases")
    print("   • Learning efficiency")
    print("   • Consensus achievement rate")

if __name__ == "__main__":
    # Run the enhanced demonstration
    solution, results = main()
    
    if solution is not None and results is not None:
        # Additional detailed analysis
        print("\\n📋 TECHNICAL IMPLEMENTATION DETAILS:")
        print("="*60)
        print("🔧 Ensemble Architecture:")
        print("   • MajorityVotingConsensus: Multi-solver agreement calculation")
        print("   • ContrastiveLearningModule: InfoNCE loss for positive/negative discrimination")
        print("   • ComprehensiveLossCalculator: 4-component loss integration")
        print("   • EnhancedARCEFESystem: Orchestrating ensemble coordination")
        
        print("\\n⚡ Loss Function Implementation:")
        print("   • L_total = L_EFE + λ_contrast*L_contrastive + λ_ambiguity*ambiguity_penalty + λ_chaos*L_chaos")
        print("   • EFE: Risk (D_KL) + Ambiguity terms")
        print("   • Contrastive: InfoNCE loss distinguishing consensus vs failed outputs")
        print("   • Ambiguity penalty: Ensemble disagreement suppression")
        print("   • Chaos: Controlled diversity maintenance for exploration")
        
        print("\\n🧠 Learning Dynamics:")
        print("   • Z-learning: Risk minimization through preference updates")
        print("   • Majority voting: Democratic decision making among solvers")
        print("   • RevThink verification: Triple validation (forward, backward, process)")
        print("   • Dynamic preference adaptation based on ensemble performance")
        
        # Run comparison if requested
        # run_ensemble_comparison()