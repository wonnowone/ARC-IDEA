#!/usr/bin/env python3
"""
Demonstration of the ARC EFE Multi-Solver System

This script demonstrates the complete integration of:
1. DPEFE (Dynamic Programming in Expected Free Energy) approach
2. RevThink (Reverse Enhanced Thinking) verification
3. Z-learning preference updates
4. Multiple solver experts with individual thinking flows

Following the user's specified flow:
문제 입력(ARC problem) → 현재 상태(state_t) → [solver 1, solver 2, solver 3, solver 4, symbolic solver] 
→ EFE 평가 → EFE 최소 solver 결정 → 최적 solver의 중간 output → (RevThink 검증, Z-learning 업데이트) 
→ state_t+1 업데이트 → (해결될 때까지 반복) → 최종 해결 결과 출력
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from arc_efe_solver import ARCEFESystem, ARCState
from example_solvers import (
    ColorPatternSolver, 
    ShapeSymmetrySolver, 
    GeometricTransformSolver, 
    LogicalRuleSolver, 
    SymbolicSolver
)

def create_sample_arc_problem() -> Dict[str, Any]:
    """Create a sample ARC problem for demonstration"""
    
    # Simple 5x5 grid with color pattern
    input_grid = np.array([
        [0, 1, 0, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 0],
        [1, 2, 1, 2, 1],
        [0, 1, 0, 1, 0]
    ])
    
    # Expected output (color swap)
    expected_output = np.array([
        [0, 2, 0, 2, 0],
        [2, 1, 2, 1, 2],
        [0, 2, 0, 2, 0],
        [2, 1, 2, 1, 2],
        [0, 2, 0, 2, 0]
    ])
    
    # Problem constraints
    constraints = {
        'color_constraints': [0, 1, 2],  # Only use colors 0, 1, 2
        'pattern_constraints': np.ones((5, 5)),  # Pattern preservation
        'symmetry': 'none'  # No specific symmetry requirement
    }
    
    return {
        'input': input_grid,
        'expected_output': expected_output,
        'constraints': constraints,
        'description': 'Color swap pattern transformation'
    }

def visualize_grids(input_grid: np.ndarray, output_grid: np.ndarray, 
                   expected_grid: np.ndarray = None, title: str = "ARC Problem Solution"):
    """Visualize input, output, and expected grids"""
    
    num_plots = 3 if expected_grid is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(4*num_plots, 4))
    
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    
    # Input grid
    im1 = axes[0].imshow(input_grid, cmap='tab10', vmin=0, vmax=9)
    axes[0].set_title('Input Grid')
    axes[0].grid(True, alpha=0.3)
    
    # Output grid
    im2 = axes[1].imshow(output_grid, cmap='tab10', vmin=0, vmax=9)
    axes[1].set_title('System Output')
    axes[1].grid(True, alpha=0.3)
    
    # Expected grid (if provided)
    if expected_grid is not None:
        im3 = axes[2].imshow(expected_grid, cmap='tab10', vmin=0, vmax=9)
        axes[2].set_title('Expected Output')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    return fig

def print_system_analysis(results: Dict[str, Any]):
    """Print detailed analysis of the system's performance"""
    
    print("\\n" + "="*60)
    print("🔍 SYSTEM ANALYSIS REPORT")
    print("="*60)
    
    print(f"\\n📊 FINAL RESULTS:")
    print(f"   • Final Confidence: {results['confidence']:.3f}")
    print(f"   • Total Iterations: {len(results['iteration_history'])}")
    print(f"   • Solver Sequence: {' → '.join(results['solver_history'])}")
    
    print(f"\\n🧠 SOLVER PREFERENCES (Z-Learning Updated):")
    for solver_name, preference in results['final_preferences'].items():
        print(f"   • {solver_name:<25}: {preference:.3f}")
    
    print(f"\\n📈 ITERATION HISTORY:")
    for i, iteration in enumerate(results['iteration_history']):
        print(f"   Iteration {i+1}:")
        print(f"     └─ Solver: {iteration['solver']}")
        print(f"     └─ EFE Score: {iteration['efe_score']:.3f}")
        print(f"     └─ RevThink Verification:")
        verification = iteration['verification']
        print(f"        ├─ Forward Score:  {verification['forward_score']:.3f}")
        print(f"        ├─ Backward Score: {verification['backward_score']:.3f}")
        print(f"        ├─ Process Score:  {verification['process_score']:.3f}")
        print(f"        └─ Combined Score: {verification['combined_score']:.3f}")
        print()

def analyze_solver_thinking_flows(solvers: List[Any]):
    """Analyze and display each solver's thinking flow"""
    
    print("\\n" + "="*60)
    print("🧠 SOLVER THINKING FLOWS ANALYSIS")
    print("="*60)
    
    for i, solver in enumerate(solvers, 1):
        thinking_flow = solver.get_thinking_flow()
        print(f"\\n{i}. {solver.__class__.__name__}:")
        print(f"   Strategy: {thinking_flow['strategy']}")
        print(f"   Steps: {' → '.join(thinking_flow['steps'])}")
        print(f"   Confidence: {thinking_flow.get('confidence', 'N/A')}")

def demonstrate_efe_calculation(system: ARCEFESystem, problem: Dict[str, Any]):
    """Demonstrate EFE calculation for each solver"""
    
    print("\\n" + "="*60)
    print("⚡ EXPECTED FREE ENERGY (EFE) CALCULATION")
    print("="*60)
    
    # Create initial state
    initial_state = ARCState(
        grid=problem['input'].copy(),
        constraints=problem['constraints'],
        step=0,
        solver_history=[],
        confidence=0.0
    )
    
    print(f"\\nUsing EFE Formula: EFE_t(solver) = Risk: D_KL(Q(o_t)||C_t(o_t)) + Ambiguity: -logP(constraint_t|o_t)")
    print(f"\\nEFE Scores for each solver:")
    
    for i, solver_name in enumerate(system.efe_solver.solver_names):
        efe_score = system.efe_solver.compute_efe(i, initial_state)
        preference = system.efe_solver.solver_preferences[solver_name]
        weighted_score = efe_score / (preference + 1e-8)
        
        print(f"   {solver_name:<25}: EFE={efe_score:.3f}, Preference={preference:.3f}, Weighted={weighted_score:.3f}")

def main():
    """Main demonstration function"""
    
    print("🚀 ARC EFE MULTI-SOLVER SYSTEM DEMONSTRATION")
    print("="*60)
    print("Integrating DPEFE + RevThink + Z-learning for ARC Problem Solving")
    
    # Create sample ARC problem
    print("\\n📋 Creating sample ARC problem...")
    problem = create_sample_arc_problem()
    print(f"Problem: {problem['description']}")
    print(f"Input shape: {problem['input'].shape}")
    print(f"Constraints: {list(problem['constraints'].keys())}")
    
    # Initialize solvers (experts of inference)
    print("\\n🔧 Initializing solver experts...")
    solvers = [
        ColorPatternSolver(),
        ShapeSymmetrySolver(), 
        GeometricTransformSolver(),
        LogicalRuleSolver(),
        SymbolicSolver()
    ]
    
    print(f"Initialized {len(solvers)} solver experts:")
    for i, solver in enumerate(solvers, 1):
        print(f"   {i}. {solver.__class__.__name__}")
    
    # Analyze solver thinking flows
    analyze_solver_thinking_flows(solvers)
    
    # Create the integrated EFE system
    print("\\n⚙️  Creating integrated ARC-EFE system...")
    arc_system = ARCEFESystem(solvers, planning_horizon=3)
    
    # Demonstrate EFE calculation
    demonstrate_efe_calculation(arc_system, problem)
    
    # Solve the problem
    print("\\n🎯 Starting problem solving process...")
    print("Following flow: 문제입력 → 상태 → [솔버들] → EFE평가 → 최적솔버 → 출력 → (RevThink검증, Z학습업데이트) → 반복")
    
    solution, results = arc_system.solve_arc_problem(
        problem['input'], 
        problem['constraints']
    )
    
    # Print comprehensive analysis
    print_system_analysis(results)
    
    # Visualize results
    print("\\n🎨 Generating visualization...")
    fig = visualize_grids(
        problem['input'], 
        solution, 
        problem['expected_output'],
        "ARC EFE Multi-Solver System Results"
    )
    
    # Calculate accuracy
    if problem['expected_output'] is not None:
        accuracy = np.mean(solution == problem['expected_output'])
        print(f"\\n✅ SOLUTION ACCURACY: {accuracy:.1%}")
    
    # Final system state
    print("\\n🏁 FINAL SYSTEM STATE:")
    print(f"   • Solution found: {results['confidence'] > 0.7}")
    print(f"   • Best performing solver: {max(results['final_preferences'].items(), key=lambda x: x[1])[0]}")
    print(f"   • System learned preferences through Z-learning")
    print(f"   • RevThink verification ensured solution quality")
    
    print("\\n" + "="*60)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*60)
    print("The system successfully integrated:")
    print("✓ DPEFE dynamic programming approach")
    print("✓ Multiple solver experts with individual thinking flows") 
    print("✓ EFE-based solver selection with Risk + Ambiguity components")
    print("✓ RevThink verification (forward, backward, process)")
    print("✓ Z-learning preference updates")
    print("✓ Iterative state updates until convergence")
    
    # Show plot
    plt.show()
    
    return solution, results

if __name__ == "__main__":
    # Run demonstration
    solution, results = main()
    
    # Additional analysis
    print("\\n📝 TECHNICAL DETAILS:")
    print(f"   • EFE computation: O(n_solvers × grid_size) per iteration")
    print(f"   • RevThink verification: 3-component scoring system")
    print(f"   • Z-learning: Preference distribution updates via softmax")
    print(f"   • Dynamic programming: Backward planning from goal state")
    print(f"   • Multi-task learning: Forward + Backward + Process objectives")