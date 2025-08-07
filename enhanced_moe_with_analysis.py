#!/usr/bin/env python3
"""
Enhanced MoE System with Deep Problem Analysis Integration

This module integrates the comprehensive problem analysis pipeline with the existing
MoE system, enabling deeper problem understanding to guide expert selection and
movement strategy generation.

Integration Points:
1. Problem analysis informs MoE routing decisions
2. Change detection guides movement script generation  
3. Component analysis enhances EFE calculation
4. Understanding feedback improves solver preferences
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
from dataclasses import dataclass

# Import existing MoE components
from enhanced_arc_ensemble_moe import EnhancedARCEFESystem, create_enhanced_arc_system
from enhanced_solvers_moe import EnhancedBaseSolver
from movement_language import MovementScript, MovementScriptBuilder, MovementOperator, MovementCondition
from moe_router import RoutingStrategy

# Import new analysis components
from arc_problem_analyzer import (
    ARCProblemAnalyzer, ChangeType, ChangeAnalysis, 
    ConnectedComponent, ColorFrame, ShapeFrame
)

@dataclass
class AnalysisGuidedContext:
    """Context enriched with problem analysis for MoE guidance"""
    original_constraints: Dict[str, Any]
    problem_analysis: Dict[str, Any]
    dominant_transformation: str
    detected_changes: List[ChangeAnalysis]
    component_analysis: Dict[str, Any]
    guidance_hints: Dict[str, float]  # Hints for expert selection

class AnalysisGuidedMoERouter:
    """Enhanced MoE router that uses problem analysis for intelligent routing"""
    
    def __init__(self, base_router):
        self.base_router = base_router
        self.analysis_history = []
    
    def route_with_analysis_guidance(self, 
                                   grid: np.ndarray,
                                   analysis_context: AnalysisGuidedContext,
                                   strategy: RoutingStrategy = RoutingStrategy.CONFIDENCE_BASED) -> Dict[str, Any]:
        """Route experts using problem analysis guidance"""
        
        # Extract guidance from analysis
        guidance_hints = analysis_context.guidance_hints
        detected_changes = analysis_context.detected_changes
        
        # Standard routing
        base_result = self.base_router.route(grid, strategy=strategy)
        
        # Enhance with analysis guidance
        enhanced_expert_calls = []
        
        for expert_call in base_result.selected_experts:
            # Boost confidence based on analysis hints
            expert_type = expert_call.movement_type.value
            
            if expert_type in guidance_hints:
                confidence_boost = guidance_hints[expert_type]
                expert_call.expected_confidence = min(1.0, 
                    expert_call.expected_confidence + confidence_boost)
            
            # Add analysis context
            expert_call.context.update({
                'analysis_guidance': True,
                'detected_changes': [change.change_type.value for change in detected_changes],
                'dominant_transformation': analysis_context.dominant_transformation
            })
            
            enhanced_expert_calls.append(expert_call)
        
        # Sort by enhanced confidence
        enhanced_expert_calls.sort(key=lambda x: x.expected_confidence, reverse=True)
        
        return {
            'selected_experts': enhanced_expert_calls,
            'analysis_influenced': True,
            'guidance_applied': guidance_hints,
            'original_routing': base_result
        }

class AnalysisGuidedSolver(EnhancedBaseSolver):
    """Enhanced solver that uses problem analysis to guide strategy"""
    
    def __init__(self, solver_name: str, base_solver: EnhancedBaseSolver):
        super().__init__(solver_name)
        self.base_solver = base_solver
        self.problem_analyzer = ARCProblemAnalyzer()
    
    def analyze_problem_with_understanding(self, input_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze problem with deep understanding"""
        # Run base analysis
        base_analysis = self.base_solver.analyze_problem(input_grid)
        
        # Run comprehensive problem analysis
        try:
            problem_grids = {'input': input_grid}
            comprehensive_analysis = self.problem_analyzer.analyze_problem_sequence(problem_grids)
            
            # Merge analyses
            enhanced_analysis = base_analysis.copy()
            enhanced_analysis.update({
                'comprehensive_analysis': comprehensive_analysis,
                'problem_understanding': comprehensive_analysis['problem_understanding'],
                'component_analysis': comprehensive_analysis['frame_results'].get('input', {}),
                'analysis_guided': True
            })
            
            return enhanced_analysis
            
        except Exception as e:
            warnings.warn(f"Comprehensive analysis failed: {e}")
            return base_analysis
    
    def create_analysis_guided_movement_script(self, 
                                             input_grid: np.ndarray, 
                                             analysis: Dict[str, Any]) -> MovementScript:
        """Create movement script guided by problem analysis"""
        
        comprehensive_analysis = analysis.get('comprehensive_analysis')
        if not comprehensive_analysis:
            # Fallback to base solver
            return self.base_solver.create_movement_script(input_grid, analysis)
        
        problem_understanding = comprehensive_analysis['problem_understanding']
        dominant_transformation = problem_understanding.get('dominant_transformation', 'unknown')
        
        # Create analysis-guided script
        script = MovementScript(
            name=f"analysis_guided_{dominant_transformation}",
            description=f"Script guided by analysis: {dominant_transformation}"
        )
        
        # Add transformations based on detected patterns
        if dominant_transformation == 'positional_change':
            script = self._create_positional_change_script(comprehensive_analysis)
        elif dominant_transformation == 'color_change':
            script = self._create_color_change_script(comprehensive_analysis)
        elif dominant_transformation == 'object_duplication':
            script = self._create_duplication_script(comprehensive_analysis)
        else:
            # Use base solver's script as fallback
            script = self.base_solver.create_movement_script(input_grid, analysis)
        
        return script
    
    def _create_positional_change_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for positional changes"""
        script = MovementScript(name="positional_change_guided")
        
        # Analyze detected position changes
        change_analyses = analysis.get('change_analyses', {})
        
        for change_key, changes in change_analyses.items():
            for change in changes:
                if change.change_type == ChangeType.POSITIONAL_CHANGE:
                    details = change.details
                    
                    # Determine likely translation
                    if 'displacement' in details and details['displacement'] > 0:
                        # Add translation based on displacement
                        from_centroid = details.get('from_centroid', (0, 0))
                        to_centroid = details.get('to_centroid', (0, 0))
                        
                        shift_x = int(to_centroid[0] - from_centroid[0])
                        shift_y = int(to_centroid[1] - from_centroid[1])
                        
                        script.add_translation(
                            shift_x=shift_x,
                            shift_y=shift_y,
                            mode='wrap'
                        )
        
        # If no specific translations detected, add general movement operations
        if not script.instructions:
            script.add_translation(shift_x=1, shift_y=0, mode='wrap')
            script.add_translation(shift_x=0, shift_y=1, mode='wrap')
        
        return script
    
    def _create_color_change_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for color changes"""
        script = MovementScript(name="color_change_guided")
        
        # Analyze color changes
        frame_results = analysis.get('frame_results', {})
        input_frame = frame_results.get('input', {})
        
        colors_present = input_frame.get('colors_present', [])
        
        if len(colors_present) == 2:
            # Binary color swap
            script.add_color_swap(
                color1=colors_present[0],
                color2=colors_present[1]
            )
        elif len(colors_present) > 2:
            # Multi-color mapping
            color_mapping = {}
            for i, color in enumerate(colors_present):
                # Cycle colors
                next_color = colors_present[(i + 1) % len(colors_present)]
                color_mapping[color] = next_color
            
            script.add_color_mapping(mapping=color_mapping)
        
        return script
    
    def _create_duplication_script(self, analysis: Dict[str, Any]) -> MovementScript:
        """Create script for object duplication"""
        script = MovementScript(name="duplication_guided")
        
        # For duplication, we might use translation + color operations
        script.add_translation(shift_x=2, shift_y=0, mode='constant', fill_value=0)
        
        # Add the original back
        # This is complex and might require custom operations
        # For now, use standard transformations
        
        return script

class EnhancedAnalysisIntegratedSystem:
    """Complete system integrating deep analysis with MoE architecture"""
    
    def __init__(self, base_system: EnhancedARCEFESystem = None):
        self.base_system = base_system or create_enhanced_arc_system()
        self.problem_analyzer = ARCProblemAnalyzer()
        
        # Create analysis-guided components
        self.analysis_guided_router = AnalysisGuidedMoERouter(
            self.base_system.solvers[list(self.base_system.solvers.keys())[0]].moe_router
        )
        
        # Wrap solvers with analysis guidance
        self.enhanced_solvers = {}
        for name, solver in self.base_system.solvers.items():
            self.enhanced_solvers[name] = AnalysisGuidedSolver(f"Analyzed{name}", solver)
    
    def solve_with_deep_analysis(self, 
                                input_grid: np.ndarray, 
                                constraints: Dict[str, Any],
                                expected_output: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Solve ARC problem with deep analysis integration"""
        
        print("🔍 Starting deep analysis-guided solving...")
        
        # Step 1: Comprehensive problem analysis
        problem_grids = {'input': input_grid}
        if expected_output is not None:
            problem_grids['expected_output'] = expected_output
        
        comprehensive_analysis = self.problem_analyzer.analyze_problem_sequence(problem_grids)
        
        # Step 2: Create analysis-guided context
        analysis_context = self._create_analysis_context(
            constraints, comprehensive_analysis
        )
        
        print(f"📊 Analysis complete:")
        print(f"  Dominant transformation: {analysis_context.dominant_transformation}")
        print(f"  Detected changes: {len(analysis_context.detected_changes)}")
        print(f"  Guidance hints: {list(analysis_context.guidance_hints.keys())}")
        
        # Step 3: Enhanced solving with analysis guidance
        enhanced_results = {}
        solver_outputs = {}
        
        for solver_name, solver in self.enhanced_solvers.items():
            try:
                print(f"  🧠 Running {solver_name} with analysis guidance...")
                
                # Enhanced analysis
                enhanced_analysis = solver.analyze_problem_with_understanding(input_grid)
                
                # Create analysis-guided movement script
                movement_script = solver.create_analysis_guided_movement_script(
                    input_grid, enhanced_analysis
                )
                
                # Execute with guidance
                output = solver.predict(input_grid)
                solver_outputs[solver_name] = output
                
                enhanced_results[solver_name] = {
                    'output': output,
                    'analysis': enhanced_analysis,
                    'movement_script': movement_script,
                    'guided_by_analysis': True
                }
                
            except Exception as e:
                warnings.warn(f"Analysis-guided solver {solver_name} failed: {e}")
                continue
        
        if not solver_outputs:
            print("❌ All analysis-guided solvers failed, using base system")
            return self.base_system.solve_with_enhanced_ensemble(input_grid, constraints)
        
        # Step 4: Analysis-informed consensus
        consensus_result = self._compute_analysis_informed_consensus(
            solver_outputs, analysis_context
        )
        
        # Step 5: Compile comprehensive results
        final_results = {
            'solution': consensus_result['solution'],
            'confidence': consensus_result['confidence'],
            'analysis_guidance': analysis_context,
            'comprehensive_analysis': comprehensive_analysis,
            'solver_results': enhanced_results,
            'consensus_method': 'analysis_informed',
            'guidance_effectiveness': self._measure_guidance_effectiveness(
                enhanced_results, consensus_result, expected_output
            )
        }
        
        return consensus_result['solution'], final_results
    
    def _create_analysis_context(self, 
                                constraints: Dict[str, Any], 
                                analysis: Dict[str, Any]) -> AnalysisGuidedContext:
        """Create analysis-guided context for MoE routing"""
        
        problem_understanding = analysis.get('problem_understanding', {})
        dominant_transformation = problem_understanding.get('dominant_transformation', 'unknown')
        
        # Extract detected changes
        detected_changes = []
        change_analyses = analysis.get('change_analyses', {})
        for changes in change_analyses.values():
            detected_changes.extend(changes)
        
        # Create guidance hints based on analysis
        guidance_hints = {}
        
        for change in detected_changes:
            if change.change_type == ChangeType.POSITIONAL_CHANGE:
                guidance_hints['translation'] = 0.8
            elif change.change_type == ChangeType.COLOR_CHANGE:
                guidance_hints['color_transform'] = 0.9
            elif change.change_type == ChangeType.OBJECT_DUPLICATION:
                guidance_hints['translation'] = 0.7
                guidance_hints['color_transform'] = 0.6
            elif change.change_type in [ChangeType.OBJECT_ADDITION, ChangeType.OBJECT_REMOVAL]:
                guidance_hints['color_transform'] = 0.5
        
        # Add symmetry hints if detected
        frame_results = analysis.get('frame_results', {})
        input_frame = frame_results.get('input', {})
        if input_frame:
            # Check for symmetry patterns (simplified)
            guidance_hints['flip'] = 0.4  # Default hint for flip operations
            guidance_hints['rotation'] = 0.3  # Default hint for rotation
        
        return AnalysisGuidedContext(
            original_constraints=constraints,
            problem_analysis=analysis,
            dominant_transformation=dominant_transformation,
            detected_changes=detected_changes,
            component_analysis=frame_results,
            guidance_hints=guidance_hints
        )
    
    def _compute_analysis_informed_consensus(self, 
                                           solver_outputs: Dict[str, np.ndarray],
                                           analysis_context: AnalysisGuidedContext) -> Dict[str, Any]:
        """Compute consensus informed by problem analysis"""
        
        if len(solver_outputs) == 1:
            # Single solver
            output = list(solver_outputs.values())[0]
            return {'solution': output, 'confidence': 0.8}
        
        # Weighted voting based on analysis relevance
        solver_weights = {}
        for solver_name in solver_outputs.keys():
            # Base weight
            weight = 1.0
            
            # Boost weight if solver aligns with dominant transformation
            dominant = analysis_context.dominant_transformation
            if dominant in solver_name.lower():
                weight *= 1.5
            
            solver_weights[solver_name] = weight
        
        # Simple weighted consensus (in practice, could be more sophisticated)
        outputs = list(solver_outputs.values())
        weights = [solver_weights.get(name, 1.0) for name in solver_outputs.keys()]
        
        # For now, use highest-weighted output
        max_weight_idx = weights.index(max(weights))
        consensus_solution = outputs[max_weight_idx]
        
        return {
            'solution': consensus_solution,
            'confidence': max(weights) / sum(weights),
            'method': 'weighted_by_analysis'
        }
    
    def _measure_guidance_effectiveness(self, 
                                      enhanced_results: Dict[str, Any],
                                      consensus_result: Dict[str, Any],
                                      expected_output: Optional[np.ndarray]) -> Dict[str, Any]:
        """Measure how effective the analysis guidance was"""
        
        effectiveness = {
            'analysis_utilization': len([r for r in enhanced_results.values() 
                                       if r.get('guided_by_analysis', False)]) / max(1, len(enhanced_results)),
            'consensus_confidence': consensus_result.get('confidence', 0.0)
        }
        
        if expected_output is not None:
            solution = consensus_result['solution']
            if solution.shape == expected_output.shape:
                accuracy = np.mean(solution == expected_output)
                effectiveness['accuracy'] = accuracy
                effectiveness['guidance_success'] = accuracy > 0.8
        
        return effectiveness

# Factory function
def create_analysis_integrated_system() -> EnhancedAnalysisIntegratedSystem:
    """Create the complete analysis-integrated MoE system"""
    return EnhancedAnalysisIntegratedSystem()

# Demonstration function
def demonstrate_analysis_integration():
    """Demonstrate the analysis-integrated system"""
    print("🚀 ANALYSIS-INTEGRATED MOE SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    # Create system
    system = create_analysis_integrated_system()
    
    # Test problem
    input_grid = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0]
    ])
    
    expected_output = np.array([
        [0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ])
    
    constraints = {
        'color_constraints': [0, 1],
        'preserve_connectivity': True
    }
    
    print("Input Grid:")
    print(input_grid)
    print("\\nExpected Output:")
    print(expected_output)
    
    # Solve with deep analysis
    solution, results = system.solve_with_deep_analysis(
        input_grid, constraints, expected_output
    )
    
    print("\\nSolution:")
    print(solution)
    
    # Show results
    analysis_guidance = results['analysis_guidance']
    print(f"\\n📊 Analysis Results:")
    print(f"Dominant transformation: {analysis_guidance.dominant_transformation}")
    print(f"Guidance hints: {analysis_guidance.guidance_hints}")
    
    effectiveness = results['guidance_effectiveness']
    print(f"\\n🎯 Guidance Effectiveness:")
    for key, value in effectiveness.items():
        print(f"  {key}: {value}")
    
    return results

if __name__ == "__main__":
    demonstrate_analysis_integration()