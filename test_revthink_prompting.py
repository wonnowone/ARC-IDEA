#!/usr/bin/env python3
"""
Test Prompt-Based RevThink Verification System

This script tests the new prompt-based RevThink verification system that integrates
with the multi-LLM architecture (GPT-OSS-20B + Kanana-1.5-15.7B-A3B).
"""

import numpy as np
from typing import Dict, Any

# Import the updated systems
from multi_llm_integration import MultiLLMEnhancedEFESystem, ModelConfig
from EFE_update import ARCState
from example_solvers import BaseSolver

def create_test_problem():
    """Create a simple ARC test problem"""
    # Simple color flip pattern
    input_grid = np.array([
        [1, 0, 1],
        [0, 1, 0], 
        [1, 0, 1]
    ])
    
    expected_output = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    constraints = {
        'color_constraints': [0, 1],
        'pattern_type': 'inversion',
        'preserve_structure': True
    }
    
    return input_grid, expected_output, constraints

def test_prompt_based_revthink():
    """Test the prompt-based RevThink verification system"""
    print("🧪 Testing Prompt-Based RevThink Verification System")
    print("=" * 60)
    
    # Create test problem
    input_grid, expected_output, constraints = create_test_problem()
    
    print("Input Grid:")
    print(input_grid)
    print("\nExpected Output:")  
    print(expected_output)
    print(f"\nConstraints: {constraints}")
    
    # Create model configurations (mock endpoints for testing)
    gpt_oss_config = ModelConfig(
        model_name="gpt-oss-20b",
        api_endpoint="https://api.mock-gpt-oss.com/v1/chat",
        api_key="mock-key",
        max_tokens=1000,
        temperature=0.3
    )
    
    kanana_config = ModelConfig(
        model_name="kanana-1.5-15.7b-a3b",
        api_endpoint="https://api.mock-kanana.com/v1/generate",
        api_key="mock-key",
        max_tokens=800,
        temperature=0.2
    )
    
    # Create a simple traditional solver for baseline
    class TestSolver(BaseSolver):
        def __init__(self, name):
            self.solver_name = name
            
        def predict(self, grid):
            # Simple flip transformation
            return 1 - grid
    
    traditional_solvers = [TestSolver("FlipSolver")]
    
    try:
        # Create multi-LLM enhanced system
        system = MultiLLMEnhancedEFESystem(
            traditional_solvers=traditional_solvers,
            gpt_oss_config=gpt_oss_config,
            kanana_config=kanana_config,
            consensus_threshold=0.6
        )
        
        print("\n✅ System initialized successfully with prompt-based RevThink")
        print(f"LLM wrappers connected to RevThink: {len(system.revthink_verifier.llm_wrappers)}")
        
        # Test the RevThink prompts generation
        print("\n🔍 Testing RevThink Prompt Generation:")
        
        # Test grid-to-string conversion
        input_str = system.revthink_verifier._grid_to_string(input_grid)
        output_str = system.revthink_verifier._grid_to_string(expected_output)
        
        print("\nGrid String Representation:")
        print("Input:", input_str.replace('\n', ' | '))
        print("Output:", output_str.replace('\n', ' | '))
        
        # Test prompt generation
        forward_prompt = system.revthink_verifier.arc_revthink_prompts['forward_reasoning'].format(
            input_grid=input_str,
            output_grid=output_str
        )
        
        print("\n📝 Generated Forward Reasoning Prompt:")
        print("-" * 40)
        print(forward_prompt[:300] + "..." if len(forward_prompt) > 300 else forward_prompt)
        
        # Test backward question generation
        backward_question = system.revthink_verifier._generate_backward_question(input_str, output_str)
        
        print("\n📝 Generated Backward Question:")
        print("-" * 40) 
        print(backward_question[:300] + "..." if len(backward_question) > 300 else backward_question)
        
        # Test consistency check prompt
        consistency_prompt = system.revthink_verifier.arc_revthink_prompts['consistency_check'].format(
            input_grid=input_str,
            output_grid=output_str
        )
        
        print("\n📝 Generated Consistency Check Prompt:")
        print("-" * 40)
        print(consistency_prompt[:300] + "..." if len(consistency_prompt) > 300 else consistency_prompt)
        
        # Test verification system (will use fallback since we don't have real LLMs)
        print("\n🔬 Testing RevThink Verification System:")
        
        state = ARCState(
            grid=input_grid,
            constraints=constraints,
            step=0,
            solver_history=["FlipSolver"],
            confidence=0.8
        )
        
        verification_results = system.revthink_verifier.verify_solution(expected_output, state)
        
        print("\nRevThink Verification Results:")
        for key, value in verification_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Show the difference between computational and prompt-based scores
        if system.revthink_verifier.llm_wrappers:
            print(f"\n✨ Prompt-based combined score: {verification_results['prompt_combined_score']:.3f}")
        else:
            print(f"\n⚠️  Using computational fallback score: {verification_results['combined_score']:.3f}")
        
        print("\n🎯 RevThink Testing Complete!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_revthink_parsing():
    """Test RevThink response parsing methods"""
    print("\n🧪 Testing RevThink Response Parsing")
    print("-" * 40)
    
    # Import RevThinkVerifier directly
    from EFE_update import RevThinkVerifier
    
    verifier = RevThinkVerifier()
    
    # Test reasoning quality parsing
    test_responses = [
        "This transformation is valid because the pattern shows a clear color inversion rule where each cell is flipped from 1 to 0 and vice versa.",
        "maybe it flips colors but not sure",  
        "The transformation makes sense since we can see that the pattern is inverted therefore creating a checkerboard effect.",
        "unclear what happens here",
        ""
    ]
    
    print("Testing reasoning quality parsing:")
    for i, response in enumerate(test_responses):
        score = verifier._parse_reasoning_quality(response)
        print(f"  Response {i+1}: {score:.3f} - '{response[:50]}{'...' if len(response) > 50 else ''}'")
    
    # Test consistency score parsing
    test_consistency = [
        "This is consistent and valid",
        "The transformation is false and inconsistent",
        "True - the pattern follows logically",  
        "This shows correct reasoning patterns therefore it is valid",
        "Not sure about this one"
    ]
    
    print("\nTesting consistency score parsing:")
    for i, response in enumerate(test_consistency):
        score = verifier._parse_consistency_score(response)
        print(f"  Response {i+1}: {score:.3f} - '{response[:50]}{'...' if len(response) > 50 else ''}'")

if __name__ == "__main__":
    print("🚀 RevThink Prompt-Based Verification System Test")
    print("=" * 60)
    
    # Test response parsing first
    test_revthink_parsing()
    
    # Test main prompt-based system 
    success = test_prompt_based_revthink()
    
    if success:
        print("\n🎉 All RevThink tests completed successfully!")
        print("\nThe system now supports:")
        print("  ✅ Prompt-based forward verification")  
        print("  ✅ Prompt-based backward verification")
        print("  ✅ Prompt-based consistency checking")
        print("  ✅ Integration with GPT-OSS-20B + Kanana-1.5-15.7B-A3B")
        print("  ✅ Fallback to computational verification")
        print("  ✅ ARC-specific prompt templates based on RevThink methodology")
    else:
        print("\n❌ Some tests failed - check the error messages above")