#!/usr/bin/env python3
"""
Simple RevThink Test - Test basic functionality without LLM initialization
"""

import numpy as np
import sys
sys.path.append('.')

def test_basic_revthink():
    """Test basic RevThink functionality"""
    print("Testing Basic RevThink Functionality")
    print("=" * 50)
    
    try:
        from EFE_update import RevThinkVerifier, ARCState
        
        # Create basic verifier without LLMs
        verifier = RevThinkVerifier()
        print("OK RevThinkVerifier created successfully")
        
        # Test grid-to-string conversion
        test_grid = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        grid_str = verifier._grid_to_string(test_grid)
        print(f"OK Grid-to-string: {grid_str.replace(chr(10), ' | ')}")
        
        # Test prompt generation
        forward_prompt = verifier._get_forward_reasoning_prompt()
        print("OK Forward reasoning prompt template created")
        
        backward_prompt = verifier._get_backward_question_prompt() 
        print("OK Backward question prompt template created")
        
        consistency_prompt = verifier._get_consistency_check_prompt()
        print("OK Consistency check prompt template created")
        
        # Test response parsing
        test_response = "This transformation is valid because the pattern shows clear logical reasoning"
        score = verifier._parse_reasoning_quality(test_response)
        print(f"OK Reasoning quality parsing: {score:.3f}")
        
        consistency_response = "True - this is consistent"
        c_score = verifier._parse_consistency_score(consistency_response)
        print(f"OK Consistency score parsing: {c_score:.3f}")
        
        # Test verification with fallback
        state = ARCState(
            grid=test_grid,
            constraints={'color_constraints': [0, 1]},
            step=0,
            solver_history=["TestSolver"],
            confidence=0.8
        )
        
        output_grid = 1 - test_grid  # Simple flip
        results = verifier.verify_solution(output_grid, state)
        
        print("OK Verification completed with results:")
        for key, value in results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
        
        print("\nSUCCESS All basic RevThink tests passed!")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_basic_revthink()