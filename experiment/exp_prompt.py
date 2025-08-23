# -*- coding: utf-8 -*-
"""
ARC Challenge Prompting System with Self-Critique and Reverse Thinking
Based on:
- Natural Language Self-Critique
- Faithful Chain-of-Thought Reasoning  
- Reverse Thinking Methodology
"""

def create_arc_prompt_template():
    """
    Creates a comprehensive prompt template for ARC challenges that incorporates:
    1. Natural language objective setting
    2. Step-by-step pixel movement planning
    3. Self-critique at each step
    4. Reverse thinking validation
    """
    
    prompt_template = """
# ARC Challenge Solving Framework

## Phase 1: Problem Objective Definition
**Task**: Analyze the input-output examples and define the transformation objective in clear, natural language.

### Step 1.1: Pattern Recognition
- Examine each input-output pair
- Identify what changes between input and output
- Describe the transformation in simple terms

### Step 1.2: Objective Statement
**Write a clear objective statement**: "The goal is to [specific transformation rule]"

**Self-Critique Questions**:
- Is this objective consistent across all training examples?
- Does it capture the essential transformation?
- Are there edge cases this objective might miss?

---

## Phase 2: Forward Chain-of-Thought Planning
**Task**: Plan every pixel movement needed to achieve the objective.

### Step 2.1: Grid Analysis
For each training example:
1. **Input Grid Analysis**:
   - Grid dimensions: [height x width]
   - Colors present: [list colors]
   - Spatial patterns: [describe patterns]
   - Shape locations: [identify shapes and positions]

2. **Output Grid Analysis**:
   - Grid dimensions: [height x width]  
   - Colors present: [list colors]
   - New patterns: [describe changes]
   - Shape transformations: [describe what happened]

### Step 2.2: Transformation Steps
For each pixel that changes:
1. **Pixel Location**: (row, col)
2. **Original Color**: [color]
3. **New Color**: [color]
4. **Reason for Change**: [explain why this specific pixel changes]
5. **Rule Application**: [how does this relate to the objective?]

**Self-Critique at Each Step**:
- **Why**: Why does this specific pixel need to change?
- **What**: What rule or pattern justifies this change?
- **How**: How does this change contribute to the overall objective?
- **Consistency**: Is this change consistent with the objective across all examples?

### Step 2.3: Rule Formalization
Based on the pixel-by-pixel analysis:
1. **Condition**: When does the transformation apply?
2. **Action**: What specific changes occur?
3. **Scope**: Which pixels/areas are affected?
4. **Constraints**: What are the limitations or boundaries?

---

## Phase 3: Reverse Thinking Validation
**Task**: Start from the desired output and work backwards to validate the reasoning.

### Step 3.1: Output-to-Input Reasoning
Starting from each output grid:
1. **Target Analysis**: What does the final result look like?
2. **Backwards Steps**: What would the input need to be to produce this output?
3. **Reverse Rule**: If I apply the transformation rule in reverse, do I get the original input?

### Step 3.2: Reverse Validation Questions
For each transformation step identified:
1. **Necessity Check**: Is this step absolutely necessary to achieve the objective?
2. **Sufficiency Check**: Are these steps sufficient to fully achieve the objective?
3. **Alternative Paths**: Could the same result be achieved through different steps?
4. **Edge Case Testing**: Does this reasoning hold for boundary conditions?

### Step 3.3: Cross-Validation
1. **Forward-Backward Consistency**: Does the forward reasoning match the backward reasoning?
2. **Objective Alignment**: Do both directions support the same objective statement?
3. **Pattern Completeness**: Are all observed patterns explained by the reasoning?

---

## Phase 4: Self-Critique and Refinement
**Task**: Critically evaluate the entire reasoning chain.

### Step 4.1: Reasoning Quality Assessment
1. **Logical Consistency**: Are there any contradictions in the reasoning?
2. **Completeness**: Does the explanation account for all observed changes?
3. **Parsimony**: Is this the simplest explanation that fits all data?
4. **Generalizability**: Will this rule work for unseen examples?

### Step 4.2: Error Detection
Common pitfalls to check:
1. **Overfitting**: Is the rule too specific to the training examples?
2. **Underfitting**: Is the rule too general to capture the pattern?
3. **Confirmation Bias**: Am I ignoring contradictory evidence?
4. **Incomplete Analysis**: Are there unexamined aspects of the pattern?

### Step 4.3: Refinement Process
If issues are found:
1. **Revise Objective**: Update the objective statement if needed
2. **Adjust Steps**: Modify the transformation steps
3. **Re-validate**: Run through reverse thinking again
4. **Iterate**: Repeat until consistency is achieved

---

## Phase 5: Final Application
**Task**: Apply the validated rule to the test input.

### Step 5.1: Test Input Analysis
1. **Grid Properties**: Analyze the test input using the same framework
2. **Pattern Matching**: Identify where the learned rule applies
3. **Condition Checking**: Verify that conditions for transformation are met

### Step 5.2: Transformation Execution
1. **Step-by-Step Application**: Apply each transformation step methodically
2. **Real-time Validation**: Check each step against the objective
3. **Consistency Monitoring**: Ensure each change aligns with the learned pattern

### Step 5.3: Final Self-Critique
1. **Result Plausibility**: Does the output make sense given the objective?
2. **Pattern Adherence**: Does the result follow the identified pattern?
3. **Confidence Assessment**: How confident am I in this solution?

---

## Implementation Notes:
- Document every decision and its justification
- Question assumptions at each step
- Use concrete examples to validate abstract rules
- Maintain consistency between forward and reverse reasoning
- Prioritize simplicity and generalizability
"""
    
    return prompt_template

def create_example_application():
    """
    Example of how to apply the prompting framework to a specific ARC challenge.
    """
    
    example = """
# Example Application: Pattern Completion Challenge

## Phase 1: Problem Objective Definition

### Input-Output Analysis:
Input: 3x3 grid with partial blue line
Output: 3x3 grid with completed blue line

### Objective Statement:
"The goal is to complete broken or partial lines of the same color to form continuous straight lines."

### Self-Critique:
- Consistent across examples?  All examples show line completion
- Captures essential transformation?  Focus is on line continuity
- Edge cases considered? Need to check diagonal lines and corners

## Phase 2: Forward Chain-of-Thought Planning

### Grid Analysis Example:
Input:  [0,1,0]
        [0,0,0]  
        [0,1,0]

Output: [0,1,0]
        [0,1,0]
        [0,1,0]

### Transformation Steps:
1. **Pixel (1,1)**: 0 � 1
   - **Why**: Creates vertical line continuity
   - **What**: Fills gap between existing blue pixels
   - **How**: Completes the vertical line pattern
   - **Consistency**: Matches line completion objective

### Rule Formalization:
- **Condition**: When pixels of same color are aligned with a gap
- **Action**: Fill the gap with the same color
- **Scope**: Direct line connections (horizontal, vertical, diagonal)
- **Constraints**: Only fill single-pixel gaps

## Phase 3: Reverse Thinking Validation

### Output-to-Input Reasoning:
Starting with completed line [0,1,0; 0,1,0; 0,1,0]:
- To create input, remove middle pixel: [0,1,0; 0,0,0; 0,1,0]
- This creates a gap that needs completion
- Reverse rule: "Identify incomplete lines and their missing segments"

### Validation:
- **Necessity**: Middle pixel necessary for line completion 
- **Sufficiency**: This step alone completes the pattern 
- **Consistency**: Forward and backward reasoning align 

## Phase 4: Self-Critique and Refinement

### Quality Assessment:
- **Logical**: Line completion is logical 
- **Complete**: All changes explained 
- **Simple**: Straightforward line completion rule 
- **Generalizable**: Should work for other line patterns 

## Phase 5: Final Application

### Test Input Application:
Apply line completion rule to test input systematically, checking each potential gap against the learned pattern.
"""
    
    return example

def save_prompt_system():
    """Save the complete prompting system to files."""
    
    # Save main template
    with open('arc_prompt_template.txt', 'w', encoding='utf-8') as f:
        f.write(create_arc_prompt_template())
    
    # Save example application
    with open('arc_prompt_example.txt', 'w', encoding='utf-8') as f:
        f.write(create_example_application())
    
    print("Prompting system saved to:")
    print("- arc_prompt_template.txt (main framework)")
    print("- arc_prompt_example.txt (example application)")

if __name__ == "__main__":
    save_prompt_system()
    print("\nARC Prompting System with Self-Critique and Reverse Thinking created successfully!")
    print("\nKey Features:")
    print("1. Natural language objective setting")
    print("2. Pixel-by-pixel transformation planning")
    print("3. Self-critique at every step (why, what, how)")
    print("4. Reverse thinking validation")
    print("5. Comprehensive error detection and refinement")