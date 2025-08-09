#!/usr/bin/env python3
"""
GPT-OSS-20B Model Integration Wrapper for ARC-IDEA System

This module provides integration between the ARC EFE system and the GPT-OSS-20B model,
allowing the large language model to act as intelligent solvers within the EFE framework.

Components:
- GPTOSSSolver: LLM-based solver using GPT-OSS-20B
- ModelInterface: API wrapper for GPT-OSS-20B communication  
- PromptTemplates: Specialized prompts for ARC problem solving
- ResponseParser: Parse and validate model responses
"""

import numpy as np
import json
import requests
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Import from our EFE system
from EFE_update import ARCState, MovementResult, MovementType

@dataclass
class ModelResponse:
    """Response from GPT-OSS-20B model"""
    raw_response: str
    parsed_grid: np.ndarray
    confidence: float
    reasoning: str
    error_message: Optional[str] = None
    processing_time: float = 0.0
    model_metadata: Optional[Dict[str, Any]] = None

@dataclass
class ModelConfig:
    """Configuration for GPT-OSS-20B model"""
    model_name: str = "gpt-oss-20b"
    api_endpoint: str = "http://localhost:8000/v1/completions"  # Adjust as needed
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.3
    top_p: float = 0.9
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0

class PromptTemplates:
    """Specialized prompts for ARC problem solving with GPT-OSS-20B"""
    
    @staticmethod
    def create_arc_solving_prompt(input_grid: np.ndarray, 
                                 constraints: Dict[str, Any],
                                 context: str = "") -> str:
        """Create prompt for ARC problem solving"""
        
        grid_str = PromptTemplates._grid_to_string(input_grid)
        constraints_str = PromptTemplates._constraints_to_string(constraints)
        
        prompt = f"""You are an expert ARC (Abstraction and Reasoning Corpus) problem solver.

INPUT GRID:
{grid_str}

CONSTRAINTS:
{constraints_str}

CONTEXT: {context}

TASK: Analyze the input grid and apply the appropriate transformation to produce the output grid.

INSTRUCTIONS:
1. Identify patterns, symmetries, and transformations in the input
2. Apply logical reasoning to determine the transformation rule
3. Generate the output grid following the identified pattern
4. Provide your confidence level (0.0 to 1.0)
5. Explain your reasoning step by step

RESPONSE FORMAT:
{{
    "output_grid": [[row1], [row2], ...],
    "confidence": 0.8,
    "reasoning": "Step-by-step explanation of the transformation",
    "pattern_identified": "Description of the main pattern"
}}

Remember:
- Grid values are integers 0-9 (0 = background/empty)
- Maintain grid dimensions unless transformation requires resizing
- Focus on geometric, color, and logical patterns
- Consider symmetries, rotations, translations, and color mappings

OUTPUT:"""
        
        return prompt
    
    @staticmethod
    def create_movement_prompt(input_grid: np.ndarray,
                              movement_type: MovementType,
                              parameters: Dict[str, Any]) -> str:
        """Create prompt for specific movement operation"""
        
        grid_str = PromptTemplates._grid_to_string(input_grid)
        
        movement_instructions = {
            MovementType.FLIP: "Flip the grid horizontally or vertically",
            MovementType.ROTATION: "Rotate the grid by specified degrees",
            MovementType.TRANSLATION: "Translate/shift grid elements",
            MovementType.COLOR_TRANSFORM: "Transform colors according to mapping",
            MovementType.SCALING: "Scale the grid up or down",
            MovementType.PATTERN: "Apply pattern-based transformation",
            MovementType.MORPHOLOGY: "Apply morphological operations",
            MovementType.LOGICAL: "Apply logical operations between grids"
        }
        
        instruction = movement_instructions.get(movement_type, "Apply transformation")
        
        prompt = f"""You are performing a specific movement operation on an ARC grid.

INPUT GRID:
{grid_str}

MOVEMENT TYPE: {movement_type.value}
INSTRUCTION: {instruction}
PARAMETERS: {json.dumps(parameters, indent=2)}

Apply the specified movement transformation and return the result.

RESPONSE FORMAT:
{{
    "output_grid": [[row1], [row2], ...],
    "confidence": 0.9,
    "success": true,
    "parameters_used": {{}},
    "operation_description": "What transformation was applied"
}}

OUTPUT:"""
        
        return prompt
    
    @staticmethod
    def create_verification_prompt(input_grid: np.ndarray,
                                  output_grid: np.ndarray,
                                  transformation_history: List[str]) -> str:
        """Create prompt for solution verification"""
        
        input_str = PromptTemplates._grid_to_string(input_grid)
        output_str = PromptTemplates._grid_to_string(output_grid)
        history_str = " → ".join(transformation_history)
        
        prompt = f"""You are verifying an ARC problem solution using reverse thinking.

ORIGINAL INPUT:
{input_str}

PROPOSED OUTPUT:
{output_str}

TRANSFORMATION HISTORY: {history_str}

VERIFICATION TASK:
1. Forward Check: Does the output logically follow from the input?
2. Backward Check: Could this input reasonably produce this output?
3. Process Check: Is the transformation sequence valid?

Analyze the transformation and provide verification scores.

RESPONSE FORMAT:
{{
    "forward_score": 0.8,
    "backward_score": 0.7,
    "process_score": 0.9,
    "overall_valid": true,
    "issues_found": ["list of any problems"],
    "confidence": 0.85,
    "explanation": "Detailed verification reasoning"
}}

OUTPUT:"""
        
        return prompt
    
    @staticmethod
    def _grid_to_string(grid: np.ndarray) -> str:
        """Convert numpy grid to readable string format"""
        rows = []
        for row in grid:
            row_str = " ".join(str(int(val)) for val in row)
            rows.append(f"[{row_str}]")
        return "\n".join(rows)
    
    @staticmethod
    def _constraints_to_string(constraints: Dict[str, Any]) -> str:
        """Convert constraints to readable string"""
        if not constraints:
            return "No specific constraints provided"
        
        constraint_strs = []
        for key, value in constraints.items():
            if key == 'color_constraints':
                constraint_strs.append(f"Colors to use: {value}")
            elif key == 'symmetry':
                constraint_strs.append(f"Symmetry type: {value}")
            elif key == 'pattern_constraints':
                constraint_strs.append("Pattern matching required")
            else:
                constraint_strs.append(f"{key}: {value}")
        
        return "; ".join(constraint_strs)

class ModelInterface:
    """Interface for communicating with GPT-OSS-20B model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.api_key}'})
    
    def query_model(self, prompt: str) -> ModelResponse:
        """Send prompt to GPT-OSS-20B and get response"""
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "stop": ["\\nUser:", "\\nHuman:", "\\n\\n"]
                }
                
                response = self.session.post(
                    self.config.api_endpoint,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    raw_response = result.get('choices', [{}])[0].get('text', '')
                    
                    processing_time = time.time() - start_time
                    
                    return self._parse_response(raw_response, processing_time)
                else:
                    error_msg = f"API error {response.status_code}: {response.text}"
                    if attempt == self.config.max_retries - 1:
                        return ModelResponse(
                            raw_response="",
                            parsed_grid=np.zeros((5, 5), dtype=int),
                            confidence=0.0,
                            reasoning="API request failed",
                            error_message=error_msg,
                            processing_time=time.time() - start_time
                        )
                    
                    warnings.warn(f"Attempt {attempt + 1} failed: {error_msg}")
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                error_msg = f"Request exception: {str(e)}"
                if attempt == self.config.max_retries - 1:
                    return ModelResponse(
                        raw_response="",
                        parsed_grid=np.zeros((5, 5), dtype=int),
                        confidence=0.0,
                        reasoning="Request failed",
                        error_message=error_msg,
                        processing_time=time.time() - start_time
                    )
                
                warnings.warn(f"Attempt {attempt + 1} failed: {error_msg}")
                time.sleep(self.config.retry_delay)
        
        # Should not reach here, but safety fallback
        return ModelResponse(
            raw_response="",
            parsed_grid=np.zeros((5, 5), dtype=int),
            confidence=0.0,
            reasoning="All attempts failed",
            error_message="Max retries exceeded",
            processing_time=time.time() - start_time
        )
    
    def _parse_response(self, raw_response: str, processing_time: float) -> ModelResponse:
        """Parse the model's response into structured format"""
        
        try:
            # Try to extract JSON from response
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Extract grid
                output_grid_data = parsed.get('output_grid', [[0]])
                parsed_grid = np.array(output_grid_data, dtype=int)
                
                # Validate grid
                if parsed_grid.size == 0:
                    parsed_grid = np.zeros((5, 5), dtype=int)
                
                confidence = float(parsed.get('confidence', 0.5))
                confidence = np.clip(confidence, 0.0, 1.0)
                
                reasoning = parsed.get('reasoning', parsed.get('explanation', 'No reasoning provided'))
                
                return ModelResponse(
                    raw_response=raw_response,
                    parsed_grid=parsed_grid,
                    confidence=confidence,
                    reasoning=reasoning,
                    processing_time=processing_time,
                    model_metadata=parsed
                )
            else:
                # Fallback parsing if JSON extraction fails
                return self._fallback_parse(raw_response, processing_time)
                
        except Exception as e:
            warnings.warn(f"Response parsing failed: {e}")
            return self._fallback_parse(raw_response, processing_time)
    
    def _fallback_parse(self, raw_response: str, processing_time: float) -> ModelResponse:
        """Fallback parsing when JSON extraction fails"""
        
        # Try to extract grid patterns from text
        grid_data = self._extract_grid_from_text(raw_response)
        
        # Extract confidence if mentioned
        confidence = 0.3  # Default low confidence for fallback
        if 'confidence' in raw_response.lower():
            import re
            conf_match = re.search(r'confidence[:\s]*([0-9.]+)', raw_response.lower())
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    confidence = np.clip(confidence, 0.0, 1.0)
                except:
                    pass
        
        return ModelResponse(
            raw_response=raw_response,
            parsed_grid=grid_data,
            confidence=confidence,
            reasoning="Fallback parsing used",
            error_message="JSON parsing failed, used fallback",
            processing_time=processing_time
        )
    
    def _extract_grid_from_text(self, text: str) -> np.ndarray:
        """Try to extract grid from unstructured text"""
        try:
            # Look for grid-like patterns
            lines = text.split('\\n')
            grid_lines = []
            
            for line in lines:
                # Look for lines with numbers separated by spaces or commas
                if any(char.isdigit() for char in line) and len([c for c in line if c.isdigit()]) >= 3:
                    # Extract digits
                    digits = [int(c) for c in line if c.isdigit()]
                    if 3 <= len(digits) <= 30:  # Reasonable grid size
                        grid_lines.append(digits)
            
            if grid_lines:
                # Try to make rectangular
                max_len = max(len(row) for row in grid_lines)
                for row in grid_lines:
                    while len(row) < max_len:
                        row.append(0)
                
                return np.array(grid_lines, dtype=int)
            else:
                # Return default grid
                return np.zeros((5, 5), dtype=int)
                
        except:
            return np.zeros((5, 5), dtype=int)

class GPTOSSSolver:
    """LLM-based solver using GPT-OSS-20B for ARC problems"""
    
    def __init__(self, config: ModelConfig = None, solver_name: str = "GPTOSSSolver"):
        self.config = config or ModelConfig()
        self.model_interface = ModelInterface(self.config)
        self.solver_name = solver_name
        
        # Performance tracking
        self.prediction_count = 0
        self.success_count = 0
        self.total_confidence = 0.0
        self.average_response_time = 0.0
        
    def predict(self, input_grid: np.ndarray, constraints: Dict[str, Any] = None) -> np.ndarray:
        """Main prediction method for EFE system integration"""
        
        self.prediction_count += 1
        
        if constraints is None:
            constraints = {}
        
        # Create context from previous performance
        context = f"This is prediction #{self.prediction_count}. "
        if self.prediction_count > 1:
            success_rate = self.success_count / (self.prediction_count - 1)
            avg_conf = self.total_confidence / max(1, self.prediction_count - 1)
            context += f"Previous success rate: {success_rate:.2f}, Average confidence: {avg_conf:.2f}"
        
        # Generate prompt
        prompt = PromptTemplates.create_arc_solving_prompt(input_grid, constraints, context)
        
        # Query model
        response = self.model_interface.query_model(prompt)
        
        # Update statistics
        self.total_confidence += response.confidence
        self.average_response_time = (
            (self.average_response_time * (self.prediction_count - 1) + response.processing_time) 
            / self.prediction_count
        )
        
        if response.confidence > 0.5 and response.error_message is None:
            self.success_count += 1
        
        return response.parsed_grid
    
    def predict_with_metadata(self, input_grid: np.ndarray, 
                            constraints: Dict[str, Any] = None) -> Tuple[np.ndarray, ModelResponse]:
        """Prediction with full metadata return"""
        
        if constraints is None:
            constraints = {}
        
        prompt = PromptTemplates.create_arc_solving_prompt(input_grid, constraints)
        response = self.model_interface.query_model(prompt)
        
        return response.parsed_grid, response
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text response for RevThink verification"""
        try:
            # Create a text-only request using the model interface
            response = self.model_interface.query_model(prompt)
            return response.raw_response if response.raw_response else ""
        except Exception as e:
            warnings.warn(f"GPTOSSSolver text generation failed: {e}")
            return "Unable to generate response due to model error."
    
    def execute_movement(self, input_grid: np.ndarray, 
                        movement_type: MovementType,
                        parameters: Dict[str, Any]) -> MovementResult:
        """Execute specific movement operation"""
        
        start_time = time.time()
        
        prompt = PromptTemplates.create_movement_prompt(input_grid, movement_type, parameters)
        response = self.model_interface.query_model(prompt)
        
        execution_time = time.time() - start_time
        
        return MovementResult(
            output_grid=response.parsed_grid,
            confidence=response.confidence,
            operation_type=movement_type.value,
            parameters=parameters,
            execution_time=execution_time,
            success=response.error_message is None,
            error_message=response.error_message,
            metadata={
                'reasoning': response.reasoning,
                'raw_response': response.raw_response,
                'model_metadata': response.model_metadata
            }
        )
    
    def verify_solution(self, input_grid: np.ndarray,
                       output_grid: np.ndarray,
                       transformation_history: List[str]) -> Dict[str, float]:
        """Use model for solution verification"""
        
        prompt = PromptTemplates.create_verification_prompt(
            input_grid, output_grid, transformation_history
        )
        
        response = self.model_interface.query_model(prompt)
        
        # Parse verification scores from response
        try:
            if response.model_metadata:
                return {
                    'forward_score': response.model_metadata.get('forward_score', 0.5),
                    'backward_score': response.model_metadata.get('backward_score', 0.5),
                    'process_score': response.model_metadata.get('process_score', 0.5),
                    'combined_score': response.confidence,
                    'model_verification': True
                }
        except:
            pass
        
        # Fallback scores
        return {
            'forward_score': response.confidence,
            'backward_score': response.confidence * 0.8,
            'process_score': response.confidence * 0.9,
            'combined_score': response.confidence,
            'model_verification': True
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get solver performance statistics"""
        
        success_rate = self.success_count / max(1, self.prediction_count)
        avg_confidence = self.total_confidence / max(1, self.prediction_count)
        
        return {
            'solver_name': self.solver_name,
            'predictions_made': self.prediction_count,
            'successful_predictions': self.success_count,
            'success_rate': success_rate,
            'average_confidence': avg_confidence,
            'average_response_time': self.average_response_time,
            'model_config': {
                'model_name': self.config.model_name,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens
            }
        }

# Factory function for easy integration
def create_gpt_oss_solver(api_endpoint: str = "http://localhost:8000/v1/completions",
                         api_key: str = None,
                         temperature: float = 0.3) -> GPTOSSSolver:
    """Factory function to create GPT-OSS solver with custom configuration"""
    
    config = ModelConfig(
        api_endpoint=api_endpoint,
        api_key=api_key,
        temperature=temperature
    )
    
    return GPTOSSSolver(config)

if __name__ == "__main__":
    print("GPT-OSS-20B Model Wrapper for ARC-IDEA")
    print("=" * 50)
    print("This module provides integration between ARC EFE system and GPT-OSS-20B.")
    print("\\nKey components:")
    print("- GPTOSSSolver: LLM-based solver")
    print("- ModelInterface: API communication")
    print("- PromptTemplates: Specialized prompts")
    print("\\nUsage:")
    print("solver = create_gpt_oss_solver('http://your-api-endpoint')")
    print("result = solver.predict(input_grid, constraints)")