#!/usr/bin/env python3
"""
Multi-LLM Model Integration Wrapper for ARC-IDEA System

This module provides integration between the ARC EFE system and multiple LLM models:
- GPT-OSS-20B: Large-scale reasoning and pattern recognition
- Kanana-1.5-15.7B-A3B: Specialized analytical and problem-solving capabilities

Components:
- MultiLLMSolver: Coordinates multiple LLM models
- KananaSolver: Specialized solver for Kanana-1.5-15.7B-A3B
- LLMEnsembleManager: Manages ensemble of different LLM models
- CrossModelConsensus: Consensus mechanism across different LLM architectures
"""

import numpy as np
import json
import requests
import time
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum

# Import from our existing systems
from EFE_update import ARCState, MovementResult, MovementType
from gpt_oss_model_wrapper import (
    ModelResponse, ModelConfig, PromptTemplates, ModelInterface, GPTOSSSolver
)

class LLMModelType(Enum):
    """Enumeration of supported LLM model types"""
    GPT_OSS_20B = "gpt-oss-20b"
    KANANA_15_7B_A3B = "kanana-1.5-15.7b-a3b"
    GENERIC_LLM = "generic-llm"

@dataclass
class MultiModelConfig:
    """Configuration for multiple LLM models"""
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    cross_validation_threshold: float = 0.7
    consensus_strategy: str = "weighted_voting"  # weighted_voting, majority, confidence_based
    timeout_per_model: float = 30.0
    max_concurrent_requests: int = 3
    enable_cross_validation: bool = True

class KananaPromptTemplates:
    """Specialized prompts for Kanana-1.5-15.7B-A3B model"""
    
    @staticmethod
    def create_kanana_arc_prompt(input_grid: np.ndarray, 
                                constraints: Dict[str, Any],
                                context: str = "") -> str:
        """Create Kanana-optimized prompt for ARC problem solving"""
        
        grid_str = PromptTemplates._grid_to_string(input_grid)
        constraints_str = PromptTemplates._constraints_to_string(constraints)
        
        prompt = f"""<|im_start|>system
You are Kanana, an expert analytical AI specialized in pattern recognition and logical reasoning for ARC (Abstraction and Reasoning Corpus) problems. Your analytical capabilities excel at identifying subtle patterns and logical transformations.
<|im_end|>

<|im_start|>user
INPUT GRID:
{grid_str}

CONSTRAINTS:
{constraints_str}

CONTEXT: {context}

Analyze this ARC problem step by step:

1. PATTERN ANALYSIS: Identify all visual patterns, symmetries, and regularities
2. TRANSFORMATION LOGIC: Determine the underlying transformation rule
3. CONSTRAINT SATISFACTION: Ensure the solution meets all given constraints
4. CONFIDENCE ASSESSMENT: Rate your confidence in the solution

Provide your analysis in this format:
{{
    "pattern_analysis": "Detailed description of identified patterns",
    "transformation_logic": "The rule governing the transformation", 
    "output_grid": [[row1], [row2], ...],
    "confidence": 0.85,
    "reasoning_steps": ["step1", "step2", "step3"],
    "alternative_interpretations": ["alt1", "alt2"]
}}

Focus on logical consistency and pattern completeness.
<|im_end|>

<|im_start|>assistant"""
        
        return prompt
    
    @staticmethod
    def create_kanana_verification_prompt(input_grid: np.ndarray,
                                        output_grid: np.ndarray,
                                        reasoning_trace: List[str]) -> str:
        """Create Kanana-optimized verification prompt"""
        
        input_str = PromptTemplates._grid_to_string(input_grid)
        output_str = PromptTemplates._grid_to_string(output_grid)
        reasoning_str = " → ".join(reasoning_trace)
        
        prompt = f"""<|im_start|>system
You are Kanana, performing critical verification analysis of ARC problem solutions. Your role is to validate logical consistency and identify potential issues.
<|im_end|>

<|im_start|>user
ORIGINAL INPUT:
{input_str}

PROPOSED SOLUTION:
{output_str}

REASONING TRACE: {reasoning_str}

Perform comprehensive verification:

1. LOGICAL CONSISTENCY: Does the solution follow from the input logically?
2. PATTERN PRESERVATION: Are essential patterns maintained or properly transformed?
3. CONSTRAINT ADHERENCE: Does the solution satisfy all implicit and explicit constraints?
4. COMPLETENESS CHECK: Is the transformation complete and consistent?

Provide verification results:
{{
    "logical_consistency": 0.9,
    "pattern_preservation": 0.85,
    "constraint_adherence": 0.95,
    "completeness_score": 0.88,
    "overall_validity": 0.87,
    "issues_identified": ["any problems found"],
    "confidence_in_verification": 0.92,
    "detailed_analysis": "Comprehensive analysis of the solution quality"
}}
<|im_end|>

<|im_start|>assistant"""
        
        return prompt
    
    @staticmethod
    def create_kanana_movement_prompt(input_grid: np.ndarray,
                                    movement_type: MovementType,
                                    parameters: Dict[str, Any]) -> str:
        """Create Kanana-optimized movement operation prompt"""
        
        grid_str = PromptTemplates._grid_to_string(input_grid)
        
        movement_descriptions = {
            MovementType.FLIP: "Apply precise geometric flip transformation",
            MovementType.ROTATION: "Execute exact rotational transformation",
            MovementType.TRANSLATION: "Perform accurate translational shift",
            MovementType.COLOR_TRANSFORM: "Apply systematic color mapping",
            MovementType.SCALING: "Execute proportional scaling transformation",
            MovementType.PATTERN: "Apply pattern-based logical transformation",
            MovementType.MORPHOLOGY: "Execute morphological structure operation",
            MovementType.LOGICAL: "Apply logical operation between grid elements"
        }
        
        description = movement_descriptions.get(movement_type, "Apply specified transformation")
        
        prompt = f"""<|im_start|>system
You are Kanana, executing precise grid transformation operations. Your analytical precision ensures accurate geometric and logical transformations.
<|im_end|>

<|im_start|>user
INPUT GRID:
{grid_str}

OPERATION: {movement_type.value}
TASK: {description}
PARAMETERS: {json.dumps(parameters, indent=2)}

Execute the transformation with analytical precision:

1. Analyze the current grid structure
2. Apply the specified transformation exactly
3. Verify the transformation correctness
4. Assess transformation quality

Return the result:
{{
    "transformed_grid": [[row1], [row2], ...],
    "transformation_quality": 0.95,
    "precision_score": 0.98,
    "success": true,
    "parameters_applied": {{}},
    "verification_notes": "Quality assessment of the transformation"
}}
<|im_end|>

<|im_start|>assistant"""
        
        return prompt

class KananaModelInterface:
    """Specialized interface for Kanana-1.5-15.7B-A3B model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = requests.Session()
        if config.api_key:
            self.session.headers.update({'Authorization': f'Bearer {config.api_key}'})
        
        # Kanana-specific parameters
        self.default_params = {
            "temperature": 0.2,  # Lower temperature for more focused analysis
            "top_p": 0.85,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "max_tokens": 2048
        }
    
    def query_kanana(self, prompt: str, custom_params: Dict = None) -> ModelResponse:
        """Send prompt to Kanana model with optimized parameters"""
        
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)
        
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "temperature": params["temperature"],
                    "top_p": params["top_p"],
                    "top_k": params.get("top_k", 50),
                    "repetition_penalty": params.get("repetition_penalty", 1.1),
                    "max_tokens": params["max_tokens"],
                    "stop": ["<|im_end|>", "\\n\\nUser:", "\\n\\nHuman:"]
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
                    
                    return self._parse_kanana_response(raw_response, processing_time)
                else:
                    error_msg = f"Kanana API error {response.status_code}: {response.text}"
                    if attempt == self.config.max_retries - 1:
                        return self._create_error_response(error_msg, time.time() - start_time)
                    
                    warnings.warn(f"Kanana attempt {attempt + 1} failed: {error_msg}")
                    time.sleep(self.config.retry_delay)
                    
            except Exception as e:
                error_msg = f"Kanana request exception: {str(e)}"
                if attempt == self.config.max_retries - 1:
                    return self._create_error_response(error_msg, time.time() - start_time)
                
                warnings.warn(f"Kanana attempt {attempt + 1} failed: {error_msg}")
                time.sleep(self.config.retry_delay)
        
        return self._create_error_response("Max retries exceeded", time.time() - start_time)
    
    def _parse_kanana_response(self, raw_response: str, processing_time: float) -> ModelResponse:
        """Parse Kanana model response with specialized handling"""
        
        try:
            # Extract JSON from Kanana response format
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Extract grid from various possible keys
                output_grid_data = (
                    parsed.get('output_grid') or 
                    parsed.get('transformed_grid') or 
                    parsed.get('solution_grid') or
                    [[0]]
                )
                
                parsed_grid = np.array(output_grid_data, dtype=int)
                
                if parsed_grid.size == 0:
                    parsed_grid = np.zeros((5, 5), dtype=int)
                
                # Extract confidence from various possible keys
                confidence = (
                    parsed.get('confidence') or
                    parsed.get('overall_validity') or
                    parsed.get('transformation_quality') or
                    0.5
                )
                confidence = float(np.clip(confidence, 0.0, 1.0))
                
                # Extract reasoning
                reasoning = (
                    parsed.get('detailed_analysis') or
                    parsed.get('pattern_analysis') or
                    parsed.get('reasoning_steps') or
                    parsed.get('verification_notes') or
                    'Kanana analytical response'
                )
                
                if isinstance(reasoning, list):
                    reasoning = '; '.join(str(r) for r in reasoning)
                
                return ModelResponse(
                    raw_response=raw_response,
                    parsed_grid=parsed_grid,
                    confidence=confidence,
                    reasoning=str(reasoning),
                    processing_time=processing_time,
                    model_metadata=parsed
                )
            else:
                return self._fallback_parse_kanana(raw_response, processing_time)
                
        except Exception as e:
            warnings.warn(f"Kanana response parsing failed: {e}")
            return self._fallback_parse_kanana(raw_response, processing_time)
    
    def _fallback_parse_kanana(self, raw_response: str, processing_time: float) -> ModelResponse:
        """Fallback parsing for Kanana responses"""
        
        # Simple grid extraction
        grid_data = self._extract_grid_from_text(raw_response)
        
        # Extract confidence mentions
        confidence = 0.4  # Lower default confidence for fallback
        confidence_patterns = [
            r'confidence[:\s]*([0-9.]+)',
            r'quality[:\s]*([0-9.]+)', 
            r'validity[:\s]*([0-9.]+)'
        ]
        
        import re
        for pattern in confidence_patterns:
            match = re.search(pattern, raw_response.lower())
            if match:
                try:
                    confidence = max(confidence, float(match.group(1)))
                    confidence = np.clip(confidence, 0.0, 1.0)
                    break
                except:
                    continue
        
        return ModelResponse(
            raw_response=raw_response,
            parsed_grid=grid_data,
            confidence=confidence,
            reasoning="Kanana fallback parsing - limited structured analysis available",
            error_message="JSON parsing failed, used fallback",
            processing_time=processing_time
        )
    
    def _extract_grid_from_text(self, text: str) -> np.ndarray:
        """Extract grid from unstructured Kanana response"""
        try:
            lines = text.split('\\n')
            grid_lines = []
            
            in_grid_section = False
            for line in lines:
                # Look for grid markers
                if 'grid' in line.lower() and (':' in line or '=' in line):
                    in_grid_section = True
                    continue
                
                if in_grid_section and line.strip():
                    # Extract digits/numbers from line
                    import re
                    numbers = re.findall(r'\\d+', line)
                    if numbers and len(numbers) >= 3:
                        row = [int(n) for n in numbers[:10]]  # Limit row length
                        if 3 <= len(row) <= 10:
                            grid_lines.append(row)
                    elif not numbers and grid_lines:
                        # End of grid section
                        break
            
            if grid_lines:
                # Normalize grid to rectangular
                max_len = max(len(row) for row in grid_lines)
                for row in grid_lines:
                    while len(row) < max_len:
                        row.append(0)
                
                return np.array(grid_lines[:10], dtype=int)  # Limit grid size
            else:
                return np.zeros((5, 5), dtype=int)
                
        except:
            return np.zeros((5, 5), dtype=int)
    
    def _create_error_response(self, error_msg: str, processing_time: float) -> ModelResponse:
        """Create error response for failed requests"""
        return ModelResponse(
            raw_response="",
            parsed_grid=np.zeros((5, 5), dtype=int),
            confidence=0.0,
            reasoning="Kanana model request failed",
            error_message=error_msg,
            processing_time=processing_time
        )

class KananaSolver:
    """Kanana-1.5-15.7B-A3B specialized solver for ARC problems"""
    
    def __init__(self, config: ModelConfig = None, solver_name: str = "KananaSolver"):
        self.config = config or ModelConfig(model_name="kanana-1.5-15.7b-a3b")
        self.model_interface = KananaModelInterface(self.config)
        self.solver_name = solver_name
        
        # Performance tracking
        self.prediction_count = 0
        self.success_count = 0
        self.total_confidence = 0.0
        self.average_response_time = 0.0
        self.analytical_scores = []
        
    def predict(self, input_grid: np.ndarray, constraints: Dict[str, Any] = None) -> np.ndarray:
        """Main prediction method using Kanana's analytical capabilities"""
        
        self.prediction_count += 1
        
        if constraints is None:
            constraints = {}
        
        # Create context highlighting analytical requirements
        context = f"Analytical task #{self.prediction_count}. "
        if self.prediction_count > 1:
            success_rate = self.success_count / (self.prediction_count - 1)
            avg_conf = self.total_confidence / max(1, self.prediction_count - 1)
            avg_analytical = np.mean(self.analytical_scores) if self.analytical_scores else 0.7
            context += f"Success rate: {success_rate:.2f}, Avg confidence: {avg_conf:.2f}, Analytical score: {avg_analytical:.2f}"
        
        # Generate Kanana-optimized prompt
        prompt = KananaPromptTemplates.create_kanana_arc_prompt(input_grid, constraints, context)
        
        # Query Kanana model
        response = self.model_interface.query_kanana(prompt)
        
        # Update statistics
        self.total_confidence += response.confidence
        self.average_response_time = (
            (self.average_response_time * (self.prediction_count - 1) + response.processing_time) 
            / self.prediction_count
        )
        
        # Track analytical quality
        if response.model_metadata:
            analytical_indicators = [
                response.model_metadata.get('logical_consistency', 0.5),
                response.model_metadata.get('pattern_preservation', 0.5),
                response.model_metadata.get('completeness_score', 0.5)
            ]
            analytical_score = np.mean([s for s in analytical_indicators if s > 0])
            self.analytical_scores.append(analytical_score)
        
        if response.confidence > 0.6 and response.error_message is None:
            self.success_count += 1
        
        return response.parsed_grid
    
    def predict_with_metadata(self, input_grid: np.ndarray, 
                            constraints: Dict[str, Any] = None) -> Tuple[np.ndarray, ModelResponse]:
        """Prediction with full Kanana metadata return"""
        
        if constraints is None:
            constraints = {}
        
        prompt = KananaPromptTemplates.create_kanana_arc_prompt(input_grid, constraints)
        response = self.model_interface.query_kanana(prompt)
        
        return response.parsed_grid, response
    
    def execute_movement(self, input_grid: np.ndarray, 
                        movement_type: MovementType,
                        parameters: Dict[str, Any]) -> MovementResult:
        """Execute movement operation with Kanana's precision"""
        
        start_time = time.time()
        
        prompt = KananaPromptTemplates.create_kanana_movement_prompt(input_grid, movement_type, parameters)
        response = self.model_interface.query_kanana(prompt)
        
        execution_time = time.time() - start_time
        
        # Extract precision metrics if available
        precision_score = 0.8
        transformation_quality = 0.8
        
        if response.model_metadata:
            precision_score = response.model_metadata.get('precision_score', 0.8)
            transformation_quality = response.model_metadata.get('transformation_quality', 0.8)
        
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
                'model_metadata': response.model_metadata,
                'precision_score': precision_score,
                'transformation_quality': transformation_quality,
                'analytical_approach': 'kanana-specialized'
            }
        )
    
    def verify_solution(self, input_grid: np.ndarray,
                       output_grid: np.ndarray,
                       transformation_history: List[str]) -> Dict[str, float]:
        """Use Kanana for comprehensive solution verification"""
        
        prompt = KananaPromptTemplates.create_kanana_verification_prompt(
            input_grid, output_grid, transformation_history
        )
        
        response = self.model_interface.query_kanana(prompt)
        
        # Parse comprehensive verification scores from Kanana
        try:
            if response.model_metadata:
                return {
                    'forward_score': response.model_metadata.get('logical_consistency', 0.6),
                    'backward_score': response.model_metadata.get('pattern_preservation', 0.6),
                    'process_score': response.model_metadata.get('constraint_adherence', 0.6),
                    'combined_score': response.model_metadata.get('overall_validity', response.confidence),
                    'completeness_score': response.model_metadata.get('completeness_score', 0.6),
                    'analytical_depth': min(1.0, np.mean(self.analytical_scores)) if self.analytical_scores else 0.7,
                    'model_verification': True,
                    'verification_confidence': response.model_metadata.get('confidence_in_verification', response.confidence)
                }
        except:
            pass
        
        # Fallback scores with Kanana characteristics
        return {
            'forward_score': response.confidence * 0.9,
            'backward_score': response.confidence * 0.85,
            'process_score': response.confidence * 0.92,
            'combined_score': response.confidence,
            'analytical_depth': 0.75,
            'model_verification': True
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get Kanana solver performance statistics"""
        
        success_rate = self.success_count / max(1, self.prediction_count)
        avg_confidence = self.total_confidence / max(1, self.prediction_count)
        avg_analytical = np.mean(self.analytical_scores) if self.analytical_scores else 0.0
        
        return {
            'solver_name': self.solver_name,
            'model_type': 'kanana-1.5-15.7b-a3b',
            'predictions_made': self.prediction_count,
            'successful_predictions': self.success_count,
            'success_rate': success_rate,
            'average_confidence': avg_confidence,
            'average_response_time': self.average_response_time,
            'analytical_capability': avg_analytical,
            'specialization': 'analytical_reasoning',
            'model_config': {
                'model_name': self.config.model_name,
                'temperature': 0.2,
                'analytical_focus': True
            }
        }
    
    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text response for RevThink verification"""
        try:
            # Create a text-only request using the Kanana interface
            response = self.model_interface.query_kanana(prompt)
            return response.raw_response if response.raw_response else ""
        except Exception as e:
            warnings.warn(f"KananaSolver text generation failed: {e}")
            return "Unable to generate response due to model error."

class MultiLLMEnsemble:
    """Ensemble manager for multiple LLM models (GPT-OSS-20B + Kanana-1.5-15.7B-A3B)"""
    
    def __init__(self, 
                 gpt_oss_config: ModelConfig,
                 kanana_config: ModelConfig,
                 ensemble_weights: Dict[str, float] = None):
        
        self.gpt_oss_config = gpt_oss_config
        self.kanana_config = kanana_config
        
        # Create LLM solvers
        self.gpt_oss_solvers = []
        self.kanana_solvers = []
        
        # Create GPT-OSS variants
        for i in range(2):
            variant_config = ModelConfig(
                model_name=gpt_oss_config.model_name,
                api_endpoint=gpt_oss_config.api_endpoint,
                api_key=gpt_oss_config.api_key,
                temperature=gpt_oss_config.temperature + (i * 0.15),
                max_tokens=gpt_oss_config.max_tokens
            )
            solver = GPTOSSSolver(variant_config, f"GPT_OSS_v{i+1}")
            self.gpt_oss_solvers.append(solver)
        
        # Create Kanana variants
        for i in range(2):
            variant_config = ModelConfig(
                model_name=kanana_config.model_name,
                api_endpoint=kanana_config.api_endpoint,
                api_key=kanana_config.api_key,
                temperature=0.15 + (i * 0.1),  # Lower temps for analytical precision
                max_tokens=kanana_config.max_tokens
            )
            solver = KananaSolver(variant_config, f"Kanana_v{i+1}")
            self.kanana_solvers.append(solver)
        
        # All LLM solvers
        self.all_llm_solvers = self.gpt_oss_solvers + self.kanana_solvers
        
        # Ensemble weights
        self.ensemble_weights = ensemble_weights or {
            'gpt_oss': 0.55,  # Slightly favor GPT-OSS for general reasoning
            'kanana': 0.45    # Kanana for analytical precision
        }
        
        # Cross-model performance tracking
        self.cross_model_agreements = 0
        self.total_comparisons = 0
        self.model_synergy_scores = []
        
    def get_all_llm_solvers(self) -> List[Union[GPTOSSSolver, KananaSolver]]:
        """Get all LLM solvers for integration"""
        return self.all_llm_solvers
    
    def compute_cross_model_consensus(self, 
                                    gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                                    kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> Dict[str, Any]:
        """Compute consensus across different LLM architectures"""
        
        all_outputs = []
        all_confidences = []
        model_types = []
        
        # Collect GPT-OSS results
        for output, response in gpt_oss_results:
            all_outputs.append(output)
            all_confidences.append(response.confidence)
            model_types.append('gpt_oss')
        
        # Collect Kanana results
        for output, response in kanana_results:
            all_outputs.append(output)
            all_confidences.append(response.confidence)
            model_types.append('kanana')
        
        if not all_outputs:
            return self._create_fallback_consensus()
        
        # Weighted consensus based on model type and confidence
        consensus_grid = self._compute_weighted_grid_consensus(all_outputs, all_confidences, model_types)
        
        # Calculate cross-model agreement
        agreements = self._calculate_cross_model_agreements(all_outputs, model_types)
        
        # Synergy analysis
        synergy_score = self._calculate_model_synergy(gpt_oss_results, kanana_results)
        
        self.total_comparisons += 1
        if agreements > 0.7:
            self.cross_model_agreements += 1
        
        self.model_synergy_scores.append(synergy_score)
        
        return {
            'consensus_grid': consensus_grid,
            'cross_model_agreement': agreements,
            'synergy_score': synergy_score,
            'gpt_oss_contribution': self._calculate_model_contribution(gpt_oss_results, consensus_grid),
            'kanana_contribution': self._calculate_model_contribution(kanana_results, consensus_grid),
            'ensemble_confidence': np.mean(all_confidences),
            'model_diversity': self._calculate_output_diversity(all_outputs),
            'consensus_quality': 'high' if agreements > 0.8 else 'medium' if agreements > 0.6 else 'low'
        }
    
    def _compute_weighted_grid_consensus(self, 
                                       outputs: List[np.ndarray], 
                                       confidences: List[float],
                                       model_types: List[str]) -> np.ndarray:
        """Compute weighted consensus across different model architectures"""
        
        if not outputs:
            return np.zeros((5, 5), dtype=int)
        
        # Ensure all grids have same shape
        target_shape = outputs[0].shape
        normalized_outputs = []
        
        for output in outputs:
            if output.shape != target_shape:
                # Resize to target shape
                normalized = np.zeros(target_shape, dtype=int)
                min_h = min(output.shape[0], target_shape[0])
                min_w = min(output.shape[1], target_shape[1])
                normalized[:min_h, :min_w] = output[:min_h, :min_w]
                normalized_outputs.append(normalized)
            else:
                normalized_outputs.append(output.copy())
        
        consensus_grid = np.zeros(target_shape, dtype=int)
        
        # Position-wise weighted voting
        for i in range(target_shape[0]):
            for j in range(target_shape[1]):
                position_votes = {}
                
                for k, output in enumerate(normalized_outputs):
                    value = output[i, j]
                    confidence = confidences[k]
                    model_type = model_types[k]
                    
                    # Apply model-specific and confidence weighting
                    weight = confidence * self.ensemble_weights.get(model_type, 0.5)
                    
                    if value not in position_votes:
                        position_votes[value] = 0.0
                    position_votes[value] += weight
                
                # Choose value with highest weighted vote
                if position_votes:
                    best_value = max(position_votes.keys(), key=lambda x: position_votes[x])
                    consensus_grid[i, j] = best_value
        
        return consensus_grid
    
    def _calculate_cross_model_agreements(self, outputs: List[np.ndarray], model_types: List[str]) -> float:
        """Calculate agreement level between different model architectures"""
        
        if len(outputs) < 2:
            return 1.0
        
        agreements = []
        gpt_oss_indices = [i for i, t in enumerate(model_types) if t == 'gpt_oss']
        kanana_indices = [i for i, t in enumerate(model_types) if t == 'kanana']
        
        # Compare across model types
        for gpt_idx in gpt_oss_indices:
            for kanana_idx in kanana_indices:
                if gpt_idx < len(outputs) and kanana_idx < len(outputs):
                    gpt_output = outputs[gpt_idx]
                    kanana_output = outputs[kanana_idx]
                    
                    # Ensure same shape for comparison
                    if gpt_output.shape == kanana_output.shape:
                        agreement = np.mean(gpt_output == kanana_output)
                        agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.5
    
    def _calculate_model_synergy(self, 
                               gpt_oss_results: List[Tuple[np.ndarray, ModelResponse]],
                               kanana_results: List[Tuple[np.ndarray, ModelResponse]]) -> float:
        """Calculate synergy score between GPT-OSS and Kanana models"""
        
        if not gpt_oss_results or not kanana_results:
            return 0.5
        
        synergy_factors = []
        
        # Confidence complementarity
        gpt_confidences = [r[1].confidence for r in gpt_oss_results]
        kanana_confidences = [r[1].confidence for r in kanana_results]
        
        avg_gpt_conf = np.mean(gpt_confidences)
        avg_kanana_conf = np.mean(kanana_confidences)
        
        # Higher synergy when both models are confident
        confidence_synergy = min(avg_gpt_conf, avg_kanana_conf) * 2.0
        synergy_factors.append(confidence_synergy)
        
        # Response diversity (healthy disagreement indicates complementary strengths)
        if len(gpt_oss_results) > 0 and len(kanana_results) > 0:
            gpt_output = gpt_oss_results[0][0]
            kanana_output = kanana_results[0][0]
            
            if gpt_output.shape == kanana_output.shape:
                diversity = 1.0 - np.mean(gpt_output == kanana_output)
                # Optimal diversity around 0.3 (some differences but not complete disagreement)
                diversity_synergy = 1.0 - abs(diversity - 0.3) / 0.7
                synergy_factors.append(diversity_synergy)
        
        # Processing time balance (both models should contribute efficiently)
        gpt_times = [r[1].processing_time for r in gpt_oss_results]
        kanana_times = [r[1].processing_time for r in kanana_results]
        
        avg_gpt_time = np.mean(gpt_times)
        avg_kanana_time = np.mean(kanana_times)
        
        time_balance = 1.0 - abs(avg_gpt_time - avg_kanana_time) / max(avg_gpt_time, avg_kanana_time, 1.0)
        synergy_factors.append(time_balance)
        
        return np.clip(np.mean(synergy_factors), 0.0, 1.0)
    
    def _calculate_model_contribution(self, 
                                    model_results: List[Tuple[np.ndarray, ModelResponse]], 
                                    consensus_grid: np.ndarray) -> float:
        """Calculate how much a specific model type contributed to consensus"""
        
        if not model_results:
            return 0.0
        
        contributions = []
        for output, response in model_results:
            if output.shape == consensus_grid.shape:
                similarity = np.mean(output == consensus_grid)
                weighted_contribution = similarity * response.confidence
                contributions.append(weighted_contribution)
        
        return np.mean(contributions) if contributions else 0.0
    
    def _calculate_output_diversity(self, outputs: List[np.ndarray]) -> float:
        """Calculate diversity among all model outputs"""
        
        if len(outputs) < 2:
            return 0.0
        
        diversities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                if outputs[i].shape == outputs[j].shape:
                    diversity = 1.0 - np.mean(outputs[i] == outputs[j])
                    diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def _create_fallback_consensus(self) -> Dict[str, Any]:
        """Create fallback consensus when computation fails"""
        
        return {
            'consensus_grid': np.zeros((5, 5), dtype=int),
            'cross_model_agreement': 0.0,
            'synergy_score': 0.0,
            'gpt_oss_contribution': 0.0,
            'kanana_contribution': 0.0,
            'ensemble_confidence': 0.1,
            'model_diversity': 0.0,
            'consensus_quality': 'failed'
        }
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get comprehensive ensemble statistics"""
        
        cross_model_agreement_rate = self.cross_model_agreements / max(1, self.total_comparisons)
        avg_synergy = np.mean(self.model_synergy_scores) if self.model_synergy_scores else 0.0
        
        return {
            'total_llm_solvers': len(self.all_llm_solvers),
            'gpt_oss_solvers': len(self.gpt_oss_solvers),
            'kanana_solvers': len(self.kanana_solvers),
            'ensemble_weights': self.ensemble_weights,
            'cross_model_agreement_rate': cross_model_agreement_rate,
            'average_synergy_score': avg_synergy,
            'total_comparisons': self.total_comparisons,
            'synergy_trend': 'improving' if len(self.model_synergy_scores) > 1 and 
                           self.model_synergy_scores[-1] > self.model_synergy_scores[0] else 'stable'
        }

# Factory functions
def create_gpt_oss_solver(api_endpoint: str, api_key: str = None, temperature: float = 0.3) -> GPTOSSSolver:
    """Factory function for GPT-OSS-20B solver"""
    config = ModelConfig(
        model_name="gpt-oss-20b",
        api_endpoint=api_endpoint,
        api_key=api_key,
        temperature=temperature
    )
    return GPTOSSSolver(config)

def create_kanana_solver(api_endpoint: str, api_key: str = None, temperature: float = 0.2) -> KananaSolver:
    """Factory function for Kanana-1.5-15.7B-A3B solver"""
    config = ModelConfig(
        model_name="kanana-1.5-15.7b-a3b",
        api_endpoint=api_endpoint,
        api_key=api_key,
        temperature=temperature
    )
    return KananaSolver(config)

def create_multi_llm_ensemble(gpt_oss_endpoint: str, 
                             kanana_endpoint: str,
                             gpt_oss_key: str = None,
                             kanana_key: str = None) -> MultiLLMEnsemble:
    """Factory function for multi-LLM ensemble"""
    
    gpt_oss_config = ModelConfig(
        model_name="gpt-oss-20b",
        api_endpoint=gpt_oss_endpoint,
        api_key=gpt_oss_key,
        temperature=0.3
    )
    
    kanana_config = ModelConfig(
        model_name="kanana-1.5-15.7b-a3b", 
        api_endpoint=kanana_endpoint,
        api_key=kanana_key,
        temperature=0.2
    )
    
    return MultiLLMEnsemble(gpt_oss_config, kanana_config)

if __name__ == "__main__":
    print("Multi-LLM Wrapper for ARC-IDEA")
    print("=" * 40)
    print("Support for:")
    print("- GPT-OSS-20B: Large-scale reasoning")
    print("- Kanana-1.5-15.7B-A3B: Analytical precision")
    print("- Multi-model ensemble consensus")
    print("- Cross-architecture synergy analysis")
    print("\\nUsage:")
    print("ensemble = create_multi_llm_ensemble(gpt_endpoint, kanana_endpoint)")
    print("solvers = ensemble.get_all_llm_solvers()")