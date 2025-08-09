#!/usr/bin/env python3
"""
ARC Training Data Loader for ARC-IDEA System

This module provides comprehensive training data loading and management for the ARC-IDEA
system with k-fold cross-validation support, integrated with the multi-LLM architecture.

Features:
- K-fold cross-validation data loading  
- Batch processing for training/validation
- Integration with EFE system and RevThink verification
- Multi-LLM training support (GPT-OSS-20B + Kanana-1.5-15.7B-A3B)
- Complexity-aware sampling and metrics
"""

import json
import numpy as np
import os
from typing import Dict, List, Tuple, Any, Optional, Iterator, Union
from dataclasses import dataclass, field
from collections import defaultdict
import random
from abc import ABC, abstractmethod

# Import ARC-IDEA components
from EFE_update import ARCState, RevThinkVerifier
from multi_llm_integration import MultiLLMEnhancedEFESystem, ModelConfig
from example_solvers import BaseSolver

@dataclass
class ARCTrainingExample:
    """Single ARC training example"""
    challenge_id: str
    train_examples: List[Dict[str, List[List[int]]]]  # List of input/output pairs
    test_examples: List[Dict[str, List[List[int]]]]   # Test inputs (no outputs)
    solutions: List[List[List[int]]]                  # Ground truth test outputs
    metadata: Dict[str, Any]
    complexity_score: float
    
    def get_training_pairs(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get input-output training pairs as numpy arrays"""
        pairs = []
        for example in self.train_examples:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])
            pairs.append((input_grid, output_grid))
        return pairs
    
    def get_test_inputs(self) -> List[np.ndarray]:
        """Get test inputs as numpy arrays"""
        return [np.array(example['input']) for example in self.test_examples]
    
    def get_test_outputs(self) -> List[np.ndarray]:
        """Get ground truth test outputs as numpy arrays"""
        return [np.array(solution) for solution in self.solutions]

@dataclass 
class ARCTrainingBatch:
    """Batch of ARC training examples"""
    examples: List[ARCTrainingExample]
    batch_id: int
    fold_number: int
    is_validation: bool
    complexity_distribution: Dict[str, float]
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def get_complexity_stats(self) -> Dict[str, float]:
        """Get complexity statistics for the batch"""
        complexities = [ex.complexity_score for ex in self.examples]
        return {
            'mean': np.mean(complexities),
            'std': np.std(complexities),
            'min': np.min(complexities),
            'max': np.max(complexities),
            'median': np.median(complexities)
        }

class ARCDatasetLoader:
    """Main dataset loader for ARC training data"""
    
    def __init__(self, data_dir: str = "arc_processed_data"):
        self.data_dir = data_dir
        self.merged_data_file = os.path.join(data_dir, "merged_arc_training_data.json")
        self.k_fold_dir = os.path.join(data_dir, "k_fold_splits")
        self.summary_file = os.path.join(data_dir, "dataset_summary.json")
        
        # Load dataset components
        self.merged_data = self._load_merged_data()
        self.dataset_summary = self._load_dataset_summary()
        self.k_fold_data = self._load_k_fold_data()
        
        print(f"Loaded ARC dataset: {len(self.merged_data)} challenges, {len(self.k_fold_data)} folds")
    
    def _load_merged_data(self) -> Dict[str, Dict]:
        """Load merged training data"""
        try:
            with open(self.merged_data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Merged data file not found: {self.merged_data_file}")
    
    def _load_dataset_summary(self) -> Dict:
        """Load dataset summary statistics"""
        try:
            with open(self.summary_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def _load_k_fold_data(self) -> Dict[str, Dict]:
        """Load all k-fold split data"""
        k_fold_data = {}
        
        for fold_file in os.listdir(self.k_fold_dir):
            if fold_file.endswith('.json') and fold_file.startswith('fold_'):
                fold_path = os.path.join(self.k_fold_dir, fold_file)
                fold_name = fold_file.replace('.json', '')
                
                with open(fold_path, 'r', encoding='utf-8') as f:
                    k_fold_data[fold_name] = json.load(f)
        
        return k_fold_data
    
    def get_fold_data(self, fold_number: int) -> Dict[str, Any]:
        """Get data for specific fold"""
        fold_key = f"fold_{fold_number}"
        if fold_key not in self.k_fold_data:
            raise ValueError(f"Fold {fold_number} not found. Available folds: {list(self.k_fold_data.keys())}")
        
        return self.k_fold_data[fold_key]
    
    def create_training_examples(self, challenge_ids: List[str]) -> List[ARCTrainingExample]:
        """Create training examples from challenge IDs"""
        examples = []
        
        for challenge_id in challenge_ids:
            if challenge_id not in self.merged_data:
                print(f"Warning: Challenge {challenge_id} not found in merged data")
                continue
            
            challenge_data = self.merged_data[challenge_id]
            
            example = ARCTrainingExample(
                challenge_id=challenge_id,
                train_examples=challenge_data['train'],
                test_examples=challenge_data['test'], 
                solutions=challenge_data['solutions'],
                metadata=challenge_data['metadata'],
                complexity_score=challenge_data['metadata']['complexity_score']
            )
            
            examples.append(example)
        
        return examples
    
    def create_batch_iterator(self, 
                             fold_number: int, 
                             batch_size: int = 10,
                             validation: bool = False,
                             shuffle: bool = True) -> Iterator[ARCTrainingBatch]:
        """Create batch iterator for training or validation"""
        
        fold_data = self.get_fold_data(fold_number)
        
        if validation:
            challenge_ids = fold_data['validation_challenges']
        else:
            challenge_ids = fold_data['train_challenges']
        
        if shuffle:
            challenge_ids = challenge_ids.copy()
            random.shuffle(challenge_ids)
        
        # Create batches
        batch_id = 0
        for i in range(0, len(challenge_ids), batch_size):
            batch_challenge_ids = challenge_ids[i:i + batch_size]
            examples = self.create_training_examples(batch_challenge_ids)
            
            # Calculate complexity distribution for this batch
            complexities = [ex.complexity_score for ex in examples]
            complexity_dist = {
                'mean': float(np.mean(complexities)),
                'std': float(np.std(complexities)),
                'range': float(np.max(complexities) - np.min(complexities))
            }
            
            batch = ARCTrainingBatch(
                examples=examples,
                batch_id=batch_id,
                fold_number=fold_number,
                is_validation=validation,
                complexity_distribution=complexity_dist
            )
            
            batch_id += 1
            yield batch
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        stats = {
            'total_challenges': len(self.merged_data),
            'total_folds': len(self.k_fold_data),
            'complexity_distribution': self.dataset_summary.get('complexity_distribution', {}),
            'shape_analysis': self.dataset_summary.get('shape_analysis', {}),
            'fold_statistics': {}
        }
        
        # Add fold-specific statistics
        for fold_name, fold_data in self.k_fold_data.items():
            stats['fold_statistics'][fold_name] = fold_data['statistics']
        
        return stats

class ARCTrainingManager:
    """Manages ARC training process with multi-LLM integration"""
    
    def __init__(self, 
                 data_loader: ARCDatasetLoader,
                 system: MultiLLMEnhancedEFESystem,
                 training_config: Dict[str, Any] = None):
        
        self.data_loader = data_loader
        self.system = system
        self.config = training_config or self._default_training_config()
        
        # Training state
        self.training_history = []
        self.validation_history = []
        self.current_fold = None
        self.current_epoch = 0
        
        # Performance tracking
        self.fold_performances = {}
        self.best_model_state = None
        self.best_validation_score = -float('inf')
        
        print(f"Training manager initialized with {len(self.data_loader.merged_data)} challenges")
    
    def _default_training_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            'epochs_per_fold': 5,
            'batch_size': 8,
            'validation_frequency': 2,  # Validate every N batches
            'early_stopping_patience': 10,
            'learning_rate_schedule': 'adaptive',
            'complexity_weighting': True,
            'revthink_integration': True,
            'save_checkpoints': True,
            'checkpoint_dir': 'arc_checkpoints'
        }
    
    def train_single_fold(self, fold_number: int) -> Dict[str, Any]:
        """Train on a single fold"""
        print(f"\nStarting training on fold {fold_number}")
        
        self.current_fold = fold_number
        fold_results = {
            'fold_number': fold_number,
            'training_batches': [],
            'validation_batches': [],
            'epoch_results': [],
            'best_epoch': -1,
            'best_validation_score': -float('inf')
        }
        
        # Create batch iterators
        train_iterator = self.data_loader.create_batch_iterator(
            fold_number=fold_number, 
            batch_size=self.config['batch_size'],
            validation=False,
            shuffle=True
        )
        
        validation_iterator = self.data_loader.create_batch_iterator(
            fold_number=fold_number,
            batch_size=self.config['batch_size'], 
            validation=True,
            shuffle=False
        )
        
        # Training loop
        for epoch in range(self.config['epochs_per_fold']):
            print(f"  Epoch {epoch + 1}/{self.config['epochs_per_fold']}")
            
            epoch_results = {
                'epoch': epoch + 1,
                'training_metrics': {},
                'validation_metrics': {},
                'batch_results': []
            }
            
            # Training phase
            train_metrics = self._train_epoch(train_iterator, fold_results)
            epoch_results['training_metrics'] = train_metrics
            
            # Validation phase
            if (epoch + 1) % self.config['validation_frequency'] == 0:
                val_metrics = self._validate_epoch(validation_iterator, fold_results)
                epoch_results['validation_metrics'] = val_metrics
                
                # Check for best model
                val_score = val_metrics.get('combined_score', -float('inf'))
                if val_score > fold_results['best_validation_score']:
                    fold_results['best_validation_score'] = val_score
                    fold_results['best_epoch'] = epoch + 1
                    
                    if val_score > self.best_validation_score:
                        self.best_validation_score = val_score
                        self._save_best_model()
            
            fold_results['epoch_results'].append(epoch_results)
            
            # Early stopping check
            if self._should_early_stop(fold_results):
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break
        
        self.fold_performances[fold_number] = fold_results
        return fold_results
    
    def _train_epoch(self, train_iterator: Iterator[ARCTrainingBatch], fold_results: Dict) -> Dict[str, float]:
        """Train for one epoch"""
        epoch_metrics = {
            'total_batches': 0,
            'successful_batches': 0,
            'average_efe_score': 0.0,
            'average_verification_score': 0.0,
            'average_complexity': 0.0,
            'processing_time': 0.0
        }
        
        batch_count = 0
        total_efe = 0.0
        total_verification = 0.0
        total_complexity = 0.0
        
        for batch in train_iterator:
            batch_count += 1
            
            try:
                batch_results = self._process_training_batch(batch)
                fold_results['training_batches'].append(batch_results)
                
                # Accumulate metrics
                total_efe += batch_results.get('average_efe', 0.0)
                total_verification += batch_results.get('average_verification', 0.0) 
                total_complexity += batch_results.get('average_complexity', 0.0)
                
                epoch_metrics['successful_batches'] += 1
                
                if batch_count % 10 == 0:
                    print(f"    Processed {batch_count} training batches...")
                
            except Exception as e:
                print(f"    Warning: Training batch {batch_count} failed: {e}")
                continue
        
        # Calculate averages
        if epoch_metrics['successful_batches'] > 0:
            epoch_metrics['average_efe_score'] = total_efe / epoch_metrics['successful_batches']
            epoch_metrics['average_verification_score'] = total_verification / epoch_metrics['successful_batches']
            epoch_metrics['average_complexity'] = total_complexity / epoch_metrics['successful_batches']
        
        epoch_metrics['total_batches'] = batch_count
        return epoch_metrics
    
    def _validate_epoch(self, val_iterator: Iterator[ARCTrainingBatch], fold_results: Dict) -> Dict[str, float]:
        """Validate for one epoch"""
        val_metrics = {
            'total_batches': 0,
            'successful_batches': 0,
            'average_accuracy': 0.0,
            'average_efe_score': 0.0,
            'average_verification_score': 0.0,
            'combined_score': 0.0
        }
        
        batch_count = 0
        total_accuracy = 0.0
        total_efe = 0.0
        total_verification = 0.0
        
        for batch in val_iterator:
            batch_count += 1
            
            try:
                batch_results = self._process_validation_batch(batch)
                fold_results['validation_batches'].append(batch_results)
                
                # Accumulate metrics  
                total_accuracy += batch_results.get('accuracy', 0.0)
                total_efe += batch_results.get('average_efe', 0.0)
                total_verification += batch_results.get('average_verification', 0.0)
                
                val_metrics['successful_batches'] += 1
                
            except Exception as e:
                print(f"    Warning: Validation batch {batch_count} failed: {e}")
                continue
        
        # Calculate averages and combined score
        if val_metrics['successful_batches'] > 0:
            val_metrics['average_accuracy'] = total_accuracy / val_metrics['successful_batches']
            val_metrics['average_efe_score'] = total_efe / val_metrics['successful_batches']
            val_metrics['average_verification_score'] = total_verification / val_metrics['successful_batches']
            
            # Combined score balances accuracy and verification quality
            val_metrics['combined_score'] = (
                val_metrics['average_accuracy'] * 0.6 +
                val_metrics['average_verification_score'] * 0.4
            )
        
        val_metrics['total_batches'] = batch_count
        return val_metrics
    
    def _process_training_batch(self, batch: ARCTrainingBatch) -> Dict[str, Any]:
        """Process a single training batch"""
        batch_results = {
            'batch_id': batch.batch_id,
            'fold_number': batch.fold_number,
            'examples_processed': 0,
            'efe_scores': [],
            'verification_scores': [],
            'complexity_scores': [],
            'processing_details': []
        }
        
        for example in batch.examples:
            try:
                # Get training pairs
                training_pairs = example.get_training_pairs()
                
                if not training_pairs:
                    continue
                
                # Use first training pair for system processing
                input_grid, expected_output = training_pairs[0]
                
                # Create constraints from metadata
                constraints = {
                    'complexity_score': example.complexity_score,
                    'challenge_id': example.challenge_id,
                    'num_train_examples': len(training_pairs)
                }
                
                # Process with multi-LLM system
                solution, results = self.system.solve_with_multi_llm_ensemble(
                    input_grid, constraints
                )
                
                # Extract metrics
                efe_score = results.get('final_loss', {}).get('efe_loss', 1.0)
                verification_score = results.get('verification_results', {}).get('combined_score', 0.5)
                
                batch_results['efe_scores'].append(efe_score)
                batch_results['verification_scores'].append(verification_score)
                batch_results['complexity_scores'].append(example.complexity_score)
                batch_results['examples_processed'] += 1
                
                # Store processing details
                batch_results['processing_details'].append({
                    'challenge_id': example.challenge_id,
                    'efe_score': efe_score,
                    'verification_score': verification_score,
                    'accuracy': float(np.mean(solution == expected_output)) if solution.shape == expected_output.shape else 0.0
                })
                
            except Exception as e:
                print(f"      Warning: Failed to process example {example.challenge_id}: {e}")
                continue
        
        # Calculate batch averages
        if batch_results['examples_processed'] > 0:
            batch_results['average_efe'] = np.mean(batch_results['efe_scores'])
            batch_results['average_verification'] = np.mean(batch_results['verification_scores'])
            batch_results['average_complexity'] = np.mean(batch_results['complexity_scores'])
        else:
            batch_results['average_efe'] = 1.0
            batch_results['average_verification'] = 0.0
            batch_results['average_complexity'] = 0.5
        
        return batch_results
    
    def _process_validation_batch(self, batch: ARCTrainingBatch) -> Dict[str, Any]:
        """Process a single validation batch"""
        batch_results = {
            'batch_id': batch.batch_id,
            'fold_number': batch.fold_number,
            'examples_processed': 0,
            'correct_predictions': 0,
            'efe_scores': [],
            'verification_scores': [],
            'accuracy_scores': [],
            'validation_details': []
        }
        
        for example in batch.examples:
            try:
                # Get test data
                test_inputs = example.get_test_inputs()
                test_outputs = example.get_test_outputs()
                
                if not test_inputs or not test_outputs:
                    continue
                
                # Use first test case
                test_input = test_inputs[0]
                expected_output = test_outputs[0]
                
                # Create constraints
                constraints = {
                    'complexity_score': example.complexity_score,
                    'challenge_id': example.challenge_id,
                    'validation_mode': True
                }
                
                # Process with system
                solution, results = self.system.solve_with_multi_llm_ensemble(
                    test_input, constraints
                )
                
                # Calculate accuracy
                accuracy = 0.0
                if solution.shape == expected_output.shape:
                    accuracy = float(np.mean(solution == expected_output))
                    if accuracy > 0.95:  # Near-perfect match
                        batch_results['correct_predictions'] += 1
                
                # Extract metrics
                efe_score = results.get('final_loss', {}).get('efe_loss', 1.0)
                verification_score = results.get('verification_results', {}).get('combined_score', 0.5)
                
                batch_results['efe_scores'].append(efe_score)
                batch_results['verification_scores'].append(verification_score)
                batch_results['accuracy_scores'].append(accuracy)
                batch_results['examples_processed'] += 1
                
                batch_results['validation_details'].append({
                    'challenge_id': example.challenge_id,
                    'accuracy': accuracy,
                    'efe_score': efe_score,
                    'verification_score': verification_score
                })
                
            except Exception as e:
                print(f"      Warning: Failed to validate example {example.challenge_id}: {e}")
                continue
        
        # Calculate batch averages
        if batch_results['examples_processed'] > 0:
            batch_results['accuracy'] = batch_results['correct_predictions'] / batch_results['examples_processed']
            batch_results['average_efe'] = np.mean(batch_results['efe_scores'])
            batch_results['average_verification'] = np.mean(batch_results['verification_scores'])
            batch_results['average_accuracy_score'] = np.mean(batch_results['accuracy_scores'])
        else:
            batch_results['accuracy'] = 0.0
            batch_results['average_efe'] = 1.0
            batch_results['average_verification'] = 0.0
            batch_results['average_accuracy_score'] = 0.0
        
        return batch_results
    
    def _should_early_stop(self, fold_results: Dict) -> bool:
        """Check if early stopping should be triggered"""
        if len(fold_results['epoch_results']) < 3:
            return False
        
        # Check if validation score hasn't improved in recent epochs
        recent_epochs = fold_results['epoch_results'][-3:]
        val_scores = [
            epoch.get('validation_metrics', {}).get('combined_score', -float('inf'))
            for epoch in recent_epochs
        ]
        
        # Early stop if no improvement in last 3 epochs
        if all(score <= fold_results['best_validation_score'] * 0.95 for score in val_scores):
            return True
        
        return False
    
    def _save_best_model(self):
        """Save the best model state"""
        if self.config['save_checkpoints']:
            checkpoint_dir = self.config['checkpoint_dir']
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model state (simplified - in practice would save actual model weights)
            checkpoint_data = {
                'best_validation_score': self.best_validation_score,
                'current_fold': self.current_fold,
                'current_epoch': self.current_epoch,
                'system_config': {
                    'model_configs': 'placeholder',  # Would store actual model configurations
                    'training_config': self.config
                }
            }
            
            checkpoint_file = os.path.join(checkpoint_dir, 'best_model.json')
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    def train_k_fold_cv(self, k_folds: int = 10) -> Dict[str, Any]:
        """Run complete k-fold cross-validation training"""
        print(f"\nStarting {k_folds}-fold cross-validation training")
        
        cv_results = {
            'total_folds': k_folds,
            'fold_results': {},
            'overall_metrics': {},
            'best_fold': -1,
            'best_score': -float('inf')
        }
        
        fold_scores = []
        
        for fold in range(1, k_folds + 1):
            print(f"\n{'='*60}")
            print(f"TRAINING FOLD {fold}/{k_folds}")
            print(f"{'='*60}")
            
            try:
                fold_results = self.train_single_fold(fold)
                cv_results['fold_results'][fold] = fold_results
                
                # Track best fold
                fold_score = fold_results['best_validation_score']
                fold_scores.append(fold_score)
                
                if fold_score > cv_results['best_score']:
                    cv_results['best_score'] = fold_score
                    cv_results['best_fold'] = fold
                
                print(f"Fold {fold} completed - Best validation score: {fold_score:.4f}")
                
            except Exception as e:
                print(f"ERROR: Fold {fold} failed: {e}")
                fold_scores.append(-1.0)  # Mark as failed
                continue
        
        # Calculate overall metrics
        valid_scores = [score for score in fold_scores if score > 0]
        if valid_scores:
            cv_results['overall_metrics'] = {
                'mean_validation_score': np.mean(valid_scores),
                'std_validation_score': np.std(valid_scores),
                'min_validation_score': np.min(valid_scores),
                'max_validation_score': np.max(valid_scores),
                'successful_folds': len(valid_scores),
                'failed_folds': k_folds - len(valid_scores)
            }
        
        print(f"\n{'='*60}")
        print(f"K-FOLD CROSS-VALIDATION COMPLETE")
        print(f"{'='*60}")
        print(f"Successful folds: {len(valid_scores)}/{k_folds}")
        print(f"Mean validation score: {cv_results['overall_metrics'].get('mean_validation_score', 0.0):.4f}")
        print(f"Best fold: {cv_results['best_fold']} (score: {cv_results['best_score']:.4f})")
        
        return cv_results

# Factory functions
def create_arc_training_setup(data_dir: str = "arc_processed_data",
                             gpt_oss_config: ModelConfig = None,
                             kanana_config: ModelConfig = None) -> Tuple[ARCDatasetLoader, ARCTrainingManager]:
    """Create complete ARC training setup"""
    
    # Load dataset
    data_loader = ARCDatasetLoader(data_dir)
    
    # Create default model configs if not provided
    if not gpt_oss_config:
        gpt_oss_config = ModelConfig(
            model_name="gpt-oss-20b",
            api_endpoint="https://api.mock-gpt-oss.com/v1/chat",
            api_key="mock-key",
            temperature=0.3
        )
    
    if not kanana_config:
        kanana_config = ModelConfig(
            model_name="kanana-1.5-15.7b-a3b", 
            api_endpoint="https://api.mock-kanana.com/v1/generate",
            api_key="mock-key",
            temperature=0.2
        )
    
    # Create traditional solvers for baseline
    from example_solvers import BaseSolver
    class MockSolver(BaseSolver):
        def __init__(self, name):
            self.solver_name = name
        def predict(self, grid):
            return grid  # Identity transformation for testing
    
    traditional_solvers = [
        MockSolver("IdentitySolver"),
        MockSolver("FlipSolver")  
    ]
    
    # Create multi-LLM system
    system = MultiLLMEnhancedEFESystem(
        traditional_solvers=traditional_solvers,
        gpt_oss_config=gpt_oss_config,
        kanana_config=kanana_config,
        consensus_threshold=0.6
    )
    
    # Create training manager
    training_config = {
        'epochs_per_fold': 3,  # Reduced for initial testing
        'batch_size': 5,       # Smaller batches for testing
        'validation_frequency': 1,
        'early_stopping_patience': 5,
        'save_checkpoints': True
    }
    
    training_manager = ARCTrainingManager(
        data_loader=data_loader,
        system=system,
        training_config=training_config
    )
    
    return data_loader, training_manager

if __name__ == "__main__":
    # Demo usage
    print("ARC Training System Demo")
    print("="*40)
    
    try:
        data_loader, training_manager = create_arc_training_setup()
        
        # Show dataset statistics
        stats = data_loader.get_dataset_statistics()
        print(f"Dataset loaded: {stats['total_challenges']} challenges across {stats['total_folds']} folds")
        
        # Test single fold training (demo with 1 fold only)
        print("\nRunning single fold demo...")
        fold_results = training_manager.train_single_fold(1)
        print(f"Fold 1 completed with validation score: {fold_results['best_validation_score']:.4f}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()