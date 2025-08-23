"""
Comprehensive Training Runner for ARC Challenge
Integrates all components: Mistral, EFE, TTA, Solvers with extracted features
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
from typing import Dict, List, Tuple, Any
import argparse
import sys

# Add parent directory to path
sys.path.append('..')

# Import all components
from mistral_training_sequence import MistralARCTrainer, TrainingConfig
from exp_tta import TestTimeAdaptationSystem, create_test_adaptation_system
from exp_EFE import ARCPromptGuidedAgent
from exp_prompt import create_arc_prompt_template
from system1_solver import System1Solver
from system2_solver import System2Solver

class ComprehensiveARCTrainer:
    """Comprehensive trainer integrating all ARC components"""
    
    def __init__(self, 
                 config: TrainingConfig,
                 enable_tta: bool = True,
                 enable_solvers: bool = True,
                 verbose: bool = True):
        
        self.config = config
        self.enable_tta = enable_tta
        self.enable_solvers = enable_solvers
        self.verbose = verbose
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.setup_components()
        
    def setup_components(self):
        """Setup all training components"""
        
        self.logger.info("Setting up comprehensive ARC training components...")
        
        # 1. Main Mistral trainer
        self.mistral_trainer = MistralARCTrainer(self.config)
        
        # 2. Test-time adaptation system
        if self.enable_tta:
            self.tta_system = create_test_adaptation_system()
            if torch.cuda.is_available() and self.config.device == "cuda":
                self.tta_system = self.tta_system.cuda()
        
        # 3. Solver systems
        if self.enable_solvers:
            self.system1 = System1Solver(debug=self.verbose)
            self.system2 = System2Solver(debug=self.verbose)
        
        # 4. Load features and data
        self.load_training_data()
        
    def load_training_data(self):
        """Load all training data and features"""
        
        self.logger.info("Loading training data and features...")
        
        # Load split training data
        with open('training_train.json', 'r') as f:
            self.train_data = json.load(f)
            
        with open('training_test.json', 'r') as f:
            self.test_data = json.load(f)
            
        # Load extracted features
        self.features_df = pd.read_csv('extracted_features.csv')
        
        self.logger.info(f"Loaded {len(self.train_data)} training problems")
        self.logger.info(f"Loaded {len(self.test_data)} test problems")
        self.logger.info(f"Loaded features for {self.features_df['prob_id'].nunique()} problems")
        
    def run_comprehensive_training(self):
        """Run comprehensive training sequence"""
        
        self.logger.info("Starting comprehensive ARC training sequence...")
        
        # Phase 1: Pre-training with Mistral + EFE
        self.logger.info("Phase 1: Mistral + EFE Pre-training")
        self.mistral_trainer.train()
        
        # Phase 2: Test-time adaptation training (if enabled)
        if self.enable_tta:
            self.logger.info("Phase 2: Test-time adaptation training")
            self.run_tta_training()
        
        # Phase 3: Solver integration training (if enabled)
        if self.enable_solvers:
            self.logger.info("Phase 3: Solver integration training")
            self.run_solver_integration()
        
        # Phase 4: Comprehensive evaluation
        self.logger.info("Phase 4: Comprehensive evaluation")
        results = self.run_comprehensive_evaluation()
        
        return results
    
    def run_tta_training(self):
        """Run test-time adaptation training"""
        
        self.logger.info("Training test-time adaptation system...")
        
        # Sample problems for TTA training
        sample_problems = list(self.train_data.keys())[:50]  # Use first 50 for TTA training
        
        tta_results = []
        
        for prob_id in sample_problems:
            prob_data = self.train_data[prob_id]
            prob_features = self.features_df[self.features_df['prob_id'] == prob_id]
            
            if len(prob_features) == 0 or len(prob_data.get('train', [])) == 0:
                continue
                
            try:
                # Get first training example
                example = prob_data['train'][0]
                input_grid = torch.tensor(example['input'], dtype=torch.long)
                target_grid = torch.tensor(example['output'], dtype=torch.long)
                
                # Create prompt embedding (simplified)
                prompt_text = self.create_prompt_from_features(prob_features)
                prompt_embedding = self.create_simple_prompt_embedding(prompt_text)
                
                # Run TTA
                tta_result = self.tta_system.test_time_adapt(
                    input_grid,
                    target_grid,
                    prompt_text,
                    prompt_embedding
                )
                
                tta_results.append({
                    'prob_id': prob_id,
                    'success': tta_result['final_surprise'] < 1.0,  # Simplified success criteria
                    'surprise': tta_result['final_surprise'],
                    'memory_size': tta_result['memory_size']
                })
                
            except Exception as e:
                self.logger.warning(f"TTA error for {prob_id}: {e}")
                continue
        
        self.logger.info(f"TTA training completed on {len(tta_results)} problems")
        
        # Save TTA results
        with open('tta_training_results.json', 'w') as f:
            json.dump(tta_results, f, indent=2)
    
    def run_solver_integration(self):
        """Run solver integration training"""
        
        self.logger.info("Training solver integration...")
        
        solver_results = {
            'system1_success': 0,
            'system2_success': 0,
            'total_problems': 0
        }
        
        # Test solvers on sample problems
        sample_problems = list(self.train_data.keys())[:100]
        
        for prob_id in sample_problems:
            prob_data = self.train_data[prob_id]
            
            if len(prob_data.get('train', [])) == 0:
                continue
                
            try:
                example = prob_data['train'][0]
                input_grid = np.array(example['input'])
                target_grid = np.array(example['output'])
                
                # System 1 attempt
                s1_output, s1_conf, s1_method = self.system1.solve(
                    input_grid, "CNN", {}
                )
                
                if s1_output is not None and np.array_equal(s1_output, target_grid):
                    solver_results['system1_success'] += 1
                else:
                    # System 2 attempt
                    from arc_dual_solver import SolverResult, SolutionConfidence, SolverType
                    s1_result = SolverResult(
                        success=False,
                        output_grid=s1_output,
                        confidence=s1_conf,
                        solver_type=SolverType.SYSTEM_1,
                        reasoning_trace=[],
                        execution_time=0.1,
                        consistency_checks={},
                        method_used=s1_method
                    )
                    
                    s2_output, s2_conf, s2_reasoning = self.system2.solve(
                        input_grid, "CNN", {}, s1_result
                    )
                    
                    if s2_output is not None and np.array_equal(s2_output, target_grid):
                        solver_results['system2_success'] += 1
                
                solver_results['total_problems'] += 1
                
            except Exception as e:
                self.logger.warning(f"Solver integration error for {prob_id}: {e}")
                continue
        
        self.logger.info(
            f"Solver integration: S1={solver_results['system1_success']}/{solver_results['total_problems']}, "
            f"S2={solver_results['system2_success']}/{solver_results['total_problems']}"
        )
        
        # Save solver results
        with open('solver_integration_results.json', 'w') as f:
            json.dump(solver_results, f, indent=2)
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on test set"""
        
        self.logger.info("Running comprehensive evaluation...")
        
        results = {
            'mistral_accuracy': 0.0,
            'tta_performance': {},
            'solver_performance': {},
            'combined_performance': {},
            'total_problems_evaluated': 0
        }
        
        # Evaluate on test problems
        test_problems = list(self.test_data.keys())[:50]  # Limit for testing
        successful_problems = 0
        
        for prob_id in test_problems:
            prob_data = self.test_data[prob_id]
            prob_features = self.features_df[self.features_df['prob_id'] == prob_id]
            
            if len(prob_features) == 0 or len(prob_data.get('train', [])) == 0:
                continue
                
            try:
                example = prob_data['train'][0]
                input_grid = np.array(example['input'])
                target_grid = np.array(example['output'])
                
                problem_success = False
                
                # Try different approaches
                approaches_tried = []
                
                # 1. Solver approach
                if self.enable_solvers:
                    try:
                        s1_output, s1_conf, _ = self.system1.solve(input_grid, "CNN", {})
                        if s1_output is not None and np.array_equal(s1_output, target_grid):
                            problem_success = True
                            approaches_tried.append("system1")
                    except:
                        pass
                
                # 2. TTA approach
                if not problem_success and self.enable_tta:
                    try:
                        prompt_text = self.create_prompt_from_features(prob_features)
                        prompt_embedding = self.create_simple_prompt_embedding(prompt_text)
                        
                        tta_result = self.tta_system.test_time_adapt(
                            torch.tensor(input_grid, dtype=torch.long),
                            torch.tensor(target_grid, dtype=torch.long),
                            prompt_text,
                            prompt_embedding
                        )
                        
                        # Simple success criteria based on surprise
                        if tta_result['final_surprise'] < 0.5:
                            problem_success = True
                            approaches_tried.append("tta")
                            
                    except Exception as e:
                        self.logger.warning(f"TTA evaluation error: {e}")
                
                if problem_success:
                    successful_problems += 1
                
                results['total_problems_evaluated'] += 1
                
            except Exception as e:
                self.logger.warning(f"Evaluation error for {prob_id}: {e}")
                continue
        
        if results['total_problems_evaluated'] > 0:
            results['combined_performance']['success_rate'] = successful_problems / results['total_problems_evaluated']
        
        self.logger.info(
            f"Comprehensive evaluation completed: "
            f"{successful_problems}/{results['total_problems_evaluated']} problems solved "
            f"({results['combined_performance'].get('success_rate', 0):.2%} success rate)"
        )
        
        # Save comprehensive results
        with open('comprehensive_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def create_prompt_from_features(self, features: pd.DataFrame) -> str:
        """Create prompt text from features"""
        
        if len(features) == 0:
            return "Analyze the input-output relationship and apply the transformation rule."
            
        # Simple prompt generation from features
        backgrounds = features['background'].unique()
        colors = features['color'].unique()
        
        prompt_parts = []
        
        if 'no' in backgrounds:
            prompt_parts.append("Focus on the pattern without background elements")
        if len(colors) <= 3:
            prompt_parts.append(f"Work with {len(colors)} main colors")
        
        if not prompt_parts:
            prompt_parts.append("Identify and apply the transformation pattern")
            
        return ". ".join(prompt_parts)
    
    def create_simple_prompt_embedding(self, prompt_text: str) -> torch.Tensor:
        """Create simple prompt embedding for testing"""
        
        # Simple hash-based embedding for testing
        import hashlib
        hash_val = int(hashlib.md5(prompt_text.encode()).hexdigest()[:8], 16)
        
        # Create deterministic embedding
        torch.manual_seed(hash_val % 10000)
        embedding = torch.randn(768)  # Standard BERT-like dimension
        
        return embedding
    
    def save_training_summary(self, results: Dict[str, Any]):
        """Save comprehensive training summary"""
        
        summary = {
            'config': {
                'max_epochs': self.config.max_epochs,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate,
                'tta_enabled': self.enable_tta,
                'solvers_enabled': self.enable_solvers
            },
            'results': results,
            'training_time': time.time(),
            'data_stats': {
                'train_problems': len(self.train_data),
                'test_problems': len(self.test_data),
                'feature_records': len(self.features_df)
            }
        }
        
        with open('comprehensive_training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info("Training summary saved to comprehensive_training_summary.json")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Comprehensive ARC Training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA training')
    parser.add_argument('--no-solvers', action='store_true', help='Disable solver integration')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    
    args = parser.parse_args()
    
    # Set working directory
    import os
    os.chdir(Path(__file__).parent)
    
    # Create configuration
    config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
    
    # Create comprehensive trainer
    trainer = ComprehensiveARCTrainer(
        config=config,
        enable_tta=not args.no_tta,
        enable_solvers=not args.no_solvers,
        verbose=not args.quiet
    )
    
    # Run comprehensive training
    start_time = time.time()
    results = trainer.run_comprehensive_training()
    total_time = time.time() - start_time
    
    # Save final summary
    results['total_training_time'] = total_time
    trainer.save_training_summary(results)
    
    print(f"\nComprehensive training completed in {total_time:.2f} seconds!")
    print(f"Results saved to comprehensive_training_summary.json")
    
    if results.get('combined_performance', {}).get('success_rate'):
        print(f"Final success rate: {results['combined_performance']['success_rate']:.2%}")

if __name__ == "__main__":
    main()