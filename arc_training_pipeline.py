#!/usr/bin/env python3
"""
ARC-IDEA Training Pipeline

Complete training pipeline for ARC-IDEA multi-LLM system with:
- K-fold cross-validation training
- GPT-OSS-20B + Kanana-1.5-15.7B-A3B integration  
- RevThink prompt-based verification
- EFE-based solver selection
- Comprehensive performance tracking and analysis

Usage:
    python arc_training_pipeline.py [options]
"""

import os
import json
import time
import argparse
from typing import Dict, List, Tuple, Any
from datetime import datetime

# Import ARC-IDEA components
from arc_training_loader import ARCDatasetLoader, ARCTrainingManager, create_arc_training_setup
from multi_llm_integration import ModelConfig
from EFE_update import RevThinkVerifier

class ARCTrainingPipeline:
    """Complete ARC training pipeline orchestrator"""
    
    def __init__(self, 
                 config_file: str = None,
                 output_dir: str = "arc_training_results"):
        
        self.config = self._load_config(config_file)
        self.output_dir = output_dir
        self.start_time = datetime.now()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.data_loader = None
        self.training_manager = None
        self.results = {}
        
        print(f"ARC Training Pipeline initialized")
        print(f"Output directory: {output_dir}")
        print(f"Configuration: {len(self.config)} parameters")
    
    def _load_config(self, config_file: str = None) -> Dict[str, Any]:
        """Load training configuration"""
        default_config = {
            # Data settings
            'data_dir': 'arc_processed_data',
            'k_folds': 10,
            
            # Model settings
            'gpt_oss_endpoint': 'https://api.mock-gpt-oss.com/v1/chat',
            'gpt_oss_key': 'mock-key',
            'kanana_endpoint': 'https://api.mock-kanana.com/v1/generate', 
            'kanana_key': 'mock-key',
            
            # Training settings
            'epochs_per_fold': 5,
            'batch_size': 8,
            'validation_frequency': 2,
            'early_stopping_patience': 10,
            'learning_rate': 0.001,
            
            # System settings
            'consensus_threshold': 0.6,
            'revthink_verification': True,
            'save_checkpoints': True,
            'checkpoint_interval': 5,
            
            # Analysis settings
            'detailed_logging': True,
            'performance_analysis': True,
            'complexity_analysis': True,
            'generate_reports': True
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def setup_training_system(self):
        """Initialize the complete training system"""
        print("\nSetting up ARC training system...")
        
        # Create model configurations
        gpt_oss_config = ModelConfig(
            model_name="gpt-oss-20b",
            api_endpoint=self.config['gpt_oss_endpoint'],
            api_key=self.config['gpt_oss_key'],
            max_tokens=1000,
            temperature=0.3
        )
        
        kanana_config = ModelConfig(
            model_name="kanana-1.5-15.7b-a3b",
            api_endpoint=self.config['kanana_endpoint'],
            api_key=self.config['kanana_key'],
            max_tokens=800,
            temperature=0.2
        )
        
        # Create training setup
        self.data_loader, self.training_manager = create_arc_training_setup(
            data_dir=self.config['data_dir'],
            gpt_oss_config=gpt_oss_config,
            kanana_config=kanana_config
        )
        
        # Update training manager configuration
        self.training_manager.config.update({
            'epochs_per_fold': self.config['epochs_per_fold'],
            'batch_size': self.config['batch_size'],
            'validation_frequency': self.config['validation_frequency'],
            'early_stopping_patience': self.config['early_stopping_patience'],
            'save_checkpoints': self.config['save_checkpoints']
        })
        
        print("✅ Training system setup complete")
        
        # Log system information
        stats = self.data_loader.get_dataset_statistics()
        print(f"   Dataset: {stats['total_challenges']} challenges, {stats['total_folds']} folds")
        print(f"   Models: GPT-OSS-20B + Kanana-1.5-15.7B-A3B")
        print(f"   RevThink verification: {'Enabled' if self.config['revthink_verification'] else 'Disabled'}")
    
    def run_full_training(self) -> Dict[str, Any]:
        """Run complete k-fold cross-validation training"""
        print(f"\n{'='*70}")
        print(f"STARTING FULL ARC-IDEA TRAINING PIPELINE")
        print(f"{'='*70}")
        
        training_start = time.time()
        
        # Setup system
        self.setup_training_system()
        
        # Run k-fold training
        print(f"\nRunning {self.config['k_folds']}-fold cross-validation training...")
        cv_results = self.training_manager.train_k_fold_cv(self.config['k_folds'])
        
        training_end = time.time()
        training_duration = training_end - training_start
        
        # Compile comprehensive results
        self.results = {
            'training_config': self.config,
            'cv_results': cv_results,
            'system_info': {
                'start_time': self.start_time.isoformat(),
                'training_duration_seconds': training_duration,
                'training_duration_formatted': f"{training_duration/3600:.1f} hours"
            },
            'dataset_statistics': self.data_loader.get_dataset_statistics(),
            'performance_analysis': self._analyze_performance(cv_results),
            'model_comparison': self._analyze_model_performance(cv_results)
        }
        
        # Save results
        self._save_results()
        
        # Generate reports
        if self.config['generate_reports']:
            self._generate_training_report()
        
        print(f"\n{'='*70}")
        print(f"TRAINING PIPELINE COMPLETE")
        print(f"{'='*70}")
        print(f"Duration: {training_duration/3600:.1f} hours")
        print(f"Results saved to: {self.output_dir}")
        
        return self.results
    
    def run_single_fold_demo(self, fold_number: int = 1) -> Dict[str, Any]:
        """Run single fold for demonstration/testing"""
        print(f"\n{'='*50}")
        print(f"RUNNING SINGLE FOLD DEMO (Fold {fold_number})")
        print(f"{'='*50}")
        
        # Setup system
        self.setup_training_system()
        
        # Run single fold
        fold_results = self.training_manager.train_single_fold(fold_number)
        
        # Compile results
        demo_results = {
            'fold_results': fold_results,
            'system_info': {
                'start_time': self.start_time.isoformat(),
                'demo_mode': True
            },
            'performance_summary': self._summarize_fold_performance(fold_results)
        }
        
        # Save demo results
        demo_file = os.path.join(self.output_dir, f'demo_fold_{fold_number}_results.json')
        with open(demo_file, 'w') as f:
            json.dump(demo_results, f, indent=2, default=str)
        
        print(f"\nDemo complete - Results saved to {demo_file}")
        return demo_results
    
    def _analyze_performance(self, cv_results: Dict) -> Dict[str, Any]:
        """Analyze overall training performance"""
        analysis = {
            'cross_validation_summary': {},
            'fold_performance_distribution': {},
            'convergence_analysis': {},
            'complexity_performance': {}
        }
        
        # CV Summary
        overall_metrics = cv_results.get('overall_metrics', {})
        analysis['cross_validation_summary'] = {
            'mean_score': overall_metrics.get('mean_validation_score', 0.0),
            'score_std': overall_metrics.get('std_validation_score', 0.0),
            'best_fold': cv_results.get('best_fold', -1),
            'best_score': cv_results.get('best_score', 0.0),
            'successful_folds': overall_metrics.get('successful_folds', 0),
            'success_rate': overall_metrics.get('successful_folds', 0) / cv_results.get('total_folds', 1)
        }
        
        # Fold performance distribution
        fold_scores = []
        convergence_epochs = []
        
        for fold_num, fold_data in cv_results.get('fold_results', {}).items():
            fold_scores.append(fold_data.get('best_validation_score', 0.0))
            convergence_epochs.append(fold_data.get('best_epoch', -1))
        
        if fold_scores:
            analysis['fold_performance_distribution'] = {
                'scores': fold_scores,
                'mean': np.mean(fold_scores),
                'std': np.std(fold_scores),
                'min': np.min(fold_scores),
                'max': np.max(fold_scores)
            }
        
        if convergence_epochs:
            analysis['convergence_analysis'] = {
                'mean_convergence_epoch': np.mean([e for e in convergence_epochs if e > 0]),
                'convergence_epochs': convergence_epochs,
                'early_convergence_rate': sum(1 for e in convergence_epochs if 0 < e <= 3) / len(convergence_epochs)
            }
        
        return analysis
    
    def _analyze_model_performance(self, cv_results: Dict) -> Dict[str, Any]:
        """Analyze multi-LLM model performance"""
        analysis = {
            'gpt_oss_performance': {},
            'kanana_performance': {},
            'ensemble_effectiveness': {},
            'revthink_effectiveness': {}
        }
        
        # Aggregate model-specific metrics across folds
        gpt_oss_scores = []
        kanana_scores = []
        ensemble_scores = []
        revthink_scores = []
        
        for fold_num, fold_data in cv_results.get('fold_results', {}).items():
            # Extract model-specific performance from training/validation batches
            training_batches = fold_data.get('training_batches', [])
            validation_batches = fold_data.get('validation_batches', [])
            
            # Aggregate scores (simplified - would extract actual model-specific metrics)
            for batch in training_batches + validation_batches:
                details = batch.get('processing_details', []) or batch.get('validation_details', [])
                for detail in details:
                    # These would be actual model-specific scores in practice
                    gpt_oss_scores.append(detail.get('efe_score', 0.5))
                    kanana_scores.append(detail.get('verification_score', 0.5))
                    ensemble_scores.append(detail.get('accuracy', 0.0))
        
        # Compile analysis
        if gpt_oss_scores:
            analysis['gpt_oss_performance'] = {
                'mean_performance': np.mean(gpt_oss_scores),
                'std_performance': np.std(gpt_oss_scores),
                'sample_count': len(gpt_oss_scores)
            }
        
        if kanana_scores:
            analysis['kanana_performance'] = {
                'mean_performance': np.mean(kanana_scores),
                'std_performance': np.std(kanana_scores),
                'sample_count': len(kanana_scores)
            }
        
        if ensemble_scores:
            analysis['ensemble_effectiveness'] = {
                'mean_accuracy': np.mean(ensemble_scores),
                'accuracy_std': np.std(ensemble_scores),
                'high_accuracy_rate': sum(1 for score in ensemble_scores if score > 0.8) / len(ensemble_scores)
            }
        
        return analysis
    
    def _summarize_fold_performance(self, fold_results: Dict) -> Dict[str, Any]:
        """Summarize single fold performance"""
        summary = {
            'best_validation_score': fold_results.get('best_validation_score', 0.0),
            'best_epoch': fold_results.get('best_epoch', -1),
            'total_epochs': len(fold_results.get('epoch_results', [])),
            'training_batches': len(fold_results.get('training_batches', [])),
            'validation_batches': len(fold_results.get('validation_batches', []))
        }
        
        # Calculate average metrics
        training_batches = fold_results.get('training_batches', [])
        if training_batches:
            avg_efe = np.mean([batch.get('average_efe', 1.0) for batch in training_batches])
            avg_verification = np.mean([batch.get('average_verification', 0.0) for batch in training_batches])
            
            summary.update({
                'average_training_efe': avg_efe,
                'average_training_verification': avg_verification
            })
        
        validation_batches = fold_results.get('validation_batches', [])
        if validation_batches:
            avg_accuracy = np.mean([batch.get('accuracy', 0.0) for batch in validation_batches])
            summary['average_validation_accuracy'] = avg_accuracy
        
        return summary
    
    def _save_results(self):
        """Save comprehensive training results"""
        # Main results file
        results_file = os.path.join(self.output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Configuration file
        config_file = os.path.join(self.output_dir, 'training_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Performance summary
        summary_file = os.path.join(self.output_dir, 'performance_summary.json')
        summary = {
            'overall_score': self.results.get('cv_results', {}).get('best_score', 0.0),
            'mean_cv_score': self.results.get('performance_analysis', {}).get('cross_validation_summary', {}).get('mean_score', 0.0),
            'successful_folds': self.results.get('cv_results', {}).get('overall_metrics', {}).get('successful_folds', 0),
            'total_folds': self.config['k_folds'],
            'training_duration': self.results.get('system_info', {}).get('training_duration_formatted', '0 hours')
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Results saved:")
        print(f"   Main results: {results_file}")
        print(f"   Configuration: {config_file}") 
        print(f"   Summary: {summary_file}")
    
    def _generate_training_report(self):
        """Generate comprehensive training report"""
        report_file = os.path.join(self.output_dir, 'training_report.md')
        
        cv_results = self.results.get('cv_results', {})
        performance = self.results.get('performance_analysis', {})
        
        report_content = f"""# ARC-IDEA Training Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Configuration
- K-folds: {self.config['k_folds']}
- Epochs per fold: {self.config['epochs_per_fold']}
- Batch size: {self.config['batch_size']}
- Models: GPT-OSS-20B + Kanana-1.5-15.7B-A3B
- RevThink verification: {'Enabled' if self.config['revthink_verification'] else 'Disabled'}

## Results Summary
- **Best validation score**: {cv_results.get('best_score', 0.0):.4f}
- **Best fold**: {cv_results.get('best_fold', -1)}
- **Mean CV score**: {performance.get('cross_validation_summary', {}).get('mean_score', 0.0):.4f}
- **Score std**: {performance.get('cross_validation_summary', {}).get('score_std', 0.0):.4f}
- **Success rate**: {performance.get('cross_validation_summary', {}).get('success_rate', 0.0):.1%}

## Training Duration
{self.results.get('system_info', {}).get('training_duration_formatted', 'Unknown')}

## Dataset Statistics
- Total challenges: {self.results.get('dataset_statistics', {}).get('total_challenges', 0)}
- Complexity distribution: Mean = {self.results.get('dataset_statistics', {}).get('complexity_distribution', {}).get('mean', 0.0):.3f}

## Model Performance Analysis
{self._format_model_analysis()}

## Convergence Analysis
{self._format_convergence_analysis()}

## Conclusions
{self._generate_conclusions()}
"""
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"✅ Training report generated: {report_file}")
    
    def _format_model_analysis(self) -> str:
        """Format model performance analysis for report"""
        model_analysis = self.results.get('model_comparison', {})
        
        gpt_oss = model_analysis.get('gpt_oss_performance', {})
        kanana = model_analysis.get('kanana_performance', {})
        ensemble = model_analysis.get('ensemble_effectiveness', {})
        
        analysis = f"""
### GPT-OSS-20B Performance
- Mean performance: {gpt_oss.get('mean_performance', 0.0):.4f}
- Standard deviation: {gpt_oss.get('std_performance', 0.0):.4f}

### Kanana-1.5-15.7B-A3B Performance  
- Mean performance: {kanana.get('mean_performance', 0.0):.4f}
- Standard deviation: {kanana.get('std_performance', 0.0):.4f}

### Ensemble Effectiveness
- Mean accuracy: {ensemble.get('mean_accuracy', 0.0):.4f}
- High accuracy rate (>80%): {ensemble.get('high_accuracy_rate', 0.0):.1%}
"""
        return analysis
    
    def _format_convergence_analysis(self) -> str:
        """Format convergence analysis for report"""
        convergence = self.results.get('performance_analysis', {}).get('convergence_analysis', {})
        
        analysis = f"""
### Training Convergence
- Mean convergence epoch: {convergence.get('mean_convergence_epoch', 0.0):.1f}
- Early convergence rate (≤3 epochs): {convergence.get('early_convergence_rate', 0.0):.1%}
"""
        return analysis
    
    def _generate_conclusions(self) -> str:
        """Generate training conclusions"""
        cv_summary = self.results.get('performance_analysis', {}).get('cross_validation_summary', {})
        
        mean_score = cv_summary.get('mean_score', 0.0)
        success_rate = cv_summary.get('success_rate', 0.0)
        
        if mean_score > 0.7 and success_rate > 0.8:
            conclusion = "✅ **Training successful** - High performance achieved across most folds with good generalization."
        elif mean_score > 0.5 and success_rate > 0.6:
            conclusion = "⚠️ **Moderate success** - Reasonable performance but room for improvement in consistency."
        else:
            conclusion = "❌ **Training challenges** - Low performance suggests need for architecture or data improvements."
        
        return conclusion

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ARC-IDEA Training Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, default='arc_training_results', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run single fold demo instead of full training')
    parser.add_argument('--fold', type=int, default=1, help='Fold number for demo mode')
    parser.add_argument('--k-folds', type=int, default=10, help='Number of folds for cross-validation')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ARCTrainingPipeline(
        config_file=args.config,
        output_dir=args.output
    )
    
    # Override k_folds if specified
    if args.k_folds != 10:
        pipeline.config['k_folds'] = args.k_folds
    
    try:
        if args.demo:
            print("Running in demo mode...")
            results = pipeline.run_single_fold_demo(args.fold)
        else:
            print("Running full training pipeline...")
            results = pipeline.run_full_training()
        
        print(f"\nTraining completed successfully!")
        return True
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import numpy as np  # Import needed for analysis functions
    success = main()
    exit(0 if success else 1)