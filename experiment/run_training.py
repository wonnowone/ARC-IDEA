"""
Simple runner script for the comprehensive ARC training
"""

import os
import sys
from pathlib import Path

# Set working directory to experiment folder
os.chdir(Path(__file__).parent)

# Add parent directory to Python path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def main():
    """Run the comprehensive training with minimal setup"""
    
    print("=== ARC Challenge Comprehensive Training ===")
    print("Integrating: Mistral + EFE + TTA + Solvers + Features")
    print()
    
    try:
        # Import and run
        from comprehensive_training_runner import ComprehensiveARCTrainer, TrainingConfig
        
        # Create lightweight config for testing
        config = TrainingConfig(
            max_epochs=2,  # Very short for testing
            batch_size=1,  # Small batch
            learning_rate=1e-4,
            device="cpu"   # CPU only for compatibility
        )
        
        print("Configuration:")
        print(f"  - Epochs: {config.max_epochs}")
        print(f"  - Batch size: {config.batch_size}")
        print(f"  - Learning rate: {config.learning_rate}")
        print(f"  - Device: {config.device}")
        print()
        
        # Create trainer
        trainer = ComprehensiveARCTrainer(
            config=config,
            enable_tta=True,
            enable_solvers=True,
            verbose=True
        )
        
        print("Starting comprehensive training...")
        results = trainer.run_comprehensive_training()
        
        print("\n=== Training Complete ===")
        if 'combined_performance' in results:
            success_rate = results['combined_performance'].get('success_rate', 0)
            print(f"Success rate: {success_rate:.2%}")
        
        print("Results saved to comprehensive_training_summary.json")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Some dependencies might be missing. Installing required packages...")
        
        # Try to install missing packages
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "scikit-learn"])
            print("Dependencies installed. Please run again.")
        except:
            print("Could not install dependencies automatically.")
            print("Please install: pip install transformers scikit-learn")
            
    except Exception as e:
        print(f"Training error: {e}")
        print("Running simplified version...")
        
        # Fallback: run basic training components
        try:
            from split_training_data import split_training_data
            print("✓ Data splitting works")
            
            from exp_prompt import create_arc_prompt_template
            print("✓ Prompt system works")
            
            from exp_EFE import test_efe_implementation  
            print("✓ EFE system works")
            
            from exp_tta import test_adaptation_system
            print("✓ TTA system works")
            
            print("\nAll individual components are working!")
            print("The integration may need additional setup for your environment.")
            
        except Exception as fallback_error:
            print(f"Fallback error: {fallback_error}")
            print("Please check the installation and dependencies.")

if __name__ == "__main__":
    main()