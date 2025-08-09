#!/usr/bin/env python3
"""
ARC Dataset Processor - Merge challenges with solutions and create K-fold validation
Unicode-safe version for Windows CP949 encoding
"""

import json
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.model_selection import KFold
import warnings

class ARCDatasetProcessor:
    """Process ARC Prize 2025 dataset for training"""
    
    def __init__(self, challenges_path: str, solutions_path: str):
        self.challenges_path = challenges_path
        self.solutions_path = solutions_path
        self.merged_data = {}
        self.k_fold_splits = {}
        
    def load_raw_data(self) -> Tuple[Dict, Dict]:
        """Load raw challenges and solutions"""
        print("Loading ARC training challenges...")
        try:
            with open(self.challenges_path, 'r', encoding='utf-8') as f:
                challenges = json.load(f)
            print(f"OK Loaded {len(challenges)} training challenges")
        except Exception as e:
            print(f"ERROR Failed to load challenges: {e}")
            return {}, {}
        
        print("Loading ARC training solutions...")
        try:
            with open(self.solutions_path, 'r', encoding='utf-8') as f:
                solutions = json.load(f)
            print(f"OK Loaded {len(solutions)} training solutions")
        except Exception as e:
            print(f"ERROR Failed to load solutions: {e}")
            return challenges, {}
        
        return challenges, solutions
    
    def merge_challenges_with_solutions(self, challenges: Dict, solutions: Dict) -> Dict:
        """Merge challenges with their corresponding solutions"""
        print("\nMerging challenges with solutions...")
        
        merged_data = {}
        matched_count = 0
        unmatched_challenges = []
        
        for challenge_id, challenge_data in challenges.items():
            if challenge_id in solutions:
                # Create merged entry
                merged_entry = {
                    'challenge_id': challenge_id,
                    'train': challenge_data['train'],
                    'test': challenge_data['test'],
                    'solutions': solutions[challenge_id]
                }
                
                # Add metadata
                merged_entry['metadata'] = {
                    'num_train_examples': len(challenge_data['train']),
                    'num_test_examples': len(challenge_data['test']),
                    'has_solutions': True,
                    'input_shapes': self._analyze_grid_shapes(challenge_data, 'input'),
                    'output_shapes': self._analyze_grid_shapes(challenge_data, 'output'),
                    'complexity_score': self._calculate_complexity_score(challenge_data)
                }
                
                merged_data[challenge_id] = merged_entry
                matched_count += 1
            else:
                unmatched_challenges.append(challenge_id)
        
        print(f"OK Successfully merged {matched_count} challenge-solution pairs")
        
        if unmatched_challenges:
            print(f"WARNING {len(unmatched_challenges)} challenges without solutions:")
            for cid in unmatched_challenges[:5]:  # Show first 5
                print(f"   - {cid}")
            if len(unmatched_challenges) > 5:
                print(f"   ... and {len(unmatched_challenges) - 5} more")
        
        return merged_data
    
    def _analyze_grid_shapes(self, challenge_data: Dict, grid_type: str) -> Dict:
        """Analyze grid shapes in challenge data"""
        shapes = defaultdict(int)
        
        # Analyze training examples (have both input and output)
        for example in challenge_data['train']:
            if grid_type in example:
                grid = example[grid_type]
                shape = f"{len(grid)}x{len(grid[0]) if grid else 0}"
                shapes[shape] += 1
        
        # Analyze test examples (only have input, not output)
        for example in challenge_data['test']:
            if grid_type in example:  # Test examples don't have 'output'
                grid = example[grid_type]
                shape = f"{len(grid)}x{len(grid[0]) if grid else 0}"
                shapes[shape] += 1
        
        return dict(shapes)
    
    def _calculate_complexity_score(self, challenge_data: Dict) -> float:
        """Calculate complexity score for a challenge"""
        complexity_factors = []
        
        # Factor 1: Number of training examples (more = easier to learn)
        num_train = len(challenge_data['train'])
        train_factor = max(0.1, 1.0 - (num_train - 1) * 0.2)  # Decreases with more examples
        complexity_factors.append(train_factor)
        
        # Factor 2: Grid size variation
        input_shapes = self._analyze_grid_shapes(challenge_data, 'input')
        output_shapes = self._analyze_grid_shapes(challenge_data, 'output')
        shape_variety = len(set(list(input_shapes.keys()) + list(output_shapes.keys())))
        shape_factor = min(1.0, shape_variety / 3.0)  # More variety = more complex
        complexity_factors.append(shape_factor)
        
        # Factor 3: Average grid size
        total_cells = 0
        total_grids = 0
        
        for example in challenge_data['train']:
            for grid_type in ['input', 'output']:
                if grid_type in example:
                    grid = example[grid_type]
                    if grid:
                        total_cells += len(grid) * len(grid[0])
                        total_grids += 1
        
        avg_size = total_cells / max(1, total_grids)
        size_factor = min(1.0, avg_size / 100.0)  # Larger grids = more complex
        complexity_factors.append(size_factor)
        
        # Combined complexity score
        return np.mean(complexity_factors)
    
    def create_k_fold_splits(self, merged_data: Dict, k: int = 10) -> Dict:
        """Create k-fold cross-validation splits"""
        print(f"\nCreating {k}-fold cross-validation splits...")
        
        # Convert to list for splitting
        challenge_ids = list(merged_data.keys())
        challenge_data = [merged_data[cid] for cid in challenge_ids]
        
        print(f"Total challenges for splitting: {len(challenge_ids)}")
        
        # Create stratified splits based on complexity
        complexities = [data['metadata']['complexity_score'] for data in challenge_data]
        
        # Convert to complexity bins for stratification
        complexity_bins = np.digitize(complexities, bins=np.linspace(0, 1, 4))  # 3 bins
        
        kfold = KFold(n_splits=k, shuffle=True, random_state=42)
        
        fold_splits = {}
        
        for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(challenge_ids, complexity_bins)):
            fold_data = {
                'fold_number': fold_idx + 1,
                'train_challenges': [challenge_ids[i] for i in train_indices],
                'validation_challenges': [challenge_ids[i] for i in val_indices],
                'train_data': {challenge_ids[i]: merged_data[challenge_ids[i]] for i in train_indices},
                'validation_data': {challenge_ids[i]: merged_data[challenge_ids[i]] for i in val_indices},
                'statistics': {
                    'train_size': len(train_indices),
                    'validation_size': len(val_indices),
                    'train_complexity_mean': np.mean([complexities[i] for i in train_indices]),
                    'validation_complexity_mean': np.mean([complexities[i] for i in val_indices])
                }
            }
            
            fold_splits[f'fold_{fold_idx + 1}'] = fold_data
            
            print(f"  Fold {fold_idx + 1}: {len(train_indices)} train, {len(val_indices)} validation")
        
        print(f"OK Created {k} stratified folds")
        return fold_splits
    
    def save_processed_data(self, output_dir: str = "arc_processed_data"):
        """Save merged data and k-fold splits"""
        print(f"\nSaving processed data to {output_dir}/...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "k_fold_splits"), exist_ok=True)
        
        # Save merged data
        merged_file = os.path.join(output_dir, "merged_arc_training_data.json")
        with open(merged_file, 'w', encoding='utf-8') as f:
            json.dump(self.merged_data, f, indent=2, ensure_ascii=False)
        print(f"OK Saved merged data: {merged_file}")
        
        # Save k-fold splits
        for fold_name, fold_data in self.k_fold_splits.items():
            fold_file = os.path.join(output_dir, "k_fold_splits", f"{fold_name}.json")
            with open(fold_file, 'w', encoding='utf-8') as f:
                json.dump(fold_data, f, indent=2, ensure_ascii=False)
            print(f"OK Saved {fold_name}: {fold_file}")
        
        # Save dataset summary
        summary = self._create_dataset_summary()
        summary_file = os.path.join(output_dir, "dataset_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"OK Saved dataset summary: {summary_file}")
        
        print(f"\nSUCCESS All processed data saved to {output_dir}/")
    
    def _create_dataset_summary(self) -> Dict:
        """Create comprehensive dataset summary"""
        summary = {
            'dataset_info': {
                'total_challenges': len(self.merged_data),
                'k_folds': len(self.k_fold_splits),
                'created_by': 'ARC-IDEA Dataset Processor',
                'source_files': {
                    'challenges': os.path.basename(self.challenges_path),
                    'solutions': os.path.basename(self.solutions_path)
                }
            },
            'complexity_distribution': {},
            'shape_analysis': {},
            'k_fold_statistics': {}
        }
        
        # Complexity distribution
        complexities = [data['metadata']['complexity_score'] for data in self.merged_data.values()]
        summary['complexity_distribution'] = {
            'mean': float(np.mean(complexities)),
            'std': float(np.std(complexities)),
            'min': float(np.min(complexities)),
            'max': float(np.max(complexities)),
            'median': float(np.median(complexities))
        }
        
        # Shape analysis
        all_input_shapes = defaultdict(int)
        all_output_shapes = defaultdict(int)
        
        for data in self.merged_data.values():
            for shape, count in data['metadata']['input_shapes'].items():
                all_input_shapes[shape] += count
            for shape, count in data['metadata']['output_shapes'].items():
                all_output_shapes[shape] += count
        
        summary['shape_analysis'] = {
            'input_shapes': dict(all_input_shapes),
            'output_shapes': dict(all_output_shapes),
            'unique_input_shapes': len(all_input_shapes),
            'unique_output_shapes': len(all_output_shapes)
        }
        
        # K-fold statistics
        for fold_name, fold_data in self.k_fold_splits.items():
            summary['k_fold_statistics'][fold_name] = fold_data['statistics']
        
        return summary
    
    def process_dataset(self, k_folds: int = 10, output_dir: str = "arc_processed_data") -> bool:
        """Complete dataset processing pipeline"""
        print("Starting ARC Dataset Processing Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load raw data
            challenges, solutions = self.load_raw_data()
            if not challenges or not solutions:
                return False
            
            # Step 2: Merge challenges with solutions
            self.merged_data = self.merge_challenges_with_solutions(challenges, solutions)
            if not self.merged_data:
                print("ERROR No data to process after merging")
                return False
            
            # Step 3: Create k-fold splits
            self.k_fold_splits = self.create_k_fold_splits(self.merged_data, k_folds)
            
            # Step 4: Save processed data
            self.save_processed_data(output_dir)
            
            # Step 5: Print final summary
            print("\nPROCESSING SUMMARY")
            print("=" * 30)
            print(f"Total merged challenges: {len(self.merged_data)}")
            print(f"K-fold splits created: {len(self.k_fold_splits)}")
            print(f"Output directory: {output_dir}")
            
            return True
            
        except Exception as e:
            print(f"ERROR Processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    # File paths
    challenges_path = r"C:\Users\SAMSUNG\Downloads\arc-prize-2025\arc-agi_training_challenges.json"
    solutions_path = r"C:\Users\SAMSUNG\Downloads\arc-prize-2025\arc-agi_training_solutions.json"
    
    # Check if files exist
    if not os.path.exists(challenges_path):
        print(f"ERROR Challenges file not found: {challenges_path}")
        return False
        
    if not os.path.exists(solutions_path):
        print(f"ERROR Solutions file not found: {solutions_path}")
        return False
    
    print(f"Input files:")
    print(f"  Challenges: {challenges_path}")
    print(f"  Solutions: {solutions_path}")
    
    # Create processor
    processor = ARCDatasetProcessor(challenges_path, solutions_path)
    
    # Process dataset
    success = processor.process_dataset(k_folds=10, output_dir="arc_processed_data")
    
    if success:
        print("\nSUCCESS ARC Dataset Processing Completed!")
        print("\nGenerated Files:")
        print("  merged_arc_training_data.json - Complete merged dataset")
        print("  k_fold_splits/ - Directory with 10 fold files")
        print("  dataset_summary.json - Comprehensive dataset statistics")
        print("\nReady for ARC-IDEA training!")
    else:
        print("\nERROR Dataset processing failed")
    
    return success

if __name__ == "__main__":
    main()