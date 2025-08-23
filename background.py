import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Dict, Tuple
import os

class NoBackgroundAnalyzer:
    """배경이 없는 ARC 문제들 분석 및 시각화"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.no_bg_problems = []
        self.bg_problems = []
        
        # ARC 색상 팔레트 (0-9)
        self.colors = [
            '#000000',  # 0: 검정
            '#0074D9',  # 1: 파랑
            '#FF4136',  # 2: 빨강
            '#2ECC40',  # 3: 초록
            '#FFDC00',  # 4: 노랑
            '#AAAAAA',  # 5: 회색
            '#F012BE',  # 6: 자홍
            '#FF851B',  # 7: 주황
            '#7FDBFF',  # 8: 하늘
            '#870C25',  # 9: 갈색
        ]
        self.cmap = mcolors.ListedColormap(self.colors)
    
    def check_background(self, grid: List[List[int]], ratio_threshold: float = 1.5) -> Tuple[bool, Dict]:
        """배경 유무 확인"""
        grid_array = np.array(grid)
        unique, counts = np.unique(grid_array, return_counts=True)
        color_counts = dict(zip(unique, counts))
        
        if len(color_counts) <= 1:
            return True, {'reason': 'single_color', 'colors': color_counts}
        
        sorted_counts = sorted(counts, reverse=True)
        ratio = sorted_counts[0] / sorted_counts[1]
        
        return ratio > ratio_threshold, {
            'ratio': ratio,
            'color_counts': color_counts,
            'most_common': unique[np.argmax(counts)],
            'pixel_distribution': sorted_counts
        }
    
    def scan_all_problems(self):
        """모든 문제 스캔하여 배경 없는 문제 찾기"""
        # ARC 데이터는 단일 JSON 파일에 모든 문제가 포함됨
        challenges_file = self.dataset_path / "arc-agi_training_challenges.json"
        
        if not challenges_file.exists():
            print(f"파일을 찾을 수 없습니다: {challenges_file}")
            return
            
        with open(challenges_file, 'r') as f:
            all_problems = json.load(f)
        
        print(f"총 {len(all_problems)}개 문제 스캔 중...")
        
        for problem_id, data in all_problems.items():
            problem_info = {
                'name': problem_id,
                'path': challenges_file,
                'train': [],
                'test': []
            }
            
            # Training 예제들 체크
            has_no_bg = False
            for idx, example in enumerate(data.get('train', [])):
                input_has_bg, input_info = self.check_background(example['input'])
                output_has_bg, output_info = self.check_background(example['output'])
                
                example_info = {
                    'index': idx,
                    'input_has_bg': input_has_bg,
                    'output_has_bg': output_has_bg,
                    'input_ratio': input_info.get('ratio', 0),
                    'output_ratio': output_info.get('ratio', 0),
                    'input_grid': example['input'],
                    'output_grid': example['output']
                }
                problem_info['train'].append(example_info)
                
                # 하나라도 배경이 없으면 표시
                if not input_has_bg or not output_has_bg:
                    has_no_bg = True
            
            # Test 예제 체크
            for idx, example in enumerate(data.get('test', [])):
                test_has_bg, test_info = self.check_background(example['input'])
                problem_info['test'].append({
                    'index': idx,
                    'has_bg': test_has_bg,
                    'ratio': test_info.get('ratio', 0),
                    'grid': example['input']
                })
            
            if has_no_bg:
                self.no_bg_problems.append(problem_info)
            else:
                self.bg_problems.append(problem_info)
        
        print(f"\n배경 없는 문제: {len(self.no_bg_problems)}개")
        print(f"배경 있는 문제: {len(self.bg_problems)}개")
    
    def visualize_problem(self, problem_info: Dict, max_examples: int = 3):
        """단일 문제 시각화"""
        name = problem_info['name']
        train_examples = problem_info['train'][:max_examples]
        
        n_examples = len(train_examples)
        fig, axes = plt.subplots(n_examples, 2, figsize=(8, 4*n_examples))
        
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f"Problem: {name}", fontsize=14, fontweight='bold')
        
        for idx, example in enumerate(train_examples):
            # Input
            input_grid = np.array(example['input_grid'])
            axes[idx, 0].imshow(input_grid, cmap=self.cmap, vmin=0, vmax=9)
            axes[idx, 0].set_title(f"Input (bg ratio: {example['input_ratio']:.2f})")
            axes[idx, 0].grid(True, alpha=0.3)
            axes[idx, 0].set_xticks(range(input_grid.shape[1]))
            axes[idx, 0].set_yticks(range(input_grid.shape[0]))
            
            # Output  
            output_grid = np.array(example['output_grid'])
            axes[idx, 1].imshow(output_grid, cmap=self.cmap, vmin=0, vmax=9)
            axes[idx, 1].set_title(f"Output (bg ratio: {example['output_ratio']:.2f})")
            axes[idx, 1].grid(True, alpha=0.3)
            axes[idx, 1].set_xticks(range(output_grid.shape[1]))
            axes[idx, 1].set_yticks(range(output_grid.shape[0]))
        
        plt.tight_layout()
        return fig
    
    def analyze_no_bg_patterns(self):
        """배경 없는 문제들의 패턴 분석"""
        patterns = {
            'grid_sizes': [],
            'color_diversity': [],
            'density': [],
            'symmetry': []
        }
        
        for problem in self.no_bg_problems:
            for example in problem['train']:
                grid = np.array(example['input_grid'])
                
                # 그리드 크기
                patterns['grid_sizes'].append(grid.shape)
                
                # 색상 다양성
                n_colors = len(np.unique(grid))
                patterns['color_diversity'].append(n_colors)
                
                # 밀도 (0이 아닌 픽셀 비율)
                density = np.mean(grid != 0)
                patterns['density'].append(density)
                
                # 대칭성 체크
                h_sym = np.array_equal(grid, np.fliplr(grid))
                v_sym = np.array_equal(grid, np.flipud(grid))
                patterns['symmetry'].append((h_sym, v_sym))
        
        return patterns
    
    def save_no_bg_list(self, output_file: str = "no_background_problems.txt"):
        """배경 없는 문제 목록 저장 - 상세 정보 포함"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("배경이 없는 ARC 문제들 (1.5배 기준)\n")
            f.write("="*50 + "\n\n")
            
            for problem in self.no_bg_problems:
                f.write(f"Problem ID: {problem['name']}\n")
                f.write("-" * 30 + "\n")
                
                # Training examples
                f.write("TRAINING EXAMPLES:\n")
                for train in problem['train']:
                    if not train['input_has_bg'] or not train['output_has_bg']:
                        f.write(f"  Example {train['index'] + 1}:\n")
                        f.write(f"    Input ratio: {train['input_ratio']:.2f}\n")
                        f.write(f"    Output ratio: {train['output_ratio']:.2f}\n")
                        
                        # Input grid
                        f.write("    INPUT:\n")
                        for row in train['input_grid']:
                            f.write(f"      {row}\n")
                        
                        # Output grid  
                        f.write("    OUTPUT:\n")
                        for row in train['output_grid']:
                            f.write(f"      {row}\n")
                        f.write("\n")
                
                # Test examples
                if problem['test']:
                    f.write("TEST EXAMPLES:\n")
                    for test in problem['test']:
                        f.write(f"  Test {test['index'] + 1}:\n")
                        f.write(f"    Ratio: {test['ratio']:.2f}\n")
                        f.write("    INPUT:\n")
                        for row in test['grid']:
                            f.write(f"      {row}\n")
                        f.write("\n")
                
                f.write("=" * 50 + "\n\n")
        
        print(f"목록 저장됨: {output_file}")
    
    def create_summary_visualization(self, n_samples: int = 10):
        """No-background problems sample visualization with real colors"""
        samples = self.no_bg_problems[:n_samples]
        
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        fig.suptitle("No-Background ARC Problems (Color Visualization)", fontsize=16, fontweight='bold')
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, problem in enumerate(samples):
            if problem['train']:
                # Find first example that meets no-background criteria
                example = None
                for train_ex in problem['train']:
                    if not train_ex['input_has_bg'] or not train_ex['output_has_bg']:
                        example = train_ex
                        break
                
                if example:
                    # Input grid with colors
                    input_grid = np.array(example['input_grid'])
                    axes[i, 0].imshow(input_grid, cmap=self.cmap, vmin=0, vmax=9)
                    axes[i, 0].set_title(f"Problem {problem['name']}\nInput (ratio: {example['input_ratio']:.2f})", fontsize=10)
                    axes[i, 0].grid(True, alpha=0.3, color='white', linewidth=0.5)
                    axes[i, 0].set_xticks(range(input_grid.shape[1]))
                    axes[i, 0].set_yticks(range(input_grid.shape[0]))
                    
                    # Output grid with colors
                    output_grid = np.array(example['output_grid'])
                    axes[i, 1].imshow(output_grid, cmap=self.cmap, vmin=0, vmax=9)
                    axes[i, 1].set_title(f"Output (ratio: {example['output_ratio']:.2f})", fontsize=10)
                    axes[i, 1].grid(True, alpha=0.3, color='white', linewidth=0.5)
                    axes[i, 1].set_xticks(range(output_grid.shape[1]))
                    axes[i, 1].set_yticks(range(output_grid.shape[0]))
                    
                    # Input color distribution
                    input_colors, input_counts = np.unique(input_grid, return_counts=True)
                    color_bars = axes[i, 2].bar(input_colors, input_counts, 
                                               color=[self.colors[c] for c in input_colors])
                    axes[i, 2].set_title("Input Color Distribution", fontsize=10)
                    axes[i, 2].set_xlabel("Color Index", fontsize=8)
                    axes[i, 2].set_ylabel("Pixel Count", fontsize=8)
                    axes[i, 2].set_xticks(input_colors)
                    
                    # Output color distribution
                    output_colors, output_counts = np.unique(output_grid, return_counts=True)
                    color_bars = axes[i, 3].bar(output_colors, output_counts,
                                               color=[self.colors[c] for c in output_colors])
                    axes[i, 3].set_title("Output Color Distribution", fontsize=10)
                    axes[i, 3].set_xlabel("Color Index", fontsize=8)
                    axes[i, 3].set_ylabel("Pixel Count", fontsize=8)
                    axes[i, 3].set_xticks(output_colors)
        
        plt.tight_layout()
        return fig
    
    def visualize_no_bg_problems(self, n_samples: int = 20, save_file: str = None):
        """Comprehensive color visualization of no-background problems"""
        if not self.no_bg_problems:
            print("No problems found matching the criteria.")
            return None
            
        samples = self.no_bg_problems[:n_samples]
        n_cols = 6  # Input, Output, Test, Input Colors, Output Colors, Test Colors
        n_rows = len(samples)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
        fig.suptitle(f"No-Background ARC Problems - Real Color Visualization (First {len(samples)} problems)", 
                     fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, problem in enumerate(samples):
            # Find first qualifying training example
            example = None
            for train_ex in problem['train']:
                if not train_ex['input_has_bg'] or not train_ex['output_has_bg']:
                    example = train_ex
                    break
            
            if example:
                # Input visualization
                input_grid = np.array(example['input_grid'])
                im1 = axes[i, 0].imshow(input_grid, cmap=self.cmap, vmin=0, vmax=9)
                axes[i, 0].set_title(f"{problem['name']}\nTrain Input", fontsize=9)
                axes[i, 0].grid(True, alpha=0.2, color='white', linewidth=0.5)
                
                # Output visualization
                output_grid = np.array(example['output_grid'])
                im2 = axes[i, 1].imshow(output_grid, cmap=self.cmap, vmin=0, vmax=9)
                axes[i, 1].set_title(f"Train Output", fontsize=9)
                axes[i, 1].grid(True, alpha=0.2, color='white', linewidth=0.5)
                
                # Test visualization (if available)
                if problem['test']:
                    test_grid = np.array(problem['test'][0]['grid'])
                    im3 = axes[i, 2].imshow(test_grid, cmap=self.cmap, vmin=0, vmax=9)
                    axes[i, 2].set_title(f"Test Input", fontsize=9)
                    axes[i, 2].grid(True, alpha=0.2, color='white', linewidth=0.5)
                else:
                    axes[i, 2].axis('off')
                    axes[i, 2].text(0.5, 0.5, 'No Test', ha='center', va='center', transform=axes[i, 2].transAxes)
                
                # Color distributions with actual colors
                input_colors, input_counts = np.unique(input_grid, return_counts=True)
                axes[i, 3].bar(input_colors, input_counts, color=[self.colors[c] for c in input_colors])
                axes[i, 3].set_title(f"Input Colors\n(ratio: {example['input_ratio']:.2f})", fontsize=8)
                axes[i, 3].set_xticks(input_colors)
                
                output_colors, output_counts = np.unique(output_grid, return_counts=True)
                axes[i, 4].bar(output_colors, output_counts, color=[self.colors[c] for c in output_colors])
                axes[i, 4].set_title(f"Output Colors\n(ratio: {example['output_ratio']:.2f})", fontsize=8)
                axes[i, 4].set_xticks(output_colors)
                
                if problem['test']:
                    test_colors, test_counts = np.unique(test_grid, return_counts=True)
                    axes[i, 5].bar(test_colors, test_counts, color=[self.colors[c] for c in test_colors])
                    axes[i, 5].set_title(f"Test Colors\n(ratio: {problem['test'][0]['ratio']:.2f})", fontsize=8)
                    axes[i, 5].set_xticks(test_colors)
                else:
                    axes[i, 5].axis('off')
            
            # Remove axis labels for cleaner look
            for j in range(n_cols):
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_file}")
        
        return fig
    
    def visualize_single_problem(self, problem_id: str, save_file: str = None):
        """Detailed color visualization of a single problem"""
        problem = None
        for p in self.no_bg_problems:
            if p['name'] == problem_id:
                problem = p
                break
        
        if not problem:
            print(f"Problem {problem_id} not found in no-background problems list.")
            return None
        
        # Count qualifying examples
        qualifying_examples = [ex for ex in problem['train'] if not ex['input_has_bg'] or not ex['output_has_bg']]
        n_examples = len(qualifying_examples)
        n_tests = len(problem['test'])
        
        # Create subplot layout
        n_rows = max(n_examples, 1)
        n_cols = 4 + (2 if n_tests > 0 else 0)  # Input, Output, Input Colors, Output Colors, [Test, Test Colors]
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.suptitle(f"Problem {problem_id} - Detailed Color Analysis", fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, example in enumerate(qualifying_examples):
            # Training input
            input_grid = np.array(example['input_grid'])
            axes[i, 0].imshow(input_grid, cmap=self.cmap, vmin=0, vmax=9)
            axes[i, 0].set_title(f"Train {example['index']+1} Input\n(ratio: {example['input_ratio']:.2f})")
            axes[i, 0].grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Training output
            output_grid = np.array(example['output_grid'])
            axes[i, 1].imshow(output_grid, cmap=self.cmap, vmin=0, vmax=9)
            axes[i, 1].set_title(f"Train {example['index']+1} Output\n(ratio: {example['output_ratio']:.2f})")
            axes[i, 1].grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Input color distribution
            input_colors, input_counts = np.unique(input_grid, return_counts=True)
            axes[i, 2].bar(input_colors, input_counts, color=[self.colors[c] for c in input_colors])
            axes[i, 2].set_title(f"Input Color Distribution")
            axes[i, 2].set_xlabel("Color Index")
            axes[i, 2].set_ylabel("Count")
            axes[i, 2].set_xticks(input_colors)
            
            # Output color distribution
            output_colors, output_counts = np.unique(output_grid, return_counts=True)
            axes[i, 3].bar(output_colors, output_counts, color=[self.colors[c] for c in output_colors])
            axes[i, 3].set_title(f"Output Color Distribution")
            axes[i, 3].set_xlabel("Color Index")
            axes[i, 3].set_ylabel("Count")
            axes[i, 3].set_xticks(output_colors)
            
            # Test cases (if available)
            if n_tests > 0 and i < n_tests:
                test = problem['test'][i]
                test_grid = np.array(test['grid'])
                axes[i, 4].imshow(test_grid, cmap=self.cmap, vmin=0, vmax=9)
                axes[i, 4].set_title(f"Test {test['index']+1} Input\n(ratio: {test['ratio']:.2f})")
                axes[i, 4].grid(True, alpha=0.3, color='white', linewidth=0.5)
                
                # Test color distribution
                test_colors, test_counts = np.unique(test_grid, return_counts=True)
                axes[i, 5].bar(test_colors, test_counts, color=[self.colors[c] for c in test_colors])
                axes[i, 5].set_title(f"Test Color Distribution")
                axes[i, 5].set_xlabel("Color Index")
                axes[i, 5].set_ylabel("Count")
                axes[i, 5].set_xticks(test_colors)
            elif n_cols > 4:
                axes[i, 4].axis('off')
                axes[i, 5].axis('off')
        
        plt.tight_layout()
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Single problem visualization saved to: {save_file}")
        
        return fig
    
    def calculate_ratio_statistics(self):
        """Calculate comprehensive ratio statistics for background vs no-background problems"""
        
        # Collect all ratios for background problems
        bg_input_ratios = []
        bg_output_ratios = []
        bg_test_ratios = []
        
        for problem in self.bg_problems:
            # Training examples
            for example in problem['train']:
                bg_input_ratios.append(example['input_ratio'])
                bg_output_ratios.append(example['output_ratio'])
            # Test examples
            for test in problem['test']:
                bg_test_ratios.append(test['ratio'])
        
        # Collect all ratios for no-background problems
        no_bg_input_ratios = []
        no_bg_output_ratios = []
        no_bg_test_ratios = []
        
        for problem in self.no_bg_problems:
            # Training examples
            for example in problem['train']:
                no_bg_input_ratios.append(example['input_ratio'])
                no_bg_output_ratios.append(example['output_ratio'])
            # Test examples
            for test in problem['test']:
                no_bg_test_ratios.append(test['ratio'])
        
        # Calculate statistics
        stats = {
            'background_problems': {
                'count': len(self.bg_problems),
                'input_ratios': {
                    'mean': np.mean(bg_input_ratios) if bg_input_ratios else 0,
                    'median': np.median(bg_input_ratios) if bg_input_ratios else 0,
                    'std': np.std(bg_input_ratios) if bg_input_ratios else 0,
                    'min': np.min(bg_input_ratios) if bg_input_ratios else 0,
                    'max': np.max(bg_input_ratios) if bg_input_ratios else 0,
                    'count': len(bg_input_ratios)
                },
                'output_ratios': {
                    'mean': np.mean(bg_output_ratios) if bg_output_ratios else 0,
                    'median': np.median(bg_output_ratios) if bg_output_ratios else 0,
                    'std': np.std(bg_output_ratios) if bg_output_ratios else 0,
                    'min': np.min(bg_output_ratios) if bg_output_ratios else 0,
                    'max': np.max(bg_output_ratios) if bg_output_ratios else 0,
                    'count': len(bg_output_ratios)
                },
                'test_ratios': {
                    'mean': np.mean(bg_test_ratios) if bg_test_ratios else 0,
                    'median': np.median(bg_test_ratios) if bg_test_ratios else 0,
                    'std': np.std(bg_test_ratios) if bg_test_ratios else 0,
                    'min': np.min(bg_test_ratios) if bg_test_ratios else 0,
                    'max': np.max(bg_test_ratios) if bg_test_ratios else 0,
                    'count': len(bg_test_ratios)
                }
            },
            'no_background_problems': {
                'count': len(self.no_bg_problems),
                'input_ratios': {
                    'mean': np.mean(no_bg_input_ratios) if no_bg_input_ratios else 0,
                    'median': np.median(no_bg_input_ratios) if no_bg_input_ratios else 0,
                    'std': np.std(no_bg_input_ratios) if no_bg_input_ratios else 0,
                    'min': np.min(no_bg_input_ratios) if no_bg_input_ratios else 0,
                    'max': np.max(no_bg_input_ratios) if no_bg_input_ratios else 0,
                    'count': len(no_bg_input_ratios)
                },
                'output_ratios': {
                    'mean': np.mean(no_bg_output_ratios) if no_bg_output_ratios else 0,
                    'median': np.median(no_bg_output_ratios) if no_bg_output_ratios else 0,
                    'std': np.std(no_bg_output_ratios) if no_bg_output_ratios else 0,
                    'min': np.min(no_bg_output_ratios) if no_bg_output_ratios else 0,
                    'max': np.max(no_bg_output_ratios) if no_bg_output_ratios else 0,
                    'count': len(no_bg_output_ratios)
                },
                'test_ratios': {
                    'mean': np.mean(no_bg_test_ratios) if no_bg_test_ratios else 0,
                    'median': np.median(no_bg_test_ratios) if no_bg_test_ratios else 0,
                    'std': np.std(no_bg_test_ratios) if no_bg_test_ratios else 0,
                    'min': np.min(no_bg_test_ratios) if no_bg_test_ratios else 0,
                    'max': np.max(no_bg_test_ratios) if no_bg_test_ratios else 0,
                    'count': len(no_bg_test_ratios)
                }
            }
        }
        
        return stats
    
    def print_ratio_statistics(self, stats):
        """Print comprehensive ratio statistics in a formatted way"""
        print("\n" + "="*80)
        print("COMPREHENSIVE RATIO STATISTICS ANALYSIS")
        print("="*80)
        
        print(f"\nOVERVIEW:")
        print(f"Total Problems Analyzed: {stats['background_problems']['count'] + stats['no_background_problems']['count']}")
        print(f"Problems with Background (>1.5 ratio): {stats['background_problems']['count']}")
        print(f"Problems with No Background (≤1.5 ratio): {stats['no_background_problems']['count']}")
        print(f"No-Background Percentage: {stats['no_background_problems']['count']/(stats['background_problems']['count'] + stats['no_background_problems']['count'])*100:.1f}%")
        
        # Background Problems Statistics
        print(f"\nBACKGROUND PROBLEMS (Dominant Color > 1.5x Second Most Common)")
        print("-" * 70)
        bg = stats['background_problems']
        
        print(f"\nInput Ratios ({bg['input_ratios']['count']} samples):")
        print(f"  Mean:   {bg['input_ratios']['mean']:.3f}")
        print(f"  Median: {bg['input_ratios']['median']:.3f}")
        print(f"  Std:    {bg['input_ratios']['std']:.3f}")
        print(f"  Range:  {bg['input_ratios']['min']:.3f} - {bg['input_ratios']['max']:.3f}")
        
        print(f"\nOutput Ratios ({bg['output_ratios']['count']} samples):")
        print(f"  Mean:   {bg['output_ratios']['mean']:.3f}")
        print(f"  Median: {bg['output_ratios']['median']:.3f}")
        print(f"  Std:    {bg['output_ratios']['std']:.3f}")
        print(f"  Range:  {bg['output_ratios']['min']:.3f} - {bg['output_ratios']['max']:.3f}")
        
        print(f"\nTest Ratios ({bg['test_ratios']['count']} samples):")
        print(f"  Mean:   {bg['test_ratios']['mean']:.3f}")
        print(f"  Median: {bg['test_ratios']['median']:.3f}")
        print(f"  Std:    {bg['test_ratios']['std']:.3f}")
        print(f"  Range:  {bg['test_ratios']['min']:.3f} - {bg['test_ratios']['max']:.3f}")
        
        # No-Background Problems Statistics
        print(f"\nNO-BACKGROUND PROBLEMS (Dominant Color ≤ 1.5x Second Most Common)")
        print("-" * 70)
        no_bg = stats['no_background_problems']
        
        print(f"\nInput Ratios ({no_bg['input_ratios']['count']} samples):")
        print(f"  Mean:   {no_bg['input_ratios']['mean']:.3f}")
        print(f"  Median: {no_bg['input_ratios']['median']:.3f}")
        print(f"  Std:    {no_bg['input_ratios']['std']:.3f}")
        print(f"  Range:  {no_bg['input_ratios']['min']:.3f} - {no_bg['input_ratios']['max']:.3f}")
        
        print(f"\nOutput Ratios ({no_bg['output_ratios']['count']} samples):")
        print(f"  Mean:   {no_bg['output_ratios']['mean']:.3f}")
        print(f"  Median: {no_bg['output_ratios']['median']:.3f}")
        print(f"  Std:    {no_bg['output_ratios']['std']:.3f}")
        print(f"  Range:  {no_bg['output_ratios']['min']:.3f} - {no_bg['output_ratios']['max']:.3f}")
        
        print(f"\nTest Ratios ({no_bg['test_ratios']['count']} samples):")
        print(f"  Mean:   {no_bg['test_ratios']['mean']:.3f}")
        print(f"  Median: {no_bg['test_ratios']['median']:.3f}")
        print(f"  Std:    {no_bg['test_ratios']['std']:.3f}")
        print(f"  Range:  {no_bg['test_ratios']['min']:.3f} - {no_bg['test_ratios']['max']:.3f}")
        
        print("\n" + "="*80)
    
    def save_ratio_statistics(self, stats, filename="ratio_statistics_analysis.txt"):
        """Save detailed ratio statistics to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE RATIO STATISTICS ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"OVERVIEW:\n")
            f.write(f"Total Problems Analyzed: {stats['background_problems']['count'] + stats['no_background_problems']['count']}\n")
            f.write(f"Problems with Background (>1.5 ratio): {stats['background_problems']['count']}\n")
            f.write(f"Problems with No Background (≤1.5 ratio): {stats['no_background_problems']['count']}\n")
            f.write(f"No-Background Percentage: {stats['no_background_problems']['count']/(stats['background_problems']['count'] + stats['no_background_problems']['count'])*100:.1f}%\n\n")
            
            # Background Problems
            f.write("BACKGROUND PROBLEMS (Dominant Color > 1.5x Second Most Common)\n")
            f.write("-" * 70 + "\n")
            bg = stats['background_problems']
            
            f.write(f"\nInput Ratios ({bg['input_ratios']['count']} samples):\n")
            f.write(f"  Mean:   {bg['input_ratios']['mean']:.3f}\n")
            f.write(f"  Median: {bg['input_ratios']['median']:.3f}\n")
            f.write(f"  Std:    {bg['input_ratios']['std']:.3f}\n")
            f.write(f"  Range:  {bg['input_ratios']['min']:.3f} - {bg['input_ratios']['max']:.3f}\n")
            
            f.write(f"\nOutput Ratios ({bg['output_ratios']['count']} samples):\n")
            f.write(f"  Mean:   {bg['output_ratios']['mean']:.3f}\n")
            f.write(f"  Median: {bg['output_ratios']['median']:.3f}\n")
            f.write(f"  Std:    {bg['output_ratios']['std']:.3f}\n")
            f.write(f"  Range:  {bg['output_ratios']['min']:.3f} - {bg['output_ratios']['max']:.3f}\n")
            
            f.write(f"\nTest Ratios ({bg['test_ratios']['count']} samples):\n")
            f.write(f"  Mean:   {bg['test_ratios']['mean']:.3f}\n")
            f.write(f"  Median: {bg['test_ratios']['median']:.3f}\n")
            f.write(f"  Std:    {bg['test_ratios']['std']:.3f}\n")
            f.write(f"  Range:  {bg['test_ratios']['min']:.3f} - {bg['test_ratios']['max']:.3f}\n")
            
            # No-Background Problems
            f.write("\nNO-BACKGROUND PROBLEMS (Dominant Color ≤ 1.5x Second Most Common)\n")
            f.write("-" * 70 + "\n")
            no_bg = stats['no_background_problems']
            
            f.write(f"\nInput Ratios ({no_bg['input_ratios']['count']} samples):\n")
            f.write(f"  Mean:   {no_bg['input_ratios']['mean']:.3f}\n")
            f.write(f"  Median: {no_bg['input_ratios']['median']:.3f}\n")
            f.write(f"  Std:    {no_bg['input_ratios']['std']:.3f}\n")
            f.write(f"  Range:  {no_bg['input_ratios']['min']:.3f} - {no_bg['input_ratios']['max']:.3f}\n")
            
            f.write(f"\nOutput Ratios ({no_bg['output_ratios']['count']} samples):\n")
            f.write(f"  Mean:   {no_bg['output_ratios']['mean']:.3f}\n")
            f.write(f"  Median: {no_bg['output_ratios']['median']:.3f}\n")
            f.write(f"  Std:    {no_bg['output_ratios']['std']:.3f}\n")
            f.write(f"  Range:  {no_bg['output_ratios']['min']:.3f} - {no_bg['output_ratios']['max']:.3f}\n")
            
            f.write(f"\nTest Ratios ({no_bg['test_ratios']['count']} samples):\n")
            f.write(f"  Mean:   {no_bg['test_ratios']['mean']:.3f}\n")
            f.write(f"  Median: {no_bg['test_ratios']['median']:.3f}\n")
            f.write(f"  Std:    {no_bg['test_ratios']['std']:.3f}\n")
            f.write(f"  Range:  {no_bg['test_ratios']['min']:.3f} - {no_bg['test_ratios']['max']:.3f}\n")
            
        print(f"통계 분석 저장됨: {filename}")


# 사용 예제
if __name__ == "__main__":

    dataset_path = "."  
    
    analyzer = NoBackgroundAnalyzer(dataset_path)
    
    # 모든 문제 스캔
    analyzer.scan_all_problems()
    
    # 배경 없는 문제 목록 저장
    analyzer.save_no_bg_list()
    
    # 패턴 분석
    patterns = analyzer.analyze_no_bg_patterns()
    print(f"\n배경 없는 문제들의 특징:")
    print(f"평균 색상 수: {np.mean(patterns['color_diversity']):.2f}")
    print(f"평균 밀도: {np.mean(patterns['density']):.2f}")
    
    # 통계 분석
    print("\n통계 분석 계산 중...")
    stats = analyzer.calculate_ratio_statistics()
    analyzer.print_ratio_statistics(stats)
    analyzer.save_ratio_statistics(stats)
    
    # Color visualization
    if analyzer.no_bg_problems:
        print(f"\n처리 완료! {len(analyzer.no_bg_problems)}개의 문제가 조건을 만족합니다.")
        print("자세한 내용은 no_background_problems.txt 파일을 확인하세요.")
        
        # Create color visualizations
        print("\n색상 시각화 생성 중...")
        
        # Summary visualization (first 10 problems)
        print("1. 요약 시각화 (첫 10개 문제)...")
        fig1 = analyzer.create_summary_visualization(min(10, len(analyzer.no_bg_problems)))
        plt.savefig("no_bg_summary_visualization.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print("   저장됨: no_bg_summary_visualization.png")
        
        # Comprehensive visualization (first 15 problems)
        print("2. 종합 시각화 (첫 15개 문제)...")
        fig2 = analyzer.visualize_no_bg_problems(n_samples=15, save_file="no_bg_comprehensive_visualization.png")
        plt.close(fig2)
        
        # Individual problem examples (first 3 problems)
        print("3. 개별 문제 상세 시각화 (첫 3개)...")
        for i, problem in enumerate(analyzer.no_bg_problems[:3]):
            problem_id = problem['name']
            save_name = f"problem_{problem_id}_detailed.png"
            fig3 = analyzer.visualize_single_problem(problem_id, save_file=save_name)
            if fig3:
                plt.close(fig3)
        
        print("\n모든 색상 시각화가 완료되었습니다!")
        print("생성된 파일들:")
        print("- no_bg_summary_visualization.png")
        print("- no_bg_comprehensive_visualization.png")
        print("- problem_[ID]_detailed.png (첫 3개 문제)")
        print("\n이 이미지들은 ARC 표준 색상 팔레트를 사용하여 실제 색상으로 표시됩니다.")