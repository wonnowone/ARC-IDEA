# arc_solver.py

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("shape_extractor", "shape extractor.py")
shape_extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shape_extractor)
ShapeExtractor = shape_extractor.ShapeExtractor
ShapeRecognizer = shape_extractor.ShapeRecognizer
Shape = shape_extractor.Shape

class ARCProblem:
    """ARC 문제 데이터 관리"""
    
    def __init__(self, json_path: str = None, json_data: Dict = None):
        """
        Args:
            json_path: JSON 파일 경로
            json_data: 이미 로드된 JSON 데이터
        """
        if json_path:
            self.data = self._load_json(json_path)
        elif json_data:
            self.data = json_data
        else:
            raise ValueError("json_path 또는 json_data 중 하나는 필요합니다")
        
        self.train_examples = self.data.get('train', [])
        self.test_examples = self.data.get('test', [])
        
    def _load_json(self, path: str) -> Dict:
        """JSON 파일 로드"""
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_train_example(self, idx: int) -> Dict:
        """특정 training 예제 반환"""
        if 0 <= idx < len(self.train_examples):
            return self.train_examples[idx]
        raise IndexError(f"Invalid training example index: {idx}")
    
    def get_test_input(self, idx: int = 0) -> List[List[int]]:
        """테스트 입력 반환"""
        if 0 <= idx < len(self.test_examples):
            return self.test_examples[idx]['input']
        raise IndexError(f"Invalid test example index: {idx}")


class ARCAnalyzer:
    """ARC 문제 분석기"""
    
    def __init__(self, problem: ARCProblem):
        self.problem = problem
        self.train_shapes = []  # [(input_shapes, output_shapes), ...]
        
    def analyze_all_examples(self, connectivity: int = 4):
        """모든 training 예제 분석"""
        print("="*60)
        print("ANALYZING TRAINING EXAMPLES")
        print("="*60)
        
        for idx, example in enumerate(self.problem.train_examples):
            print(f"\n--- Training Example {idx + 1} ---")
            
            input_shapes = self.analyze_grid(example['input'], "Input")
            output_shapes = self.analyze_grid(example['output'], "Output")
            
            self.train_shapes.append((input_shapes, output_shapes))
            
            # 변화 분석
            self.analyze_changes(input_shapes, output_shapes)
    
    def analyze_grid(self, grid: List[List[int]], label: str) -> List[Shape]:
        """단일 그리드 분석"""
        print(f"\n{label} Grid ({len(grid)}x{len(grid[0])}):")
        
        # Shape 추출
        extractor = ShapeExtractor(grid)
        shapes = extractor.extract_all_shapes()
        
        # 색상별 요약
        for color, color_shapes in extractor.shapes_by_color.items():
            print(f"  Color {color}: {len(color_shapes)} shapes")
            for shape in color_shapes:
                pattern = ShapeRecognizer.identify_pattern(shape)
                print(f"    - {pattern} ({shape.pixel_count} pixels) at {shape.centroid}")
        
        return shapes
    
    def analyze_changes(self, input_shapes: List[Shape], output_shapes: List[Shape]):
        """입출력 간 변화 분석"""
        print("\nChanges detected:")
        
        # 색상별 개수 변화
        input_colors = {}
        output_colors = {}
        
        for shape in input_shapes:
            input_colors[shape.color] = input_colors.get(shape.color, 0) + 1
        
        for shape in output_shapes:
            output_colors[shape.color] = output_colors.get(shape.color, 0) + 1
        
        # 변화 출력
        all_colors = set(input_colors.keys()) | set(output_colors.keys())
        
        for color in all_colors:
            in_cnt = input_colors.get(color, 0)
            out_cnt = output_colors.get(color, 0)
            
            if in_cnt != out_cnt:
                print(f"  Color {color}: {in_cnt} → {out_cnt} shapes")
        
        # 전체 픽셀 수 변화
        input_pixels = sum(s.pixel_count for s in input_shapes)
        output_pixels = sum(s.pixel_count for s in output_shapes)
        
        if input_pixels != output_pixels:
            print(f"  Total pixels: {input_pixels} → {output_pixels}")
    
    def find_pattern(self):
        """모든 예제에서 공통 패턴 찾기"""
        print("\n" + "="*60)
        print("FINDING COMMON PATTERNS")
        print("="*60)
        
        # 여기에 패턴 찾기 로직 구현
        # 예: 모든 예제에서 동일한 변환이 일어나는지 확인
        
        common_changes = []
        
        # 색상 변화 패턴
        color_mappings = self._find_color_mappings()
        if color_mappings:
            print("\nColor mappings found:")
            for src, dst in color_mappings.items():
                print(f"  {src} → {dst}")
        
        # 위치 변화 패턴
        position_changes = self._find_position_changes()
        if position_changes:
            print("\nPosition changes found:")
            for change in position_changes:
                print(f"  {change}")
        
        return common_changes
    
    def _find_color_mappings(self) -> Dict[int, int]:
        """색상 매핑 찾기"""
        # 간단한 구현 - 실제로는 더 복잡한 로직 필요
        mappings = {}
        
        for input_shapes, output_shapes in self.train_shapes:
            # 각 예제에서 색상 변화 확인
            for in_shape in input_shapes:
                for out_shape in output_shapes:
                    if (in_shape.pixel_count == out_shape.pixel_count and 
                        in_shape.width == out_shape.width and 
                        in_shape.height == out_shape.height):
                        if in_shape.color != out_shape.color:
                            mappings[in_shape.color] = out_shape.color
        
        return mappings
    
    def _find_position_changes(self) -> List[str]:
        """위치 변화 패턴 찾기"""
        changes = []
        
        for input_shapes, output_shapes in self.train_shapes:
            # 각 shape의 이동 확인
            for in_shape in input_shapes:
                for out_shape in output_shapes:
                    if (in_shape.color == out_shape.color and 
                        in_shape.pixel_count == out_shape.pixel_count):
                        dx = out_shape.centroid[0] - in_shape.centroid[0]
                        dy = out_shape.centroid[1] - in_shape.centroid[1]
                        
                        if abs(dx) > 0.5 or abs(dy) > 0.5:
                            changes.append(f"Move by ({dx:.1f}, {dy:.1f})")
        
        return changes
    
    def visualize_grid(self, grid: List[List[int]], title: str = ""):
        """그리드를 시각적으로 출력"""
        if title:
            print(f"\n{title}")
        
        # 색상 매핑 (0-9를 다른 문자로)
        color_map = {
            0: '.',  # 배경
            1: '#',
            2: '@',
            3: '*',
            4: '%',
            5: '&',
            6: 'O',
            7: '^',
            8: '=',
            9: '+'
        }
        
        for row in grid:
            print(' '.join(color_map.get(cell, str(cell)) for cell in row))


def solve_arc_problem(json_path: str):
    """ARC 문제 해결 메인 함수"""
    
    # 1. 문제 로드
    problem = ARCProblem(json_path)
    print(f"Loaded problem with {len(problem.train_examples)} training examples")
    print(f"and {len(problem.test_examples)} test examples")
    
    # 2. 분석
    analyzer = ARCAnalyzer(problem)
    
    # 모든 training 예제 시각화 (선택적)
    for idx, example in enumerate(problem.train_examples):
        analyzer.visualize_grid(example['input'], f"Training {idx+1} - Input")
        analyzer.visualize_grid(example['output'], f"Training {idx+1} - Output")
    
    # 3. Shape 분석
    analyzer.analyze_all_examples()
    
    # 4. 패턴 찾기
    patterns = analyzer.find_pattern()
    
    # 5. 테스트 입력 분석
    print("\n" + "="*60)
    print("ANALYZING TEST INPUT")
    print("="*60)
    
    test_input = problem.get_test_input()
    analyzer.visualize_grid(test_input, "Test Input")
    test_shapes = analyzer.analyze_grid(test_input, "Test")
    
    # 6. 솔루션 생성 (TODO: 실제 변환 로직 구현)
    print("\n" + "="*60)
    print("GENERATING SOLUTION")
    print("="*60)
    print("TODO: Apply discovered patterns to generate solution")
    
    return test_shapes


# 사용 예제
if __name__ == "__main__":
    # 로컬 JSON 파일들 로드
    challenges_path = "arc-agi_training_challenges.json"
    solutions_path = "arc-agi_training_solutions.json"
    
    # JSON 파일 로드
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)
    
    print(f"Loaded {len(challenges)} training challenges")
    print(f"Loaded {len(solutions)} training solutions")
    
    # 첫 번째 문제 분석 예제
    if challenges:
        first_problem_id = list(challenges.keys())[0]
        print(f"\nAnalyzing problem: {first_problem_id}")
        
        # 문제 데이터 구성
        problem_data = challenges[first_problem_id]
        if first_problem_id in solutions:
            # 솔루션을 train 예제에 추가
            for i, solution in enumerate(solutions[first_problem_id]):
                if i < len(problem_data['train']):
                    problem_data['train'][i]['output'] = solution
        
        # 문제 분석
        problem = ARCProblem(json_data=problem_data)
        analyzer = ARCAnalyzer(problem)
        
        # 모든 training 예제 시각화
        for idx, example in enumerate(problem.train_examples):
            analyzer.visualize_grid(example['input'], f"Training {idx+1} - Input")
            if 'output' in example:
                analyzer.visualize_grid(example['output'], f"Training {idx+1} - Output")
        
        # Shape 분석 및 패턴 찾기
        analyzer.analyze_all_examples()
        analyzer.find_pattern()
        
        # 테스트 입력 분석
        if problem.test_examples:
            print("\n" + "="*60)
            print("ANALYZING TEST INPUT")
            print("="*60)
            
            test_input = problem.get_test_input()
            analyzer.visualize_grid(test_input, "Test Input")
            test_shapes = analyzer.analyze_grid(test_input, "Test")
    
    # 전체 데이터셋 통계
    print(f"\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    total_shapes_by_color = {}
    total_patterns = {}
    
    for problem_id in list(challenges.keys())[:5]:  # 처음 5개 문제만 분석
        problem_data = challenges[problem_id]
        if problem_id in solutions:
            for i, solution in enumerate(solutions[problem_id]):
                if i < len(problem_data['train']):
                    problem_data['train'][i]['output'] = solution
        
        problem = ARCProblem(json_data=problem_data)
        analyzer = ARCAnalyzer(problem)
        
        for example in problem.train_examples:
            if 'output' in example:
                shapes = analyzer.analyze_grid(example['input'], f"Problem {problem_id}")
                
                # 색상별 통계
                for shape in shapes:
                    color = shape.color
                    total_shapes_by_color[color] = total_shapes_by_color.get(color, 0) + 1
                    
                    # 패턴 인식
                    pattern = ShapeRecognizer.identify_pattern(shape)
                    total_patterns[pattern] = total_patterns.get(pattern, 0) + 1
    
    print("\nShape colors found:")
    for color, count in sorted(total_shapes_by_color.items()):
        print(f"  Color {color}: {count} shapes")
    
    print("\nShape patterns found:")
    for pattern, count in sorted(total_patterns.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pattern}: {count} occurrences")