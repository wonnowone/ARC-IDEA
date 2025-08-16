# shape_extractor.py

import numpy as np
from collections import deque
from typing import List, Tuple, Dict, Set

class Shape:
    """연결된 픽셀들로 구성된 하나의 모양"""
    
    def __init__(self, pixels: List[Tuple[int, int]], color: int):
        self.pixels = pixels  # [(x1,y1), (x2,y2), ...]
        self.color = color
        self.pixel_count = len(pixels)
        
        # 자동 계산 속성
        self._calculate_properties()
        
    def _calculate_properties(self):
        """Shape의 기본 속성 계산"""
        if not self.pixels:
            return
            
        xs = [p[0] for p in self.pixels]
        ys = [p[1] for p in self.pixels]
        
        # Bounding box: (min_x, min_y, max_x, max_y)
        self.bounding_box = (min(xs), min(ys), max(xs), max(ys))
        self.width = self.bounding_box[2] - self.bounding_box[0] + 1
        self.height = self.bounding_box[3] - self.bounding_box[1] + 1
        
        # 중심점
        self.centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        
        # 정규화된 매트릭스 형태
        self.shape_matrix = self._to_matrix()
        
    def _to_matrix(self):
        """Shape를 2D 매트릭스로 변환 (원점 기준)"""
        if not self.pixels:
            return [[]]
            
        min_x, min_y = self.bounding_box[0], self.bounding_box[1]
        
        # 빈 매트릭스 생성
        matrix = [[0] * self.width for _ in range(self.height)]
        
        # 픽셀 채우기
        for x, y in self.pixels:
            matrix[y - min_y][x - min_x] = 1
            
        return matrix
    
    def __repr__(self):
        return f"Shape(color={self.color}, pixels={self.pixel_count}, size={self.width}x{self.height}, pos=({self.centroid[0]:.1f},{self.centroid[1]:.1f}))"


class ShapeExtractor:
    """그리드에서 연결된 모양들을 추출"""
    
    def __init__(self, grid):
        self.grid = np.array(grid) if not isinstance(grid, np.ndarray) else grid
        self.height, self.width = self.grid.shape
        self.shapes_by_color = {}  # {color: [Shape1, Shape2, ...]}
        self.all_shapes = []
        
    def extract_all_shapes(self, connectivity: int = 4, include_background: bool = False) -> List[Shape]:
        """
        모든 색상에 대해 연결된 모양 추출
        
        Args:
            connectivity: 4 (상하좌우) or 8 (대각선 포함)
            include_background: 0(배경)도 포함할지 여부
        """
        # 초기화
        self.shapes_by_color = {}
        self.all_shapes = []
        
        # 고유 색상 찾기
        unique_colors = np.unique(self.grid)
        
        for color in unique_colors:
            if color == 0 and not include_background:
                continue
                
            # 해당 색상의 연결된 요소들 찾기
            shapes = self._find_connected_components(color, connectivity)
            
            if shapes:
                self.shapes_by_color[color] = shapes
                self.all_shapes.extend(shapes)
        
        return self.all_shapes
    
    def _find_connected_components(self, color: int, connectivity: int) -> List[Shape]:
        """특정 색상의 연결된 요소들 찾기 (BFS)"""
        visited = set()
        shapes = []
        
        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == color and (i, j) not in visited:
                    # BFS로 연결된 픽셀 찾기
                    component = self._bfs(i, j, color, visited, connectivity)
                    if component:
                        # Shape 객체 생성 (x,y 좌표로 변환 - j가 x, i가 y)
                        pixels = [(y, x) for x, y in component]
                        shapes.append(Shape(pixels, color))
        
        return shapes
    
    def _bfs(self, start_i: int, start_j: int, color: int, visited: Set, connectivity: int) -> List[Tuple[int, int]]:
        """BFS로 연결된 같은 색상 픽셀 찾기"""
        queue = deque([(start_i, start_j)])
        component = []
        
        while queue:
            i, j = queue.popleft()
            
            if (i, j) in visited:
                continue
                
            if i < 0 or i >= self.height or j < 0 or j >= self.width:
                continue
                
            if self.grid[i][j] != color:
                continue
            
            visited.add((i, j))
            component.append((i, j))
            
            # 이웃 픽셀 추가
            neighbors = self._get_neighbors(i, j, connectivity)
            queue.extend(neighbors)
        
        return component
    
    def _get_neighbors(self, i: int, j: int, connectivity: int) -> List[Tuple[int, int]]:
        """이웃 픽셀 좌표 반환"""
        neighbors = []
        
        # 4-방향 (상하좌우)
        directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # 8-방향 (대각선 포함)
        directions_8 = directions_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        directions = directions_8 if connectivity == 8 else directions_4
        
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.height and 0 <= nj < self.width:
                neighbors.append((ni, nj))
        
        return neighbors
    
    def get_shape_summary(self) -> Dict:
        """추출된 모양들의 요약 정보"""
        summary = {
            'total_shapes': len(self.all_shapes),
            'colors': {},
        }
        
        for color, shapes in self.shapes_by_color.items():
            summary['colors'][color] = {
                'count': len(shapes),
                'pixel_counts': [s.pixel_count for s in shapes],
                'sizes': [(s.width, s.height) for s in shapes]
            }
        
        return summary
    
    def print_shapes(self):
        """추출된 모양 정보 출력 (simplified)"""
        print(f"=== Total {len(self.all_shapes)} shapes found ===\n")
        
        for color, shapes in self.shapes_by_color.items():
            print(f"Color {color}: {len(shapes)} shapes")
            for i, shape in enumerate(shapes, 1):
                print(f"  Shape {i}: {shape.pixel_count} pixels at ({shape.centroid[0]:.1f},{shape.centroid[1]:.1f}) [size: {shape.width}x{shape.height}]")
            print()


class ShapeRecognizer:
    """Simplified shape analysis focusing on essential properties"""
    
    @staticmethod
    def identify_pattern(shape: Shape) -> str:
        """Return basic shape description with essential properties"""
        return f"{shape.pixel_count}px_shape"


def compare_shapes(shape1: Shape, shape2: Shape) -> float:
    """두 Shape의 유사도 계산 (0.0 ~ 1.0)"""
    
    # 색상이 다르면 0
    if shape1.color != shape2.color:
        return 0.0
    
    # 픽셀 개수 동일성
    pixel_score = 1.0 if shape1.pixel_count == shape2.pixel_count else 0.5
    
    # 크기 유사도
    size_diff = abs(shape1.width - shape2.width) + abs(shape1.height - shape2.height)
    size_score = 1.0 / (1.0 + size_diff * 0.5)
    
    # 형태 유사도 (같은 크기인 경우만)
    shape_score = 0.0
    if shape1.width == shape2.width and shape1.height == shape2.height:
        matching = sum(
            1 for i in range(shape1.height)
            for j in range(shape1.width)
            if shape1.shape_matrix[i][j] == shape2.shape_matrix[i][j]
        )
        total = shape1.width * shape1.height
        shape_score = matching / total if total > 0 else 0.0
    
    # 가중 평균
    return pixel_score * 0.3 + size_score * 0.3 + shape_score * 0.4


# 사용 예제
if __name__ == "__main__":
    # 테스트 그리드
    test_grid = [
        [0, 1, 1, 0, 0, 2, 2, 2],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 3, 0, 3],
        [0, 0, 0, 0, 0, 3, 3, 3],
        [4, 4, 0, 0, 0, 0, 3, 0],
        [4, 4, 0, 0, 0, 0, 0, 0],
    ]
    
    # Shape 추출
    extractor = ShapeExtractor(test_grid)
    shapes = extractor.extract_all_shapes(connectivity=4)
    
    # 결과 출력
    extractor.print_shapes()
    
    # 패턴 인식
    recognizer = ShapeRecognizer()
    print("=== Pattern Recognition ===")
    for shape in shapes:
        pattern = recognizer.identify_pattern(shape)
        print(f"Shape (color={shape.color}, pixels={shape.pixel_count}): {pattern}")