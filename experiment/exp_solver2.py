# -*- coding: utf-8 -*-
"""
Permanent Memory Solver with DBSCAN-Style Problem Classification
Implements long-term memory storage and retrieval with clustering-based problem classification.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any

class ProblemObjectiveExtractor:
    """Extracts objectives and movement patterns from ARC problems."""
    
    def __init__(self):
        self.movement_patterns = {
            'translation': ['shift', 'move', 'translate'],
            'rotation': ['rotate', 'turn', 'pivot'],
            'reflection': ['mirror', 'flip', 'reflect'],
            'scaling': ['resize', 'scale', 'expand', 'shrink'],
            'completion': ['fill', 'complete', 'extend'],
            'filtering': ['remove', 'filter', 'select'],
            'grouping': ['group', 'cluster', 'organize'],
            'pattern': ['repeat', 'pattern', 'sequence']
        }
    
    def extract_movement_features(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Dict[str, float]:
        """Extract movement pattern features from input-output pair."""
        features = {}
        
        # Basic grid properties
        features['size_change'] = float(torch.numel(output_grid) / torch.numel(input_grid))
        features['color_diversity_in'] = len(torch.unique(input_grid))
        features['color_diversity_out'] = len(torch.unique(output_grid))
        
        # Shape preservation
        if input_grid.shape == output_grid.shape:
            features['shape_preserved'] = 1.0
            features['pixel_change_ratio'] = float(torch.sum(input_grid != output_grid)) / torch.numel(input_grid)
        else:
            features['shape_preserved'] = 0.0
            features['pixel_change_ratio'] = 1.0
        
        # Color preservation
        input_colors = set(torch.unique(input_grid).tolist())
        output_colors = set(torch.unique(output_grid).tolist())
        features['color_preservation'] = len(input_colors.intersection(output_colors)) / len(input_colors.union(output_colors))
        
        # Spatial correlation (if same shape)
        if input_grid.shape == output_grid.shape:
            flat_in = input_grid.flatten().float()
            flat_out = output_grid.flatten().float()
            features['spatial_correlation'] = float(torch.corrcoef(torch.stack([flat_in, flat_out]))[0, 1])
            if torch.isnan(torch.tensor(features['spatial_correlation'])):
                features['spatial_correlation'] = 0.0
        else:
            features['spatial_correlation'] = 0.0
        
        # Symmetry detection
        features['input_symmetry'] = self._detect_symmetry(input_grid)
        features['output_symmetry'] = self._detect_symmetry(output_grid)
        
        return features
    
    def _detect_symmetry(self, grid: torch.Tensor) -> float:
        """Detect symmetry in grid (horizontal, vertical, rotational)."""
        symmetry_score = 0.0
        
        # Horizontal symmetry
        if torch.equal(grid, torch.flip(grid, [0])):
            symmetry_score += 0.33
        
        # Vertical symmetry
        if torch.equal(grid, torch.flip(grid, [1])):
            symmetry_score += 0.33
        
        # Rotational symmetry (90 degrees)
        if grid.shape[0] == grid.shape[1]:
            if torch.equal(grid, torch.rot90(grid, k=2)):
                symmetry_score += 0.34
        
        return symmetry_score
    
    def classify_objective(self, features: Dict[str, float]) -> str:
        """Classify problem objective based on extracted features."""
        # Rule-based classification
        if features['size_change'] != 1.0:
            return 'scaling'
        elif features['shape_preserved'] == 0.0:
            return 'transformation'
        elif features['pixel_change_ratio'] < 0.1:
            return 'minor_edit'
        elif features['color_preservation'] < 0.5:
            return 'recoloring'
        elif features['spatial_correlation'] < 0.3:
            return 'reconstruction'
        elif abs(features['input_symmetry'] - features['output_symmetry']) > 0.3:
            return 'symmetry_operation'
        else:
            return 'pattern_completion'

class PermanentMemoryBank:
    """Long-term memory storage with DBSCAN clustering for problem classification."""
    
    def __init__(self, feature_dim: int = 256, max_memories: int = 10000):
        self.feature_dim = feature_dim
        self.max_memories = max_memories
        
        # Memory storage
        self.memories = []  # List of memory dictionaries
        self.feature_vectors = []  # Corresponding feature vectors
        self.objectives = []  # Problem objectives
        self.success_rates = []  # Success tracking
        
        # Clustering
        self.clusterer = DBSCAN(eps=0.3, min_samples=3, metric='cosine')
        self.cluster_centers = {}
        self.cluster_objectives = {}
        
        # Problem classifier
        self.objective_extractor = ProblemObjectiveExtractor()
        
    def store_memory(self, 
                    problem_features: torch.Tensor,
                    solution_features: torch.Tensor,
                    input_grid: torch.Tensor,
                    output_grid: torch.Tensor,
                    success: bool,
                    metadata: Dict[str, Any] = None):
        """Store a problem-solution pair in permanent memory."""
        
        # Extract movement features and objective
        movement_features = self.objective_extractor.extract_movement_features(input_grid, output_grid)
        objective = self.objective_extractor.classify_objective(movement_features)
        
        # Create memory entry
        memory = {
            'problem_features': problem_features.clone(),
            'solution_features': solution_features.clone(),
            'input_grid': input_grid.clone(),
            'output_grid': output_grid.clone(),
            'movement_features': movement_features,
            'objective': objective,
            'success': success,
            'timestamp': len(self.memories),
            'metadata': metadata or {}
        }
        
        # Combine features for clustering
        combined_features = torch.cat([
            problem_features.flatten(),
            torch.tensor(list(movement_features.values()), dtype=torch.float32)
        ])
        
        self.memories.append(memory)
        self.feature_vectors.append(combined_features.numpy())
        self.objectives.append(objective)
        self.success_rates.append(float(success))
        
        # Maintain memory limit
        if len(self.memories) > self.max_memories:
            self._prune_memories()
        
        # Update clustering periodically
        if len(self.memories) % 50 == 0:
            self._update_clusters()
    
    def retrieve_similar_problems(self, 
                                 problem_features: torch.Tensor,
                                 input_grid: torch.Tensor,
                                 k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve similar problems from memory using clustering."""
        
        if len(self.memories) == 0:
            return []
        
        # Extract current problem features
        current_movement = self.objective_extractor.extract_movement_features(
            input_grid, torch.zeros_like(input_grid)
        )
        current_objective = self.objective_extractor.classify_objective(current_movement)
        
        # Combine features
        combined_features = torch.cat([
            problem_features.flatten(),
            torch.tensor(list(current_movement.values()), dtype=torch.float32)
        ]).numpy().reshape(1, -1)
        
        # Find most similar memories
        similarities = cosine_similarity(combined_features, self.feature_vectors)[0]
        
        # Filter by objective if available
        objective_mask = np.array([obj == current_objective for obj in self.objectives])
        if np.any(objective_mask):
            similarities = similarities * objective_mask
        
        # Get top-k similar memories
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        similar_memories = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                memory = self.memories[idx].copy()
                memory['similarity'] = similarities[idx]
                similar_memories.append(memory)
        
        return similar_memories
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get statistics about problem clusters."""
        if not hasattr(self, 'cluster_labels_') or len(self.memories) == 0:
            return {}
        
        stats = {
            'total_clusters': len(set(self.cluster_labels_)) - (1 if -1 in self.cluster_labels_ else 0),
            'noise_points': sum(1 for label in self.cluster_labels_ if label == -1),
            'cluster_sizes': {},
            'cluster_success_rates': {},
            'objective_distribution': defaultdict(int)
        }
        
        # Calculate cluster statistics
        for label in set(self.cluster_labels_):
            if label == -1:
                continue
            
            cluster_indices = [i for i, l in enumerate(self.cluster_labels_) if l == label]
            stats['cluster_sizes'][label] = len(cluster_indices)
            
            # Success rate for this cluster
            cluster_successes = [self.success_rates[i] for i in cluster_indices]
            stats['cluster_success_rates'][label] = np.mean(cluster_successes)
        
        # Objective distribution
        for obj in self.objectives:
            stats['objective_distribution'][obj] += 1
        
        return stats
    
    def _update_clusters(self):
        """Update DBSCAN clustering with current memories."""
        if len(self.feature_vectors) < 3:
            return
        
        # Perform clustering
        self.cluster_labels_ = self.clusterer.fit_predict(self.feature_vectors)
        
        # Update cluster centers and objectives
        self.cluster_centers = {}
        self.cluster_objectives = {}
        
        for label in set(self.cluster_labels_):
            if label == -1:  # Noise points
                continue
            
            cluster_indices = [i for i, l in enumerate(self.cluster_labels_) if l == label]
            
            # Calculate cluster center
            cluster_features = [self.feature_vectors[i] for i in cluster_indices]
            self.cluster_centers[label] = np.mean(cluster_features, axis=0)
            
            # Determine dominant objective
            cluster_objectives = [self.objectives[i] for i in cluster_indices]
            most_common_obj = max(set(cluster_objectives), key=cluster_objectives.count)
            self.cluster_objectives[label] = most_common_obj
    
    def _prune_memories(self):
        """Remove least useful memories to maintain size limit."""
        # Keep successful memories and recent memories
        keep_indices = []
        
        # Always keep successful memories
        for i, success in enumerate(self.success_rates):
            if success > 0.5:
                keep_indices.append(i)
        
        # Keep recent memories
        recent_start = max(0, len(self.memories) - self.max_memories // 2)
        for i in range(recent_start, len(self.memories)):
            if i not in keep_indices:
                keep_indices.append(i)
        
        # Randomly keep some older memories for diversity
        remaining_slots = self.max_memories - len(keep_indices)
        if remaining_slots > 0:
            older_indices = [i for i in range(recent_start) if i not in keep_indices]
            if older_indices:
                np.random.shuffle(older_indices)
                keep_indices.extend(older_indices[:remaining_slots])
        
        # Sort indices to maintain order
        keep_indices = sorted(set(keep_indices))
        
        # Prune all lists
        self.memories = [self.memories[i] for i in keep_indices]
        self.feature_vectors = [self.feature_vectors[i] for i in keep_indices]
        self.objectives = [self.objectives[i] for i in keep_indices]
        self.success_rates = [self.success_rates[i] for i in keep_indices]
    
    def save_memory_bank(self, filepath: str):
        """Save memory bank to disk."""
        data = {
            'memories': self.memories,
            'feature_vectors': self.feature_vectors,
            'objectives': self.objectives,
            'success_rates': self.success_rates,
            'cluster_centers': self.cluster_centers,
            'cluster_objectives': self.cluster_objectives
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_memory_bank(self, filepath: str):
        """Load memory bank from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.memories = data['memories']
        self.feature_vectors = data['feature_vectors']
        self.objectives = data['objectives']
        self.success_rates = data['success_rates']
        self.cluster_centers = data.get('cluster_centers', {})
        self.cluster_objectives = data.get('cluster_objectives', {})

class PermanentSolver(nn.Module):
    """Solver with permanent memory and DBSCAN-style problem classification."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 512, max_grid_size: int = 30):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_grid_size = max_grid_size
        
        # Memory system
        self.memory_bank = PermanentMemoryBank(feature_dim=input_dim)
        
        # Core solver network
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Memory-guided adaptation
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Solution generator
        self.solution_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_grid_size * max_grid_size * 10)  # 10 colors
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                problem_features: torch.Tensor,
                input_grid: torch.Tensor,
                target_shape: Tuple[int, int] = None) -> Dict[str, torch.Tensor]:
        """Forward pass with permanent memory guidance."""
        
        batch_size = problem_features.shape[0]
        
        # Encode problem features
        encoded_features = self.feature_encoder(problem_features)
        
        # Retrieve similar problems from memory
        memory_guidance = []
        confidences = []
        
        for i in range(batch_size):
            similar_memories = self.memory_bank.retrieve_similar_problems(
                problem_features[i], input_grid[i]
            )
            
            if similar_memories:
                # Average successful solutions
                successful_solutions = [
                    mem['solution_features'] for mem in similar_memories 
                    if mem['success'] > 0.5
                ]
                
                if successful_solutions:
                    avg_solution = torch.stack(successful_solutions).mean(dim=0)
                    memory_guidance.append(self.feature_encoder(avg_solution))
                    
                    # Weight by similarity and success
                    weights = [mem['similarity'] * mem['success'] for mem in similar_memories[:3]]
                    confidences.append(np.mean(weights))
                else:
                    memory_guidance.append(torch.zeros_like(encoded_features[0]))
                    confidences.append(0.1)
            else:
                memory_guidance.append(torch.zeros_like(encoded_features[0]))
                confidences.append(0.1)
        
        memory_guidance = torch.stack(memory_guidance)
        memory_confidence = torch.tensor(confidences, device=problem_features.device)
        
        # Apply memory attention
        attended_features, attention_weights = self.memory_attention(
            encoded_features.unsqueeze(1),
            memory_guidance.unsqueeze(1),
            memory_guidance.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Combine with original features
        combined_features = torch.cat([encoded_features, attended_features], dim=-1)
        
        # Generate solution
        solution_logits = self.solution_generator(combined_features)
        
        # Estimate confidence
        solution_confidence = self.confidence_estimator(combined_features).squeeze(-1)
        
        # Adjust confidence by memory confidence
        final_confidence = solution_confidence * (0.5 + 0.5 * memory_confidence)
        
        # Reshape solution to grid format
        if target_shape is None:
            target_shape = (self.max_grid_size, self.max_grid_size)
        
        solution_grid = solution_logits.view(
            batch_size, target_shape[0], target_shape[1], 10
        )
        
        return {
            'solution_grid': solution_grid,
            'confidence': final_confidence,
            'memory_guidance': memory_guidance,
            'attention_weights': attention_weights,
            'memory_confidence': memory_confidence
        }
    
    def update_memory(self,
                     problem_features: torch.Tensor,
                     solution_features: torch.Tensor,
                     input_grid: torch.Tensor,
                     output_grid: torch.Tensor,
                     success: bool,
                     metadata: Dict[str, Any] = None):
        """Update permanent memory with new problem-solution pair."""
        
        for i in range(problem_features.shape[0]):
            self.memory_bank.store_memory(
                problem_features[i],
                solution_features[i] if solution_features is not None else problem_features[i],
                input_grid[i],
                output_grid[i],
                success,
                metadata
            )
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory bank statistics."""
        return self.memory_bank.get_cluster_statistics()
    
    def save_solver(self, filepath: str):
        """Save solver state including memory bank."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'memory_bank_data': {
                'memories': self.memory_bank.memories,
                'feature_vectors': self.memory_bank.feature_vectors,
                'objectives': self.memory_bank.objectives,
                'success_rates': self.memory_bank.success_rates
            }
        }, filepath)
    
    def load_solver(self, filepath: str):
        """Load solver state including memory bank."""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore memory bank
        memory_data = checkpoint['memory_bank_data']
        self.memory_bank.memories = memory_data['memories']
        self.memory_bank.feature_vectors = memory_data['feature_vectors']
        self.memory_bank.objectives = memory_data['objectives']
        self.memory_bank.success_rates = memory_data['success_rates']

def test_permanent_solver():
    """Test the permanent solver implementation."""
    
    # Create solver
    solver = PermanentSolver(input_dim=256, hidden_dim=512)
    
    # Generate test data
    batch_size = 4
    problem_features = torch.randn(batch_size, 256)
    input_grids = torch.randint(0, 10, (batch_size, 10, 10))
    target_grids = torch.randint(0, 10, (batch_size, 10, 10))
    
    print("Testing Permanent Solver with DBSCAN Classification...")
    
    # Test forward pass
    results = solver(problem_features, input_grids, target_shape=(10, 10))
    
    print(f"Solution grid shape: {results['solution_grid'].shape}")
    print(f"Confidence shape: {results['confidence'].shape}")
    print(f"Memory confidence: {results['memory_confidence']}")
    
    # Test memory updates
    for i in range(10):
        success = np.random.random() > 0.3
        solver.update_memory(
            problem_features,
            problem_features,  # Using same as solution for test
            input_grids,
            target_grids,
            success,
            {'test_iteration': i}
        )
    
    # Check memory statistics
    stats = solver.get_memory_statistics()
    print(f"\nMemory Statistics:")
    print(f"Total memories: {len(solver.memory_bank.memories)}")
    print(f"Objective distribution: {dict(stats.get('objective_distribution', {}))}")
    
    print("Permanent solver test completed successfully!")

if __name__ == "__main__":
    test_permanent_solver()