"""
Mistral Training Sequence for ARC Challenge
Integrates:
- extracted_features.csv
- exp_prompt (natural language objectives)  
- exp_EFE (Expected Free Energy loss)
- system1_solver and system2_solver
- exp_tta.py (test-time adaptation)
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
import pickle
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import time
import os

# Import our components
from exp_prompt import create_arc_prompt_template
from exp_EFE import ARCPromptGuidedAgent, EFELoss
from exp_tta import TestTimeAdaptationSystem, create_test_adaptation_system

# Import solvers
import sys
sys.path.append('..')
from system1_solver import System1Solver
from system2_solver import System2Solver

@dataclass
class TrainingConfig:
    """Configuration for the training sequence"""
    model_name: str = "mistralai/Mistral-7B-v0.1"  # Mistral model
    max_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    efe_weight: float = 1.0
    solver_weight: float = 0.5
    tta_weight: float = 0.3
    max_sequence_length: int = 2048
    warmup_steps: int = 100
    gradient_clip_norm: float = 1.0
    save_every_n_epochs: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ARCDataset(Dataset):
    """Dataset for ARC problems with features and prompts"""
    
    def __init__(self, 
                 training_data: Dict,
                 features_df: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 max_length: int = 2048):
        
        self.training_data = training_data
        self.features_df = features_df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Prepare problem data
        self.problems = []
        self._prepare_problems()
        
    def _prepare_problems(self):
        """Prepare problems with features and prompts"""
        
        for prob_id, prob_data in self.training_data.items():
            # Get features for this problem
            prob_features = self.features_df[self.features_df['prob_id'] == prob_id]
            
            if len(prob_features) == 0:
                continue
                
            # Create natural language objective from features
            objective = self._create_objective_from_features(prob_features)
            
            # Process train examples
            train_examples = prob_data.get('train', [])
            for i, example in enumerate(train_examples):
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])
                
                problem_entry = {
                    'prob_id': prob_id,
                    'example_id': i,
                    'input_grid': input_grid,
                    'output_grid': output_grid,
                    'objective': objective,
                    'features': prob_features.to_dict('records')
                }
                
                self.problems.append(problem_entry)
    
    def _create_objective_from_features(self, features: pd.DataFrame) -> str:
        """Create natural language objective from extracted features"""
        
        # Analyze features to generate objective
        backgrounds = features['background'].unique()
        colors = features['color'].unique()
        shapes = features['shapes'].dropna().unique()
        
        # Generate objective based on patterns
        objectives = []
        
        if 'no' in backgrounds and len(backgrounds) == 1:
            objectives.append("Transform the pattern without background elements")
        elif 'yes' in backgrounds:
            objectives.append("Work with the pattern considering background elements")
            
        if len(colors) <= 3:
            objectives.append(f"Focus on color relationships with {len(colors)} main colors")
        elif len(colors) > 5:
            objectives.append("Handle complex multi-color pattern transformation")
            
        # Check for geometric patterns
        has_lines = any(features[['horizontal_lines', 'vertical_lines', 'diagonal_lines']].notna().any())
        if has_lines:
            objectives.append("Consider line patterns and geometric relationships")
            
        if len(shapes) > 0:
            objectives.append("Transform the spatial arrangement of shapes")
            
        # Default objective
        if not objectives:
            objectives.append("Analyze input-output relationships and apply the transformation rule")
            
        return ". ".join(objectives)
    
    def __len__(self):
        return len(self.problems)
    
    def __getitem__(self, idx):
        problem = self.problems[idx]
        
        # Create prompt with grid data
        prompt = self._create_prompt(problem)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'input_grid': torch.tensor(problem['input_grid'], dtype=torch.float32),
            'output_grid': torch.tensor(problem['output_grid'], dtype=torch.long),
            'objective': problem['objective'],
            'prob_id': problem['prob_id'],
            'features': problem['features']
        }
    
    def _create_prompt(self, problem: Dict) -> str:
        """Create a structured prompt for the problem"""
        
        input_grid = problem['input_grid']
        output_grid = problem['output_grid']
        objective = problem['objective']
        
        prompt = f"""
# ARC Challenge Problem Analysis

## Objective
{objective}

## Input Grid ({input_grid.shape[0]}x{input_grid.shape[1]})
{self._grid_to_text(input_grid)}

## Output Grid ({output_grid.shape[0]}x{output_grid.shape[1]})  
{self._grid_to_text(output_grid)}

## Analysis
Observe the transformation from input to output and identify the underlying rule.

## Rule Identification
Based on the input-output pair, the transformation rule is:
"""
        return prompt
    
    def _grid_to_text(self, grid: np.ndarray) -> str:
        """Convert grid to text representation"""
        lines = []
        for row in grid:
            lines.append(' '.join(str(int(cell)) for cell in row))
        return '\n'.join(lines)

class MistralARCTrainer:
    """Main trainer for Mistral with ARC components"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._setup_model_and_tokenizer()
        self._setup_arc_components()
        
    def _setup_model_and_tokenizer(self):
        """Setup Mistral model and tokenizer"""
        
        self.logger.info(f"Loading Mistral model: {self.config.model_name}")
        
        # For this implementation, we'll use a lighter model for demonstration
        # In practice, you'd use the full Mistral model
        model_name = "microsoft/DialoGPT-medium"  # Lighter alternative for testing
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Add ARC-specific head
        hidden_size = self.base_model.config.hidden_size
        self.arc_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 10 * 30 * 30),  # Max 30x30 grid with 10 colors
            nn.Reshape(-1, 10, 30, 30)
        ).to(self.device)
        
    def _setup_arc_components(self):
        """Setup ARC-specific components"""
        
        # EFE-guided agent
        self.arc_agent = ARCPromptGuidedAgent(
            max_grid_size=30,
            num_colors=10,
            hidden_dim=256,
            prompt_dim=self.base_model.config.hidden_size,
            num_reasoning_steps=5
        ).to(self.device)
        
        # Test-time adaptation system
        self.tta_system = create_test_adaptation_system().to(self.device)
        
        # Solvers
        self.system1_solver = System1Solver(debug=True)
        self.system2_solver = System2Solver(debug=True)
        
        # Feature scaler for numerical features
        self.feature_scaler = StandardScaler()
        
    def load_data(self):
        """Load training and test data"""
        
        # Load split data
        with open('training_train.json', 'r') as f:
            train_data = json.load(f)
            
        with open('training_test.json', 'r') as f:
            test_data = json.load(f)
            
        # Load features
        features_df = pd.read_csv('extracted_features.csv')
        
        self.logger.info(f"Loaded {len(train_data)} training problems")
        self.logger.info(f"Loaded {len(test_data)} test problems")
        self.logger.info(f"Loaded {len(features_df)} feature records")
        
        # Create datasets
        train_dataset = ARCDataset(train_data, features_df, self.tokenizer, self.config.max_sequence_length)
        test_dataset = ARCDataset(test_data, features_df, self.tokenizer, self.config.max_sequence_length)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.logger.info(f"Created {len(self.train_loader)} training batches")
        self.logger.info(f"Created {len(self.test_loader)} test batches")
        
    def setup_training(self):
        """Setup optimizers and schedulers"""
        
        # Combine all parameters
        all_params = list(self.base_model.parameters()) + \
                    list(self.arc_head.parameters()) + \
                    list(self.arc_agent.parameters()) + \
                    list(self.tta_system.parameters())
                    
        self.optimizer = optim.AdamW(
            all_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs * len(self.train_loader)
        )
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        
        self.base_model.train()
        self.arc_head.train()
        self.arc_agent.train()
        self.tta_system.train()
        
        epoch_losses = {
            'total': 0.0,
            'efe': 0.0,
            'solver': 0.0,
            'tta': 0.0,
            'consistency': 0.0
        }
        
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            input_grids = batch['input_grid'].to(self.device)
            output_grids = batch['output_grid'].to(self.device)
            
            try:
                # Forward pass through Mistral
                model_outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get prompt embeddings
                prompt_embeddings = model_outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
                
                # Generate grid predictions
                grid_logits = self.arc_head(prompt_embeddings)  # [B, colors, H, W]
                
                losses = {}
                
                # 1. EFE Loss - train the agent with prompt guidance
                for i in range(input_grids.shape[0]):
                    if input_grids[i].sum() > 0:  # Skip empty grids
                        input_grid = input_grids[i].cpu().numpy().astype(int)
                        target_grid = output_grids[i].cpu().numpy().astype(int)
                        prompt_emb = prompt_embeddings[i]
                        
                        # Resize grids to match input dimensions
                        h, w = input_grid.shape
                        if h <= 30 and w <= 30:
                            agent_losses = self.arc_agent.train_episode(
                                torch.tensor(input_grid, dtype=torch.long),
                                torch.tensor(target_grid, dtype=torch.long),
                                batch['objective'][i],
                                prompt_emb,
                                num_steps=3
                            )
                            
                            if 'total' in agent_losses:
                                losses[f'efe_{i}'] = agent_losses['total']
                
                # 2. Solver Integration Loss
                solver_loss = torch.tensor(0.0, device=self.device)
                consistency_loss = torch.tensor(0.0, device=self.device)
                
                for i in range(input_grids.shape[0]):
                    if input_grids[i].sum() > 0:
                        input_grid = input_grids[i].cpu().numpy().astype(int)
                        
                        # Get solver predictions (simplified for training)
                        try:
                            # System 1 attempt
                            s1_output, s1_conf, s1_method = self.system1_solver.solve(
                                input_grid, "CNN", {}
                            )
                            
                            if s1_output is not None:
                                s1_tensor = torch.tensor(s1_output, dtype=torch.float32, device=self.device)
                                target_tensor = output_grids[i].float()
                                
                                # Resize for comparison
                                if s1_tensor.shape == target_tensor.shape:
                                    solver_loss += nn.MSELoss()(s1_tensor, target_tensor)
                                    
                        except Exception as e:
                            self.logger.warning(f"Solver error: {e}")
                            continue
                
                # 3. TTA Loss (simplified)
                tta_loss = torch.tensor(0.0, device=self.device)
                
                # Combine losses
                total_loss = torch.tensor(0.0, device=self.device)
                
                # Add EFE losses
                for key, loss in losses.items():
                    if 'efe' in key and isinstance(loss, torch.Tensor):
                        total_loss += self.config.efe_weight * loss
                        epoch_losses['efe'] += loss.item()
                
                # Add solver loss
                if solver_loss.item() > 0:
                    total_loss += self.config.solver_weight * solver_loss
                    epoch_losses['solver'] += solver_loss.item()
                
                # Add TTA loss
                if tta_loss.item() > 0:
                    total_loss += self.config.tta_weight * tta_loss
                    epoch_losses['tta'] += tta_loss.item()
                
                # Consistency loss between grid predictions and targets
                for i in range(min(grid_logits.shape[0], output_grids.shape[0])):
                    h, w = output_grids[i].shape
                    if h <= 30 and w <= 30:
                        pred_slice = grid_logits[i, :, :h, :w]  # [colors, h, w]
                        target = output_grids[i]  # [h, w]
                        
                        consistency_loss += nn.CrossEntropyLoss()(
                            pred_slice.view(10, -1).transpose(0, 1),  # [h*w, colors]
                            target.view(-1)  # [h*w]
                        )
                
                if consistency_loss.item() > 0:
                    total_loss += consistency_loss
                    epoch_losses['consistency'] += consistency_loss.item()
                
                if total_loss.item() > 0:
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in all_params if p.requires_grad], 
                        self.config.gradient_clip_norm
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    
                    epoch_losses['total'] += total_loss.item()
                
            except Exception as e:
                self.logger.error(f"Batch {batch_idx} error: {e}")
                continue
                
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {total_loss.item():.4f}"
                )
        
        # Average losses
        for key in epoch_losses:
            if num_batches > 0:
                epoch_losses[key] /= num_batches
                
        return epoch_losses
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on test set"""
        
        self.base_model.eval()
        self.arc_head.eval()
        self.arc_agent.eval()
        self.tta_system.eval()
        
        eval_losses = {'total': 0.0, 'accuracy': 0.0}
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                input_grids = batch['input_grid'].to(self.device)
                output_grids = batch['output_grid'].to(self.device)
                
                try:
                    # Forward pass
                    model_outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    prompt_embeddings = model_outputs.last_hidden_state.mean(dim=1)
                    grid_logits = self.arc_head(prompt_embeddings)
                    
                    # Simple accuracy calculation
                    for i in range(min(grid_logits.shape[0], output_grids.shape[0])):
                        h, w = output_grids[i].shape
                        if h <= 30 and w <= 30:
                            pred_slice = grid_logits[i, :, :h, :w]
                            pred_grid = pred_slice.argmax(dim=0)
                            
                            correct_predictions += (pred_grid == output_grids[i]).float().mean().item()
                            total_predictions += 1
                            
                except Exception as e:
                    self.logger.warning(f"Evaluation error: {e}")
                    continue
        
        if total_predictions > 0:
            eval_losses['accuracy'] = correct_predictions / total_predictions
            
        return eval_losses
    
    def save_model(self, epoch: int, losses: Dict[str, float]):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'base_model_state_dict': self.base_model.state_dict(),
            'arc_head_state_dict': self.arc_head.state_dict(),
            'arc_agent_state_dict': self.arc_agent.state_dict(),
            'tta_system_state_dict': self.tta_system.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': losses,
            'config': self.config
        }
        
        checkpoint_path = f'mistral_arc_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        
        self.logger.info("Starting Mistral ARC training sequence...")
        
        # Load data
        self.load_data()
        
        # Setup training
        self.setup_training()
        
        best_accuracy = 0.0
        
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            
            # Train
            train_losses = self.train_epoch(epoch)
            
            # Evaluate
            eval_losses = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Eval Accuracy: {eval_losses['accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if eval_losses['accuracy'] > best_accuracy:
                best_accuracy = eval_losses['accuracy']
                self.save_model(epoch, {'train': train_losses, 'eval': eval_losses})
            
            # Save periodic checkpoints
            if epoch % self.config.save_every_n_epochs == 0:
                self.save_model(epoch, {'train': train_losses, 'eval': eval_losses})
        
        self.logger.info(f"Training completed! Best accuracy: {best_accuracy:.4f}")

def main():
    """Main training function"""
    
    # Set working directory to experiment folder
    os.chdir(Path(__file__).parent)
    
    # Create config
    config = TrainingConfig(
        max_epochs=5,  # Reduced for testing
        batch_size=2,  # Smaller batch size
        learning_rate=1e-4,
        device="cpu"  # Use CPU for testing
    )
    
    # Create trainer
    trainer = MistralARCTrainer(config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()