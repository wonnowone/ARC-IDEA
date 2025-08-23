"""
Expected Free Energy (EFE) Implementation for ARC Challenge
Based on the episode loss formulation with bi-directional planning and Z-learning anchoring.

Mathematical formulation (Equation 1):
L = Σ[t=1 to T] [λ_risk D_KL(Q→(o_t)||C) + λ_amb E_Q→(s_t)H(P(o_t | s_t))]
    + λ_step T + λ_cons CE(Q→(o_T), δ_o_T*)
    + λ_bi JS(Q→(o_t) || Q←(o_t)) + λ_Z D_KL(σ(c) || Ĉ)

Components:
- EFE term: risk + expected ambiguity (Eq. A)
- step/risk budget: λ_step T
- future-plan consistency: λ_cons CE(Q→(o_T), δ_o_T*)
- bi-directional agreement per step: λ_bi JS(Q→(o_t) || Q←(o_t))
- Z-learning anchoring: λ_Z D_KL(σ(c) || Ĉ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

class EFELoss(nn.Module):
    """
    Expected Free Energy Loss for ARC challenge planning.
    Implements bi-directional planning with preference learning.
    """
    
    def __init__(self, 
                 lambda_risk: float = 1.0,
                 lambda_amb: float = 0.5,
                 lambda_step: float = 0.1,
                 lambda_cons: float = 1.0,
                 lambda_bi: float = 0.5,
                 lambda_z: float = 0.2,
                 lambda_prompt: float = 0.3,
                 max_grid_size: int = 30,
                 num_colors: int = 10):
        """
        Initialize EFE Loss function with variable grid size support.
        
        Args:
            lambda_risk: Weight for risk term (preference matching)
            lambda_amb: Weight for ambiguity reduction
            lambda_step: Weight for step penalty
            lambda_cons: Weight for future-plan consistency
            lambda_bi: Weight for bi-directional agreement
            lambda_z: Weight for Z-learning anchoring
            lambda_prompt: Weight for prompt consistency term
            max_grid_size: Maximum grid size for ARC (for memory allocation)
            num_colors: Number of possible colors in ARC
        """
        super().__init__()
        
        # Loss weights
        self.lambda_risk = lambda_risk
        self.lambda_amb = lambda_amb
        self.lambda_step = lambda_step
        self.lambda_cons = lambda_cons
        self.lambda_bi = lambda_bi
        self.lambda_z = lambda_z
        self.lambda_prompt = lambda_prompt
        
        # ARC-specific parameters
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        
        # Multi-scale preference learning (different sizes have different preferences)
        self.preference_networks = nn.ModuleDict({
            f"{h}x{w}": nn.Linear(h*w*num_colors, h*w*num_colors) 
            for h in range(1, max_grid_size+1) 
            for w in range(1, max_grid_size+1)
        })
        
        # Global preference network for unseen sizes
        self.global_preference = nn.Sequential(
            nn.Linear(max_grid_size*max_grid_size*num_colors, 256),
            nn.ReLU(),
            nn.Linear(256, max_grid_size*max_grid_size*num_colors)
        )
        
        # Target preferences storage (updated from successful episodes)
        self.target_preferences = {}
        
    def forward(self, 
                forward_predictions: torch.Tensor,  # Q→(o_t) - forward predictions
                backward_predictions: torch.Tensor,  # Q←(o_t) - backward predictions
                state_predictions: torch.Tensor,     # Q→(s_t) - state predictions
                observation_probs: torch.Tensor,     # P(o_t|s_t) - observation probabilities
                final_prediction: torch.Tensor,      # Q→(o_T) - final outcome prediction
                target_outcome: torch.Tensor,        # δ_o_T* - target outcome (delta function)
                episode_length: int,
                prompt_embedding: Optional[torch.Tensor] = None,  # Natural language objective embedding
                grid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:  # Mask for variable sizes
        """
        Compute the complete EFE loss according to Equation (1) with prompt integration.
        
        L = Σ[t=1 to T] [λ_risk D_KL(Q→(o_t)||C) + λ_amb E_Q→(s_t)H(P(o_t | s_t))]
            + λ_step T + λ_cons CE(Q→(o_T), δ_o_T*)
            + λ_bi JS(Q→(o_t) || Q←(o_t)) + λ_Z D_KL(σ(c) || Ĉ)
            + λ_prompt L_prompt(prompt, predictions)
        
        Args:
            forward_predictions: [T, H, W, C] - Q→(o_t) forward predicted outcomes
            backward_predictions: [T, H, W, C] - Q←(o_t) backward predicted outcomes
            state_predictions: [T, H, W, C] - Q→(s_t) state predictions
            observation_probs: [T, H, W, C] - P(o_t|s_t) observation probabilities
            final_prediction: [H, W, C] - Q→(o_T) final predicted outcome distribution
            target_outcome: [H, W] - δ_o_T* target outcome (delta function)
            episode_length: T - number of steps
            prompt_embedding: [D] - Natural language objective embedding
            grid_mask: [H, W] - Binary mask for valid grid positions
            
        Returns:
            Dictionary with loss components and total loss
        """
        
        T = episode_length
        losses = {}
        H, W = forward_predictions.shape[1], forward_predictions.shape[2]
        
        # Get current preference distribution C = σ(c) for this grid size
        current_preference = self._get_preference_distribution(H, W, prompt_embedding)  # σ(c)
        
        # Apply grid mask if provided
        if grid_mask is not None:
            forward_predictions = forward_predictions * grid_mask.unsqueeze(0).unsqueeze(-1)
            backward_predictions = backward_predictions * grid_mask.unsqueeze(0).unsqueeze(-1)
            current_preference = current_preference * grid_mask.unsqueeze(-1)
        
        # 1. EFE Term: risk + expected ambiguity (Eq. A)
        risk_loss = self._compute_risk_loss(forward_predictions, current_preference, grid_mask)  # D_KL(Q→(o_t)||C)
        ambiguity_loss = self._compute_ambiguity_loss(state_predictions, observation_probs, grid_mask)  # E_Q→(s_t)H(P(o_t|s_t))
        efe_term = self.lambda_risk * risk_loss + self.lambda_amb * ambiguity_loss
        
        losses['risk'] = risk_loss
        losses['ambiguity'] = ambiguity_loss
        losses['efe'] = efe_term
        
        # 2. step/risk budget: λ_step T (scaled by grid size for fairness)
        grid_scale = (H * W) / (self.max_grid_size * self.max_grid_size)
        step_penalty = self.lambda_step * T * grid_scale
        losses['step_penalty'] = step_penalty
        
        # 3. future-plan consistency: λ_cons CE(Q→(o_T), δ_o_T*)
        consistency_loss = self._compute_consistency_loss(final_prediction, target_outcome, grid_mask)
        losses['consistency'] = self.lambda_cons * consistency_loss
        
        # 4. bi-directional agreement per step: λ_bi JS(Q→(o_t) || Q←(o_t))
        bidirectional_loss = self._compute_bidirectional_loss(forward_predictions, backward_predictions, grid_mask)
        losses['bidirectional'] = self.lambda_bi * bidirectional_loss
        
        # 5. Z-learning anchoring: λ_Z D_KL(σ(c) || Ĉ)
        z_anchoring_loss = self._compute_z_anchoring_loss(current_preference, H, W)
        losses['z_anchoring'] = self.lambda_z * z_anchoring_loss
        
        # 6. Prompt consistency: λ_prompt L_prompt(prompt, predictions)
        prompt_loss = torch.tensor(0.0, device=forward_predictions.device)
        if prompt_embedding is not None:
            prompt_loss = self._compute_prompt_consistency_loss(forward_predictions, prompt_embedding)
        losses['prompt_consistency'] = self.lambda_prompt * prompt_loss
        
        # Total Loss (Extended Equation 1)
        total_loss = (efe_term + step_penalty + losses['consistency'] + 
                     losses['bidirectional'] + losses['z_anchoring'] + losses['prompt_consistency'])
        losses['total'] = total_loss
        
        return losses
    
    def _get_preference_distribution(self, H: int, W: int, prompt_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get preference distribution C = σ(c) for specific grid size with optional prompt guidance.
        """
        grid_key = f"{H}x{W}"
        
        if grid_key in self.preference_networks:
            # Use size-specific network
            dummy_input = torch.randn(H*W*self.num_colors, device=next(self.parameters()).device)
            pref_logits = self.preference_networks[grid_key](dummy_input)
            pref_logits = pref_logits.view(H, W, self.num_colors)
        else:
            # Use global network with padding/cropping
            dummy_input = torch.randn(self.max_grid_size*self.max_grid_size*self.num_colors, 
                                    device=next(self.parameters()).device)
            pref_logits = self.global_preference(dummy_input)
            pref_logits = pref_logits.view(self.max_grid_size, self.max_grid_size, self.num_colors)
            pref_logits = pref_logits[:H, :W, :]  # Crop to actual size
        
        # Apply prompt guidance if available
        if prompt_embedding is not None:
            # Simple approach: use prompt to modulate preferences
            prompt_influence = torch.sigmoid(prompt_embedding.mean())  # Scale factor from prompt
            pref_logits = pref_logits * prompt_influence
        
        return F.softmax(pref_logits, dim=-1)
    
    def _compute_prompt_consistency_loss(self, predictions: torch.Tensor, prompt_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute prompt consistency loss: ensures predictions align with natural language objective.
        """
        # Simple implementation: cosine similarity between prediction features and prompt
        pred_features = predictions.mean(dim=[1, 2])  # [T, C] - average over spatial dimensions
        prompt_features = prompt_embedding.unsqueeze(0).expand(pred_features.shape[0], -1)  # [T, D]
        
        # Project to same dimensionality if needed
        if pred_features.shape[-1] != prompt_features.shape[-1]:
            if not hasattr(self, 'prompt_projector'):
                self.prompt_projector = nn.Linear(prompt_features.shape[-1], pred_features.shape[-1])
                self.prompt_projector = self.prompt_projector.to(pred_features.device)
            prompt_features = self.prompt_projector(prompt_features)
        
        # Compute cosine similarity loss (maximize similarity = minimize negative similarity)
        similarity = F.cosine_similarity(pred_features, prompt_features, dim=-1)
        return -similarity.mean()  # Negative because we want to maximize similarity
    
    def _compute_risk_loss(self, predictions: torch.Tensor, preferences: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute risk loss: Σ_t D_KL(Q→(o_t) || C)
        Measures how well forward predictions match learned preferences.
        """
        # Ensure predictions are probabilities
        pred_probs = F.softmax(predictions, dim=-1)
        
        # Compute KL divergence for each timestep
        kl_divs = []
        for t in range(predictions.shape[0]):
            if mask is not None:
                # Apply mask and normalize
                masked_pred = pred_probs[t] * mask.unsqueeze(-1)
                masked_pref = preferences * mask.unsqueeze(-1)
                # Renormalize
                masked_pred = masked_pred / (masked_pred.sum(dim=-1, keepdim=True) + 1e-8)
                masked_pref = masked_pref / (masked_pref.sum(dim=-1, keepdim=True) + 1e-8)
                
                kl_div = F.kl_div(masked_pref.log(), masked_pred, reduction='batchmean')
            else:
                kl_div = F.kl_div(preferences.log(), pred_probs[t], reduction='batchmean')
            
            kl_divs.append(kl_div)
        
        return torch.stack(kl_divs).sum()
    
    def _compute_ambiguity_loss(self, states: torch.Tensor, obs_probs: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute expected ambiguity: Σ_t E_Q→(s_t)[H(P(o_t|s_t))]
        Reduces uncertainty in observation predictions given states.
        """
        # Compute entropy of observation probabilities
        epsilon = 1e-8
        obs_probs_safe = torch.clamp(obs_probs, epsilon, 1.0)
        entropy = -torch.sum(obs_probs_safe * torch.log(obs_probs_safe), dim=-1)
        
        # Weight by state probabilities and sum over time
        state_probs = F.softmax(states, dim=-1)
        expected_entropy = torch.sum(state_probs * entropy.unsqueeze(-1), dim=-1)
        
        return expected_entropy.sum()
    
    def _compute_consistency_loss(self, final_pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute future-plan consistency: CE(Q→(o_T), δ_o_T*)
        Ensures final prediction matches target outcome (delta function).
        """
        if target.dtype == torch.long:
            # Target is indices
            return F.cross_entropy(final_pred.view(-1, final_pred.shape[-1]), 
                                 target.view(-1))
        else:
            # Target is one-hot or probabilities
            final_probs = F.log_softmax(final_pred, dim=-1)
            return F.kl_div(final_probs, target, reduction='batchmean')
    
    def _compute_bidirectional_loss(self, forward_pred: torch.Tensor, backward_pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute bi-directional agreement per step: Σ_t JS(Q→(o_t) || Q←(o_t))
        Ensures forward and backward predictions agree at each timestep.
        """
        # Convert to probabilities
        forward_probs = F.softmax(forward_pred, dim=-1)
        backward_probs = F.softmax(backward_pred, dim=-1)
        
        # Compute Jensen-Shannon divergence
        js_divs = []
        for t in range(forward_pred.shape[0]):
            # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q)
            m = 0.5 * (forward_probs[t] + backward_probs[t])
            
            kl1 = F.kl_div(m.log(), forward_probs[t], reduction='batchmean')
            kl2 = F.kl_div(m.log(), backward_probs[t], reduction='batchmean')
            
            js_div = 0.5 * (kl1 + kl2)
            js_divs.append(js_div)
        
        return torch.stack(js_divs).sum()
    
    def _compute_z_anchoring_loss(self, current_pref: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Compute Z-learning anchoring: D_KL(σ(c) || Ĉ)
        Keeps learned preferences σ(c) close to target preferences Ĉ.
        """
        grid_key = f"{H}x{W}"
        if grid_key in self.target_preferences:
            target_pref = self.target_preferences[grid_key]
            return F.kl_div(current_pref.log(), target_pref, reduction='batchmean')
        else:
            # No target preference for this size yet, return zero loss
            return torch.tensor(0.0, device=current_pref.device)
    
    def update_target_preference(self, successful_outcomes: List[torch.Tensor], grid_sizes: List[Tuple[int, int]], smoothing: float = 0.1):
        """
        Update target preference based on successful episode outcomes for different grid sizes.
        
        Args:
            successful_outcomes: List of successful outcome grids
            grid_sizes: List of (H, W) tuples corresponding to each outcome
            smoothing: Smoothing factor for exponential moving average
        """
        if not successful_outcomes or len(successful_outcomes) != len(grid_sizes):
            return
        
        # Group outcomes by grid size
        size_groups = {}
        for outcome, (H, W) in zip(successful_outcomes, grid_sizes):
            grid_key = f"{H}x{W}"
            if grid_key not in size_groups:
                size_groups[grid_key] = []
            size_groups[grid_key].append(outcome)
        
        # Update target preferences for each grid size
        for grid_key, outcomes in size_groups.items():
            H, W = map(int, grid_key.split('x'))
            
            # Convert outcomes to preference distribution
            outcome_counts = torch.zeros(H, W, self.num_colors, device=outcomes[0].device)
            
            for outcome in outcomes:
                # Convert grid to one-hot and accumulate
                if outcome.dtype == torch.long:
                    one_hot = F.one_hot(outcome, num_classes=self.num_colors).float()
                    outcome_counts += one_hot
            
            # Normalize to probability distribution
            new_preference = outcome_counts / (len(outcomes) + 1e-8)
            
            # Exponential moving average update
            if grid_key in self.target_preferences:
                self.target_preferences[grid_key] = (1 - smoothing) * self.target_preferences[grid_key] + smoothing * new_preference
            else:
                self.target_preferences[grid_key] = new_preference


class ARCPromptGuidedAgent(nn.Module):
    """
    ARC Agent using Expected Free Energy with prompt-guided planning and learning.
    Integrates natural language objectives with bi-directional reasoning.
    """
    
    def __init__(self, 
                 max_grid_size: int = 30, 
                 num_colors: int = 10, 
                 hidden_dim: int = 256,
                 prompt_dim: int = 768,  # e.g., BERT embedding dimension
                 num_reasoning_steps: int = 5):
        super().__init__()
        
        self.max_grid_size = max_grid_size
        self.num_colors = num_colors
        self.hidden_dim = hidden_dim
        self.prompt_dim = prompt_dim
        self.num_reasoning_steps = num_reasoning_steps
        
        # Prompt encoder for natural language objectives
        self.prompt_encoder = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Grid transformer for spatial reasoning
        self.grid_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        
        # Forward planning model Q→(o_t): predicts next state given current state and prompt
        self.forward_model = nn.ModuleDict({
            'grid_encoder': nn.Sequential(
                nn.Conv2d(num_colors, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU()
            ),
            'prompt_fusion': nn.MultiheadAttention(hidden_dim, 8, batch_first=True),
            'predictor': nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, num_colors, 1)
            )
        })
        
        # Backward planning model Q←(o_t): predicts previous state given current state and prompt
        self.backward_model = nn.ModuleDict({
            'grid_encoder': nn.Sequential(
                nn.Conv2d(num_colors, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU()
            ),
            'prompt_fusion': nn.MultiheadAttention(hidden_dim, 8, batch_first=True),
            'predictor': nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_dim, num_colors, 1)
            )
        })
        
        # Self-critique module for step-by-step validation
        self.critique_module = nn.Sequential(
            nn.Linear(hidden_dim + prompt_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # [why, what, how] scores
            nn.Softmax(dim=-1)
        )
        
        # EFE Loss function with prompt support
        self.efe_loss = EFELoss(
            max_grid_size=max_grid_size, 
            num_colors=num_colors,
            lambda_prompt=0.3
        )
        
    def forward_planning(self, 
                        initial_state: torch.Tensor, 
                        prompt_embedding: torch.Tensor,
                        num_steps: int,
                        grid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generate forward predictions Q→(o_t) with prompt guidance and self-critique.
        
        Args:
            initial_state: [H, W] - Initial grid state
            prompt_embedding: [D] - Natural language objective embedding
            num_steps: Number of planning steps
            grid_mask: [H, W] - Binary mask for valid positions
            
        Returns:
            predictions: [T, H, W, C] - Forward predictions
            critique_scores: List[T] of [3] - Self-critique scores for each step
        """
        H, W = initial_state.shape
        
        # Convert to one-hot and add batch dimension
        current_state = F.one_hot(initial_state, num_classes=self.num_colors).float()
        current_state = current_state.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Encode prompt
        encoded_prompt = self.prompt_encoder(prompt_embedding)  # [hidden_dim]
        
        predictions = []
        critique_scores = []
        
        for step in range(num_steps):
            # Encode current grid state
            grid_features = self.forward_model['grid_encoder'](current_state)  # [1, hidden_dim, H, W]
            
            # Reshape for attention: [1, H*W, hidden_dim]
            grid_flat = grid_features.flatten(2).transpose(1, 2)
            
            # Apply prompt guidance through attention
            prompt_expanded = encoded_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            attended_features, _ = self.forward_model['prompt_fusion'](
                grid_flat, prompt_expanded, prompt_expanded
            )
            
            # Reshape back to spatial: [1, hidden_dim, H, W]
            attended_features = attended_features.transpose(1, 2).view(1, self.hidden_dim, H, W)
            
            # Predict next state
            next_state_logits = self.forward_model['predictor'](attended_features)
            
            # Apply mask if provided
            if grid_mask is not None:
                next_state_logits = next_state_logits * grid_mask.unsqueeze(0).unsqueeze(0)
            
            predictions.append(next_state_logits.squeeze(0).permute(1, 2, 0))  # [H, W, C]
            
            # Self-critique: why, what, how does this step relate to objective?
            step_features = attended_features.mean(dim=[2, 3]).squeeze(0)  # [hidden_dim]
            critique_input = torch.cat([step_features, prompt_embedding], dim=0)
            critique_score = self.critique_module(critique_input)  # [3]: [why, what, how]
            critique_scores.append(critique_score)
            
            # Update current state (using predicted probabilities)
            current_state = F.softmax(next_state_logits, dim=1)
        
        return torch.stack(predictions), critique_scores  # [T, H, W, C], List[T] of [3]
    
    def backward_planning(self, 
                         target_state: torch.Tensor, 
                         prompt_embedding: torch.Tensor,
                         num_steps: int,
                         grid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate backward predictions Q←(o_t) with prompt guidance for reverse thinking validation.
        
        Args:
            target_state: [H, W] - Target grid state
            prompt_embedding: [D] - Natural language objective embedding
            num_steps: Number of planning steps
            grid_mask: [H, W] - Binary mask for valid positions
            
        Returns:
            predictions: [T, H, W, C] - Backward predictions (reversed order)
        """
        H, W = target_state.shape
        
        # Convert to one-hot and add batch dimension
        current_state = F.one_hot(target_state, num_classes=self.num_colors).float()
        current_state = current_state.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        
        # Encode prompt
        encoded_prompt = self.prompt_encoder(prompt_embedding)  # [hidden_dim]
        
        predictions = []
        for _ in range(num_steps):
            # Encode current grid state
            grid_features = self.backward_model['grid_encoder'](current_state)  # [1, hidden_dim, H, W]
            
            # Reshape for attention: [1, H*W, hidden_dim]
            grid_flat = grid_features.flatten(2).transpose(1, 2)
            
            # Apply prompt guidance through attention
            prompt_expanded = encoded_prompt.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
            attended_features, _ = self.backward_model['prompt_fusion'](
                grid_flat, prompt_expanded, prompt_expanded
            )
            
            # Reshape back to spatial: [1, hidden_dim, H, W]
            attended_features = attended_features.transpose(1, 2).view(1, self.hidden_dim, H, W)
            
            # Predict previous state
            prev_state_logits = self.backward_model['predictor'](attended_features)
            
            # Apply mask if provided
            if grid_mask is not None:
                prev_state_logits = prev_state_logits * grid_mask.unsqueeze(0).unsqueeze(0)
            
            predictions.append(prev_state_logits.squeeze(0).permute(1, 2, 0))  # [H, W, C]
            
            # Update current state
            current_state = F.softmax(prev_state_logits, dim=1)
        
        # Reverse to match forward planning order
        return torch.stack(predictions[::-1])  # [T, H, W, C]
    
    def train_episode(self, 
                     initial_state: torch.Tensor, 
                     target_state: torch.Tensor,
                     prompt_text: str,
                     prompt_embedding: torch.Tensor,
                     num_steps: int,
                     grid_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Train on a single episode using EFE loss with prompt guidance and self-critique.
        
        Args:
            initial_state: [H, W] - Initial grid
            target_state: [H, W] - Target grid
            prompt_text: Natural language objective (for logging)
            prompt_embedding: [D] - Encoded natural language objective
            num_steps: Number of planning steps
            grid_mask: [H, W] - Binary mask for valid positions
            
        Returns:
            Dictionary of losses and critique information
        """
        # Generate forward and backward plans with prompt guidance
        forward_preds, critique_scores = self.forward_planning(
            initial_state, prompt_embedding, num_steps, grid_mask
        )
        backward_preds = self.backward_planning(
            target_state, prompt_embedding, num_steps, grid_mask
        )
        
        # Use forward predictions as state predictions
        state_preds = forward_preds
        
        # Observation probabilities (could be learned separately)
        obs_probs = F.softmax(forward_preds, dim=-1)
        
        # Final prediction is the last forward prediction
        final_pred = forward_preds[-1]
        
        # Compute EFE loss with prompt consistency
        losses = self.efe_loss(
            forward_predictions=forward_preds,
            backward_predictions=backward_preds,
            state_predictions=state_preds,
            observation_probs=obs_probs,
            final_prediction=final_pred,
            target_outcome=target_state,
            episode_length=num_steps,
            prompt_embedding=prompt_embedding,
            grid_mask=grid_mask
        )
        
        # Add self-critique analysis
        critique_analysis = self._analyze_critique_scores(critique_scores, prompt_text)
        losses.update(critique_analysis)
        
        return losses
    
    def _analyze_critique_scores(self, critique_scores: List[torch.Tensor], prompt_text: str) -> Dict[str, torch.Tensor]:
        """
        Analyze self-critique scores for interpretability.
        
        Args:
            critique_scores: List[T] of [3] - [why, what, how] scores for each step
            prompt_text: Natural language objective for context
            
        Returns:
            Dictionary with critique analysis
        """
        if not critique_scores:
            return {}
        
        # Stack scores: [T, 3]
        scores_tensor = torch.stack(critique_scores)
        
        # Compute average critique scores
        avg_why = scores_tensor[:, 0].mean()
        avg_what = scores_tensor[:, 1].mean()
        avg_how = scores_tensor[:, 2].mean()
        
        # Compute critique consistency (how stable are scores across steps)
        critique_std = scores_tensor.std(dim=0).mean()
        
        return {
            'critique_why': avg_why,
            'critique_what': avg_what,
            'critique_how': avg_how,
            'critique_consistency': critique_std,
            'prompt_text': prompt_text  # For logging purposes
        }


def create_sample_training_data():
    """Create sample ARC-like training data for testing."""
    # Simple pattern: copy input to output
    inputs = []
    outputs = []
    
    for _ in range(10):
        # Create random 3x3 grid
        grid = torch.randint(0, 3, (3, 3))
        inputs.append(grid)
        outputs.append(grid.clone())  # Simple copy task
    
    return inputs, outputs


def test_efe_implementation():
    """Test the EFE implementation with sample data."""
    print("Testing EFE Implementation for ARC with Prompting...")
    
    # Create agent
    agent = ARCPromptGuidedAgent(max_grid_size=10, num_colors=3, hidden_dim=64, prompt_dim=256)
    optimizer = torch.optim.Adam(agent.parameters(), lr=0.001)
    
    # Create sample data
    inputs, outputs = create_sample_training_data()
    
    # Create sample prompt embeddings
    sample_prompts = [
        "Copy the input grid exactly as shown",
        "Replicate the pattern from input to output", 
        "Transform input by maintaining the same structure"
    ]
    
    # Simple prompt embedding (in practice, use BERT/similar)
    def create_prompt_embedding(text):
        # Simple hash-based embedding for testing
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        torch.manual_seed(hash_val % 1000)
        return torch.randn(256)
    
    # Training loop
    for epoch in range(5):
        total_loss = 0
        
        for i, (input_grid, target_grid) in enumerate(zip(inputs, outputs)):
            optimizer.zero_grad()
            
            # Get prompt for this example
            prompt_text = sample_prompts[i % len(sample_prompts)]
            prompt_embedding = create_prompt_embedding(prompt_text)
            
            # Train on episode
            losses = agent.train_episode(
                input_grid, 
                target_grid, 
                prompt_text,
                prompt_embedding,
                num_steps=3
            )
            
            # Backward pass
            losses['total'].backward()
            optimizer.step()
            
            total_loss += losses['total'].item()
            
            if i == 0:  # Print first example details
                print(f"Epoch {epoch}, Example {i}:")
                print(f"  Prompt: {prompt_text}")
                for key, value in losses.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        print(f"  {key}: {value.item():.4f}")
                    elif key == 'prompt_text':
                        continue  # Skip string values
        
        print(f"Epoch {epoch}, Average Loss: {total_loss/len(inputs):.4f}")
        print()
    
    print("EFE Implementation with Prompting test completed!")


if __name__ == "__main__":
    test_efe_implementation()