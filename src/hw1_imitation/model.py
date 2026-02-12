"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""






#######################################################################################
class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
           
        # Build MLP layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        output_dim = chunk_size * action_dim
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        # Predict action chunk
        batch_size = state.shape[0]
        pred_flat = self.network(state) 
        # Reshape prediction to match action_chunk shape
        pred_chunk = pred_flat.reshape(batch_size, self.chunk_size, self.action_dim)
        # MSE loss
        loss = nn.functional.mse_loss(pred_chunk, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        
        # Predict action chunk
        batch_size = state.shape[0]
        pred_flat = self.network(state)  
        # Reshape 
        action_chunk = pred_flat.reshape(batch_size, self.chunk_size, self.action_dim)
        return action_chunk
#######################################################################################


#######################################################################################

class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        
        # Build MLP layers
        layers = []        
        input_dim = state_dim + (chunk_size * action_dim) + 1  # +1 for tau
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        output_dim = chunk_size * action_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        
        batch_size = state.shape[0]
        device = state.device
        
        # Nosie Sampling
        noise = torch.randn_like(action_chunk) 
        # Sample flow matching Tau!
        tau = torch.rand(batch_size, 1, device=device)  
        
        # Interpolatation
        tau_expanded = tau.unsqueeze(2)  
        noisy_action = tau_expanded * action_chunk + (1 - tau_expanded) * noise
        
        # Flatten noisy actions 
        noisy_action_flat = noisy_action.reshape(batch_size, -1)
        network_input = torch.cat([state, noisy_action_flat, tau], dim=1)
        
        # Velocity Extraction
        pred_velocity_flat = self.network(network_input)  
        pred_velocity = pred_velocity_flat.reshape(batch_size, self.chunk_size, self.action_dim)
        target_velocity = action_chunk - noise
        
        # Flow matching loss
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)
        return loss
    
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Generate action chunks via Euler integration (denoising).
        """
        batch_size = state.shape[0]
        device = state.device
        
        # Sample a noise
        action_chunk = torch.randn(
            batch_size, self.chunk_size, self.action_dim, device=device
        )
        step_size = 1.0 / num_steps
        for i in range(num_steps):
            tau = torch.full((batch_size, 1), i * step_size, device=device)
            action_flat = action_chunk.reshape(batch_size, -1)
            network_input = torch.cat([state, action_flat, tau], dim=1)
            
            # Predict velocity
            pred_velocity_flat = self.network(network_input)
            pred_velocity = pred_velocity_flat.reshape(batch_size, self.chunk_size, self.action_dim)
            
            # Derivative update
            action_chunk = action_chunk + step_size * pred_velocity

        return action_chunk
    
#######################################################################################

PolicyType: TypeAlias = Literal["mse", "flow"]

def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")