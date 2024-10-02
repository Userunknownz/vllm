from enum import IntEnum

import torch
import torch.nn as nn

from vllm.model_executor.pooling_metadata import (PoolingMetadata,
                                                  PoolingTensors)
from vllm.sequence import EmbeddingSequenceGroupOutput, PoolerOutput


class PoolingType(IntEnum):
    """Enumeration for different types of pooling methods."""
    LAST = 0
    AVERAGE = 1


class Pooler(nn.Module):
    """A layer that pools specific information from hidden states.

    This layer does the following:
    1. Extracts specific tokens or aggregates data based on pooling method.
    2. Normalizes output if specified.
    3. Returns structured results as `PoolerOutput`.

    Attributes:
        pooling_type: The type of pooling to use (LAST, AVERAGE, MAX).
        normalize: Whether to normalize the pooled data.
    """

    def __init__(self, pooling_type: PoolingType, normalize: bool):
        super().__init__()
        self.pooling_type = pooling_type
        self.normalize = normalize

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Pools specific information from hidden states based on metadata."""
        prompt_lens = PoolingTensors.from_pooling_metadata(
            pooling_metadata, hidden_states.device).prompt_lens

        if self.pooling_type == PoolingType.LAST:
            last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
            pooled_data = hidden_states[last_token_flat_indices]
        
        elif self.pooling_type == PoolingType.AVERAGE:
            last_token_flat_indices = torch.cumsum(prompt_lens, dim=0) - 1
            pooled_data = []

            # Iterate through each prompt
            start_idx = 0
            for i, end_idx in enumerate(last_token_flat_indices):
                # Extract embeddings for the current prompt
                prompt_embeddings = hidden_states[start_idx:end_idx + 1]  # From start_idx to end_idx (inclusive)
                
                # Compute the average embedding for the current prompt
                average_embedding = prompt_embeddings.mean(dim=0)
                
                # Append the average embedding to the pooled_data list
                pooled_data.append(average_embedding)
                
                # Update the start_idx for the next prompt
                start_idx = end_idx + 1
              
              # Convert the list of pooled_data back to a tensor
            pooled_data = torch.stack(pooled_data)
            
        else:
            raise ValueError(f"Invalid pooling type: {self.pooling_type}")

        if self.normalize:
            pooled_data = nn.functional.normalize(pooled_data, p=2, dim=1)

        pooled_outputs = [
            EmbeddingSequenceGroupOutput(data.tolist()) for data in pooled_data
        ]

        return PoolerOutput(outputs=pooled_outputs)
