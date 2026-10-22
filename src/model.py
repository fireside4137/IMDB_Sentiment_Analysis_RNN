from typing import cast

import torch
import torch.nn as nn


class RNN(nn.Module):
    """Many‑to‑one RNN for binary sentiment classification.

    The model expects input of shape ``(batch, seq_len, input_size)`` where
    ``seq_len`` is ``1`` because we are feeding a TF‑IDF vector as a single
    timestep.  While this architecture is not ideal for text, we keep it to
    preserve the original educational example.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        ``x`` is expected to have shape ``(batch, 1, input_size)``.
        """
        # Initialise hidden state to zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # Take the output from the last timestep
        return cast(torch.Tensor, out)  # NOTE: logits (no sigmoid applied)


__all__ = ["RNN"]
