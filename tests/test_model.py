import torch

from src.model import RNN


def test_rnn_forward_shape() -> None:
    input_size = 50
    model = RNN(input_size=input_size, hidden_size=32)
    # Batch size 4, seq_len 1, input_size 50
    x = torch.randn(4, 1, input_size)
    out = model(x)
    assert out.shape == (4, 1)
