import numpy as np
import torch


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size):
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.1
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.b = np.zeros(hidden_size)

    def forward(self, x_t, h_prev):
        return np.tanh(self.W_xh @ x_t + self.W_hh @ h_prev + self.b)


if __name__ == "__main__":
    cell = SimpleRNNCell(2, 2)
    cell.W_xh = np.array([[0.5, 0.2], [0.1, 0.4]])
    cell.W_hh = np.array([[0.3, 0.1], [0.2, 0.5]])
    x = np.array([1., 0.])
    h = np.array([0.5, 0.2])

    ref = torch.nn.RNNCell(2, 2, dtype=torch.float64)
    with torch.no_grad():
        ref.weight_ih.copy_(torch.tensor(cell.W_xh))
        ref.weight_hh.copy_(torch.tensor(cell.W_hh))
        ref.bias_ih.zero_()
        ref.bias_hh.zero_()

    actual = cell.forward(x, h)
    expected = ref(torch.tensor(x), torch.tensor(h)).detach().numpy()
    print("RNN cell:", actual)
    print("Khớp PyTorch:", np.allclose(actual, expected))
