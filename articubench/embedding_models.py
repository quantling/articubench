import torch

class MelEmbeddingModel(torch.nn.Module):
    """
        EmbedderModel
        - Initial Conv1d Layers for Convolution over time and neighbouring Mel Channels with residual connections
        - stacked LSTM-Cells
        - Post upsammpling layer
    """

    def __init__(self, input_size=60,
                 output_size=300,
                 hidden_size=720,
                 num_lstm_layers=2,
                 post_activation=torch.nn.LeakyReLU(),
                 post_upsampling_size=0,
                 dropout=0.7):
        super().__init__()

        self.post_upsampling_size = post_upsampling_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_lstm_layers, batch_first=True,
                                  dropout=dropout)
        if post_upsampling_size > 0:
            self.post_linear = torch.nn.Linear(hidden_size, post_upsampling_size)
            self.linear_mapping = torch.nn.Linear(post_upsampling_size, output_size)
            self.post_activation = post_activation
        else:
            self.linear_mapping = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lens, *args):
        output, (h_n, _) = self.lstm(x)
        output = torch.stack([output[i, (last - 1).long(), :] for i, last in enumerate(lens)])
        if self.post_upsampling_size > 0:
            output = self.post_linear(output)
            output = self.post_activation(output)
        output = self.linear_mapping(output)

        return output
