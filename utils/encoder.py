import torch.nn as nn

class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=1, bias=True, *args, **kwargs):
        super(LinearEncoder, self).__init__(*args, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.input_dim, self.output_dim, bias=bias))

    def forward(self, batch):
        x = batch.x
        for i in range(self.num_layers):
            x = self.layers[i](x)
        batch.x = x
        return batch

