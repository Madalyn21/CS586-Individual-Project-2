# model.py

import torch.nn as nn

class TaxiDriverClassifier(nn.Module):
    """
    Input:
        Data: the output of process_data function.
        Model: your model.
    Output:
        prediction: the predicted label(plate) of the data, an int value.
    """
    def __init__(self, input_dim, output_dim):
        super(TaxiDriverClassifier, self).__init__()

        ###########################
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(64, output_dim)
        ###########################

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        lstm_out, (hidden, cell) = self.lstm(x)

        # hidden shape: (num_layers, batch_size, hidden_size)
        last_hidden = hidden[-1]   # shape: (batch_size, 64)

        x = self.fc(last_hidden)   # shape: (batch_size, output_dim)

        ###########################
        return x
    
