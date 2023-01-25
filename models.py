import torch
import math
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from math import sqrt

#torch.manual_seed(314)
#########################
# BASELINE CLASS
#########################


class BASELINE(nn.Module):
    def __init__(self, input_size, num_classes, drop):
        super(BASELINE, self).__init__()

        # Linear model for the slope
        self.linearA1 = nn.Linear(input_size, input_size*2)
        self.linearA2 = nn.Linear(input_size*2, input_size//3)
        self.linearA3 = nn.Linear(input_size//3, 1)

        # Linear model for the intercept
        self.linearB1 = nn.Linear(input_size, input_size*2)
        self.linearB2 = nn.Linear(input_size*2, input_size//3)
        self.linearB3 = nn.Linear(input_size//3, 1)

        # Linear combinatio
        self.linearY = nn.Linear(input_size+2, num_classes)

        # Activation function
        self.relu = nn.ReLU()

        # Dropout
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        a = self.relu(self.linearA1(x))
        a = self.relu(self.linearA2(a))
        a = self.drop(a)
        a = self.relu(self.linearA3(a))

        b = self.relu(self.linearB1(x))
        b = self.relu(self.linearB2(b))
        b = self.drop(b)
        b = self.relu(self.linearB3(b))

        axb = torch.concat([a, x, b], axis=2)  # to change a*x + b
        out = self.linearY(axb)
        out = torch.squeeze(out, -1)
        return out

    
#########################
# LSTM CLASS
#########################

class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch, device, drop):
        # Class properties:
        # - input_size: the number of features of the input
        # - hidden_size: the output size of the linear layer and the in/output size of the LSTM cells
        # - num_layers: number of consecutives LSTM layers
        # - num_classes: size of final output (this is a regression task, in this case = 1)
        # - batch: the size of the batches 
        # - device: 'CPU' or 'GPU'
        # - drop: dropout probability

        super(LSTM_Model, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # the short and long memories of the LSTM
        self.h0 = torch.rand(self.num_layers, batch, self.hidden_size).to(device)
        self.c0 = torch.rand(self.num_layers, batch, self.hidden_size).to(device)

        
        self.linear1 = nn.Linear(input_size, hidden_size) # enlarging the degrees of freedom of the inputs (14 -> 64)
        self.LSTM = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear2 = nn.Linear(hidden_size, int(hidden_size//2))
        self.linear3 = nn.Linear(int(hidden_size//2), num_classes)
        self.relu = nn.ReLU() # activation function
        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        # Defining the data flow in the model
        x = self.linear1(x)
        x = self.relu(self.LSTM(x, (self.h0, self.c0))[0])
        x = self.drop(x)
        x = self.relu(self.linear2(x))
        x = self.drop(x)
        out = self.linear3(x)
        out = torch.squeeze(out, -1) # reshaping to 1D vector
        return out

#########################
# TRANS CLASS
#########################


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len: int = 5000):
        # Class needed for the positional encoding. It generates the positional information, though sinusoidal curves, to add to the input
        # d_model: size of the input
        # max_len: max size for the input
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Trans_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, heads, drop):
        super(Trans_Model, self).__init__()
        # enlarging input
        self.linear1 = nn.Linear(input_size, hidden_size)

        # ADD POSITIONAL ENCODING
        self.pos_encoder = PositionalEncoding(d_model=hidden_size, dropout=drop)

        # encoders
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=heads, dropout=drop) 
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers) 

        # restricting input
        self.linear2 = nn.Linear(hidden_size, int(hidden_size//2)) 
        self.linear3 = nn.Linear(int(hidden_size//2), num_classes)

        # act function
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=drop)

    def init_weights(self):
        # conditioning the initial weights seems to perform better
        initrange = 0.1
        self.transformer_encoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        x = self.linear1(x)
        x = self.pos_encoder(x)
        x = self.relu(self.transformer_encoder(x))
        x = self.drop(x)
        x = self.relu(self.linear2(x))
        x = self.drop(x)
        x = self.linear3(x)
        out = torch.squeeze(x, -1)
        return out


#########################
# MODEL TESTING
#########################
def predict(model, serie, device, sq_len):
    # Using a model on a serie to get the predictions
    # sq_len: len of the input sequence
    model = model.to(device)
    predictions = []
    with torch.no_grad():
        for inputs, labels in serie:
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)

            # taking only the last prediction at the end of the serie
            y_pred = output[:, sq_len-1]
            for j in range(len(y_pred)):
                predictions.append(y_pred[j].item())

    return np.array(predictions)



# metrics computations
def ts_metrics(y, y_pred, verbose):
    # computing the metrics to evaluate the model
    y = np.array(y)
    y_pred = np.array(y_pred)

    if len(y) > len(y_pred):
        y = y[len(y) - len(y_pred):]
    elif len(y) < len(y_pred):
        y_pred = y_pred[len(y_pred) - len(y):]

    # MAPE
    mape = mean_absolute_percentage_error(y, y_pred)  

    # RSME
    rmse = sqrt(mean_squared_error(y, y_pred))

    # Score
    score = 0
    h = y-y_pred
    for h_i in h:
        if h_i < 0:
            e = np.exp(-h_i/13)-1
        else:
            e = np.exp(h_i/10)-1
        score += e

    if verbose:
        print(f"MAPE: {round(mape,3)}")
        print(f"RMSE: {round(rmse,3)}")
        print(f"Score: {round(score,3)}")

    return round(mape, 3), round(rmse, 3), round(score, 3)
