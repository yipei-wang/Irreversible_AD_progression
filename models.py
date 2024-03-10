from torch import nn

class MLP(nn.Module):
    def __init__(self, input_size = 68, output_size = 4, hidden_layers = [64],dropout_ratio=0.5):
        super(MLP, self).__init__()

        hidden_layers.insert(0, input_size)
        layers = []
        for i in range(len(hidden_layers)- 1):
            if i != 0:
                layers.append(nn.Dropout(dropout_ratio))
            layers.append(nn.Linear(hidden_layers[i],hidden_layers[i+1]))
            if i < len(hidden_layers) - 2:
                layers.append(nn.ReLU(inplace = False))
        self.prediction = nn.Linear(hidden_layers[-1], output_size)
    
        self.fc = nn.Sequential(*layers)
    def forward(self,x):
        feature = self.fc(x)
        x = self.prediction(feature)
        return x, feature
