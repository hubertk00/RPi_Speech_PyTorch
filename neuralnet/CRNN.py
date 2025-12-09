import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, input_channels, num_classes, k=1.0):
        super(CRNN, self).__init__()
        
        conv_filters = int(32 * k)
        gru_units = int(64 * k)
        dense_units = int(32 * k)
        
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=conv_filters, 
            kernel_size=3, 
            stride=1, 
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.1)
        
        self.gru = nn.GRU(
            input_size=conv_filters,
            hidden_size=gru_units,
            batch_first=True,
            num_layers=1
        )
        
        self.bn2 = nn.BatchNorm1d(gru_units)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(gru_units, dense_units)
        self.dropout3 = nn.Dropout(0.4)
        self.fc_out = nn.Linear(dense_units, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        x = F.pad(x, (2, 0)) 
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 1) 
        
        output, _ = self.gru(x)
        x = output[:, -1, :] 
        
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        x = self.fc_out(x)
        return x