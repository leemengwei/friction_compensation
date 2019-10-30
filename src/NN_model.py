import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_depth, output_size, device):
        super(NeuralNet, self).__init__()
        self.fc_first1 = nn.Linear(input_size, hidden_size*2)
        self.fc_first2 = nn.Linear(hidden_size*2, hidden_size)
        self.hidden_depth = hidden_depth
        self.fc_last1 = nn.Linear(hidden_size, 10)
        self.fc_last2 = nn.Linear(10, output_size)
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.th = nn.Tanh()
        self.ths = nn.Tanhshrink()
        self.sg = nn.Sigmoid()
        self.fcs = nn.ModuleList()   #collections.OrderedDict()
        self.bns = nn.ModuleList()   #collections.OrderedDict()
        for i in range(self.hidden_depth):
            self.bns.append(nn.BatchNorm1d(hidden_size, track_running_stats=True).to(device))
            self.fcs.append(nn.Linear(hidden_size, hidden_size).to(device))
    def forward(self, x):
        out = self.fc_first1(x)
        out = self.fc_first2(out)
        for i in range(self.hidden_depth):
            out = self.bns[i](out)
            out = self.fcs[i](out)
            out = self.relu(out)
        #out = self.th(out)
        out = self.fc_last1(out)
        out = self.fc_last2(out)
        out = self.ths(out)
        return out


