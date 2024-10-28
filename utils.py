import torch

def build_mlp(hidden_dims,dropout=0,activation=torch.nn.ReLU,with_bn=True,no_act_last_layer=True):
    modules = []
    for i in range(len(hidden_dims)-1):
        modules.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        if not (no_act_last_layer and i == len(hidden_dims)-2):
            if with_bn:
                modules.append(torch.nn.BatchNorm1d(hidden_dims[i+1]))
            modules.append(activation())
            if dropout > 0.:
                modules.append(torch.nn.Dropout(p=dropout))
    return torch.nn.Sequential(*modules)