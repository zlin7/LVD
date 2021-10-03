#Adapted from https://github.com/ahmedmalaa/discriminative-jackknife for comparability
import torch
import torch.nn as nn
import numpy as np


ACTIVATION_DICT = {"ReLU": torch.nn.ReLU(), "Hardtanh": torch.nn.Hardtanh(),
                   "ReLU6": torch.nn.ReLU6(), "Sigmoid": torch.nn.Sigmoid(),
                   "Tanh": torch.nn.Tanh(), "ELU": torch.nn.ELU(),
                   "CELU": torch.nn.CELU(), "SELU": torch.nn.SELU(),
                   "GLU": torch.nn.GLU(), "LeakyReLU": torch.nn.LeakyReLU(),
                   "LogSigmoid": torch.nn.LogSigmoid(), "Softplus": torch.nn.Softplus()}


def build_architecture(base_model):

    modules          = []

    if base_model.dropout_active:

        modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

    modules.append(torch.nn.Linear(base_model.n_dim, base_model.num_hidden))
    modules.append(ACTIVATION_DICT[base_model.activation])

    for u in range(base_model.num_layers - 1):

        if base_model.dropout_active:

            modules.append(torch.nn.Dropout(p=base_model.dropout_prob))

        modules.append(torch.nn.Linear(base_model.num_hidden, base_model.num_hidden))
        modules.append(ACTIVATION_DICT[base_model.activation])

    modules.append(torch.nn.Linear(base_model.num_hidden, base_model.output_size))

    _architecture    = nn.Sequential(*modules)

    return _architecture

class DJKP_DNN(nn.Module):

    def __init__(self,
                 n_dim=-1,
                 dropout_prob=0.0,
                 dropout_active=False,
                 num_layers=1,
                 num_hidden=100,
                 output_size=1,
                 activation="Tanh",
                 mode="Regression"
                 ):

        super(DJKP_DNN, self).__init__()

        self.n_dim = n_dim
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.mode = mode
        self.activation = activation
        self.device = torch.device('cpu')  # Make this an option
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.dropout_active = dropout_active
        self.model = build_architecture(self)

    def fit(self, X, y, learning_rate=1e-2, loss_type="MSE", batch_size=100, num_iter=1000, verbosity=False,
            weight_decay=0, CV=None):
        self.model.train()
        if self.n_dim != X.shape[1]:
            self.n_dim = X.shape[1]
            self.model = build_architecture(self)

        self.X = torch.tensor(X.reshape((-1, self.n_dim))).float()
        self.y = torch.tensor(y).float()

        loss_dict = {"MSE": torch.nn.MSELoss}
        try:
            self.loss_fn = loss_dict[loss_type](reduction='mean')
        except:
            self.loss_fn = loss_type
        self.loss_trace = []

        batch_size = np.min((batch_size, X.shape[0]))

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        for _ in range(num_iter):

            batch_idx = np.random.choice(list(range(X.shape[0])), batch_size)

            y_pred = self.model(self.X[batch_idx, :])

            self.model.zero_grad()

            optimizer.zero_grad()  # clear gradients for this training step

            self.loss = self.loss_fn(y_pred.reshape((batch_size, -1)), self.y[batch_idx].reshape((batch_size, -1)))

            self.loss.backward(retain_graph=True)  # backpropagation, compute gradients
            optimizer.step()

            self.loss_trace.append(self.loss.detach().numpy())

            if verbosity:
                print("--- Iteration: %d \t--- Loss: %.3f" % (_, self.loss.item()))
        self.model.eval()

    def predict(self, X, numpy_output=True):

        X = torch.tensor(X.reshape((-1, self.n_dim))).float()

        if numpy_output:

            prediction = self.model(X).detach().numpy()

        else:

            prediction = self.model(X)
        #if self.output_size == 1: return prediction[...,0]
        return prediction

    def embed(self, X, numpy_output=True):
        X = torch.tensor(X.reshape((-1, self.n_dim))).float()
        if numpy_output:
            prediction = self.model[:-1](X).detach().numpy()
        else:
            prediction = self.model[:-1](X)
        return prediction


    def update_loss(self):

        self.loss = self.loss_fn(self.predict(self.X, numpy_output=False), self.y)


def get_DNN_and_trainkwargs(setting_id=0):
    if setting_id == 0:
        return DJKP_DNN, {'activation': 'ReLU'}, {}
    raise NotImplementedError()