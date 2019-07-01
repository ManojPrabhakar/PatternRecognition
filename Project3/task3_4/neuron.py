import numpy as np

class Neuron():
    def __init__(self, lr_w = .005, lr_theta = .001, dim = 2):
        self.w = np.random.uniform(-1, 1, dim)[..., np.newaxis]
        self.theta = np.random.uniform(-1,1, 1)

        self.lr_w = lr_w
        self.lr_theta = lr_theta

    def activation_function(self, x):
        self._temp = np.dot(self.w.T, x) - self.theta
        return 2 * np.exp(- .5 * self._temp ** 2) - 1

    def train(self, data, epochs = 1):
        losses = []

        for epoch in range(epochs):
            for i, (x, label) in enumerate(data):
                x = x[..., np.newaxis]
                out = self.activation_function(x)
                assert out.shape[0] == 1 and out.shape[1] == 1, print(out)
                out = out[0]
                loss = self.loss(out, label)
                losses.append(loss[0])
                self.update(out, label, x)

        return losses

    def loss(self, y, label):
        return .5 * (y - label) ** 2

    def loss_vectorized(self, y, label):
        return .5 * np.sum((y - label) ** 2)

    def gradient_w(self, out, label, x):
        temp = - (out - label) * (out + 1) * self._temp
        return temp * x

    def gradient_theta(self, out, label):
        return (out - label) * (out + 1) * self._temp

    def update(self, out, label, x):
        self.w = self.w - self.lr_w * self.gradient_w(out, label, x)
        self.theta = self.theta - self.lr_theta * self.gradient_theta(out, label)

    def test(self, data):
        predicted = []
        for i, (x, label) in enumerate(data):
            out = self.activation_function(x)
            predicted.append( 1 if out > 0 else -1)

        return predicted

