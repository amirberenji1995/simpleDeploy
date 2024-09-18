import torch

class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv = torch.nn.Conv1d(in_channels = 1, out_channels = 5, kernel_size = 100)
        self.avgPool = torch.nn.AvgPool1d(kernel_size = 50)
        self.fc = torch.nn.Linear(5 * 38, 4)

    def forward(self, x):

        z = self.conv(x)
        z = torch.tanh(z)
        z = self.avgPool(z)

        z = z.view(-1, 5 * 38)

        z = self.fc(z)

        return z