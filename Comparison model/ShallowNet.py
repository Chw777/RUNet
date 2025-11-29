import torch
import torch.nn as nn

class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, dropout_rate=0.5):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 25), stride=(1, 1))
        self.conv2 = nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1), groups=40)
        self.bn1 = nn.BatchNorm2d(40, eps=1e-03, momentum=0.99)
        self.elu = nn.ELU(alpha=1.0, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(2440, n_classes)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool(x)
        x = self.drop_out(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    inp = torch.rand(1,22, 1000)
    model =ShallowConvNet()
    out = model(inp)
    print(out.shape)
