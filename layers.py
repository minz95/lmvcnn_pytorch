import torch
import torch.nn as nn

class LRN(nn.Module):
    """
    Adapted from jiecaoyu's implementation
    """
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class ScaleLayer(nn.Module):

    def __init__(self, init_value=1e-2):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale

class SplitLayer(nn.Module):
    def __init__(self, size):
        super(SplitLayer, self).__init__()
        self.view_size = size

    def forward(self, x):
        splits = torch.split(x, self.view_size, dim=0)
        return splits[0], splits[1]

class EltMaxLayer(nn.Module):
    def __init__(self):
        super(EltMaxLayer, self).__init__()
        self.view_size = 48
        self.split_layer = nn.Sequential(
            SplitLayer(self.view_size),
        )

    def forward(self, x):
        positive, negative = self.split_layer(x)
        positive = torch.max(positive, 0, keepdim=True)[0]
        negative = torch.max(negative, 0, keepdim=True)[0]

        return torch.cat((positive, negative), 0)

class FeatureLayer(nn.Module):
    def __init__(self):
        super(FeatureLayer, self).__init__()
        self.view_size = 48
        """
        self.fc_layer = nn.Linear(in_features=4096, out_features=128, bias=False)
        torch.nn.init.normal_(self.fc_layer.weight, 0, 0.01)
        """
        self.features = nn.Sequential(
            EltMaxLayer(),
            ScaleLayer(),
            nn.Linear(4096, 128),
            SplitLayer(1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        splits = torch.split(x, self.view_size, dim=0)
        positives = torch.max(splits[0], 0, keepdim=True)[0]
        negatives = torch.max(splits[1], 0, keepdim=True)[0]

        positives = self.features(positives)
        negatives = self.features(negatives)
        """
        #result = torch.cat((positives, negatives), dim=0)
        positives, negatives = self.features(x)
        return torch.squeeze(positives), torch.squeeze(negatives)

class AlexNet(nn.Module):
    """
    Adapted from jiecaoyu's implementation
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        elem_features = nn.Sequential(
            *list(self.classifier.children())[0:3]
        )
        x = elem_features(x)
        
        return x

def alexnet(pretrained=False):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet()
    if pretrained:
        model_path = './model/alexnet.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model
