import torch
from torch import nn
from torchvision import transforms

from layers import FeatureLayer, alexnet
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DeployPredictor(nn.Module):
    def __init__(self, original_model=alexnet(pretrained=True)):
        super(DeployPredictor, self).__init__()
        self.original_model = original_model
        self.transform = transforms.Compose([
            transforms.Resize([227, 227]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.prev_features = nn.Sequential(
            *list(self.original_model.features.children()),
        )
        self.classifiers = nn.Sequential(
            *list(self.original_model.classifier.children())[0:3],
        )
        self.post_features = nn.Sequential(
            FeatureLayer(),
        )

    def forward(self, x):   
        out = self.prev_features(x)
        out = out.view(out.size(0), 256 * 6 * 6)
        out = self.classifiers(out)
        p, n = self.post_features(out)
        
        return p, n
