import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        tensor = torch.ones(0)
        self.margin = tensor.new_tensor(margin)
        self.margin = self.margin.to(device)
        self.eps = tensor.new_tensor(0.0005)
        self.eps = self.eps.to(device)

    def forward(self, output1, output2, label, size_average=False):
        distances = (output2 - output1).pow(2).sum() # squared distances
        losses = distances
        if label == 1:
            losses = distances
            #print("label: " + str(label) + ", loss: " + str(distances))
        else:
            #print("label: " + str(label) + ", dist: " + str(distances))
            losses = torch.clamp(self.margin-distances.sqrt(), min=0).pow(2)
            #print("label: " + str(label) + ", loss: " + str(losses))
        losses += self.eps
        return losses.mean() if size_average else losses.sum()
