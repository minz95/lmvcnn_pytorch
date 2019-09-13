from __future__ import print_function, division

import os
import time
import copy
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms

from pair_loader import PairLoader
from layers import FeatureLayer, alexnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
d_path = 'model/'

class PairPredictor(nn.Module):
    def __init__(self, original_model=alexnet(pretrained=True)):
        super(PairPredictor, self).__init__()
        self.dataset_sizes = 480
        self.view_size = 48
        self.transform = transforms.Compose([
            transforms.Resize([227, 227]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.category_list = [
            'Airplane',
            'Bag',
            'Cap',
            'Car',
            'Chair',
            'Earphone',
            'Guitar',
            'Knife',
            'Lamp',
            'Laptop',
            'Motorbike',
            'Mug',
            'Pistol',
            'Rocket',
            'Skateboard',
            'Table'
        ]
        self.data_loader = PairLoader(self.transform)
        self.losses = []
        self.iters = []
        self.original_model = original_model
        self.prev_features = nn.Sequential(
            *list(self.original_model.features.children()),
        )
        self.classifiers = nn.Sequential(
            *list(self.original_model.classifier.children())[0:3],
        )
        self.post_features = nn.Sequential(
            FeatureLayer(),
        )
        self.snapshot = 1
        self.count = 0

    def forward(self, x):
        
        out = self.prev_features(x)
        out = out.view(out.size(0), 256 * 6 * 6)
        out = self.classifiers(out)
        p, n = self.post_features(out)
        
        return p, n

    def train_model(self, criterion, optimizer, scheduler, num_epochs=8):
        """
        Adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        """
        since = time.time()
        self.count = 0
        #epoch_count = 0
        #iters = []
        #losses = []

        #model = alexnet(pretrained=True)
        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = float('inf')
        
        #for param_group in optimizer.param_groups:
        #print("learning rate: " + str(param_group['lr']))

        self.losses.clear()
        #self.iters = range(num_epochs * 10 * len(self.data_loader.dataset.corr_list))
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    self.train()  # Set model to training mode
                else:
                    self.eval()   # Set model to evaluate mode

                running_loss = 0.0
                
                # Iterate over data (until one epoch)
                for i in range(10 * len(self.data_loader.dataset.corr_list)):
                    """
                    old_state_dict = {}
                    for key in self.state_dict():
                        old_state_dict[key] = self.state_dict()[key].clone()
                    """
                    start = time.time()
                    a_batch, b_batch, label = self.data_loader.next_batch()
                    print("data loading time: " + str(time.time() - start))
                    a_inputs = a_batch.to(device)
                    b_inputs = b_batch.to(device)
                    labels = label.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        a_positive, a_negative = self(a_inputs) 
                        b_positive, b_negative = self(b_inputs)

                        loss_positive = criterion(a_positive, b_positive, 1)
                        loss_negative = criterion(a_negative, b_negative, 0)
                        loss = loss_positive + loss_negative
                        print("loss: " + str(loss))

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                        # check if the model weights are updated
                        """
                        for p in list(self.parameters()):
                            print(p)
                        """

                        # statistics
                        running_loss += loss.item()
                        self.count += 1
                        """
                        if self.count % self.snapshot == 0:
                            self.losses.append(loss.item())
                            self.iters.append(self.count)
                            self.draw_loss()

                        
                        new_state_dict = {}
                        for key in self.state_dict():
                            new_state_dict[key] = self.state_dict()[key].clone()

                        # Compare params
                        
                        if self.count % self.snapshot == 0:
                            for key in old_state_dict:
                                if not (old_state_dict[key] == new_state_dict[key]).all():
                                    print('Diff in {}'.format(key))
                        """
                    print("iteration: " + str(self.count))
            
                epoch_loss = running_loss / self.dataset_sizes
                if phase == 'train':
                    self.iters.append(self.count)
                    self.losses.append(epoch_loss)
                    self.draw_loss()
                #epoch_count += 1
                #iters.append(epoch_count)
                #losses.append(epoch_loss)
                #_draw_loss(iters, losses)

                with open(os.path.join(d_path, 'epoch_loss.txt').replace("\\", "/"), 'a') as f:
                    f.write(str(epoch_loss))
                    f.write('\n')
                #epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.state_dict())
                    torch.save(self.state_dict(), os.path.join(d_path, 'lmvcnn_model.pt').replace("\\", "/"))

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Loss: {:4f}'.format(best_loss))

        #self.iters = range(num_epochs * 10 * len(self.data_loader.dataset.corr_list))
        self.draw_loss()

        # load best model weights
        self.load_state_dict(best_model_wts)
        torch.save(self.state_dict(), os.path.join(d_path, 'lmvcnn_model.pt').replace("\\", "/"))
        return self

    def draw_loss(self):
        """
        draw loss variation during the training period
        """
        if not self.iters or not self.losses:
            return

        plt.plot(self.iters, self.losses, '-b')
        plt.xlabel("iterations")
        plt.ylabel("loss")

        #plt.legend(loc="upper left")
        plt.title("Loss")
        plt.savefig("Loss" + ".png")

        #plt.show()

def _draw_loss(iters, losses):
    """
    draw loss variation during the training period
    """
    if not iters or not losses:
        return

    plt.plot(iters, losses, '-b')
    plt.xlabel("iterations")
    plt.ylabel("loss")

    #plt.legend(loc="upper left")
    plt.title("Loss")
    plt.savefig("_Loss" + ".png")

if __name__ == '__main__':
    pp = PairPredictor()
    input_x = []
    pp.forward(input_x)
