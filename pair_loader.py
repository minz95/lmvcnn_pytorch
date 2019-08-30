import torch
from torchvision import transforms
from lmvcnn.model.pair_datasets import ImagePairDataset

class PairLoader():
    """
    make a batch using Pair Dataset
    returns a batch each time next function is called
    """
    def __init__(self, transform=None):
        """
        """
        self.dataset = ImagePairDataset(transform=transform)
        self.transform = transform
        self.batch_size = 48
        self.width = 227
        self.height = 227
        self.channel = 3
        self.index = 0      # index of getting item from the dataset

    def next_batch(self):
        """
        """
        print("called next batch")
        zeros = torch.zeros(())
        a_batch = zeros.new_empty((self.height, self.width, self.channel, 2*self.batch_size))
        b_batch = zeros.new_empty((self.height, self.width, self.channel, 2*self.batch_size))
        label = zeros.new_full((1, 1, 1, 2*self.batch_size), 0)
        #a_batch = np.ndarray((self.height, self.width, self.channel, self.batch_size), dtype=np.float32)
        #b_batch = np.ndarray((self.height, self.width, self.channel, self.batch_size), dtype=np.float32)
        #label = np.zeros((1, 1, 1, self.batch_size), dtype=np.float32)
        
        if self.index >= self.dataset.__len__():
            self.index = 0

        for idx in range(self.batch_size):
            # pimg and nimg are of type torch.Tensor
            pimg, nimg = self.dataset.__getitem__(self.index)
            assert (pimg is not None and nimg is not None)

            a_batch[:, :, :, idx*2] = pimg[0].permute(2, 1, 0)
            b_batch[:, :, :, idx*2] = pimg[1].permute(2, 1, 0)
            a_batch[:, :, :, idx*2 + 1] = nimg[0].permute(2, 1, 0)
            b_batch[:, :, :, idx*2 + 1] = nimg[1].permute(2, 1, 0)
            label[:, :, :, idx*2] = 1       # positive
            label[:, :, :, idx*2 + 1] = 0   # negative

            self.index += 1
        return a_batch.permute(3, 2, 0, 1), b_batch.permute(3, 2, 0, 1), label

if __name__ == '__main__':
    dl = PairLoader(transform=transforms.Compose([
            transforms.Resize([227, 227]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ]))
    for i in range(10):
        dl.next_batch()
    