import os
import random
import numpy as np
from PIL import Image
import torch.utils.data as data_utils

class ImagePairDataset(data_utils.dataset.Dataset):
    """
    """
    def __init__(self, data_dir=u'image_data/', 
                view_size=48, train=True, transform=None):
        self.data_dir = data_dir.replace("'", "")
        self.transform = transform
        self.train = train
        self.view_size = view_size
        self.batch_iter = 0
        self.data_iter = 0
        self.data_generator = DataGeneration()
        self.corr_list = self.get_corr_list()

        #print("data generation begin")
        """
        corr = self.corr_list[self.data_iter]
        self.data_generator.delete_sample()
        self.data_generator.delete_image()
        self.data_generator.generate_pair_view(corr[0], corr[1], corr[2])
        self.data_iter += 1
        """
        #print("data generation end")
        self.data_iter += 1
        self.pos_data = []
        self.neg_data = []
        self.get_data()

    def __getitem__(self, index):
        """
        """
        #print("getitem:: " + str(index) + ", " + str(self.__len__()))
        #print("pos length: " + str(len(self.pos_data)))
        if self.data_iter >= len(self.corr_list):
            self.data_iter = 0
            random.shuffle(self.corr_list)

        if self.batch_iter >= self.view_size:
            self.batch_iter = 0
            #corr = self.corr_list[self.data_iter]
            #self.data_generator.delete_sample()
            #self.data_generator.delete_image()
            #self.data_generator.generate_pair_view(corr[0], corr[1], corr[2])
            self.data_iter += 1
            self.get_data()
        
        # make pairs (1 positive pair + 1 negative pair)
        p_images = self.pos_data[index]
        n_images = self.neg_data[index]
        
        p_images[0] = Image.open(p_images[0]).convert('RGB')
        p_images[1] = Image.open(p_images[1]).convert('RGB')
        n_images[0] = Image.open(n_images[0]).convert('RGB')
        n_images[1] = Image.open(n_images[1]).convert('RGB')
        if self.transform is not None:    
            p_images[0] = self.transform(p_images[0])
            p_images[1] = self.transform(p_images[1])
            n_images[0] = self.transform(n_images[0])
            n_images[1] = self.transform(n_images[1])

        self.batch_iter += 1

        return p_images, n_images

    def __len__(self):
        """
        """
        return len(self.pos_data)

    def get_corr_list(self, corr_path='Corr/'):
        """
        """
        clist_path = 'train_corr_list.npy'
        if os.path.isfile(clist_path):
            return np.load(clist_path).tolist()

        corr_list = []
        for dir_path, _, f_name in os.walk(corr_path):
            cat_list = []
            for f in f_name:
                corr = []
                idx = dir_path.find('Corr')
                category = dir_path[idx+5:]
                corr.append(category)

                model_idx = f.find('___')
                model_a = f[:model_idx]
                model_b = f[model_idx+10:len(f)-4]
                corr.append(model_a)
                corr.append(model_b)
                #corr_list.append(corr)
                cat_list.append(corr)
            if not cat_list:
                continue

            """
            for c in cat_list:
                print(c)
            """
            # make a corr_list without redundant objects, only 3 corr for one category
            visited = []
            cnt = 0
            while cnt < 15:
                '''
                if cnt < 11:
                    cnt += 1
                    continue
                '''
                sample = cat_list[random.randrange(len(cat_list))]
                if sample[1] in visited or sample[2] in visited:
                    continue
                else:
                    corr_list.append(sample)
                    cnt += 1
                    visited.append(sample[1])
                    visited.append(sample[2])
                    cat_list.remove(sample)

            """
            sample = random.sample(cat_list, 25)
            for s in sample:
                corr_list.append(s)
            """

        random.shuffle(corr_list)
        # use this line to generate pair views using samples corr list
        self.data_generator.generate_sample_view(corr_list)

        clist_path = os.path.join('C:/Users/HCIL/InfoVisProject/PolySquare/server/lmvcnn/model/', 'corr_list')
        np.save(clist_path, np.asarray(corr_list))
        return corr_list

    def get_data(self):
        """
        store image names in pos, neg lists
        only store data for one batch (this is for a full training purpose)
        """
        a_path = os.path.join(self.data_dir, self.corr_list[self.data_iter-1][1])
        b_path = os.path.join(self.data_dir, self.corr_list[self.data_iter-1][2])
        
        self.pos_data.clear()
        self.neg_data.clear()
        for d, _, f_name in os.walk(a_path):
            for f in f_name:
                a_file = os.path.join(d, f).replace("\\", "/")
                b_file = os.path.join(b_path, f).replace("\\", "/")
                
                idx = f.find('_')
                if f[0:idx] == 'positive':
                    self.pos_data.append([a_file, b_file])
                else:
                    self.neg_data.append([a_file, b_file])

    def get_fulldata(self):
        """
        store image names in pos, neg lists
        store full data in the list (this is for small dataset only, 
        for a transfer learning purpose)
        prone to memory error
        """
        for corr in self.corr_list:
            a_path = os.path.join(self.data_dir, corr[1])
            b_path = os.path.join(self.data_dir, corr[2])
            
            #self.pos_data.clear()
            #self.neg_data.clear()
            for d, _, f_name in os.walk(a_path):
                for f in f_name:
                    a_file = os.path.join(d, f).replace("\\", "/")
                    b_file = os.path.join(b_path, f).replace("\\", "/")
                    
                    idx = f.find('_')
                    if f[0:idx] == 'positive':
                        self.pos_data.append([a_file, b_file])
                    else:
                        self.neg_data.append([a_file, b_file])

if __name__ == '__main__':
    ipd = ImagePairDataset()
    ipd.get_corr_list()
    # ipd.get_corr_list()
    #ipd.get_data()
    #loader = data_utils.DataLoader(ImagePairDataset(), batch_size=1, shuffle=False, num_workers=1)
    #print(len(loader))
    # code for testing a data loader
    #dataloader = data_utils.DataLoader(ipd, batch_size=48, num_workers=4)
    #print(len(dataloader))

    """
    for i, data in enumerate(loader, 0):
        pos, neg, label = data
        num = len(pos)
        ax = plt.subplot(1, image_show, i+1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        
        for n in range(num):
            p_img = pos[n]
            n_img = neg[n]
            l = label[n]
            plt.imshow(p_img)

        if i == image_show-1:
            break
    """
