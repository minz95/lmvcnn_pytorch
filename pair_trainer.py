import os
import torch
import torch.optim as optim

from lmvcnn.model.pair_fcn import PairPredictor
from lmvcnn.model.loss import ContrastiveLoss
from lmvcnn.model.layers import alexnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model_type='pair_fcn'):
    model = None
    if model_type == 'pair_fcn':
        model_conv = alexnet(pretrained=True)
        print("model type: " + str(type(model_conv)))
        #num_ftrs = model_conv.fx.in_features
        #model_conv.fc = nn.Linear(num_ftrs, 2)
        model_conv = model_conv.to(device)

        pair_model = PairPredictor(model_conv)
        for param in pair_model.parameters():
            param.requires_grad = True

        model_path = 'C:/Users/HCIL/InfoVisProject/PolySquare/server/lmvcnn/model/lmvcnn_model.pt'
        if os.path.isfile(model_path):
            pair_model.load_state_dict(model_path)

        model = pair_model.to(device)
        contra_loss = ContrastiveLoss(1)
        
        """
        print("PARAMETERS############# ")
        for p in model_conv.parameters():
            print(p)
        """
        
        optimizer_conv = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_conv, step_size=10000, gamma=0.1)
        best_model = model.train_model(contra_loss, optimizer_conv, exp_lr_scheduler)

        # visualize the best model
        #pair_model.visualize_model(best_model)
    
    return best_model

if __name__ == '__main__':
    result = train()
    #print(result[0])   
