import numpy as np
import torch
from torch import nn
from pytorch_lightning import LightningModule
from torchvision import transforms
from utils.dataloader import TrajDataSet, Center, ToTensor

class FCNet(nn.Module):
    def __init__(self, window, nb_feature):
        super(FCNet, self).__init__()

        self.name = 'FCNet'
        self.window = window
        self.nb_feature = nb_feature

        self.cnn = nn.Sequential(
            nn.Linear(self.nb_feature*self.window,100),
            nn.ReLU(),
            nn.Linear(100,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,1)
        )
    def forward(self, x):
        out = x.reshape(x.shape[0], 1, self.nb_feature*self.window)
        out = self.cnn(out)
        return out.squeeze(1)

class CNNet(nn.Module):
    def __init__(self, nb_feature):
        super(CNNet, self).__init__()

        self.name = 'CNNet'
        self.nb_feature = nb_feature

        self.cnn = nn.Sequential(
            nn.Conv1d(self.nb_feature, 8, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            nn.Conv1d(8, 16, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 32, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 1, kernel_size = 5, stride = 1, padding = 2, dilation = 1)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out.squeeze()

class UNet(nn.Module):
    def __init__(self, nb_feature):
        super(UNet, self).__init__()

        self.name = 'UNet'
        self.nb_feature = nb_feature

        self.cnn_input_1 = nn.Sequential(
            nn.BatchNorm1d(self.nb_feature),
            nn.Conv1d(self.nb_feature, 8, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU()
        )

        self.pooling_1 = nn.Sequential(
            nn.MaxPool1d(kernel_size = 5, stride = 2, padding = 2, dilation = 1)
        )

        self.cnn_input_2 = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Conv1d(8, 16, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU()
        )

        self.pooling_2 = nn.Sequential(
            nn.MaxPool1d(kernel_size = 5, stride = 2, padding = 2, dilation = 1)
        )

        self.cnn_input_3 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Conv1d(16, 32,  kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(32, 32,  kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU()
        )

        self.upconv_2 = nn.Sequential(
             nn.ConvTranspose1d(32, 16, kernel_size = 6, stride = 2, padding = 2, dilation = 1)
         )

        self.cnn_output_2 = nn.Sequential(
            nn.BatchNorm1d(2*16),
            nn.Conv1d(2*16, 16,  kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(16, 16,  kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU()
        )

        self.upconv_1 = nn.Sequential(
             nn.ConvTranspose1d(16, 8, kernel_size = 6, stride = 2, padding = 2, dilation = 1)
         )

        self.cnn_output_1 = nn.Sequential(
            nn.BatchNorm1d(2*8),
            nn.Conv1d(2*8, 8,  kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size = 5, stride = 1, padding = 2, dilation = 1),
            nn.ReLU(),
            nn.Conv1d(2, 1,  kernel_size = 5, stride = 1, padding = 2, dilation = 1)
        )

    def forward(self, x):
        out_1 = self.cnn_input_1(x)
        out = self.pooling_1(out_1)
        out_2 = self.cnn_input_2(out)
        out = self.pooling_2(out_2)
        out = self.cnn_input_3(out)

        out = self.upconv_2(out)
        out = torch.cat((out, out_2), 1)
        out = self.cnn_output_2(out)

        out = self.upconv_1(out)
        out = torch.cat((out, out_1), 1)
        out = self.cnn_output_1(out)
        return out.squeeze()


class Lightning(LightningModule):
    def __init__(self, model, weight):
        super().__init__()
        self.model = model
        self.lr = 1e-3
        self.register_buffer("weight", torch.FloatTensor([weight])) 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x,y = batch
        out = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight = self.weight)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        out = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight = self.weight)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x,y = batch
        out = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(out, y, pos_weight = self.weight)
        # self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    #--------------------------
    # Model Prediction Results
    def predict(self, data, window, variable):
      dive_estim = []

      if self.model.name != 'FCNet':
        for i in data.trip.unique():
            # create dataset for a trajectory
            t = data[data.trip == i].copy()
            test_set = TrajDataSet(t, window, variable, transform = ToTensor())

            # Test the model
            estim = np.zeros(len(t))
            nb = np.zeros(len(t))
            k = 0
            with torch.no_grad():
                for (x, y) in test_set:
                    out = self.model(x.unsqueeze(0))
                    estim[k:k + round(window)] += out.squeeze().numpy()
                    nb[k:k + round(window)] += 1
                    k+=1
            # # add to list by trajectory
            dive_estim.append(estim/nb)

      if self.model.name == 'FCNet':
          for i in data.trip.unique():
              # create dataset for a trajectory
              t = data[data.trip == i].copy()
              test_set = TrajDataSet(t, window, variable, transform = transforms.Compose([Center(), ToTensor()]))
              estim = [0 for i in range(int(window/2))]
              k = 0
              with torch.no_grad():
                  for (x, y) in test_set:
                      # Run the forward pass
                      out = self.model(x.unsqueeze(0))
                      estim.append(out.squeeze().numpy())
              estim = estim + [0 for i in range(int(window/2))]
              dive_estim.append(estim)

      data[self.model.name] = 1/(1+np.exp(-np.hstack(dive_estim)))
      return data

    def roc(self, input, target):
      # globally
      TP = []
      FP = []

      for tt in np.arange(0,1,0.001):
          all_estim = 1* (input > tt)
          true_positive = np.mean(all_estim[target == 1])
          true_negative = 1-np.mean(all_estim[target == 0])
          TP.append(true_positive)
          FP.append(1-true_negative)

      return (np.array(FP), np.array(TP))
