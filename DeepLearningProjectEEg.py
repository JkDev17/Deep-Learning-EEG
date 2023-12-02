# Data preperation
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import signal
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score as auc

# With the 2 following commands, we unzip the zipped version of the train/test data uploaded on my google drive account.
!unzip drive/My\Drive/Dataset/train.zip
!unzip drive/My\Drive/Dataset/test1.zip

# load data from google colab's content file to a list.

# train data path.
train_path = r'/content/train'

# test data path.
test_path = r'/content/test1'

# glob helps us to search for files where the filename matches a certain pattern.

data_csv = glob.glob(train_path + "/*subj1_series*_data*")

events_csv = glob.glob(train_path + "/subj1_*_events*")

test_csv = glob.glob(test_path + "/subj1_series*_data*")

# loop through files and append every cvs file's content to to our training list.
train_list = []
for  f in data_csv:
  df = pd.read_csv(f, index_col=None, header=0)# header specifies which row of the CSV file should be used as the header of the resulting df.
                                               # index_col is used to specify which column should be used as the index of the resulting df.
  print(f"I am appending the file {f}")
  train_list.append(df)

events_list = []
for f in events_csv:
  df = pd.read_csv(f)
  print(f"I am appending the file {f}")
  events_list.append(df)

test_list = []
for  f in test_csv:
  df = pd.read_csv(f, index_col=None, header=0)
  print(f"I am appending the file {f}")
  test_list.append(df)

# concatatenate  a list of DataFrames verifically and skip the first column(skip the id column).
train_dataframes = pd.concat(train_list, axis=0, ignore_index=True) # axis=0 ---> the objects will be concatenated vertically, i.e. they will be stacked on top of each other.
                                                                    # ignore_index=True --> a new index will be created for the concatenated object.
train_dataframes = train_dataframes.iloc[:,1:]                      # keep all the rows, remove first column and keep the others.

events_dataframes = pd.concat(events_list, axis=0, ignore_index=True)
events_dataframes = events_dataframes.iloc[:,1:]

test_dataframes = pd.concat(test_list, axis=0, ignore_index=True)
test_dataframes = test_dataframes.iloc[:,1:]

def apply_low_pass_filter(eeg_signal, cutoff_freq, sampling_rate):
  # Normalize the cutoff frequency
  normalized_cutoff = cutoff_freq / (0.5 * sampling_rate)

  # Design the low pass filter
  b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False, output='ba')

  # Apply the filter to the EEG signal
  filtered_signal = signal.lfilter(b, a, eeg_signal)

  return filtered_signal

# Standardization of the features in the dataset
Xsc = StandardScaler().fit_transform(train_dataframes)
Xsc=pd.DataFrame(Xsc)

x_data = Xsc.copy()

x_low_pass_filtered_train_data = apply_low_pass_filter(x_data, cutoff_freq=40, sampling_rate=500)

Xsc.max()
Xsc.min()
Xsc.shape[1]

# Applying pca for dimensionality reduction and noise reduction.
pca = PCA(n_components = 16)
pca.fit(x_low_pass_filtered_train_data)
X_train_pca = pca.transform(x_low_pass_filtered_train_data)

X_train_pca.max()
X_train_pca.min()
X_train_pca.shape[1]


# Applying low pass filter & pca on test data.
test_dataframes.max()
test_dataframes.min()

Ysc = StandardScaler().fit_transform(test_dataframes)
Ysc=pd.DataFrame(Ysc)

low_pass_filtered_test_data = apply_low_pass_filter(Ysc, cutoff_freq=40, sampling_rate=500)

Ysc.max()
Ysc.min()
Ysc.shape[1]

pca = PCA(n_components = 16)
pca.fit(low_pass_filtered_test_data)
test_pca = pca.transform(low_pass_filtered_test_data)

test_pca.max()
test_pca.min()
test_pca.shape[1]



# CNN Model, Batching, Training
num_features = 16
window_size = 1024
batch_size = 2000

# Batching improves efficiency, optimizes memory usage, enables parallelization, and provides stable gradient estimates.
# It plays a significant role in accelerating the training

def get_batch(data_train, data_events, batch_size=2000, index=None):
    index = random.randint(window_size, len(data_train) - 16 * batch_size)
    indexes = np.arange(index, index + 16*batch_size, 16)

    batch = np.zeros((batch_size, num_features, window_size//4))
    b = 0

    for i in indexes:
        start = i - window_size if i - window_size > 0 else 0
        tmp = data_train[start:i, :num_features]
        batch[b, :, :] = tmp[::4].transpose()
        b += 1

    targets = data_events.iloc[indexes]
    numpy_targets = targets.to_numpy()

    return torch.DoubleTensor(batch), torch.DoubleTensor(numpy_targets)


class convmodel(nn.Module):

    def __init__(self, out_classes, drop=0.5, d_linear=124): # drop helps to prevent overfitting
                                                             # d_linear = 124 -> the linear layer will have 124 neurons
                                                             # all of them will be connected to previous layer's neurons
        super().__init__() #init the superclass

        self.conv2 = nn.Conv1d(16, 64, kernel_size=3, padding=0, stride=1) # init a 1d conv layer with input 16 channels
                                                                           # outputs 64 channels
                                                                           # kernel_size= 3 -> three elements at a time
                                                                           # padding = 0 -> we don't add numbers at the right and the left of the tensor
                                                                           # stride = 1 -> move one element at a a time
        self.bn = nn.BatchNorm1d(64) # Normalize the  64 channel activations outputted by the prev conv layer
        self.pool = nn.MaxPool1d(2, stride=2) # Apply max Pooling for every 2 elements
        self.linear1 = nn.Linear(8128, d_linear) # The linear layer takes as input 8128 values and outputs 124
                                                 # using matrix multiplication between the input data and weights.
                                                 # the weight values are learned during the training

        self.linear3 = nn.Linear(d_linear, out_classes) # 2nd linear layout reducing the 124 values to our 6 possible results
        self.dropout1 = nn.Dropout(drop) # prevents overfitting
        self.dropout2 = nn.Dropout(drop) # randomly sets a fraction of input units to zero during training
        self.dropout3 = nn.Dropout(drop) # thus it helps nn to learn more robust and generalizable features.

        # A sequential container is a way to organize multiple layers in a specific order
        # The layers within the sequential container will be executed in the same order as they are defined.
        # The purpose of this sequential container is to define the forward pass operations for the convolutional layers of the model.

        self.conv = nn.Sequential(self.conv2, nn.ReLU(inplace=True), self.bn, # inplace = true --> modifying the input tensor directly.
                                    self.pool, self.dropout1)
        self.dense = nn.Sequential(self.linear1, nn.ReLU(inplace=True),self.dropout2,
                                    self.dropout3, self.linear3)

    # A forward pass in deep learning involves feeding input data through a neural network model.
    # During this process, the input data is transformed through a series of mathematical operations, weights, and activation functions to produce
    # an output prediction. The forward pass facilitates information flow and enables the network to learn and make predictions based on the given input.

    def forward(self, x):
        bs = x.size(0)   # get size of the batch  --> 2000
        x = self.conv(x) # Pass the input tensor x through the layers defined in self.conv(the sequential container for convolutional layers)
                         # It applies the convolutional, activation, batch normalization, pooling, and dropout operations in the defined order.

        x = x.view(bs, -1) # reshape the tensor --> -1 means (100, 20, 30) --> (100, 600)
        output = self.dense(x) # Pass the input ternsor through the linear layouts just as the conv layer.

        return torch.sigmoid(output) # Apply a sigmoid activation function to the output tensor.
                                     # Squash the output values between 0 and 1 to perform the classification.


model = convmodel(6).double().cuda() # double --> convert from torch.float32 to 64

optim = torch.optim.Adadelta(model.parameters(), lr=1, eps=1e-10) # model.parameters() --> update them while training
                                                                  # learning rate --> adjusts the model's parameters during gradient descent--> small value
                                                                  # epsilon --> improve numerical stability --> very small value


bs = batch_size

def train(traindata, train_events, epochs, printevery=1, shuffle=True): # shuffle=True --> reduce any potential bias or patterns that may exist in the data
  model.train() # enter train mode.

  for epoch in range(epochs):
    total_loss = 0 # the loss for each batch of training data.
    for i in range(len(traindata)//bs):
      optim.zero_grad() # gradients --> minimize the loss function by adjusting the model's parameters (weights and biases) using gradients
                        # optim.zero_grad() --> resets all the gradients of the model's parameters to zero.
      x, y = get_batch(traindata,train_events)
      x = Variable(x).cuda()
      y = Variable(y).cuda()
      preds = model(x) # trigger forward pass and compute the predictions of the model for the input data
      loss = F.binary_cross_entropy(preds.reshape(-1), y.reshape(-1)) # calculates the binary cross-entropy loss between the predictions
                                                                      # and the actual labels (0 or 1 for the 6 channels we have as results)
      loss.backward() # computes the gradients of the loss using backpropagation.
      total_loss += loss.item() # accumulates the loss value for the current batch.
      optim.step() # updates the model's parameters using the computed gradients.
      print("epoch: %d, iter %d/%d, loss %.4f"%(epoch + 1, i + 1, len(traindata)//2000, total_loss/printevery))
      total_loss = 0 # resets total_loss for the next iteration.

train(X_train_pca, events_dataframes, 30)



# Make predictions, evaluate the model

def get_test_batch(dataset, batch_size=2000, index=False):

  indexes = np.arange(index, index + batch_size)

  batch = np.zeros((batch_size, num_features, window_size//4))
  b = 0

  for i in indexes:
      start = i - window_size if i - window_size > 0 else 0
      tmp = dataset[start:i]
      batch[b, :, :] = tmp[::4].transpose()
      b += 1

  targets = events_dataframes.iloc[indexes]
  numpy_targets = targets.to_numpy()

  return torch.DoubleTensor(batch), torch.DoubleTensor(numpy_targets)

def getPredictions(data):
    model.eval()
    p = []
    res = []
    i = window_size
    bs = 2000
    while i < len(data):
        if i + bs > len(data):
            bs = len(data) - i
        x, y = get_test_batch(data, bs, index=i)
        x = x.cuda()  # Move input tensor to the GPU
        preds = model(x)
        preds = preds.squeeze(1)
        p.append(np.array(preds.data.cpu()))  # Move predictions to CPU
        res.append(np.array(y.data.cpu()))    # Move targets to CPU
        i += bs
    preds = p[0]
    for i in p[1:]:
        preds = np.vstack((preds, i))
    targs = res[0]
    for i in res[1:]:
        targs = np.vstack((targs, i))
    return preds, targs

def valscore(data):

    preds, targs = getPredictions(data)
    aucs = [auc(targs[:, j], preds[:, j]) for j in range(6)]
    total_loss = np.mean(aucs)
    print(total_loss)
    return preds, targs

valscore(test_pca)