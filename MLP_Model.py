import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F


# get data from file
df = pd.read_csv('team_stats_2003_2023.csv')

# Initialize a list to store PyTorch tensors
tensor_list = []
TrueLabelList = []

# Iterate over each row in the file
for index, row in df.iterrows():
    # Convert the row into a Python list
    row_list = row.tolist()

    # don't take first row
    if(row_list[0]=='year'):
        continue

    # don't take season with 17 games
    if row_list[9] == 17:
        continue

    # delete some colomns like (wins, loss, win percentage)
    del row_list[1]

    #---wins for True Label
    TrueLabelList.append(row_list.pop(1))
    #---

    # continue deleting colomns (like wins, loss, win percentage)
    del row_list[1]
    del row_list[1]
    del row_list[4]
    del row_list[0]

    del row_list[28]





    # Convert the Python list into a PyTorch tensor
    row_tensor = torch.tensor(row_list, dtype=torch.float32)
    
    # Append the tensor to the list
    tensor_list.append(row_tensor)


# convert list of tensors into tensor of tensors
TrueLabelTensor = torch.tensor(TrueLabelList, dtype=torch.float32)
torch_tensors = torch.stack(tensor_list)
print(torch_tensors.shape)
print(TrueLabelTensor.shape)



#-----Normalization of Data----------#
# z-score standardization around axis = 0
mean_X = torch_tensors.mean()
mean_Y = TrueLabelTensor.mean()

std_X = torch_tensors.std()
std_y = TrueLabelTensor.std()

normalized_tensor = (torch_tensors - mean_X) / std_X
normalized_tensor_y = (TrueLabelTensor - mean_Y) / std_y


#---split data into training (0.8) and testing (0.2) Data

split = int(len(normalized_tensor)*0.8)
training_Data = normalized_tensor[:split]
training_Data_Y = normalized_tensor_y[:split]

testing_Data = normalized_tensor[split:]
testing_Data_Y = normalized_tensor_y[split:]


#------------Data set--------------#

class myDataSet(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

#        self.features = self.features.view(-1, 1)  # reshape it into a tensor with column 1 and row -1. -1 means, that torch finds the number of rows on its own

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return self.labels.shape[0]
    
trainDataSet = myDataSet(training_Data, training_Data_Y)

#------------Data Loader--------------#
trainDataLoader = DataLoader(trainDataSet, batch_size=2, shuffle=True)


#------------NN-Model--------------#
# with 3 layers (2 hidden and one output)
class MLP_regression(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linearLayer1 = torch.nn.Linear(num_features, 64)
        self.linearLayer2 = torch.nn.Linear(64, 32)
        self.linearLayer3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.linearLayer1(x))
        x = torch.relu(self.linearLayer2(x))
        x = self.linearLayer3(x)
        return x
    

model = MLP_regression(num_features=28)


#--------Plot training Loss as a function of epoch-------#
    
def plotLoss(epoch, loss, numEpochs):
    plt.plot(epoch, loss ,label="Loss as a function of epoch")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc='upper left')
    #plt.ylim(-1, 2)
    plt.ylim(-5, 5)
    plt.xlim(-1,numEpochs)
    plt.grid()
    plt.show()

#------------Training Loop----------------#
torch.manual_seed(123)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

numEpochs = 400

lossList = []
epochList = []
for epoch in range(numEpochs):

    model = model.train()
    
    for batch_idx, (features, classLables) in enumerate(trainDataLoader):
        # update the parameters after every batch_size

        yPredict = model(features)  # calls the forward() method on the model

        loss = F.mse_loss(yPredict, classLables.view(yPredict.shape))


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    lossList.append(loss.item())
    
    epochList.append(epoch)


plotLoss(epochList, lossList, numEpochs)



#-------Evaluate Model with testing data---------#

yValues = []
dataList = []
for data in testing_Data:
    yResult = model(data)

    yValue = (yResult.item() *  std_y) + mean_Y  #un-normalize y value
    yValues.append(yValue)
    dataList.append((data *  std_X) + mean_X)    #un-normalize x value


newY = (testing_Data_Y * std_y) + mean_Y


for i in range(len(yValues)):
    print('---------')
    print('predicted Wins:' + str(yValues[i]) + '  True Wins:' + str(newY[i]))

