import torch
import torch.nn as nn
import os
import torchvision
import torch.optim as optim

class Partial_Fully_connect(nn.Module):
    def __init__(self, start_channel, end_channel, input_example):
        super(Partial_Fully_connect, self).__init__()
        self.start_channel = start_channel
        self.end_channel = end_channel
        self.dense_num=int(input_example.numel()/input_example.shape[0]/input_example.shape[1])
        self.fc = nn.Linear(self.dense_num, 1)

    def forward(self, x):
        # Slicing the tensor to extract desired channels
        x = x[:, self.start_channel:self.end_channel, :]
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class FC_net(nn.Module):
    def __init__(self, input_example, ratio=16):
        super(FC_net, self).__init__()
        self.fc_1 = nn.Linear(channel_number, channel_number)
        self.fc_2 = nn.Linear(channel_number, 10)
        self.fc_out = nn.Linear(10, 10)
        self.sigmoid = nn.Sigmoid()
        self.channel_number = input_example.shape[1]
        self.CFC = nn.ModuleList([Partial_Fully_connect(i, i + 1, input_example) for i in range(channel_number)])
    def forward(self, x):
        channel_size = x.shape[1]
        channel_outs = []
        for i in range(channel_number):
            channel_out = self.CFC[i](x)
            channel_outs.append(channel_out)
        out = torch.cat(channel_outs, dim=1)
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_out(out)
        return self.sigmoid(out)

#%% 
# Load MNIST data
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024,
                                          shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024,
                                         shuffle=False)

for data in trainloader:
    # Unpack the data
    inputs, labels = data

    # Get the first sample and its label
    first_sample = inputs[0]
    first_label = labels[0]
    break

#%%
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
model = FC_net(first_sample)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criterion and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100  # Number of epochs
load_start=[]
load_end=[]
fit_start=[]
fit_end=[]
print('train start')
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.reshape(inputs.shape[0],inputs.shape[2],inputs.shape[3])
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}")
        running_loss = 0.0
            
print('Finished Training')

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.reshape(images.shape[0],images.shape[2],images.shape[3])
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

