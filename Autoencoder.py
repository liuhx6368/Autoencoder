import sys,os,torch,torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X=np.load(sys.argv[1]+'.npy',allow_pickle=True)
y=np.zeros(len(X))

N=len(X)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(N,16),
                                     nn.Sigmoid(),
                                     nn.Linear(16,2),
                                     nn.Sigmoid()
                                     )
 
    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(nn.Linear(2,16),
                                     nn.Sigmoid(),
                                     nn.Linear(16,N),
                                     nn.Sigmoid()
                                     )
 
    def forward(self, x):
        x = self.decoder(x)
        return x

encoder=Encoder()
decoder=Decoder()
criterion=nn.MSELoss(reduction='mean')
optimizer=optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

num_epochs=100000
best_loss=float('inf')
early_stop_patience=20
counter=0

for epoch in range(num_epochs):
    X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.3,shuffle=False)    
        
    inputs=torch.from_numpy(X_train).float()

    optimizer.zero_grad()

    encoded=encoder(inputs).float()
    decoded=decoder(encoded).float()

    loss=criterion(decoded, inputs)
    loss.backward()
    optimizer.step()

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        inputs=torch.from_numpy(X_val).float()
        encoded=encoder(inputs).float()
        decoded=decoder(encoded).float()
        val_loss=criterion(decoded, inputs)

    val_loss=val_loss**0.5

    if epoch%1000==0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, val_loss))
    
    if val_loss<best_loss:
        best_loss=val_loss
        counter=0
    else:
        counter += 1
        if counter >= early_stop_patience:
            print(f'Early stopping after {epoch+1} epochs without improvement on validation set.')
            break

    encoder.train()
    decoder.train()

print(f'\nBest validation loss:{best_loss}\n')

inputs=torch.from_numpy(X).float()
encoded=encoder(inputs).float()
encoded=encoded.detach().numpy()

np.save(sys.argv[1]+'encoded',encoded)


