import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
from models.SimpleModel import SimpleModel
from models.BaseModel import BaseModel

from matplotlib import pyplot as plt



def load_data():
    shapes = os.listdir('data')

    data = None

    for shape in shapes:
        files = os.listdir(os.path.join('data', shape))
        for file in files:
            data_path = os.path.join('data', shape, file)
            label = int(shape[0])
            unlabeled = np.loadtxt(data_path, delimiter=',')
            labeled = np.hstack((unlabeled, label * np.ones(shape=(unlabeled.shape[0], 1))))
            data = labeled if data is None else np.vstack((data, labeled))

    return data

def main():
    # prepare data
    data = load_data()
    X_train, X_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=.20)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # define the model
    model = SimpleModel()
    model_name = "fist_model"
    
    # train the model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    
    n_epochs = 100
    batch_size = 100

    training_loss = []
    validation_loss = []
    
    for epoch in range(n_epochs):
        for i in range(0, X_train.shape[0], batch_size):
            Xbatch = X_train[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y_train[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finished epoch {epoch}, latest loss {loss}')
        training_loss.append(loss.item())
        # breakpoint()
        with torch.no_grad():
            validation_loss.append(loss_fn(model(X_test), y_test).item())
    
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.argmax(y_pred, axis=1)

    

    accuracy = torch.sum(y_pred == y_test) / y_pred.shape[0]
    print(f"Accuracy: {accuracy:.2f}")

    torch.save(model.state_dict(), os.path.join("models", "saved_models", model_name))

    # plt.plot(training_loss, label='train_loss')
    # plt.plot(validation_loss,label='val_loss')
    # plt.legend()
    # plt.show()
if __name__ == "__main__":
    main()