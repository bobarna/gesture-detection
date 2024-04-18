import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
from models.SimpleModel import SimpleModel

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
    model_name = "results"

    # track loss and accuracy
    loss_train_values = []
    loss_test_values = []
    acc_train_values = []
    acc_test_values = []

    
    # train the model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.0001)
    
    n_epochs = 50
    batch_size = 100
    
    for epoch in range(n_epochs):
        # trainlosstotal = 0
        # trainacctotal = 0
        for i in range(0, X_train.shape[0], batch_size):
            Xbatch = X_train[i:i+batch_size]
            y_pred = model(Xbatch)
            ybatch = y_train[i:i+batch_size]
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    #         trainlosstotal += loss
    #         trainacctotal += torch.sum(torch.argmax(y_pred, axis=1) == ybatch) / ybatch.shape[0]

    #     # compute accuracy (no_grad is optional)
    #     with torch.no_grad():
    #         trainlosstotal /= X_train.shape[0] // batch_size
    #         trainacctotal /= X_train.shape[0] // batch_size

    #         y_pred = model(X_test)
    #         test_loss = loss_fn(y_pred, y_test)
    #         y_pred = torch.argmax(y_pred, axis=1)
    #         accuracy = torch.sum(y_pred == y_test) / y_pred.shape[0]

    #     # breakpoint()
    #     # add avg values to tracking
    #     loss_train_values.append(trainlosstotal.item())
    #     loss_test_values.append(test_loss.item())
    #     acc_train_values.append(trainacctotal.item())
    #     acc_test_values.append(accuracy.item())
    #     print(f'Finished epoch {epoch}, latest loss {loss}')
    
    # # compute accuracy (no_grad is optional)
    # with torch.no_grad():
    #     y_pred = model(X_test)
    #     y_pred = torch.argmax(y_pred, axis=1)

    accuracy = torch.sum(y_pred == y_test) / y_pred.shape[0]
    print(f"Accuracy: {accuracy:.2f}")

    # torch.save(model.state_dict(), os.path.join("models", "saved_models", model_name))

    # plt.figure(1)
    # plt.subplot(121)
    # plt.plot(loss_train_values, label="Train")
    # plt.plot(loss_test_values, label="Test")
    # plt.title("Loss Plot")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend(loc="upper right")

    # plt.subplot(122)
    # plt.plot(acc_train_values, label="Train")
    # plt.plot(acc_test_values, label="Test")
    # plt.title("Accuracy Plot")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(loc="lower right")
    # plt.show()


if __name__ == "__main__":
    main()