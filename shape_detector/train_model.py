import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split 
from shape_detector.models.SimpleModel import SimpleModel


def load_data():
    pass


def main():
    right_point = np.loadtxt('pointing_right.txt', delimiter=',')
    other = np.loadtxt('other.txt', delimiter=',')

    right_labels = np.ones(shape=(right_point.shape[0], 1))
    other_labels = np.zeros(shape=(other.shape[0], 1))

    right_point = np.hstack((right_point, right_labels))
    other = np.hstack((other, other_labels))

    training_data = np.vstack((right_point, other))
    X_train, X_test, y_train, y_test = train_test_split(training_data[:,:-1], training_data[:,-1], test_size=.20)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # define the model
    model = SimpleModel()
    
    # train the model
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    n_epochs = 250
    batch_size = 100
    
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
    
    # compute accuracy (no_grad is optional)
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.argmax(y_pred, axis=1)

    accuracy = torch.sum(y_pred == y_test) / y_pred.shape[0]
    print(f"Accuracy: {accuracy:.2f}")

    torch.save(model.state_dict(), 'simple_model')

if __name__ == "__main__":
    main()