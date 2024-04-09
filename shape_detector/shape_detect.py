import torch
import os



def classify(pred):
    data_dir = os.path.abspath(os.path.join(__file__, os.pardir, "data"))
    labels =  os.listdir(data_dir)
    labels.sort()
    idx = torch.argmax(pred).item()

    shape = labels[idx][2:] if pred[idx] > .8 else "none"
    return shape