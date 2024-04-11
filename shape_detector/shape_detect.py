import torch
import os



def classify(pred):
    data_dir = os.path.abspath(os.path.join(__file__, os.pardir, "data"))
    labels =  os.listdir(data_dir)
    labels.sort()
    idx = torch.argmax(pred).item()

    with torch.no_grad():
        p = torch.softmax(pred, dim=0)

    shape = labels[idx][2:] if p[idx] > .9999 else "none"
    return shape