import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, r2_score

class PerformanceCalculator:
    def __init__(self):
        self.complete_labels = []
        self.complete_outputs = []

    def extend(self, outputs, labels):
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().tolist()

        self.complete_outputs.extend(outputs)
        self.complete_labels.extend(labels)

    def mse(self):
        np_outputs = np.array(self.complete_outputs)
        np_labels = np.array(self.complete_labels)
        
        calc_mse = []
        for i in range(len(np_outputs[0])):
            calc_mse.append(mean_squared_error(np_labels[:,i], np_outputs[:, i]))

        return calc_mse
    
    def r2(self):
        np_outputs = np.array(self.complete_outputs)
        np_labels = np.array(self.complete_labels)
        
        calc_r2 = []
        for i in range(len(np_outputs[0])):
            calc_r2.append(r2_score(np_labels[:,i], np_outputs[:, i]))

        return calc_r2
    
    def plot(self, path, name="outputs"):
        plt.rcParams['figure.figsize'] = (15,15)
        fig, axs = plt.subplots(2, 2)
        np_outputs = np.array(self.complete_outputs)
        np_labels = np.array(self.complete_labels)
        for i in range(len(np_outputs[0])):
            r = i % 2
            c = (i // 2) % 2

            y, x = np_labels[:,i], np_outputs[:, i]
            axs[r, c].scatter(x, y, alpha=0.1)
            axs[r, c].set_title(f'Command [{i}]')
            axs[r, c].set_xlabel("outputs")
            axs[r, c].set_ylabel("labels")
        
        fig.suptitle(name)
        fig.savefig(os.path.join(path, name+".png"))

    
    def __str__(self) -> str:
        string = "Computed metrics:" + "\n"
        string += "MSE:\t" + str(self.mse()) + "\n"
        string += "R2:\t" + str(self.r2()) + "\n"
        string += "\n"

        return string
    
