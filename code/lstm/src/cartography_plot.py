import os
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import matplotlib.pyplot as plt 


def load_logits(dir_path: str):
    file_list = os.listdir(dir_path)
    file_list.sort()
    print("Loading files in:", dir_path)
    labels, idxs, logits = [], [], []
    for file_name in file_list:
        file_path = f"{dir_path}/{file_name}"
        if "idxs" in file_path:
            idxs.append(np.array(torch.load(file_path)))
        elif "logits" in file_path:
            logits.append(np.array(torch.load(file_path)))
        elif "labels" in file_path:
            labels.append(np.array(torch.load(file_path)))
        else:
            raise Exception("Wrong Files!")
    logits_ordered = np.zeros(np.array(logits).shape)
    true_labels = np.zeros(np.array(labels).shape)
    
    for epoch in range(len(idxs)):
      # print(idxs[epoch])
      logits_ordered[epoch][idxs[epoch]] = logits[epoch]

      # true_labels[epoch][idxs[epoch]] = labels[epoch]
    
    return logits_ordered, labels


def cartography(logits, true_labels):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predictions = probs.numpy()
    
    corr_probs = np.sum(predictions * np.expand_dims(torch.nn.functional.one_hot(true_labels, num_classes=2).numpy(), axis=0), axis=-1)
    
    confs = np.mean(corr_probs[0], axis=0)
    
    variabilities = np.std(corr_probs[0], axis=0)
    # print(len(variabilities))
    return confs, variabilities


def plot(x, y):
	# plotting points as a scatter plot
	plt.scatter(x, y, label= "stars", color= "green",
	      marker= ".", s=30)
	
	plt.xlim(0, 0.5)
	plt.ylim(0, 1)
	# x-axis label
	plt.xlabel('x - variability')
	# frequency label
	plt.ylabel('y - confidence')
	# plot title
	plt.title('Dataset Cartography')
	# showing legend
	plt.legend()
	# function to show the plot
	#plt.show()
	# function to save the figure
	plt.savefig('cartography.png')

def main():
	# outputs_folder
	logits, labels = load_logits('../outputs_folder/cartographyOut')
	true_labels = a_tensor = torch.Tensor(labels)
	true_labels2 = true_labels.type(torch.int64)
	logits_t = torch.Tensor(logits)
	conf, vari = cartography(logits_t, true_labels2)
	plot(vari, conf)


if __name__ == "__main__":
	main()
