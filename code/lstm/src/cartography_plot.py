import os
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
import argparse


def load_logits(dir_path: str):
    file_list = os.listdir(dir_path)
    file_list.sort()
    print("Loading files in:", dir_path)
    labels, idxs, logits = [], [], []
    output_csv_name = None
    for file_name in file_list:
        file_path = f"{dir_path}/{file_name}"
        print(file_path)
        if "idxs" in file_path:
            idxs.append(np.array(torch.load(file_path)))
        elif "logits" in file_path:
            logits.append(np.array(torch.load(file_path)))
        elif "labels" in file_path:
            labels.append(np.array(torch.load(file_path)))
        else:
            output_csv_name = file_path
    logits_ordered = np.zeros(np.array(logits).shape)
    #true_labels = np.zeros(np.array(labels).shape)
    
    for epoch in range(len(idxs)):
      # print(idxs[epoch])
      logits_ordered[epoch][idxs[epoch]] = logits[epoch]

      # true_labels[epoch][idxs[epoch]] = labels[epoch]
    
    return logits_ordered, labels, idxs, output_csv_name


def cartography(logits, true_labels):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predictions = probs.numpy()
    
    corr_probs = np.sum(predictions * np.expand_dims(torch.nn.functional.one_hot(true_labels, num_classes=2).numpy(), axis=0), axis=-1)
    
    confs = np.mean(corr_probs[0], axis=0)
    
    variabilities = np.std(corr_probs[0], axis=0)
    # print(len(variabilities))
    return confs, variabilities


def correctness(logits, true_labels):
  crrctnss = []
  epochs = len(true_labels)

  probs = torch.nn.functional.softmax(logits, dim=-1)
  predictions = probs.numpy()
  
  corr_probs = np.sum(predictions * np.expand_dims(torch.nn.functional.one_hot(true_labels, num_classes=2).numpy(), axis=0), axis=-1)

  y = torch.argmax(probs, dim=-1)
  # print(y)
  for i in range(len(y[0])):
    correct = 0
    for j in range(len(y)):
      if y[j][i] == true_labels[j][i]:
        correct += 1
    crrctnss.append(correct/epochs)

  return crrctnss

def create_plot_dataset(path, conf, vari, crrectness):
	print(path)
	df = pd.read_csv(path)
	new_df = pd.DataFrame({
	    'Confidence': conf,
	    'Variability': vari,
	    'Correctness': crrectness,
	    'idx': range(0, len(conf))
	}, columns= ['Confidence', 'Variability', 'Correctness','idx'])
	cartography = pd.merge(df, new_df, on="idx")
	return cartography

'''
def plot_old(x, y):
	# plotting points as a scatter plot
	plt.scatter(x, y, label= "stars", color= "green",
	      marker= ".", s=30)
	
	plt.xlim(0, 0.5)
	plt.ylim(0, 1)
	# x-axis label
	plt.xlabel('Variability')
	# frequency label
	plt.ylabel('Confidence')
	# plot title
	plt.title('Cartography Plot for SCAN Simple Split')
	# showing legend
	plt.legend()
	# function to show the plot
	#plt.show()
	# function to save the figurea
	plt.savefig('cartography.png')
'''

def plot(df, path_name, extra_path_info):
	fig = px.scatter(df, x="Variability", y="Confidence", color='Correctness')
	fig.write_image(f"{path_name}/cartography_plot_{extra_path_info}.pdf")
	

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument('-logits_path', required=True, type=str)
	parser.add_argument('-plot_path', required=True, type=str)
	parser.add_argument('-extra_path_info', default="", type=str)
	args = parser.parse_args()
	logits_path = args.logits_path
	plot_path = args.plot_path
	extra_path_info = args.extra_path_info

	logits, labels, idx, output_name = load_logits(logits_path)
	true_labels = torch.Tensor(np.array(labels)).type(torch.int64)
	logits = torch.Tensor(logits)
	conf, vari = cartography(logits, true_labels)
	crrctnss = correctness(logits, true_labels)
	df = create_plot_dataset(output_name, conf, vari, crrctnss)
	plot(df, plot_path, extra_path_info)
	

if __name__ == "__main__":
	main()
