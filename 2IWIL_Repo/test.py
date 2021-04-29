import numpy as np
import random

def add_noise(num):
	return num + random.random()*1e-6 if num < 0.5 else num - random.random()*1e-6


x = np.load('mixedConfidences.npz')
sorted_confidences = []

for i in range(len(x['train_idx'])):
	sorted_confidences.append([x['train_idx'][i], add_noise(x['train_conf'][i])])

for i in range(len(x['val_idx'])):
	sorted_confidences.append([x['val_idx'][i], add_noise(x['val_conf'][i])])

sorted_confidences = np.array(sorted(sorted_confidences, key=lambda x: x[0]))
np.save(open('sorted_confidences.npy', 'wb'), sorted_confidences)
split_sorted_cofidences = [[i[0] for i in sorted_confidences], [i[1] for i in sorted_confidences]]
print(split_sorted_cofidences)
np.save(open('split_sorted_confidences.npy', 'wb'), np.array(split_sorted_cofidences))
