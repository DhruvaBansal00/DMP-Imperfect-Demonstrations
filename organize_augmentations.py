import pickle
import numpy as np
import glob
import tqdm

def save_organized_augmentations(original_augmentations):
    total_agumentations = 10
    for curr_augmentation in range(total_agumentations):
        one_augmentation_all_points = []
        for curr_file in tqdm.tqdm(glob.glob(original_augmentations)):
            if 'rewards' not in curr_file:
                with open(curr_file, 'rb') as curr_state_augmentations:
                    curr_state_augmentation_arr = pickle.load(curr_state_augmentations)
                    one_augmentation_all_points.append(curr_state_augmentation_arr[curr_augmentation])
        pickle.dump(one_augmentation_all_points, open(f'/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/Augmentations/Ant-v2/{curr_augmentation}.pkl', 'wb'))
                    


save_organized_augmentations('/home/dhruva/Desktop/DMP-Imperfect-Demonstrations/Augmentations/Ant-v2-separate/*')