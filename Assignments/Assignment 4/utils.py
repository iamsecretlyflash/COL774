import nltk
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time
import pandas as pd
from collections import defaultdict

def sbleu(GT,PRED):
    score = 0
    for i in range(len(GT)):
        Lgt = len(GT[i].split(' '))
        if Lgt > 4 :
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=(0.25,0.25,0.25,0.25),smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        else:
            weight_lst = tuple([1.0/Lgt]*Lgt)
            cscore = nltk.translate.bleu_score.sentence_bleu([GT[i].split(' ')],PRED[i].split(' '),weights=weight_lst,smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method4)
        score += cscore
    return score/(len(GT))

def make_vocabulary(paths):
    vocab = defaultdict(lambda : -1)
    vocab["[PAD]"] = 0
    vocab["<SOS>"] = 1
    vocab["<EOS>"] = 2
    run_index = 3
    for file in paths:
        csv_file = pd.read_csv(file)
        for formula in csv_file['formula'].values:
            formula_split = formula.split()
            for character in formula_split:
                if character not in vocab:
                    vocab[character] = run_index
                    run_index += 1
    return vocab

def load_data(path_to_images, path_to_csv, vocabulary, max_length_formula = 128):
    images = []
    labels = []
    lengths = []
    trans = transforms.ToTensor()
    label_csv = pd.read_csv(path_to_csv)[0:256]
    print(label_csv.__len__())
    label_split = [label_csv.iloc[i]['formula'].split() for i in range(len(label_csv))]
    print(len(label_split))
    x = [True if len(label) <= max_length_formula-2 else False for label in label_split]
    label_csv = label_csv.loc[x]
    label_split2 = label_split
    label_split = []
    for label in label_split2:
        if len(label) <= max_length_formula-2:
            label_split.append(label)
    t = time.time()
    images = ([(trans(Image.open(os.path.join(path_to_images, fname)).resize((224, 224)))) for fname in label_csv['image']])
    print(f"Images done in :{time.time() -t:.3}s")
    lengths = np.array([len(label)+2 for label in label_split])
    labels = np.zeros((len(label_split), max_length_formula))
    labels[:,0] = vocabulary['<SOS>']
    for i in range(len(label_split)):
        labels[i,1:len(label_split[i])+1] = np.array([vocabulary[char] for char in label_split[i]])
        labels[i,len(label_split[i])+1] = vocabulary['<EOS>']
    return images, labels, vocabulary, lengths

class image_latex_dataset2(torch.utils.data.Dataset):

    def __init__(self, images, labels, vocabulary, lengths, max_length_formula = 128):
        self.images, self.labels, self.vocabulary, self.lengths = images, labels, vocabulary, lengths
        self.inverse_vocabulary = {v:k for k,v in self.vocabulary.items()}

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.images[idx].shape[0] == 1:
            self.images[idx] = torch.cat((self.images[idx], self.images[idx], self.images[idx]), dim=0)
        return self.images[idx], self.labels[idx], self.lengths[idx]
    
def load_data_subHW(path_to_images, path_to_csv):
    images = []
    index = []
    trans = transforms.ToTensor()
    label_csv = pd.read_csv(path_to_csv)
    images = ([(trans(Image.open(os.path.join(path_to_images, fname)).resize((224, 224)))) for fname in label_csv['image']])
    for i in range(len(images)):
        if images[i].shape[0] == 1:
            images[i] = torch.cat([images[i], images[i], images[i]], dim = 0)
    return images
