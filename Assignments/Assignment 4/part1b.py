import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, sgd
from torch.utils.data import DataLoader, Dataset
import os
from matplotlib import pyplot as plt
import numpy as np
import time
import pandas as pd
from utils import *
import sys
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dataset_dir_path = sys.argv[1]
path_to_synth_images = dataset_dir_path + '/SyntheticData/images'
path_to_synth_test = dataset_dir_path + '/SyntheticData/test.csv'
path_to_synth_train = dataset_dir_path + '/SyntheticData/train.csv'
path_to_synth_val = dataset_dir_path + '/SyntheticData/val.csv'
path_to_hw_images = dataset_dir_path + '/HandwrittenData/images'
path_to_hw_train = dataset_dir_path + '/HandwrittenData/train_hw.csv'
path_to_hw_val = dataset_dir_path + '/HandwrittenData/val_hw.csv'

vocab_train = make_vocabulary([path_to_synth_train,path_to_hw_train])
vocab_test = make_vocabulary([path_to_synth_test])
vocab_val = make_vocabulary([path_to_synth_val])

test_vocab_inverse = {v:k for (k,v) in vocab_test.items()}
train_vocab_inverse = {v:k for (k,v) in vocab_train.items()}
val_vocab_inverse = {v:k for (k,v) in vocab_val.items()}

t = time.time()
train_data_load = load_data(path_to_synth_images,path_to_synth_train,vocab_train)
train_data2 = image_latex_dataset2(train_data_load[0],train_data_load[1],train_data_load[2],train_data_load[3])
print(f'Train data loaded in : {time.time() -t:.3}s')

t = time.time()
test_data_load = load_data(path_to_synth_images,path_to_synth_test,vocab_test)
test_data2 = image_latex_dataset2(test_data_load[0],test_data_load[1],test_data_load[2],test_data_load[3])
print(f'Test data loaded in : {time.time() -t:.3}s')

t = time.time()
val_data_load = load_data(path_to_synth_images,path_to_synth_val,vocab_val)
val_data2 = image_latex_dataset2(val_data_load[0],val_data_load[1],val_data_load[2],val_data_load[3])
print(f'Val data loaded in : {time.time() -t:.3}s')

t = time.time()
train_data_loadHw = load_data(path_to_hw_images+'/train',path_to_hw_train,vocab_train)
train_dataHw = image_latex_dataset2(train_data_loadHw[0],train_data_loadHw[1],train_data_loadHw[2],train_data_loadHw[3])
print(f'Handwritten Training data loaded in : {time.time() -t:.3}s')

train_loader_hw = DataLoader(train_dataHw, batch_size = 128, shuffle = True)
test_loader = DataLoader(test_data2, batch_size=128, shuffle=True)
train_loader = DataLoader(train_data2, batch_size=256, shuffle=True)
val_loader = DataLoader(val_data2, batch_size=128, shuffle=True)
print("loaders set")

class Encoder(nn.Module):
    def __init__(self, embed_size):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels= 32, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 32, out_channels= 64, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 64, out_channels= 128, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 128, out_channels= 256, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),

            nn.Conv2d(in_channels = 256, out_channels= 512, kernel_size = 5),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = 2),
            
            nn.AvgPool2d(kernel_size = 3),
        )

    def forward(self, image):
        # get the 3D feature map from image
        feature_map = (self.cnn(image).view(image.size(0), -1))
        return feature_map
    
class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers, vocabulary, max_seq_length=128 ):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2 , hidden_size, num_layers = num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.max_seg_length = max_seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab = dict(vocabulary)
        
    def forward(self, encoder_outputs,target_tensor=None, teacher_forcing_prob = 0.5):
        # if target_tensor is None then prediction mode
        #teacher_forcing_prob : if prob <x then use teacher forcing
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.vocab["<SOS>"])
        decoder_hidden = (encoder_outputs.view(1,encoder_outputs.shape[0],encoder_outputs.shape[1]),encoder_outputs.view(1,encoder_outputs.shape[0],encoder_outputs.shape[1]))
        decoder_outputs = []

        for i in range(self.max_seg_length):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            if torch.rand(1).item() > teacher_forcing_prob :
                    # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input
            else:
                decoder_input = target_tensor[:, i].unsqueeze(1)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = self.softmax(decoder_outputs)
        return decoder_outputs, decoder_hidden, None
    

    def forward_step(self, input, hidden, encoder_repr):
        output = self.embedding(input)
        output = torch.cat((encoder_repr.unsqueeze(1), output), dim=-1)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden
    

class LatexNet2(nn.Module):

    def __init__(self,hidden_size, vocab_size, num_layers, vocabulary, max_seq_length=128):
        #hidden_size == embed_size
        super(LatexNet2, self).__init__()
        self.encoder = Encoder(hidden_size)
        self.decoder = DecoderLSTM(hidden_size, vocab_size, num_layers, vocabulary, max_seq_length)
    
    def forward(self, image, formula = None, teacher_forcing_prob = 0.5):
        encoder_outputs = self.encoder(image)
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, formula, teacher_forcing_prob)
        return decoder_outputs, decoder_hidden, encoder_outputs
    def predict(self,image):
        encoder_outputs = self.encoder(image)
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, teacher_forcing_prob = 0)
        return decoder_outputs.argmax(-1), decoder_outputs
    
def get_val_score(model, val_loader):
    predict_val = []
    true_out_val = []
    img_c_val = []
    for data in val_loader:
        decoder_outputs = model.predict(data[0])[0]
        for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
            s = []
            img_c_val.append(img)
            for i in sent:
                if train_vocab_inverse[i.item()] == "<EOS>":
                    break
                s.append(train_vocab_inverse[i.item()])
            predict_val.append(' '.join(s))
            s = []
            for i in true_sent:
                if val_vocab_inverse[i.item()] == "<EOS>":
                    break
                s.append(val_vocab_inverse[i.item()])
            true_out_val.append(' '.join(s[1:]))

    return sbleu(true_out_val, predict_val)

#training loop
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
path_pre = f'LatexNetSaved_{dt_string}'
os.mkdir(path_pre)
prev_loss = float('inf')
print("Lol")
stochastic_losses = []
best_val_score = 0
epoch_val_score = []
best_model = None
model = LatexNet2(512, len(vocab_train), 1, vocab_train)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss(ignore_index = 0)
model.to(device)
model.train()
print("Starting Training")
PRE_TRAIN_EPOCH_MAX = 30
for epoch in range(1, PRE_TRAIN_EPOCH_MAX +1):
    model.train()
    epoch_train_loss = []
    epoch_time = time.time()
    for i, (imgs, labls, lenghts) in enumerate(train_loader):
        imgs = imgs.to(device)
        labls = labls.long().to(device)
        optimizer.zero_grad()
        decoderOutput, decoderHidden, encoderOutput = model(imgs, labls)
        loss = criterion(decoderOutput[:,:-1,:].reshape(-1, len(vocab_train)), labls[:,1:].long().reshape(-1))
        loss.backward()
        epoch_train_loss.append(loss.item())
        stochastic_losses.append(loss.item())
        optimizer.step()
        if i % 100 == 0:
            plt.figure(figsize=(30,18))
            plt.plot(stochastic_losses)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend(["Training Loss"])
            plt.savefig("StochasticLoss_pre_nc.png")
            plt.close()
            print("Epoch: {} Iteration: {} Loss: {}".format(epoch, i, loss.item()))
    torch.save(model.state_dict, os.path.join(path_pre,f'model_train_{epoch}.pth'))
    print("Epoch: {} Loss: {} Time :{}".format(epoch, np.mean(epoch_train_loss), time.time() -epoch_time))
    print("Calculating Validation Loss:")
    t = time.time()
    val_score = get_val_score(model, val_loader)
    print(f'Validation loss calculated in {time.time() -t:.3}s : {val_score:.3f}')
    print()
    if val_score > best_val_score:
        best_model = model.state_dict
        best_val_score = val_score
    epoch_val_score.append(val_score)

t = time.time()
vocab_val_hw  = make_vocabulary([path_to_hw_val])
vocab_val_hw_inv = {h:k for (h,k) in vocab_val_hw.items()}
val_data_load_hw = load_data(path_to_hw_images+'/train',path_to_hw_val,vocab_train)
print(f'Handwritten Training data loaded in : {time.time() -t:.3}s')
val_dataHW = image_latex_dataset2(val_data_load_hw[0],val_data_load_hw[1],val_data_load_hw[2],val_data_load_hw[3])

val_hw_loader = DataLoader(val_dataHW, batch_size=128, shuffle=True)
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
path_fine = f'LatexNet_fine_tune_{dt_string}'
os.mkdir(path_pre)
prev_loss = float('inf')
print("Lol")
stochastic_losses_fine = []
best_val_score = 0
epoch_val_score_fine = []
best_model = None
model = LatexNet2(512, len(vocab_train), 1, vocab_train)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss(ignore_index = 0)
PRE_TRAIN_EPOCH = PRE_TRAIN_EPOCH_MAX #select pre train epoch using validation
model.load_state_dict(torch.load(os.path.join(path_pre,f'model_train_{PRE_TRAIN_EPOCH}.pth')))
model.to(device)
model.train()

print("Starting Fine Tuning")
FINE_TRAIN_EPOCH_MAX = 30 
for epoch in range(1, FINE_TRAIN_EPOCH_MAX +1):
    model.train()
    epoch_train_loss = []
    epoch_time = time.time()
    for i, (imgs, labls, lenghts) in enumerate(train_loader_hw):
        imgs = imgs.to(device)
        labls = labls.long().to(device)
        optimizer.zero_grad()
        decoderOutput, decoderHidden, encoderOutput = model(imgs, labls)
        loss = criterion(decoderOutput[:,:-1,:].reshape(-1, len(vocab_train)), labls[:,1:].long().reshape(-1))
        loss.backward()
        epoch_train_loss.append(loss.item())
        stochastic_losses.append(loss.item())
        optimizer.step()
        if i % 100 == 0:
            plt.figure(figsize=(30,18))
            plt.plot(stochastic_losses_fine)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend(["Training Loss"])
            plt.savefig("StochasticLoss_pre_fine.png")
            plt.close()
            print("Epoch: {} Iteration: {} Loss: {}".format(epoch, i, loss.item()))
    torch.save(model.state_dict, os.path.join(path_fine,f'model_train_{epoch}.pth'))
    print("Epoch: {} Loss: {} Time :{}".format(epoch, np.mean(epoch_train_loss), time.time() -epoch_time))
    print("Calculating Validation Loss:")
    t = time.time()
    val_score = get_val_score(model, val_hw_loader)
    print(f'Validation loss calculated in {time.time() -t:.3}s : {val_score:.3f}')
    print()
    if val_score > best_val_score:
        best_model = model.state_dict
        best_val_score = val_score
    epoch_val_score_fine.append(val_score)

print("fine tuning complete")

model.eval()
model.to(device)
predict_val = []
true_out_val = []
img_c = []
for data in val_loader:
    decoder_outputs = model.predict(data[0])[0]
    for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
        s = []
        img_c.append(img)
        for i in sent:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict_val.append(' '.join(s))
        s = []
        for i in true_sent:
            if val_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(val_vocab_inverse[i.item()])
        true_out_val.append(' '.join(s[1:]))

print("Macro Bleu for Synthetic Val Data set: ", sbleu(true_out_val, predict_val))

predict_test = []
true_out_test = []
img_c = []
for data in test_loader:
    decoder_outputs = model.predict(data[0])[0]
    for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
        s = []
        img_c.append(img)
        for i in sent:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict_test.append(' '.join(s))
        s = []
        for i in true_sent:
            if test_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(test_vocab_inverse[i.item()])
        true_out_test.append(' '.join(s[1:]))

print("Macro Bleu for Synthetic Test Data set: ", sbleu(true_out_test, predict_test))

model.eval()
model.to(device)
predict_val_hw = []
true_out_val_hw = []
img_c = []
for data in val_hw_loader:
    decoder_outputs = model.predict(data[0].to(device))[0]
    for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
        s = []
        img_c.append(img)
        for i in sent:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict_val_hw.append(' '.join(s))
        s = []
        for i in true_sent:
            if vocab_val_hw_inv[i.item()] == "<EOS>":
                break
            s.append(vocab_val_hw_inv[i.item()])
        true_out_val_hw.append(' '.join(s[1:]))

print("Macro Bleu for Handwritten Val Data set: ", sbleu(true_out_val_hw, predict_val_hw))


def load_data_imgs(path_to_images, path_to_csv):
    images = []
    trans = transforms.ToTensor()
    label_csv = pd.read_csv(path_to_csv)
    images = ([(trans(Image.open(os.path.join(path_to_images, fname)).resize((224, 224)))) for fname in label_csv['image']])
    for i in range(len(images)):
        if images[i].shape[0] == 1:
            images[i] = torch.cat([images[i], images[i], images[i]], dim = 0)
    return images
t = time.time()

test_data_load_hw = load_data_imgs(path_to_hw_images+'/train',os.path.join(dataset_dir_path,'sample_sub.csv'))
test_data_load_synth = load_data_imgs(path_to_synth_images,path_to_synth_test)
print(time.time() -t)
testHW_Loader = DataLoader(test_data_load_hw, batch_size=128, shuffle=False)
testSynth_Loader = DataLoader(test_data_load_synth, batch_size=128, shuffle=False)
model.eval()
model.to(device)
model.max_seg_length = 1000
predict_test_hw = []
for data in testHW_Loader:
    decoder_outputs = model.predict(data.to(device))[0]
    for sent in (decoder_outputs):
        s = []
        for i in sent:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict_test_hw.append(' '.join(s))

csv = pd.read_csv(os.path.join(dataset_dir_path,'sample_sub.csv'))
new_dict = {'image_id':[], 'formula':[]}
for i in range(len(predict_test_hw)):
    new_dict['image_id'].append(csv['image'][i])
    new_dict['formula'].append(predict_test_hw[i])
new_pd = pd.DataFrame(new_dict)
new_pd.to_csv('pred1a.csv', index = False)

predict_test_synth = []
for data in testSynth_Loader:
    decoder_outputs = model.predict(data.to(device))[0]
    for sent in (decoder_outputs):
        s = []
        for i in sent:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict_test_synth.append(' '.join(s))

csv = pd.read_csv(path_to_synth_test)
new_dict = {'image_id':[], 'formula':[]}
for i in range(len(predict_test_synth)):
    new_dict['image_id'].append(csv['image'][i])
    new_dict['formula'].append(predict_test_synth[i])
new_pd = pd.DataFrame(new_dict)
new_pd.to_csv('pred1b.csv', index = False)