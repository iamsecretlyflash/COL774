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
train_data_loadHw = load_data('col_774_A4_2023/HandwrittenData/images/train','col_774_A4_2023/HandwrittenData/train_hw.csv',vocab_train)
train_dataHw = image_latex_dataset2(train_data_loadHw[0],train_data_loadHw[1],train_data_loadHw[2],train_data_loadHw[3])
print(f'Handwritten Training data loaded in : {time.time() -t:.3}s')

train_loader_hw = DataLoader(train_dataHw, batch_size = 128, shuffle = True)
test_loader2 = DataLoader(test_data2, batch_size=128, shuffle=True)
train_loader2 = DataLoader(train_data2, batch_size=256, shuffle=True)
val_loader2 = DataLoader(val_data2, batch_size=128, shuffle=True)
print("loaders set")

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.cnn = torch.load('resnest50.model') #assuming resnest is availble in local path

    def forward(self, image):
        # get the 3D feature map from image
        feature_map = (self.cnn(image))
        return feature_map
    
class AttentionUnit(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionUnit, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)
        return context, weights

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers, vocabulary, max_seq_length=128):

        super(AttnDecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 3 , hidden_size, num_layers = num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.relu = nn.ReLU()
        self.attention = AttentionUnit(hidden_size)

        self.max_seg_length = max_seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.vocab = dict(vocabulary)

    def forward(self, encoder_outputs, target_tensor=None, teacher_forcing_prob = 0.5):
        batch_size = encoder_outputs.size(0)
        lstm_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(self.vocab["<SOS>"])
        hidden_state = (encoder_outputs.view(1,encoder_outputs.shape[0],encoder_outputs.shape[1]),encoder_outputs.view(1,encoder_outputs.shape[0],encoder_outputs.shape[1]))
        decoder_outputs = []
        attentions = []

        for i in range(self.max_seg_length):
            decoder_output, hidden_state, attn_weights = self.forward_step(
                lstm_input, hidden_state, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if torch.rand(1).item() > teacher_forcing_prob :
                # Teacher forcing: Feed the target as the next input
                lstm_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                lstm_input = topi.squeeze(-1).detach()  # detach from history as input
        
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, hidden_state, attentions


    def forward_step(self, input, hidden, encoder_repr):
        embedded =  (self.embedding(input))
        query = hidden[0].permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_repr.unsqueeze(1))
        input_lstm = torch.cat((embedded, context), dim=-1)
        input_lstm = torch.concat((encoder_repr.unsqueeze(1), input_lstm), dim=-1)
        output, hidden = self.lstm(input_lstm, hidden)
        output = self.out(output)
        return output, hidden, attn_weights
    
    
class LatexNet_ResNest(nn.Module):

    def __init__(self,hidden_size, vocab_size, num_layers, vocabulary, max_seq_length=128):
        #hidden_size == embed_size
        super(LatexNet_ResNest, self).__init__()
        self.encoder = Encoder()
        self.decoder = AttnDecoderLSTM(hidden_size, vocab_size, num_layers, vocabulary, max_seq_length)
    
    def forward(self, image, formula = None, teacher_forcing_prob = 0.5):
        encoder_outputs = self.encoder(image)
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, formula, teacher_forcing_prob)
        return decoder_outputs, decoder_hidden, encoder_outputs
    def predict(self,image):
        encoder_outputs = self.encoder(image.to(device))
        decoder_outputs, decoder_hidden, _ = self.decoder(encoder_outputs, teacher_forcing_prob = 0)
        return decoder_outputs.argmax(-1), decoder_outputs
        
# TRAINING
model = LatexNet_ResNest(1000, len(vocab_train), 1, vocab_train)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index = 0)
model.to(device)
model.train()

#training loop
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
path_pre_train_models = f'LatexNet_ResNest_pre_training{dt_string}'
os.mkdir(path_pre_train_models)
prev_loss = float('inf')
stochastic_losses_pre = []
use_adaptive_tf = False
print(f"Saving models to {path_pre_train_models}")
teacher_forcing_prob = 0.5
PRE_TRAIN_EPOCH_MAX = 30
for epoch in range(1,PRE_TRAIN_EPOCH_MAX+1):
    model.train()
    epoch_train_loss = []
    epoch_time = time.time()
    for i, (imgs, labls, lenghts) in enumerate(train_loader2):
        imgs = imgs.to(device)
        labls = labls.long().to(device)
        optimizer.zero_grad()
        decoderOutput, decoderHidden, encoderOutput = model(imgs, labls, teacher_forcing_prob)
        decoderOutput.shape
        loss = criterion(decoderOutput.reshape(-1, len(vocab_train)), labls.long().reshape(-1))
        loss.backward()
        epoch_train_loss.append(loss.item())
        stochastic_losses_pre.append(loss.item())
        optimizer.step()
        if i % 100 == 0:
            plt.figure(figsize=(30,18))
            plt.plot(stochastic_losses_pre)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend(["Training Loss"])
            plt.savefig(f"StochasticLoss_res_nest_pre_maxEp_{PRE_TRAIN_EPOCH_MAX}.png")
            print("Epoch: {} Iteration: {} Loss: {}".format(epoch, i, loss.item()))
            plt.close()
    torch.save(model.state_dict, os.path.join(path_pre_train_models,f'model_train_{epoch}.pth'))
    print("Epoch: {} Loss: {} Time :{} | TFP : ".format(epoch, np.mean(epoch_train_loss), time.time() -epoch_time), teacher_forcing_prob)

test_vocab_inverse = {v:k for (k,v) in vocab_test.items()}
train_vocab_inverse = {v:k for (k,v) in vocab_train.items()}
val_vocab_inverse = {v:k for (k,v) in vocab_val.items()}

PRE_TRAIN_MODEL_INDEX = PRE_TRAIN_EPOCH_MAX # Change to select model of suitable epoch

model.load_state_dict(torch.load(os.path.join(path_pre_train_models,f'model_train_{PRE_TRAIN_MODEL_INDEX}.pth')))
model.eval()
model.to(device)
predict_test = []
true_out_test = []
img_c_test = []
for data in test_loader2:
    decoder_outputs = model.forward(data[0].to(device),teacher_forcing_prob =1)[0].argmax(dim = -1)
    for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
        s = []
        img_c_test.append(img)
        for i in sent[1:]:
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

predict_val = []
true_out_val = []
img_c_val = []
for data in val_loader2:
    decoder_outputs = model.forward(data[0].to(device),teacher_forcing_prob =1)[0].argmax(dim = -1)
    for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
        s = []
        img_c_val.append(img)
        for i in sent[1:]:
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
        

print("Macro Bleu for Valdiation Data set: ", sbleu(true_out_val, predict_val))

predict_train = []
true_out_train = []
img_c_train = []
for data in train_loader2:
    decoder_outputs = model.forward(data[0].to(device),teacher_forcing_prob =1)[0].argmax(dim = -1)
    for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
        s = []
        img_c_train.append(img)
        for i in sent[1:]:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict_train.append(' '.join(s))
        s = []
        for i in true_sent:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        true_out_train.append(' '.join(s[1:]))
        
print("Macro Bleu over train: ", sbleu(true_out_train, predict_train))


vocab_val_hw  = make_vocabulary(['col_774_A4_2023/HandwrittenData/val_hw.csv'])
vocab_val_hw_inv = {v:k for (k,v) in vocab_val_hw.items()}
t = time.time()
val_data_load_hw = load_data('col_774_A4_2023/HandwrittenData/images/train','col_774_A4_2023/HandwrittenData/val_hw.csv',vocab_val_hw,128)
val_dataHW = image_latex_dataset2(val_data_load_hw[0],val_data_load_hw[1],val_data_load_hw[2],val_data_load_hw[3])
print(f'Handwritten Validation loaded in : {time.time() -t}')
val_hw_loader = DataLoader(val_dataHW, batch_size=128, shuffle=True)

# Check validation over handwritten data to select best pre-trained model
modelTest = LatexNet_ResNest(1000, len(vocab_train), 1, vocab_train)
for loop_var in range(51,71):
    modelTest.load_state_dict(torch.load(os.path.join(path_pre_train_models,f'model_train_{loop_var}.pth'))())
    modelTest.eval()
    modelTest.to(device)
    predict_hw_val = []
    true_out_hw_val = []
    img_c_hw_val = []
    for data in val_hw_loader:
        decoder_outputs = modelTest.forward(data[0].to(device),teacher_forcing_prob =1)[0].argmax(dim = -1)
        for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
            s = []
            img_c_hw_val.append(img)
            for i in sent[1:]:
                if train_vocab_inverse[i.item()] == "<EOS>":
                    break
                s.append(train_vocab_inverse[i.item()])
            predict_hw_val.append(' '.join(s))
            s = []
            for i in true_sent:
                if vocab_val_hw_inv[i.item()] == "<EOS>":
                    break
                s.append(vocab_val_hw_inv[i.item()])
            true_out_hw_val.append(' '.join(s[1:]))
    print(f"Macro Bleu Over HW Validations Set w/o fine-tuning :Epoch : {loop_var}: ",sbleu(true_out_hw_val, predict_hw_val))


BEST_MODEL_FOR_FINE_TUNE = PRE_TRAIN_EPOCH_MAX # Change to best handwritten validation
#training loop
from datetime import datetime
# datetime object containing current date and time
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
path_hw = f'LatexNet_ResNest_Fine_epoch_{BEST_MODEL_FOR_FINE_TUNE}_{dt_string}'
os.mkdir(path_hw)
prev_loss = float('inf')
stochastic_losses_fine_tune = []
print(f"Saving models to {path_hw}")
teacher_forcing_prob = 0.5
 
LatxResNetFine = LatexNet_ResNest(1000, len(vocab_train), 1, vocab_train)
LatxResNetFine.load_state_dict(torch.load(os.path.join(path_pre_train_models,f'model_train_{BEST_MODEL_FOR_FINE_TUNE}.pth')))

optimizer_hw = Adam(LatxResNetFine.parameters(), lr=0.001)
criterion_hw = nn.CrossEntropyLoss(ignore_index = 0)
LatxResNetFine.to(device)
LatxResNetFine.train()
MAX_FINE_TUNE_EPOCH = 30

for epoch in range(1,MAX_FINE_TUNE_EPOCH+1):
    LatxResNetFine.train()
    epoch_train_loss = []
    epoch_time = time.time()
    for i, (imgs, labls, lenghts) in enumerate(train_loader_hw):
        imgs = imgs.to(device)
        labls = labls.long().to(device)
        optimizer_hw.zero_grad()
        decoderOutput, decoderHidden, encoderOutput = LatxResNetFine(imgs, labls, teacher_forcing_prob)
        decoderOutput.shape
        loss = criterion_hw(decoderOutput.reshape(-1, len(vocab_train)), labls.long().reshape(-1))
        loss.backward()
        epoch_train_loss.append(loss.item())
        stochastic_losses_fine_tune.append(loss.item())
        optimizer_hw.step()
        if i % 100 == 0:
            plt.figure(figsize=(30,18))
            plt.plot(stochastic_losses_fine_tune)
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend(["Training Loss"])
            plt.savefig("StochasticLoss_ResNest_fine_epMax_{MAX_FINE_TUNE_EPOCH}.png")
            print("Epoch: {} Iteration: {} Loss: {}".format(epoch, i, loss.item()))
            plt.close()
    torch.save(LatxResNetFine.state_dict, os.path.join(path_hw,f'model_train_fine_{epoch}.pth'))

    print("Epoch: {} Loss: {} Time :{} | TFP : ".format(epoch, np.mean(epoch_train_loss), time.time() -epoch_time), teacher_forcing_prob)


val_hw_loader = DataLoader(val_dataHW, batch_size=128, shuffle=True)
for loop_var in range(1,MAX_FINE_TUNE_EPOCH+1):
    LatxResNetFine.load_state_dict(torch.load(os.path.join(path_hw,f'model_train_fine_{loop_var}.pth'))())
    LatxResNetFine.eval()
    LatxResNetFine.to(device)
    predict_hw_val = []
    true_out_hw_val = []
    img_c_hw_val = []
    for data in val_hw_loader:
        decoder_outputs = LatxResNetFine.forward(data[0].to(device),teacher_forcing_prob =1)[0].argmax(dim = -1)
        for sent,true_sent,img in zip(decoder_outputs,data[1],data[0]):
            s = []
            img_c_hw_val.append(img)
            for i in sent[1:]:
                if train_vocab_inverse[i.item()] == "<EOS>":
                    break
                s.append(train_vocab_inverse[i.item()])
            predict_hw_val.append(' '.join(s))
            s = []
            for i in true_sent:
                if vocab_val_hw_inv[i.item()] == "<EOS>":
                    break
                s.append(vocab_val_hw_inv[i.item()])
            true_out_hw_val.append(' '.join(s[1:]))

    print(f"Macro Bleu Over HW Validations Set with fine-tuning steps {loop_var}: ", sbleu(true_out_hw_val, predict_hw_val))

# Check validation over handwritten data to select best fine-trained model
BEST_FINE_TUNE_INDEX = MAX_FINE_TUNE_EPOCH # change as desired
LatxResNetFine.load_state_dict(torch.load(os.path.join(path_hw,f'model_train_fine_{BEST_FINE_TUNE_INDEX}.pth'))())

t = time.time()
sub_dataHW = load_data_subHW('col_774_A4_2023/HandwrittenData/images/test','sample_sub.csv')
print(f'Submission Images loaded in : {time.time() -t:.3}s')
testHW_Loader = DataLoader(sub_dataHW, batch_size=128, shuffle=False)
LatxResNetFine.max_seg_length = 1000
predict = []
true_out = []
img_c = []
for data in testHW_Loader:
    decoder_outputs = LatxResNetFine.forward(data.to(device),teacher_forcing_prob =1)[0].argmax(dim = -1)
    for i,sent in enumerate(decoder_outputs):
        s = []
        img_c.append(data[i])
        for i in sent[1:]:
            if train_vocab_inverse[i.item()] == "<EOS>":
                break
            s.append(train_vocab_inverse[i.item()])
        predict.append(' '.join(s))

csv = pd.read_csv(os.path.join(dataset_dir_path,'sample_sub.csv'))
new_dict = {'image':[], 'formula':[]}
for i in range(len(predict)):
    new_dict['image'].append(csv['image'][i])
    new_dict['formula'].append(predict[i])
new_pd = pd.DataFrame(new_dict)
new_pd.to_csv('comp.csv', index = False)