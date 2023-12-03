The assignmnet had competititve and non-competititve parts. The non-comp part used a standard CNN-LSTM encoder decoder architecture which is described in the assingment statement. 

For the competitive part I used ResNeST-50 as my encoder model, and LSTM with Bahdanau Attention as the decoder model. 

To use the code provided in this repo, install [ResNeST-50](https://pytorch.org/hub/pytorch_vision_resnest/) from PyTorch Hub and use torch.save to store the complete model in the same directory. Name the saved model as 'resnest50.model'. Alternatively, you can edit the __init__() module to load ResNeST using torch.hub using the following code:
~~~
import torch
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
model.eval()
~~~

The dataset for the task can be found [here](https://paperswithcode.com/dataset/im2latex-100k) (Due to the size of the dataset, the link does not lead to the exact contest dataset. Will update once resolved). Download the dataset and keep it in the root directory.
