import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Hidden size:  512
        self.hidden_size = hidden_size  
        
        # Vocab size:  9955
        self.vocab_size = vocab_size 
        
        # Embedding:  Embedding(9955, 300)
        self.embed = nn.Embedding(num_embeddings = vocab_size, 
                                  embedding_dim = embed_size)  
        
        # LSTM:  LSTM(300, 512)
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_layers,
                            batch_first = True)  
        
        # Initializing the output
        # Linear:  Linear(in_features=512, out_features=9955, bias=True)
        self.linear = nn.Linear(in_features = hidden_size, 
                                out_features = vocab_size)
    
    def forward(self, features, captions):
        # Deleting the last column in captions
        # Reshaping from "torch.Size([10, 17])" to "torch.Size([10, 16])"
        captions = captions[:, :-1] 
        
        # 1.1 - Getting the embedding
        # embedding's size: torch.Size([10, 16, 300])
        embedding = self.embed(captions) 
        
        # 1.2 - Concatenating features to embedding
        # features's size: torch.Size([10, 256]) # size of 
        # embedding's size: torch.Size([10, 17, 256]) 
        embedding = torch.cat((features.unsqueeze(dim = 1), embedding), features.unsqueeze(dim = 1): torch.Size([10, 1, 256])
                              dim = 1) 

        # 2. - Running through the the LSTM layer
        # Output's shape:  torch.Size([10, 17, 512]) # hidden[0] shape:  torch.Size([1, 17, 512]) / hidden[1] shape:  torch.Size([1, 17, 512])
        lstm_out, hidden = self.lstm(embedding)  
        
        # 3.1 - Running through the linear layer
        # output's shape: torch.Size([10, 17, 9955])
        outputs = self.linear(lstm_out) 
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # Initializing and empty list for predictions
        predicted_sentence = []
        
        # iterating max_len times
        for index in range(max_len):
            
            # Running through the LSTM layer
            lstm_out, states = self.lstm(inputs, states)

            # Running through the linear layer
            lstm_out = lstm_out.squeeze(1)
            outputs = self.linear(lstm_out)
            
            # Getting the maximum probabilities
            target = outputs.max(1)[1]
            
            # Appending the result into a list
            predicted_sentence.append(target.item())
            
            # Updating the input
            inputs = self.embed(target).unsqueeze(1)
            
        return predicted_sentence