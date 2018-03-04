from data_prep import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import random


#define LSTM class
class Music(nn.Module):
    def __init__(self,input_size, hidden_size,num_layers, dropout = None):
        super(Music, self).__init__()
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden = self.init_hidden()
        self.hidden2out = nn.Linear(hidden_size,input_size)
        if dropout == None:
            self.drop_out = None
        else:
            self.drop_out = nn.Dropout(dropout)
        
        
    def init_hidden(self):
        h0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).cuda()
        c0 = Variable(torch.zeros(self.num_layers, 1, self.hidden_size)).cuda()
        return h0,c0
    
    def forward(self,input_tensor):
        lstm_out, self.hidden = self.lstm(input_tensor,self.hidden)
        output_tensor = self.hidden2out(lstm_out.view(len(input_tensor),-1))
        if self.drop_out != None:
            return self.drop_out(output_tensor)
        else:
            return output_tensor
    

#sample from one piece of music 
def random_sample(piece, sample_size):
    ind = random.randint(0,len(piece) - sample_size)
    return piece[ind:ind + sample_size]

#make training set and test set
def data_loader(all_data):
    test_set = random.sample(all_data, int(len(all_data)*0.2))
    training_set = []
    for x in all_data:
        if x not in test_set:
            training_set.append(x)
    return training_set, test_set

#convert ascii char to tensor
def ch2tensor(ch_set,ch):
    x = Variable(torch.zeros(1,len(ch_set))).cuda()
    x[0,ch_set.index(ch)] = 1
    return x
    
def train(net, max_epoch, sample_size, learning_rate = 0.2,print_loss = False):
    net.cuda()
    net.train()
    all_data = get_music()
    ch_set = get_ch_set()
    n_letters = len(ch_set)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr = learning_rate)
    counter = 0
    print("Finish Initialization")
    print(net.drop_out)
    
    trainin_loss_container = []
    test_loss_container = []
    
    for epoch in range(max_epoch):
        training_set, test_set = data_loader(all_data)
        epoch_loss = 0
#         average_loss_container = []
        net.train()
        for piece in training_set:
            if len(piece) > sample_size:
                #prepare 
                net.zero_grad()
                net.hidden = net.init_hidden()
                training_sample = random_sample(piece, sample_size)
                training_in = Variable(torch.from_numpy(input_vector(ch_set,training_sample)).view(len(training_sample) - 1,1,n_letters).type(torch.FloatTensor)).cuda()
                training_target = Variable(torch.from_numpy(target_vector(ch_set,training_sample)).view(len(training_sample) - 1).type(torch.LongTensor)).cuda()
                net.hidden = net.init_hidden()
                
                #training
                net_pred = net(training_in)
#                 print([training_pred.data.size(), training_target.data.size()])
                training_loss = loss_func(net_pred, training_target)
                counter += 1
                training_loss.backward()
                optimizer.step()
                epoch_loss += training_loss.data[0]
    
        trainin_loss_container.append(epoch_loss / len(training_set))
        test_loss_container.append(test_loss(net, test_set, sample_size))
        
        print("Epoch[%d] Training Loss : %s\tTest Loss : %s "%(epoch,trainin_loss_container[-1], test_loss_container[-1]))
    return trainin_loss_container,test_loss_container
        


def test_loss(net, test_set, sample_size):
    loss_func = nn.CrossEntropyLoss()
    epoch_loss = 0
    net.eval()
    ch_set = get_ch_set()
    n_letters = len(ch_set)
    for piece in test_set:
        if len(piece) > sample_size:
            test_sample = random_sample(piece, sample_size)
            test_in = Variable(torch.from_numpy(input_vector(ch_set,test_sample)).view(len(test_sample) - 1,1,n_letters).type(torch.FloatTensor)).cuda()
            #test mode
            net.zero_grad()
            net.hidden = net.init_hidden()
            test_target = Variable(torch.from_numpy(target_vector(ch_set,test_sample)).view(len(test_sample) - 1).type(torch.LongTensor)).cuda()
            net_pred = net(test_in)
            test_loss = loss_func(net_pred, test_target)
            epoch_loss += test_loss.data[0]
    return epoch_loss / len(test_set)
        
        

def compose(net, piece_size, T):
    ch_set = get_ch_set()
    net.eval()
    net.cuda()
#     activation_container = []
    st = 'X'
    n_letters = len(ch_set)
    x = ch2tensor(ch_set,st)
    net.hidden = net.init_hidden()
    for i in range(piece_size):
        pred = net(x.view(1,1,len(ch_set)))
        pred_ch = ch_set[np.random.choice(np.arange(n_letters),p = F.softmax(pred / T).cpu().data.numpy()[0])]
        x = ch2tensor(ch_set,pred_ch)
        st += pred_ch
        del pred
    return st


def heatmap(net, piece_size, T):
    ch_set = get_ch_set()
    net.eval()
    net.cuda()
    st = 'X'
    n_letters = len(ch_set)
    x = ch2tensor(ch_set,st)
    activation_container = []
    net.hidden = net.init_hidden()
    for i in range(piece_size):
        lstm_out, net.hidden = net.lstm(x.view(1,1,len(ch_set)), net.hidden)
        
        #add to container
        #activation_container.append(lstm_out)
        pred = net.hidden2out(lstm_out.view(len(x.view(1,1,len(ch_set))),-1))
        activation_container.append(pred)
        
        #focus on the firs neuron
        pred_ch = ch_set[np.random.choice(np.arange(n_letters),p = F.softmax(pred / T).cpu().data.numpy()[0])]
        x = ch2tensor(ch_set,pred_ch)
        st += pred_ch
    return st,activation_container
