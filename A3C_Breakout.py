
# Import Packages
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initializing and setting the variance of a tensor of weights
def normalized_columns_initializer(weights, std = 1.0):
    # initialize a torch tensor with random weights that follow a normal distribution
    out = torch.randn(weights.size())
    # set the std, normalize the torch tensor of weights; squared sum of the weights of our vector
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out)) # var(out) = std^2
    return out

# Initializing the weights of the neural netowrk for an optimal learning
def weights_init(m):
    classname = m.__class__.__name__
    # intiailiztation of the convolution connections
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4]) # dim1 * dim2 * dim3
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0] # dim0 * dim2 * dim3
        w_bound = np.sqrt(6. / fan_in + fan_out) # size of the tensor of weights 
        m.weight.data.uniform_(-w_bound, w_bound) # generate random weights that are inversely proportional to the size of tensor of weights
        m.bias.data.fill_(0) # initialize the tensor bias with zeros
    # intiailiztation of the full connections
    elif classname.find('Linear') != -1:
        weights_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / fan_in + fan_out)
        m.bias.data.fill_(0)
    
# Making the A3C brain
class ActorCritic(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        # simple convolutional architecture : 32 feature detectors, size 3 X 3, stride of 2, padding of 1
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        # learn the temporal properties of the input image, if the ball hits a brick the lstm will encode the bounce, records the temporal dependencies t, t-1, t-2, t-3, t-n
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1) # output = V(S)
        self.actor_linear = nn.Linear(256, num_outputs) # output = Q(S,A)
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data, 1)
        self.critic_linear.bias.data.fill_(0)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()

    def forward(self, inputs): # inputs also consider input and hidden nodes of the LSTM
        inputs, (hx, cx) = inputs # hx = hidden states; cx = cell states
        x = F.elu(self.conv1(inputs)) # propagates signal from input images to first layer, and activates nonlinear neurons of different layers - elu = exponential linear unit
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x)) # Output signal of layer 4
        x = x.view(-1, 32 * 3 * 3) # flattened vector: -1 = flattened one-dimenaional vector, second arguement is number of elements in this vector
        (hx, cx) = self.lstm(x, (hx, cx)) # flattened output vector after 4 convolutional networks for lstm; output hidden nodes and cell nodes
        x = hx
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)










