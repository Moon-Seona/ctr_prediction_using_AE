import torch
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear
import numpy as np
import torch.nn.functional as F

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, vocab_size, embed_dim):
        super().__init__()

        self.char_embedding_dim = embed_dim
        self.embed_dim = embed_dim
        self.output_dim = 1

        # exclude three
        field_dims = np.hstack((field_dims[0:2],field_dims[4:6], field_dims[12], field_dims[17:21], field_dims[26:]))

        # for linear
        self.fc = torch.nn.Embedding(sum(field_dims), self.output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((self.output_dim,)))
        
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm = FactorizationMachine(reduce_sum=True) # 32 x 23 x 16

        self.rnn = torch.nn.LSTM(self.char_embedding_dim, embed_dim, num_layers=4, batch_first=True) #num_layers=1
        self.char_embedding = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = self.char_embedding_dim) #vocab_size+2

        self.linear = FeaturesLinear(field_dims)

        # when use lstm
        self.embed_output_dim = (len(field_dims)) * embed_dim
        #self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        #self.init_weights() error


    def forward(self, x, additional, column):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = torch.cat((x[:,0:2],x[:,4:6], x[:,12:13], x[:,17:21], x[:,26:]), dim=1)
        
        feat_embedding = self.embedding(x) # batch x 21 x embed_dim
        
        linear_x = (torch.sum(torch.sum(feat_embedding, dim=2), dim=1) + self.bias).unsqueeze(1)

        # when no use ifa and ip, modify self.embed_output_dim in __init__
        x = linear_x + self.fm(feat_embedding) #+ self.mlp(embed_x.view(-1, self.embed_output_dim)) # batch x embed_dim + batch_size x 1
        
        return torch.sigmoid(x.squeeze(1))
