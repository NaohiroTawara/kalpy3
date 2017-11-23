
import chainer
from kaldi.io import KaldiArk,KaldiScp
from kaldi.commands import KaldiCommand
from kaldi.data import load_nnet, apply_cmvn, splice, load_labeled_data
import numpy as np

#def load_labeled_data_kaldi(dirname,offsets=[0], bias=None,scale=None, folding=True):
#    return x, np.asarray(y, dtype=np.int32)

class LabeledKaldiDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dirname, offsets=[0], nnet_transf=None, bias=None, scale=None, cmvn=None, folding=True):
        scp=KaldiScp(dirname+'/feats.scp')
        ark=KaldiArk(dirname+'/ali.ark')
        feats={key:value for key,value in scp}
        pdf  ={key:value for key,value in ark}
        x, y =load_labeled_data(feats, pdf, offsets, bias, scale, folding)
        self.x = x
        self.y = np.asarray(y, dtype=np.int32)[0,:]
        self.data_size=len(self.x)

    def __len__(self):
        return self.data_size

    def get_example(self, i):
        return self.x[i], self.y[i]

class LabeledKaldiSegmentDataset(chainer.dataset.DatasetMixin):
    def __init__(self, dirname, offsets=[0], nnet_transf=None, bias=None, scale=None, cmvn=None, folding=True):

        scp=KaldiScp(dirname+'/feats.scp')
        ark=KaldiArk(dirname+'/ali.ark')
        self.x = [value for key,value in scp]
        self.y = [np.asarray(value, dtype=np.int32)[0,:] for key,value in ark]
        self.segments = np.asarray(x)
        self.y = np.asarray(y)
        self.data_size=len(self.segments)

    def __len__(self):
        return self.data_size

    def get_example(self, i):
        return self.segments[i], self.y[i]
