
import chainer
from kaldi.io import KaldiArk,KaldiScp
from kaldi.commands import KaldiCommand
from kaldi.data import load_nnet, apply_cmvn, splice, load_labeled_data
import numpy as np


def load_labeled_data(ark,pdf,offsets=[0], bias=None,scale=None, folding=True):
    X=[]
    L=[]
    for key in ark:
        data=ark[key]
        dif= data.shape[0]- pdf[key].shape[0]
        L.append(np.asarray(np.r_[pdf[key], np.ones(dif)*pdf[key][-1]],dtype=np.int32))
        for x in splice(data,offsets,folding):
            if bias is not None:
                x += bias[0]
            if scale is not None:
                x *= scale[0]
            X.append(x)
    return np.vstack(X), np.hstack(L)

class KaldiDataset(chainer.dataset.DatasetMixin):
    def __init__(self, feats_dir, pdf_dir, offsets=[0], nnet_transf=None, bias=None, scale=None, cmvn=None, folding=True):
        """
        Load the TIMIT frames with their labels from directory
        Each frame is concatinated with its neighbor frames as described in nnet_transf.
        
        When pdf_dir is given, this function returns:
          feature matrix, labels
        Otherwise, this function returns:
          feature matrix, {utt_id: (s_frame, e_frame)}
        """
        if nnet_transf is not None:
            for layer in load_nnet(nnet_transf):
                if layer['func'] == 'Splice':
                    offsets = layer['context']
                elif layer['func'] == 'AddShift':
                    bias = layer['b']
                elif layer['func'] == 'Rescale':
                    scale=layer['b']
        assert cmvn==None or 'global' or 'utterance' or 'speaker', 'Unknown cmvn option: %s' % cmvn
        if cmvn is not None:
            if cmvn=='global':
                feats = apply_cmvn(feats_dir+'/feats.scp', utt2spk='global',var_norm=True)
            elif cmvn=='utterance':
                feats = apply_cmvn(feats_dir+'/feats.scp', var_norm=True)
            elif cmvn=='speaker':
                feats = apply_cmvn(feats_dir+'/feats.scp', utt2spk=feats_dir+'/utt2spk',var_norm=True)
            else:
                logger.error('Unknown cmvn option: %s' % cmvn)
        else:
            feats = {key:v for key ,v in KaldiCommand('featbin/copy-feats') << feats_dir}
        pdf = {key:v[0] for key ,v in KaldiCommand('featbin/copy-feats') << pdf_dir}

        self.train_x, self.train_y \
            = load_labeled_data(feats, pdf, offsets, bias, scale, folding=True)
        self.data_size = self.train_x.shape[0]

    def __len__(self):
        return self.data_size

    def get_example(self, i):
        return self.train_x[i], self.train_y[i]
