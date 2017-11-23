"""
Functions that wlap KALDI functions.

Author: Naohiro Tawara
Contact: tawara@ttic.edu
Date: 2016
"""

import os
import logging
import numpy as np
import numpy.matlib
import sys
from kaldi.io import KaldiScp, KaldiArk, KaldiNnet
from kaldi.commands import KaldiCommand


logger = logging.getLogger('__name__')
ch=logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

#-----------------------------------------------------------------------------#
#                         TIMIT  DATASET I/O FUNCTIONS                        #
#-----------------------------------------------------------------------------#

def super_splice(X, offsets, t):
    """High speed splice.
    For online data generation.

    # Arguments
        t: id of data.
    """
    N, D = X.shape
    L = len(offsets)
    return np.reshape(X[list(map(lambda x: min(max(0, t+x), N-1), offsets))], (L*D,))

def read_utt2spk(filename):
    utt={}
    with open(filename) as f:
        for line in f.readlines():
            line=line.replace('\n','').split(' ')
            utt[line[0]] = line[1]
    return utt

def read_spk2utt(filename):
    spk={}
    with open(filename) as f:
        for line in f.readlines():
            line=line.replace('\n','').split(' ')
            spk[line[0]] = line[1:]
    return spk

def __get_word_to_i__(labels):
    i_type = 0
    word_to_i_map = {}
    for label in labels:
        if not label in word_to_i_map:
            word_to_i_map[label] = i_type
            i_type += 1
    return word_to_i_map

def load_labeled_data_kaldi(dirname):
    feats={key:value for key,value in KaldiScp(dirname+'/feats.scp')}
    pdf  ={key:value for key,value in KaldiArk(dirname+'/ali.ark')}
    return load_labeled_data(feats, pdf)

def load_labeled_data(ark, ali, offsets=[0], bias=None,scale=None, folding=True):
    X = []
    labels = []
    if type(ali) is not dict:
        ali = {key:vec for key, vec in ali}
    if type(ark) is not dict:
        ark = {key:vec for key, vec in ark}
    for key in ark:
        data=ark[key]
        label=ali[key]
        shortage=data.shape[0]-label.shape[1]
        if shortage != 0:
            logger.warning('Warning: Length missmatch in %s, feats: %d, labels: %d' % (key, data.shape[0], label.shape[1]) )
            assert shortage >= 0, "Error: feats must be longer than labels %d,%d"% (data.shape[0], label.shape[1]) 
            logger.warning('Warning: Missed labels are supplemented with -1')
            label = np.c_[label, np.ones([1,shortage])]
        labels.append(label)
        for x in splice(data, offsets, folding):
            if bias is not None:
                x +=  bias[0]
            if scale is not None:
                x *=  scale[0]
            X.append(x)
    return np.vstack(X), np.hstack(labels)

def load_labeled_data_with_spk(ark, ali, spk, offsets=[0], bias=None,scale=None, folding=True):
    X = []
    labels = []
    speakers=  []
    if type(ali) is not dict:
        ali = {key:vec for key, vec in ali}
    if type(ark) is not dict:
        ark = {key:vec for key, vec in ark}

    speaker_labels = __get_word_to_i__(spk.values())
    for key in ali:
        data=ark[key]
        label=ali[key]
        speaker=speaker_labels[spk[key]]
        #shortage=data.shape[0]-label.shape[1]
        #if shortage != 0:
        #    logger.warning('Warning: Length missmatch in %s, feats: %d, labels: %d' % (key, data.shape[0], label.shape[1]) )
        #    assert shortage >= 0, "Error: feats must be longer than labels %d,%d"% (data.shape[0], label.shape[1]) 
        #    logger.warning('Warning: Missed labels are supplemented with -1')
        #    label = np.c_[label, np.ones([1,shortage])]
        labels.append(label)
        speakers.append([speaker]*len(label))
        for x in splice(data, offsets, folding):
            if bias is not None:
                x +=  bias[0]
            if scale is not None:
                x *=  scale[0]
            X.append(x)
    return np.vstack(X), np.hstack(labels), np.hstack(speakers)


def load_data(ark, offsets=[0], bias=None,scale=None, folding=True):
    X = []
    Nf={}
    s_index =0
    if type(ark) is not dict:
        ark = {key:vec for key, vec in ark}

    for key  in ark:
        data=ark[key]
        for x in splice(data, offsets, folding):
            if bias is not None:
                x += bias[0]
            if scale is not None:
                x *= scale[0]
            X.append (x)
        Nf[key] = (s_index,s_index+data.shape[0]-1)
        s_index +=data.shape[0]

    return  np.vstack(X), Nf

def load_timit_labelled_kaldi(feats_dir, pdf_dir=None, offsets=[0], nnet_transf=None, bias=None, scale=None, cmvn=None, folding=True):
    """
    Load the TIMIT frames with their labels from directory
    Each frame is concatinated with its neighbor frames as described in nnet_transf.

    When pdf_dir is given, this function returns:
       feature matrix, labels
    Otherwise, this function returns
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
        feats = {key:v for key ,v in KaldiScp(feats_dir+'/feats.scp')}
    if pdf_dir is not None:
        train_x, train_y = \
            load_labeled_data(feats, KaldiArk(pdf_dir), offsets, bias, scale, folding=True)
    else:
        train_x, train_y = \
            load_data(feats, offsets, bias, scale, folding=True)

    return train_x, train_y

def load_timit_labelled_kaldi_KaldiCommand(feats_dir, pdf_dir=None, offsets=[0], nnet_transf=None, bias=None,scale=None, cmvn=False, folding=True):
    """
    Load the TIMIT frames with their labels from directory
    Each frame is concatinated with its neighbor frames as described in nnet_transf.

    When pdf_dir is given, this function returns:
       feature matrix, labels
    Otherwise, this function returns
       feature matrix, {utt_id: (s_frame, e_frame)}
    """
    def load_labeled_data(ark, pdf, offsets=[0], bias=None,scale=None, folding=True):
        X = []
        labels = []
        ali = {key:vec for key, vec in pdf}
        for key, data in ark:
            labels.append(ali[key])
            for x in splice(data, offsets, folding):
                if bias is not None:
                    x +=  bias[0]
                if scale is not None:
                    x *=  scale[0]
                X.append(x)
        return np.vstack(X), np.hstack(labels)

    if nnet_transf is not None:
        for layer in load_nnet(nnet_transf):
            if layer['func'] == 'Splice':
                offsets = layer['context']
            elif layer['func'] == 'AddShift':
                bias = layer['b']
            elif layer['func'] == 'Rescale':
                scale=layer['b']

    if cmvn:
        logger.info("Applying CMVN: %s"%  (feats_dir+'/cmvn.scp'))
        assert os.path.exists(feats_dir+'/cmvn.scp'), 'cmvn file is not found: %s' % (feats_dir+'/cmvn.scp')
        feats=KaldiCommand('featbin/apply-cmvn', option='--norm-means=true --norm-vars=true --utt2spk=ark:'+feats_dir+'/utt2spk scp:'+feats_dir+'/cmvn.scp')
    else:
        feats=KaldiCommand('featbin/copy-feats')
    if pdf_dir is not None:
        train_x, train_y = load_labeled_data(feats<<feats_dir+'/feats.scp', KaldiArk(pdf_dir), offsets, bias, scale, folding=True)
    else:
        train_x, train_y = load_data(feats<<feats_dir+'/feats.scp', offsets, bias, scale, folding=True)

    return train_x, train_y

def load_nnet(filename):
    return [layer for layer in KaldiNnet(filename)]

def utt2spk_to_dict(utt2spk):
    mapping={}
    with open(utt2spk) as f:
        for line in f.readlines():
            key, value = line.rstrip().split(' ')
            mapping[key] =value
    return mapping

def spk2utt_to_dict(spk2utt):
    mapping={}
    with open(spk2utt) as f:
        for line in f.readlines():
            key, values = (line.rstrip().split(' ')[0], line.rstrip().split(' ')[1:])
            mapping[key] =values
    return Mapping

def load_timit_labeled_posterior(post_file, pdf_ark):
    """
    Load the TIMIT frames with their labels from .scp file.
    Each frame is concatinated with its neighbor frames
    (e.x. offsets=[-2,-1,0,1,2] then each frame is concatenated with +-2 frames 
     and generating dim*5 dimensional vectors).
    If ratio is less than 1, some labels will be removed for semi-supervised experiment.
    The ratio determines the ratio of labeled data to whole data (unlabeled + labeled).
    In this time, the minimum number of samples in each class will be min_count.
    If foldings_file is given, some phones are folded into specific phone.

    Return: feature matrix, 
    """

    feats=KaldiCommand('featbin/apply-cmvn', option='--norm-means=true --norm-vars=true --utt2spk=ark:'+feats_dir+'/utt2spk scp:'+feats_dir+'/cmvn.scp')
    
    train_x, train_y = load_labeled_data(feats<<feats_dir+'/feats.scp', KaldiArk(pdf_ark), offsets, bias, scale)

        
#    if word_to_i is None:
#        word_to_i = __get_word_to_i__(train_labels)

    # Convert labels to word IDs
#    train_y = np.asarray([word_to_i[label] for label in train_labels],dtype=np.int32)

    return train_x, train_y

#-----------------------------------------------------------------------------#
#                         KALDI COMMAND-LIKE FUNCTIONS                        #
#-----------------------------------------------------------------------------#

def compute_cmvn_stats(filename, utt2spk=None):
    """
    Compute cepstral mean and variance normalization statistics.
    if utt2spk-file  provided, per-speaker; 
    'global' provided, global; 
    otherwise, per-utterance
    ref: compute-cmvn-stats
    """
    cmvn={}
    if utt2spk == 'global':
        for key, value in rspecifier(filename):
            size,dim = value.shape
            if 'global' in cmvn:
                cmvn['global'] += np.c_[np.r_[value.sum(0),size], np.r_[(value**2).sum(0),0]].T
            else:
                cmvn['global'] = np.c_[np.r_[value.sum(0),size], np.r_[(value**2).sum(0),0]].T
                
    elif utt2spk is not None:
        assert os.path.exists(utt2spk), 'cmvn file is not found: %s' % utt2spk
        mapping = utt2spk_to_dict(utt2spk)
        for key, value in rspecifier(filename):
            spkr = mapping[key]
            size,dim = value.shape
            if spkr in cmvn:
                cmvn[spkr] += np.c_[np.r_[value.sum(0),size], np.r_[(value**2).sum(0),0]].T
            else:
                cmvn[spkr] = np.c_[np.r_[value.sum(0), size], np.r_[(value**2).sum(0),0]].T
    else:
        for key, value in rspecifier(filename):
            size,dim = value.shape
            cmvn[key] = np.c_[np.r_[value.mean(0) * size,size], np.r_[(value**2).mean(0) *size,0]].T
    return cmvn

def apply_cmvn(filename, var_norm=False, utt2spk=None):
    """
    Apply cepstral mean and (optionally) variance normalization
    Per-utterance by default, or per-speaker if utt2spk option provided
    ref: apply-cmvn-stats
    """
    cmvn = compute_cmvn_stats(filename, utt2spk)
    if utt2spk == 'global':
        logger.info('Apply CMVN with global statistics')
    elif utt2spk is not None:
        assert os.path.exists(utt2spk), 'cmvn file is not found: %s' % utt2spk
        mapping = utt2spk_to_dict(utt2spk)
        logger.info('Apply CMVN per speaker')
    else:
        logger.info('Apply CMVN per utterance')
    result={}
    for key, value in rspecifier(filename):
        org_key = key
        if utt2spk == 'global':
            key = 'global'
        elif utt2spk is not None:
            key = mapping[key]
        assert key in cmvn, 'there is no key (%s) in cmvn file %s' % (key, filename)
        stats = cmvn[key]
        count = stats[0,-1]
        mean  = stats[0,:-1] / count
        if not var_norm:
            scale  = 1.0
            offset = -mean
        else:
            var = stats[1,:-1] / count - mean*mean
            floor = 1.0e-20
            if (var < floor).sum() >0:
                var.flags.writeable=True
                logger.Warnings('Flooring cepstral variance from %f to %f' % (var, floor))
                var[var < floor] = floor
                var.flags.writeable=False
            scale= 1.0 / np.sqrt(var)
            assert (scale==scale).sum()==scale.shape[0] and (1/scale).sum()!=0.0, \
                'NaN or infinity in cepstral mean/variance computation'
            offset = -(mean*scale)

        if var_norm:
            value = value * scale
        value = value + offset
        result[org_key]=value
    return result

def feat_to_len(filename):
    """
    Get frame-length of each utterance described in pdf.ark
    ref: feat-to-len
    """
    d={key:value.shape[0] for key, value in rspecifier(filename)}
    return d

def feat_to_dim(filename):
    """
    Get dimension of each utterance described in pdf.ark
    ref: feat-to-dim
    """
    d={key:value.shape[1] for key, value in rspecifier(filename)}
    return d


def splice(X, offsets, folding=True):
    """
    Concatinate some frames.
    if folding is activated, head and tail are folded.
    Otherwise, padded by zero.
    """
    XX=[]
    N, D =X.shape
    L=len(offsets)
    if folding:
        r=range(N)
    else:
        X= np.r_[np.zeros([1,D]),X,np.zeros([1,D])]
        N=N+2
        r=range(1,N-1)
    for t in r:
        XX.append(np.reshape(X[list(map(lambda x: min(max(0,t+x), N-1), offsets))], (1, L*D)))
    return np.vstack(XX)


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def rspecifier(filename):
    if os.path.splitext(filename)[-1] == ".ark":
        return KaldiArk(filename)
    elif os.path.splitext(filename)[-1] == ".scp":
        return KaldiScp(filename)

def convert_nnet_to_conf(filename):
    layers_specs =[]
    layers = load_nnet(filename)
    i =0
    while i <len(layers):
        if layers[i]['func'] == 'AffineTransform':
            d={}
            d['dimensions'] = np.asarray([layers[i]['dim'][1],layers[i]['dim'][0]])
            d['type'] = 'full'
            d['activation'] = layers[i+1]['func']
            d['W'] = layers[i]['W']
            d['b'] = layers[i]['b']
            layers_specs.append(d)
            i=i+1
        i=i+1
    return layers_specs

def get_mss_set(train_x, train_y, ratio=1.0, min_count=1):
    # Remove labels from some samples for semi-supervised setting
    # The number of non-labeled samples is determined to be equal among clusters.
    # Return: Indexes of labeled data
    ndata=train_x.shape[0]
    if ratio == 1.0:
        logger.info("Using all data: %d" % ndata)
        i_labeled=np.array(range(0,ndata))
    else:
        n_classes = train_y.max() + 1
        min_samples=sys.maxint
        max_samples=0
        avg_samples=0
        indices=np.array(range(0,ndata))
        i_labeled = []
        n_labeled_all = 0
        for c in range(n_classes):
            n_samples= round(sum(train_y==c) * ratio) \
                if round(sum(train_y==c) * ratio) >= min_count else min_count
            i = (indices[train_y == c])[:n_samples]
            i_labeled += list(i)
            avg_samples += n_samples
            min_samples = n_samples if n_samples < min_samples else min_samples
            max_samples = n_samples if n_samples > max_samples else max_samples
            n_labeled_all += n_samples
        logger.info('Minimum number of labeled samples: %d' % min_samples)
        logger.info('Maximum number of labeled samples: %d' % max_samples)
        logger.info('Average number of labeled samples: %d' % int(avg_samples / n_classes))
        logger.info('Total   number of labeled samples: %d' % n_labeled_all)
    return i_labeled

#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

if __name__ == "__main__":
#    logging.basicConfig(level=logging.DEBUG)
    feats_dir= '/data2/tawara/work/ttic/MyPython/src/kaldi/timit/data/fbank/train_tr90'
    nnet_transf='/data2/tawara/work/ttic/MyPython/src/kaldi/timit/final.feature_transform'
    train_x, train_y = load_timit_labelled_kaldi(feats_dir, 'src/vat/models/pdf.ark',  nnet_transf= nnet_transf)

#    ark.reset()
#    train, nf = load_data(ark,[0])

#    filename ='timit/exp/fbank/dnn4_nn5_1024_cmvn_splice10_pretrain-dnn/final.nnet'
#    layers = load_nnet(filename)
#    print layers

