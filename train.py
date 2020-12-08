import sugartensor as tf
import tensorflow
from data import SpeechCorpus, voca_size
from model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 16    # total batch size

#
# inputs
#

# corpus input tensor
data = SpeechCorpus(batch_size=batch_size * tf.sg_gpus())
print('data fetch complete')
print(data.mfcc)
print(data.label)
# mfcc feature of audio
inputs = tf.split(data.mfcc, tf.sg_gpus(), axis=0)
# target sentence label
labels = tf.split(data.label, tf.sg_gpus(), axis=0)

# sequence length except zero-padding
seq_len = []
for input_ in inputs:
    seq_len.append(tf.not_equal(input_.sg_sum(axis=2), 0.).sg_int().sg_sum(axis=1))


# parallel loss tower
@tf.sg_parallel
def get_loss(opt):
    # encode audio feature
    logit = get_logit(opt.input[opt.gpu_index], voca_size=voca_size)
    # CTC loss
    return logit.sg_ctc(target=opt.target[opt.gpu_index], seq_len=opt.seq_len[opt.gpu_index])

#
# train
#


cf = tf.ConfigProto()
cf.gpu_options.allow_growth = True
cf.gpu_options.per_process_gpu_memory_fraction = 0.4
cf.log_device_placement = True
cf.allow_soft_placement=True
print(cf)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session(config=cf)
tf.sg_train(allow_growth=True, lr=0.0001, loss=get_loss(input=inputs, target=labels, seq_len=seq_len),
                ep_size=data.num_batch, max_ep=35, early_stop=False)
