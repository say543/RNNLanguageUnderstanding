import requests
import os

import math
import numpy as np


# good github
# spanish might be useless
# https://github.com/chsasank/ATIS.keras
# https://chsasank.github.io/spoken-language-understanding.html
# multi turn
# https://github.com/yvchen/ContextualSLU
# data set download location
# https://github.com/Microsoft/CNTK/tree/v2.0/Examples/LanguageUnderstanding/ATIS

# cntj
import cntk as C

# Using TensorFlow backend.
# so need to pip install tensorflow
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input
from keras.callbacks import EarlyStopping

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': '..\\atis.train.ctf', 'location': 0 },
  'test': { 'file': '..\\atis.test.ctf', 'location': 0 },
  'query': { 'file': '..\\query.wl', 'location': 1 },
  'slots': { 'file': '..\\slots.wl', 'location': 1 }
}

if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        C.device.try_set_default_device(C.device.cpu())
    else:
        C.device.try_set_default_device(C.device.gpu(0))




# Test for CNTK version
# i use 2.5.1 version 
#if not C.__version__ == "2.0":
#    raise Exception("this notebook was designed to work with 2.0. Current Version: " + C.__version__) 





# number of words in vocab, slot labels, and intent labels
vocab_size = 943 ; num_labels = 129 ; num_intents = 26    

# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 150
hidden_dim = 300

# Create the containers for input feature (x) and the label (y)
x = C.sequence.input_variable(vocab_size)
y = C.sequence.input_variable(num_labels)

def create_model():

    # cntk layer embedding
    # https://www.cntk.ai/pythondocs/layerref.html#embedding
    # dense
    # https://www.cntk.ai/pythondocs/layerref.html#dense

    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False),
            C.layers.Dense(num_labels, name='classify')
        ])

def create_keras_model():

    model = Sequential()
    # embedding
    #  (batch, input_length).
    # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # https://keras.io/layers/embeddings/




    #model.add(Embedding(1000, 64, input_length=10))
    # only use the input and output accordingly
    model.add(Embedding(vocab_size, emb_dim))
    # no the first layer so need to specify input_shape
    # go backwards is useless just follow CNF
    model.add(GRU(hidden_dim, go_backwards =False))
    # Dense
    # https://www.cntk.ai/pythondocs/layerref.html#dense
    #  Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # should be output labels size i think
    # do not care avtivation function at this point
    model.add(Dense(num_labels))
    
    

def create_reader(path, is_training):

    query = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True)
    intent_unused = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True)
    slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)

    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         query         = query,
         intent_unused = intent_unused,  
         slot_labels   = slot_labels
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)



z = create_model()
print(z.embed.E.shape)
print(z.classify.b.value)


# Pass an input and check the dimension
z = create_model()
print(z(x).embed.E.shape)


# peek
reader = create_reader(data['train']['file'], is_training=True)
print(reader.streams.keys())




def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)


def train(reader, model_func, max_epochs=10):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Instantiate the loss and error function
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 
    minibatch_size = 70
    
    # LR schedule over epochs 
    # In CNTK, an epoch is how often we get out of the minibatch loop to
    # do other stuff (e.g. checkpointing, adjust learning rate, etc.)
    # (we don't run this many epochs, but if we did, these are good values)
    lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]
    lr_per_minibatch = [lr * minibatch_size for lr in lr_per_sample]
    lr_schedule = C.learning_rate_schedule(lr_per_minibatch, C.UnitType.minibatch, epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = C.momentum_as_time_constant_schedule(700)
    
    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from 
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: reader.streams.query,
                y: reader.streams.slot_labels
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()

def do_train():
    global z
    z = create_model()
    reader = create_reader(data['train']['file'], is_training=True)
    train(reader, z)
do_train()




def evaluate(reader, model_func):
    
    # Instantiate the model function; x is the input (feature) variable 
    model = model_func(x)
    
    # Create the loss and error functions
    loss, label_error = create_criterion_function_preferred(model, y)

    # process minibatches and perform evaluation
    progress_printer = C.logging.ProgressPrinter(tag='Evaluation', num_epochs=0)

    while True:
        minibatch_size = 500
        data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
            x: reader.streams.query,
            y: reader.streams.slot_labels
        })
        if not data:                                 # until we hit the end
            break

        evaluator = C.eval.Evaluator(loss, progress_printer)
        evaluator.test_minibatch(data)
     
    evaluator.summarize_test_progress()


# z is a global variable storing models    

def do_test():
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, z)
do_test()
print(z.classify.b.value)


#########################################
# below is for one sequence test
#########################################

# load dictionaries
# from * .wl
# ? why needs * w.l
query_wl = [line.rstrip('\n') for line in open(data['query']['file'])]
slots_wl = [line.rstrip('\n') for line in open(data['slots']['file'])]
query_dict = {query_wl[i]:i for i in range(len(query_wl))}
slots_dict = {slots_wl[i]:i for i in range(len(slots_wl))}

# for debug
#print (query_wl)
#print (slots_wl)
#print (query_dict)
#print (slots_dict)


# after traningi and test 
# run a sinlge query and get the result
# let's run a sequence through
seq = 'BOS flights from new york to seattle EOS'
w = [query_dict[w] for w in seq.split()] # convert to word indices
print(w)
onehot = np.zeros([len(w),len(query_dict)], np.float32)
for t in range(len(w)):
    onehot[t,w[t]] = 1

#x = C.sequence.input_variable(vocab_size)
pred = z(x).eval({x:[onehot]})[0]
print(pred.shape)
best = np.argmax(pred,axis=1)

# print best mapping index
print(best)
# print best output result
print(list(zip(seq.split(),[slots_wl[s] for s in best])))
