import requests
import os

import math
import numpy as np


# https://blog.csdn.net/dcrmg/article/details/78180692
# serialization
import pickle

# requested by ATSI.keras
# for showing training bar
import progressbar

# ? not sure which package it is
# https://github.com/mesnilgr/is13
# this is another file
# i do not create a directory for that so just use file name is enough
#from accuracy import conlleval

# i prefer to use other things to do precision and recall calculation


# for update pip from 9.0.1 to 10.0.1
# using pip --upgrade in visual studio


# good github
# spanish might be useless
# this link has snipper to prevent the data for keras
# https://chsasank.github.io/spoken-language-understanding.html
# corresponding github
# https://github.com/chsasank/ATIS.keras
# multi turn
# https://github.com/yvchen/ContextualSLU
# data set download location
# https://github.com/Microsoft/CNTK/tree/v2.0/Examples/LanguageUnderstanding/ATIS

# cnovert keras model into tensorflow
# http://www.pythonexample.com/code/convert-keras-model-to-tensorflow/

# dump keras model and load_model
# https://jovianlin.io/saving-loading-keras-models/
# Import dependencies
import json
from keras.models import model_from_json, load_model




# cntk scripts
#https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/BrainScript

# cntj
import cntk as C

# Using TensorFlow backend.
# so need to pip install tensorflow
#from keras.models import Model, Sequential
#from keras.layers import GRU, Dense, RepeatVector, TimeDistributed, Flatten, Input, Embedding, Dropout, SimpleRNN

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed


from keras.callbacks import EarlyStopping

locations = ['Tutorials/SLUHandsOn', 'Examples/LanguageUnderstanding/ATIS/BrainScript']

data = {
  'train': { 'file': '..\\atis.train.ctf', 'location': 0 },
  'test': { 'file': '..\\atis.test.ctf', 'location': 0 },
   # ? query.wl seems useless
   # but it is 
  'c': { 'file': '..\\query.wl', 'location': 1 },
  'slots': { 'file': '..\\slots.wl', 'location': 1 }
}

####################################################
# running node configuration
LOADING_MODEL = True
DUMPING_MODEL = True
####################################################


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


    # cntk model description
    # https://cntk.ai/pythondocs/layerref.html


    with C.layers.default_options(initial_state=0.1):
        return C.layers.Sequential([
            C.layers.Embedding(emb_dim, name='embed'),
            C.layers.Recurrence(C.layers.LSTM(hidden_dim), go_backwards=False),
            C.layers.Dense(num_labels, name='classify')
        ])
# u
def create_keras_model():




    ##########################################################
    # if using simple GRU
    # non yet completed and might have bugs
    ##########################################################

    '''
    model = Sequential()
    # embedding
    #  (batch, input_length).
    # https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
    # https://keras.io/layers/embeddings/




    #model.add(Embedding(1000, 64, input_length=10))
    # only use the input and output accordingly
    model.add(Embedding(vocab_size, emb_dim))
    # if not the first layer no need to specify input_shape
    # go backwards is useless just follow CNF
    model.add(GRU(hidden_dim, go_backwards =False))
    # Dense
    # https://www.cntk.ai/pythondocs/layerref.html#dense
    #  Dense(64) 是一个具有 64 个隐藏神经元的全连接层。
    # should be output labels size i think
    # do not care avtivation function at this point
    model.add(Dense(num_labels))


    # optimization function
    # add by myself , you can change it in the future
    # https://github.com/say543/RNNForTimeSeriesForecastTutorial/blob/master/slides/RNN%20For%20Time%20Series%20Forecasting%20Tutorial.pdf
    # mse : Mean-squared-error
    # how to select optimization ?
    # i used to learn that is it provided by Maximal likelihood optimization
    # https://keras.io/getting-started/sequential-model-guide/#compilation
    # having adagrad, studying in th future
    model.compile(optimizer='RMSprop', loss='mse')
    '''

    ##########################################################
    # if using simple RNN
    # https://chsasank.github.io/spoken-language-understanding.html
    # refer ti this link to do 
    # i go with my own vocab_size and embeed dimention
    ###########################################################
    model = Sequential()
    model.add(Embedding(vocab_size, emb_dim))
    # follow link to dropout, not sure if it is useful or not
    model.add(Dropout(0.25))
    # i go with  the hidden_dim
    # since it is unnecessary the same as emb_dim
    # return sequences measns it is on top of embedding layer
    model.add(SimpleRNN(emb_dim, return_sequences=True))
    model.add(TimeDistributed(Dense(num_labels, activation='softmax')))
    model.compile('rmsprop', 'categorical_crossentropy')

    # ouput model for detail
    model.summary()
    
    return model

def create_reader(path, is_training):

    query = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True)
    intent_unused = C.io.StreamDef(field='S1', shape=num_intents, is_sparse=True)
    slot_labels   = C.io.StreamDef(field='S2', shape=num_labels,  is_sparse=True)

    # https://www.cntk.ai/pythondocs/Manual_How_to_feed_data.html?highlight=minibatchsource
    # Using built-in MinibatchSource class is the choice when data cannot be loaded in 
    # refer to Speech section in [13] having a detail example
    # in [13] : it pass speech_feature / speech_label as similar examples
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
# three keys
#'intent_unused', 'query', 'slot_labels'
print(reader.streams.keys())



def atisfull():
    f = open('..\\atis.pkl','rb')
    
    try:
        train_set, test_set, dicts = pickle.load(f)
    except UnicodeDecodeError:
        train_set, test_set, dicts = pickle.load(f, encoding='latin1')
    return train_set, test_set, dicts


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
            # https://www.cntk.ai/pythondocs/Manual_How_to_feed_data.html?highlight=minibatchsource
            # mini batch in B.2 Feeding data with full-control over each minibatch update
            # has aan example
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: reader.streams.query,
                y: reader.streams.slot_labels
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()


'''
def do_train():
    global z
    z = create_model()
    reader = create_reader(data['train']['file'], is_training=True)
    train(reader, z)
do_train()
'''

'''
def train_keras(reader, max_epochs=10):

    # training config
    epoch_size = 18000        # 18000 samples is half the dataset size 

    # modify to 1
    # minibatch_size = 70
    minibatch_size = 1

    # for data exraction
    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
                x: reader.streams.query,
                y: reader.streams.slot_labels
            })

            # https://www.tutorialspoint.com/python/dictionary_keys.htm
            #print(data)
            print(data.keys())
            print(data.values())

            for key in data.keys():
                # cntk cntk class types
                # cntk.variables.Variable
                print (type(key))
                #print (key)

            # if min batch = 1
            # 1 X 11 X 943 , 1 X 11 X 129
            # if min batch = 70
            # 5 X 21 X 943 , 5 X 21 X 129
            # 5(0-4) dpends on min batchm so it decicde how many sentenses can ve covered
            # it looks like mini batch should at least cover a whole sense row
            # this is where 21 comes from (21 row) 21 row is the largest sequence 

            
            #  if mni brach = 30
            # still 1 x 11 x 943


            #print (data[x].num_samples)
            #print (data[y].num_samples)
            print (data[x])
            print (data[y])

            # no train at this point
            #trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()
'''


def train_keras(train_x, train_label):

    # extend global as range
    global kerasZ


    if LOADING_MODEL:
        try:
            with open('kerasZ_architecture.json', 'r') as f:
                kerasZ = model_from_json(f.read())
            kerasZ.load_weights('kerasZ_weights.h5')
        except ValueError as excep:
            print ("input argument in valid: %s" % (excep))
        except Exception:
            print ("unknown exception, something wrong")
        finally:
            if f is not None:
                f.close()

        return



    #We will pass each sentence as a batch to the model. We cannot use model.fit() as it expects all the sentences to be of same size. We will therefore use model.train_on_batch(). 

    # to speed up, only two epochs
    n_epochs = 2
    #n_epochs = 30



    for i in range(n_epochs):
        print("Training epoch {}".format(i))

        #not sure how progressbar package works here
        # i think this is fpr display

        print(len(train_x))
        # max_value is used in progressbar2
        # here i use odd oversion
        # if using 2 then use max_value
        #bar = progressbar.ProgressBar(max_value=len(train_x))
        bar = progressbar.ProgressBar(maxval=len(train_x))

        for n_batch, sent in bar(enumerate(train_x)):
            label = train_label[n_batch]
            # Make labels one hot



            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.eye.html
            # ? not sure how it work
            # numpy eye can create identiy metric
            # np.eye(num_labels)
            # create identity based on dimention of num_labels


            # https://blog.csdn.net/lanchunhui/article/details/49725065
            # np.newaxis == None
            # create another axix for a numpy.ndarray

            # how to use array to index
            # generate array for easy access, not based on shift operation
            # it will based on labe lvalue and get a list of vectors as array
            # temp = np.eye(num_labels)[label]
            # and then np.newaxis add one dimention
            # ? not really sure how to use np.newaxis
            #print 
            label = np.eye(num_labels)[label][np.newaxis,:]


            
            # View each sentence as a batch
            # ? why sentence no needs to one hot
            # becasye  inbuilt Embedding layer for word embeddings. It expects integer indices. 
            sent = sent[np.newaxis,:]

            # ? not sure why ignore one word sentense
            # it is a bug in keras
            if sent.shape[1] > 1: #ignore 1 word sentences
                # https://keras.io/models/sequential/
                kerasZ.train_on_batch(sent, label)


    # save the model
    if DUMPING_MODEL:
        kerasZ.save_weights('kerasZ_weights.h5')
        with open('kerasZ_architecture.json', 'w') as f:
            f.write(kerasZ.to_json())
   


def prepare_keras_data(filename):
    #######################################################
    # if going with atixfull another internet data
    # compare to ms
    # no BOS end EOS
    # also number of volcabulary are words are different
    # also extra things like DIGITDIGITDIGITDIGIT

    # for a training example
    #array([232, 542, 502, 196, 208,  77,  62,  10,  35,  40,  58, 234, 137,  DIGITDIGITDIGITDIGIT      62,  11, 234, 481, 321])

    #        i   want  to  fly  from boston at 838   an   and arrive in denver      at  1110  in the morning

    #array([  0,   0,   0,   0,   0,  18,   0,  94, 126,   0,   0,   0,  18,  'o'       0,  52,   0,   0,  76])
                                      #tag B-city_name
                                      #but Ms B-fromloc.city_name
    #######################################################
    '''
    train_set, valid_set, dicts = atisfull()

    # if you look at the file *.pl
    # S'labels2idx' 
    #S'B-time_relative'
    # p17634
    # I74 <= this is index 

    # ssS'tables2idx'
    # ignore this part at first
    # ? not sure what does tihs mean
    # S'day_number'
    #p17762
    #I134 <= this is index

    # ssS'words2idx'
    # S'all'
    # p17905
    #I32 <= this is index


    w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']
    # Create index to word/label dicts
    idx2w  = {w2idx[k]:k for k in w2idx}
    #idx2ne = {ne2idx[k]:k for k in ne2idx}
    idx2la = {labels2idx[k]:k for k in labels2idx}
    # 127
    # ? this one has something weird since according to atis.pkl
    # it does have more than 127
    n_classes = len(idx2la)
    # 572
    # also one hot encoding i guess
    n_vocab = len(idx2w)

    ### Ground truths etc for conlleval
    train_x, train_ne, train_label = train_set
    val_x, val_ne, val_label = valid_set

    # for type debug
    # all of them are lists
    #print(type(train_x))
    #print(type(train_ne))
    print(type(train_label))
    # inside element
    # it is numpy nd array
    # https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.ndarray.html
    print(type(train_label[0]))

    words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
    groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
    words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
    groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]
    '''

    #######################################################
    # if going with microsoft data
    #######################################################

    
    train_x = []
    train_label = []
    # these two is for debug. no need to have it
    train_x_word = []
    train_label_word = []

    #intFileName = data['train']['file']
    intFileName = filename
    intFile = None
    seq = -1
    try:
        intFile = open(intFileName, "r")
        for line in intFile:
            # skip the ending line character
            lineParts = line[:-1].split("\t");
            if int(lineParts[0]) != seq:
                train_x_per = []
                train_x_word_per = []
                train_label_per = []
                train_label_word_per = []
                seq = int(lineParts[0])

            endOfSeq = False
            for part in lineParts[1:]:
                if part.startswith('|S0'):

                    subparts = part.split(" ")

                    # one-hot encoding
                    word_encoding = int((subparts[1])[0:(subparts[1]).find(':')])
                    train_x_per.append(word_encoding)

                    # real word for comment
                    word = subparts[-1]
                    train_x_word_per.append(word)

                    endOfSeq = endOfSeq or word == 'EOS'

                #elif str.startswith("|S1"):  
                elif part.startswith("|S2"):
                    subparts = part.split(" ")

                    # one-hot encoding
                    label_encoding = int((subparts[1])[0:(subparts[1]).find(':')])
                    train_label_per.append(label_encoding)

                    # real word for comment
                    label = subparts[-1]
                    train_label_word_per.append(label)

            if endOfSeq:
                #train_x_per = np.asfarray(train_x_per)
                #train_label_per = np.asfarray(train_label_per)
                train_x_per = np.asarray(train_x_per)
                train_label_per = np.asarray(train_label_per)
                train_x_word_per = np.asarray(train_x_word_per)
                train_label_word_per = np.asarray(train_label_word_per)
                train_x.append(train_x_per)
                train_label.append(train_label_per)
                # these two is for debug. no need to have it
                train_x_word.append(train_x_word_per)
                train_label_word.append(train_label_word_per)


    except ValueError as excep:
        print ("input argument in valid: %s" % (excep))
    except Exception:
        print ("unknown exception, something wrong")
    finally:
        if intFile is not None:
            intFile.close()

    return train_x, train_label, train_x_word, train_label_word

def do_keras_train():



    global kerasZ
    kerasZ = create_keras_model()

    train_x, train_label, train_x_word, train_label_word = prepare_keras_data(data['train']['file'])


    # Create index to word/label dicts
    # for 
    #global idx2w 
    #idx2w  = {w2idx[k]:k for k in w2idx}
    #global idx2la
    #idx2la = {labels2idx[k]:k for k in labels2idx}
    # instead
    # i assume training data has all words and labels necessary


    # do not use cntk reader anymore
    #reader = create_reader(data['train']['file'], is_training=True)

    # output reader information for keras to use
    #'intent_unused', 'query', 'slot_labels'
    # still cannot see how data looks like
    #print (reader.streams.query)
    #print (reader.streams.intent_unused)
    #print (reader.streams.slot_labels)
    train_keras(train_x, train_label)

do_keras_train()


'''
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
'''

# z is a global variable storing models    
'''
def do_test():
    reader = create_reader(data['test']['file'], is_training=False)
    evaluate(reader, z)
do_test()
print(z.classify.b.value)
'''


def do_keras_model_dump():
    # https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    print ("finish here in the future") 

def do_keras_test():


    val_x, val_label, val_x_word, val_label_word = prepare_keras_data(data['test']['file'])

    labels_pred_val = []

    # max_value is used in progressbar2
    # here i use odd oversion
    # if using 2 then use max_value
    #bar = progressbar.ProgressBar(max_value=len(train_x))
    bar = progressbar.ProgressBar(maxval=len(val_x))

    for n_batch, sent in bar(enumerate(val_x)):
        label = val_label[n_batch]

        # check function do_keras_train for comment for below lines
        label = np.eye(num_labels)[label][np.newaxis,:]

        # ? why here does not need to check if a sentence has only one word as training
        sent = sent[np.newaxis,:]

        # https://keras.io/models/sequential/
        # sinlge batch
        pred = kerasZ.predict_on_batch(sent)


        # return 
        # Numpy arrays
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html
        # return the indicse of the maximum values along an axis
        # applying here is to find the index of 1 since outpout will be one-hot encoding
        pred = np.argmax(pred,-1)[0]

        labels_pred_val.append(pred)


    # i go wtih my own precision and recall calculation
    # sklearn precision and recall
    # https://blog.csdn.net/sinat_26917383/article/details/75199996
    # https://phpcoderblog.wordpress.com/2017/11/02/how-to-calculate-accuracy-precision-recall-and-f1-score-deep-learning-precision-recall-f-score-calculating-precision-recall-python-precision-recall-scikit-precision-recall-ml-metrics-to-use-bi/

    # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=0ahUKEwjR4ILJ66ncAhW4BjQIHTf1ADoQFggrMAA&url=https%3A%2F%2Fblog.argcv.com%2Farticles%2F1036.c&usg=AOvVaw0ojZNwWKXfO9Cesm2i9KZ8
    # based on here
    #  i calculate the whole dataset instead of query level
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # hardcode 128 for 'O'
    # create index to label mapping ni the future
    for val_label_per, labels_pred_val_per in zip(val_label, labels_pred_val):
        for val_label_ele_per , labels_pred_val_ele_per in zip(val_label_per, labels_pred_val_per):
            if val_label_ele_per == labels_pred_val_ele_per:
                if labels_pred_val_ele_per != 128:
                    true_positives += 1
                else:
                    true_negatives +=1
            else:
                if labels_pred_val_ele_per != 128:
                    false_positives += 1
                if val_label_ele_per != 128:
                    false_negatives += 1

    # a ratio of correctly predicted observation to the total observations
    accuracy = (true_positives + true_negatives) \
               / (true_positives + true_negatives + false_positives + false_negatives)
 
    # precision is "how useful the search results are"
    precision = true_positives / (true_positives + false_positives)
    # recall is "how complete the results are"
    recall = true_positives / (true_positives + false_negatives)
 
    f1_score = 2 / ((1 / precision) + (1 / recall))

    print("accuracy = %.6f" % (accuracy))
    print("precision = %.6f" % (precision))
    print("recall = %.6f" % (recall))
    print("f1_score = %.6f" % (f1_score))
 
    return accuracy, precision, recall, f1_score

do_keras_test()

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
