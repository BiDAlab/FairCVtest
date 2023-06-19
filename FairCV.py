# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 11:23:04 2021

@author: BiDA-lab
"""

import os
import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Concatenate, Dense, Embedding, LSTM, Bidirectional, Dropout, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import svm, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE

MAX_WORDS = 20000
NUM_EPOCHS = 16
NUM_EPOCHS_DEMO = 100
BATCH_SIZE = 128
NUM_OCC = 12

#Load the FairCV dataset based on the experiment configuration
def loadDataset(data_path, database_file, config = 'neutral'):
    
    fairCV = np.load(os.path.join(data_path, database_file), allow_pickle = True).item()
    profiles_train = fairCV['Profiles Train']
    profiles_test = fairCV['Profiles Test']
    
    if config == 'neutral':
        labels_train = fairCV['Blind Labels Train']
        labels_test = fairCV['Blind Labels Test']
        labels_train = np.expand_dims(labels_train, axis = 1)
        labels_test = np.expand_dims(labels_test, axis = 1)
        demo_labels_train = profiles_train[:,0:2]
        demo_labels_test = profiles_test[:,0:2]
        
        labels_train = np.concatenate((labels_train, demo_labels_train), axis = 1)
        labels_test = np.concatenate((labels_test, demo_labels_test), axis = 1)
        
        profiles_train = profiles_train[:,4:31]
        profiles_test = profiles_test[:,4:31]
        
        bios_train = fairCV['Bios Train'][:,0]
        bios_test = fairCV['Bios Test'][:,0]
        
    elif config == 'gender':
        labels_train = np.expand_dims(fairCV['Biased Labels Train (Gender)'], axis = 1)
        labels_test = np.expand_dims(fairCV['Biased Labels Test (Gender)'], axis = 1)
        gender_labels_train = np.expand_dims(profiles_train[:,1], axis = 1)
        gender_labels_test = np.expand_dims(profiles_test[:,1], axis = 1)
        
        labels_train = np.concatenate((labels_train, gender_labels_train), axis = 1)
        labels_test = np.concatenate((labels_test, gender_labels_test), axis = 1)
        
        profiles_train = profiles_train[:,4:31]
        profiles_test = profiles_test[:,4:31]
        
        bios_train = fairCV['Bios Train'][:,0]
        bios_test = fairCV['Bios Test'][:,0]
        
    elif config == 'ethnicity':
        labels_train = np.expand_dims(fairCV['Biased Labels Train (Ethnicity)'], axis = 1)
        labels_test = np.expand_dims(fairCV['Biased Labels Test (Ethnicity)'], axis = 1)
        ethnicity_labels_train = np.expand_dims(profiles_train[:,0], axis = 1)
        ethnicity_labels_test = np.expand_dims(profiles_test[:,0], axis = 1)
        
        labels_train = np.concatenate((labels_train, ethnicity_labels_train), axis = 1)
        labels_test = np.concatenate((labels_test, ethnicity_labels_test), axis = 1)
        
        profiles_train = profiles_train[:,4:31]
        profiles_test = profiles_test[:,4:31]
        
        bios_train = fairCV['Bios Train'][:,0]
        bios_test = fairCV['Bios Test'][:,0]
        
    elif config == 'agnostic ethnicity':
        labels_train = np.expand_dims(fairCV['Biased Labels Train (Ethnicity)'], axis = 1)
        labels_test = np.expand_dims(fairCV['Biased Labels Test (Ethnicity)'], axis = 1)
        ethnicity_labels_train = np.expand_dims(profiles_train[:,0], axis = 1)
        ethnicity_labels_test = np.expand_dims(profiles_test[:,0], axis = 1)
        
        labels_train = np.concatenate((labels_train, ethnicity_labels_train), axis = 1)
        labels_test = np.concatenate((labels_test, ethnicity_labels_test), axis = 1)
        
        profiles_train = np.concatenate((profiles_train[:,4:11], profiles_train[:,31:]),
                                        axis = 1)
        profiles_test = np.concatenate((profiles_test[:,4:11], profiles_test[:,31:]),
                                        axis = 1)
        
        bios_train = fairCV['Bios Train'][:,1]
        bios_test = fairCV['Bios Test'][:,1]
        
    elif config == 'agnostic gender':
        labels_train = np.expand_dims(fairCV['Biased Labels Train (Gender)'], axis = 1)
        labels_test = np.expand_dims(fairCV['Biased Labels Test (Gender)'], axis = 1)
        gender_labels_train = np.expand_dims(profiles_train[:,1], axis = 1)
        gender_labels_test = np.expand_dims(profiles_test[:,1], axis = 1)
        
        labels_train = np.concatenate((labels_train, gender_labels_train), axis = 1)
        labels_test = np.concatenate((labels_test, gender_labels_test), axis = 1)
        
        profiles_train = np.concatenate((profiles_train[:,4:11], profiles_train[:,31:]),
                                        axis = 1)
        profiles_test = np.concatenate((profiles_test[:,4:11], profiles_test[:,31:]),
                                        axis = 1)
        
        bios_train = fairCV['Bios Train'][:,1]
        bios_test = fairCV['Bios Test'][:,1]
        
    elif config == 'evaluation':
        labels_train_blind = np.expand_dims(fairCV['Blind Labels Train'], axis = 1)
        labels_test_blind = np.expand_dims(fairCV['Blind Labels Test'], axis = 1)
        labels_train_gender = np.expand_dims(fairCV['Biased Labels Train (Gender)'], axis = 1)
        labels_test_gender = np.expand_dims(fairCV['Biased Labels Test (Gender)'], axis = 1)
        labels_train_ethnicity = np.expand_dims(fairCV['Biased Labels Train (Ethnicity)'], axis = 1)
        labels_test_ethnicity = np.expand_dims(fairCV['Biased Labels Test (Ethnicity)'], axis = 1)
        labels_train = np.concatenate((labels_train_blind, labels_train_gender,
                                       labels_train_ethnicity), axis = 1)
        labels_test = np.concatenate((labels_test_blind, labels_test_gender,
                                       labels_test_ethnicity), axis = 1)
        
        profiles_train = profiles_train[:,:11]
        profiles_test = profiles_test[:,:11]
        
        bios_train = fairCV['Bios Train'][:,0]
        bios_test = fairCV['Bios Test'][:,0]
        
    return labels_train, labels_test, profiles_train, profiles_test, bios_train, bios_test

# Process bios and return tokenized sequences
def processBios(bios_raw, t = None, max_len = 0):
    
    if t:
        sequences = t.texts_to_sequences(bios_raw)
        bios = pad_sequences(sequences, maxlen = max_len)
        
    else:
        t = Tokenizer(num_words = MAX_WORDS)
        t.fit_on_texts(bios_raw)
        sequences = t.texts_to_sequences(bios_raw)
        bios = pad_sequences(sequences)
    
    return bios, t, bios.shape[1]

# Load the fastText word embeddings
def load_vectors(fname):
    
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype ='float32')
        
    return d, data

# Define the network
def baseNetwork(num_feat, max_len_bios, data_path, word_emb_file, t):
    
    # Load the word embeddings
    embedding_dim, word_embeddings = load_vectors(os.path.join(data_path, word_emb_file))
    
    # Definie the embedding matrix
    embedding_matrix = np.zeros((MAX_WORDS, embedding_dim))
    
    for word, i in t.word_index.items():
        if i < MAX_WORDS:
            embedding_vector = word_embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
    del word_embeddings
    
    #Define the recurrent network
    text_input = Input(shape = (max_len_bios,))
    embedding = Embedding(MAX_WORDS, embedding_dim, input_length = max_len_bios,
                          weights = [embedding_matrix], trainable = False)(text_input)
    text_embedding = Bidirectional(LSTM(32, activation = 'tanh'))(embedding)
    
    #Define the feature network and merge the two networks
    feature_input = Input(shape = (num_feat,))
    concat = Concatenate()([feature_input, text_embedding])
    x = Dense(40, activation = 'relu', input_shape = (concat.shape),
              name = 'embedding_layer')(concat)
    x = Dropout(0.3)(x)
    x = Dense(20, activation = 'relu')(x)
    output = Dense(1, activation = 'sigmoid')(x)
    
    model = keras.Model(inputs = [feature_input, text_input], outputs = output)
    model.summary()
    # keras.utils.plot_model(model, "FairCVtest_Architecture.png", show_shapes = False)
    
    return model

#Train the hiring tool
def trainHiringTool(model, profiles_train, bios_train, labels_train):
    
    model.compile(optimizer = keras.optimizers.Adam(), loss = keras.losses.MeanAbsoluteError())
    
    history = model.fit([profiles_train, bios_train], labels_train, batch_size = BATCH_SIZE,
                        epochs = NUM_EPOCHS)
    
    print('##########TRAINING##########')
    print('Training loss (MAE): {:.3f}'.format(history.history['loss'][-1]))
    
    return model, history

# Evaluate the hiring tool
def evaluateHiringTool(model, profiles_test, bios_test, labels_test):
    
    scores_loss = model.evaluate([profiles_test, bios_test], labels_test)
    
    print('##########EVALUATION##########')
    print('Test loss (MAE): {:.3f}'.format(scores_loss))
    
    scores = model.predict([profiles_test, bios_test])
    
    return scores

# Save the results
def saveResults(data_path, config, model, t, history, scores, max_len):
    
    fairCVtest_results = {}
    fairCVtest_results['History'] = history.history['loss']
    fairCVtest_results['Scores'] = scores
    fairCVtest_results['Len Bios'] = max_len
    
    save_file =  'FairCVtest ' + config + '.npy'
    np.save(os.path.join(data_path, save_file), fairCVtest_results, allow_pickle = True)
    
    model_file = 'HiringTool ' + config + '.h5'
    model.save(os.path.join(data_path, model_file))
    
    tokenizer_file = 'Tokenizer ' + config + '.pickle'
    with open(os.path.join(data_path, tokenizer_file), 'wb') as handle:
        pickle.dump(t, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
    #with open(path, 'rb') as handle:
        #t = pickle.load(handle)
    

# Compute the KL divergence
def kl_divergence(p, q):
    return np.sum(np.where(np.logical_and(p > 0,q > 0), p * np.log(p / q), 0))

    # The KL divergence is defined if P and Q are normalized and Q!=0 where P !=0
    
# Compute distributions from the raw scores, and return the KL divergence
def computeKL(x,y):
    
    p = []
    q = []
    
    # Compute scores distributions
    for i in range(0,98,2):
        i = i/100
        j = i + 0.02
        p.append(np.sum(np.where(np.logical_and(x >= i, x <j), 1, 0)))
        q.append(np.sum(np.where(np.logical_and(y >= i, y <j), 1, 0)))
        
    p = np.asarray(p).reshape(1,-1)
    q = np.asarray(q).reshape(1,-1)
    
    # Normalize distributions
    p = p/x.shape[0]
    q = q/y.shape[0]
    
    return kl_divergence(p,q)

#Generate feature embeddings from the profiles
def generateCVEmbeddings(config, profiles_train, profiles_test, bios_train, bios_test):
    
    model_file = 'HiringTool ' + config + '.h5'
    model = keras.models.load_model(model_file)
    
    embedding = model.get_layer('embedding_layer').output
    embedding = Flatten(name = 'Flatten')(embedding)
    feature_extractor = keras.Model(inputs = model.input, outputs = embedding)
    
    embeddings_train = feature_extractor.predict([profiles_train, bios_train])
    embeddings_test = feature_extractor.predict([profiles_test, bios_test])
    
    embeddings_train = preprocessing.normalize(embeddings_train, norm = 'l2', axis = 1,
                                               copy = True, return_norm = False)
    embeddings_test = preprocessing.normalize(embeddings_test, norm = 'l2', axis = 1,
                                               copy = True, return_norm = False)
    
    return embeddings_train, embeddings_test

def evaluateSVM(embeddings_train, embeddings_test, labels_train, labels_test):
    
    svm_clf = svm.SVC(probability = True, kernel = 'rbf')
    svm_clf.fit(embeddings_train, labels_train)
    
    y_pred_svm = svm_clf.predict(embeddings_test)
    acc_svm = accuracy_score(labels_test, y_pred_svm)
    print('SVM (rbf kernel) Acc: {:.4f}'.format(acc_svm))
    
def evaluateRandomForest(embeddings_train, embeddings_test, labels_train, labels_test):
    
    rf_clf = RandomForestClassifier(max_depth = 1000, random_state = 0)
    rf_clf.fit(embeddings_train, labels_train)
    
    y_pred_rf = rf_clf.predict(embeddings_test)
    acc_rf = accuracy_score(labels_test, y_pred_rf)
    print('RF Acc: {:.4f}'.format(acc_rf))
    
def evaluateNeuralNetwork(embeddings_train, embeddings_test, labels_train, labels_test, config):
    
    labels_train = keras.utils.to_categorical(labels_train)
    labels_test = keras.utils.to_categorical(labels_test)
    
    # Define neural network
    embedding_input = Input(shape = (embeddings_train.shape[1],))
    x = Dense(10, activation = 'sigmoid', name = 'embedding_layer')(embedding_input)
    x = Dense(labels_train.shape[1], activation = 'softmax')(x)
        
    nn_clf = keras.Model(inputs = embedding_input, outputs = x)

    nn_clf.compile(optimizer = keras.optimizers.Adam(),
                      loss = keras.losses.CategoricalCrossentropy(), metrics = ['acc'])
        
    # Train and evaluate network
    callback = keras.callbacks.EarlyStopping(monitor = 'val_acc', patience = 10)
    
    nn_clf.fit(embeddings_train, labels_train, batch_size = BATCH_SIZE,
                  epochs = NUM_EPOCHS_DEMO, callbacks = [callback],
                  validation_data = (embeddings_test, labels_test))
    
    results_nn = nn_clf.evaluate(embeddings_test, labels_test,
                                    batch_size = BATCH_SIZE)
    
    print('Neural Network Acc: {:.4f}'.format(results_nn[1]))
    
    # Generate embeddings
    embedding = nn_clf.get_layer('embedding_layer').output
    embedding = Flatten(name = 'Flatten')(embedding)
    feature_extractor = keras.Model(inputs = nn_clf.input, outputs = embedding)
    
    embeddings = feature_extractor.predict(embeddings_test)
    
    embeddings = preprocessing.normalize(embeddings, norm = 'l2', axis = 1,
                                               copy = True, return_norm = False)
    
    return embeddings

def representTSNE(data_path, embeddings, labels, config):
    
    tsne = TSNE(n_components = 2, verbose = 1, perplexity = 20, n_iter = 500)
    tsne_transf = tsne.fit_transform(embeddings)
    
    save_file = 'tsne ' + config + '.png'
    
    plt.figure(figsize = (7.3,7.3))
    ax = plt.axes()
    ax.set_facecolor('white')
    
    if config in ['ethnicity', 'agnostic ethnicity', 'neutral']:
        E1 = (labels == 0)
        E2 = (labels == 1)
        E3 = (labels == 2)
        
        plt.scatter(tsne_transf[E1,0], tsne_transf[E1,1], color = 'y', alpha = .3,
                    lw = 2.5, s = 50, label = 'Group 1')
        plt.scatter(tsne_transf[E2,0], tsne_transf[E2,1], color = 'g', alpha = .3,
                    lw = 2.5, s = 50, label = 'Group 2')
        plt.scatter(tsne_transf[E3,0], tsne_transf[E3,1], color = 'b', alpha = .3,
                    lw = 2.5, s = 50, label = 'Group 3')
        plt.title('t-SNE representation by Ethnicity')
        
    else:
        male = (labels == 0)
        female = (labels == 1)
        
        plt.scatter(tsne_transf[male,0], tsne_transf[male,1], color = 'r', alpha = .3,
                    lw = 2.5, s = 50, label = 'Male')
        plt.scatter(tsne_transf[female,0], tsne_transf[female,1], color = 'c', alpha = .3,
                    lw = 2.5, s = 50, label = 'Female')
        plt.title('t-SNE representation by Gender')
    
    plt.legend(loc='upper right')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True,  linestyle=':',color = 'lightgrey')
    plt.savefig(os.path.join(data_path, save_file), dpi = 1000)
    plt.show()
        
#Evaluate the demographic information extracted by the Hiring Tool
def evaluateDemographics(data_path, database_file, config):
    
    keras.backend.clear_session()
    os.chdir(data_path)
    
    labels_train, labels_test, profiles_train, profiles_test, bios_train, bios_test = loadDataset(data_path, database_file, config)
    demo_labels_train = labels_train[:,1]
    demo_labels_test = labels_test[:,1]
    
    tokenizer_file = 'Tokenizer ' + config + '.pickle'
    with open(tokenizer_file, 'rb') as f:
        t = pickle.load(f)
    
    results_file = 'FairCVtest ' + config + '.npy'
    results = np.load(results_file, allow_pickle = True).item()
    max_len = results['Len Bios']
    
    bios_train, _, _ = processBios(bios_train, t, max_len)
    bios_test, _, _ = processBios(bios_test, t, max_len)
    
    embeddings_train, embeddings_test = generateCVEmbeddings(config, profiles_train,
                                                             profiles_test, bios_train, bios_test)
    
    # Evaluate SVM
    evaluateSVM(embeddings_train, embeddings_test, demo_labels_train, demo_labels_test)
    
    # Evaluate RF
    evaluateRandomForest(embeddings_train, embeddings_test, demo_labels_train, demo_labels_test)
    
    # Evaluate NN
    embeddings = evaluateNeuralNetwork(embeddings_train, embeddings_test, demo_labels_train,
                                       demo_labels_test, config)
    
    # t-SNE representation
    representTSNE(data_path, embeddings, demo_labels_test, config)
    
    #The neutral case has two options, gender and ethnicity
    if config == 'neutral':
        
        demo_labels_train = labels_train[:,2]
        demo_labels_test = labels_test[:,2]
        
        # Evaluate SVM
        evaluateSVM(embeddings_train, embeddings_test, demo_labels_train, demo_labels_test)
        
        # Evaluate RF
        evaluateRandomForest(embeddings_train, embeddings_test, demo_labels_train, demo_labels_test)
        
        # Evaluate NN
        embeddings = evaluateNeuralNetwork(embeddings_train, embeddings_test, demo_labels_train,
                                           demo_labels_test, config)
        
        representTSNE(data_path, embeddings, demo_labels_test, 'neutral gender')
        

def computeTopScore(scores, demo_att, config):
    
    scores_sorted = np.argsort(scores, axis = 0)
    scores_sorted = scores_sorted[::-1]
    
    demo_att_sorted = demo_att[scores_sorted]
    top100 = demo_att_sorted[:1000]
      
    if config in ['gender', 'agnostic gender', 'neutral gender']:
        top100_female = np.sum(top100)/1000
        top100_male = 1 - top100_female
        
        print('Top 1000 Male: {:.2f}%'.format(top100_male * 100))
        print('Top 1000 Female: {:.2f}%'.format(top100_female * 100))
         
    elif config in ['ethnicity', 'agnostic ethnicity', 'neutral ethnicity']:
        top100_g2 = np.sum(top100 % 2)/1000
        top100_g3 = np.sum(top100//2)/1000
        top100_g1 = 1 - top100_g2 - top100_g3

        print('Top 1000 G1: {:.2f}%'.format(top100_g1*100))
        print('Top 1000 G2: {:.2f}%'.format(top100_g2*100))
        print('Top 1000 G3: {:.2f}%'.format(top100_g3*100))
        
# Test demographic parity by computing p value in the top 1000 candidates.
def testDemographicParity(scores, demo_att, config):
    
    scores_sorted = np.argsort(scores, axis = 0)
    scores_sorted = scores_sorted[::-1]
    
    demo_att_sorted = demo_att[scores_sorted]
    top1000 = demo_att_sorted[:1000]
    
    if config in ['gender', 'agnostic gender','neutral gender']:
        top1000_female = np.sum(top1000)/1000
        top1000_male = 1 - top1000_female
        
        p_val = min(top1000_female/top1000_male, top1000_male/top1000_female)
        print('P-% Value: {:.2f}%'.format(p_val*100))
    
    elif config in ['ethnicity', 'agnostic ethnicity','neutral ethnicity']:
        top1000_g2 = np.sum(top1000 % 2)/1000
        top1000_g3 = np.sum(top1000//2)/1000
        top1000_g1 = 1 - top1000_g2 - top1000_g3
        
        p_val_1 = min(top1000_g2/top1000_g1,top1000_g1/top1000_g2)
        p_val_2 = min(top1000_g3/top1000_g1,top1000_g1/top1000_g3)
        p_val_3 = min(top1000_g3/top1000_g2,top1000_g2/top1000_g3)
        
        print('P-% Value (G2/G1): {:.2f}%'.format(p_val_1*100))
        print('P-% Value (G3/G1): {:.2f}%'.format(p_val_2*100))
        print('P-% Value (G3/G2): {:.2f}%'.format(p_val_3*100))
        
# Test equality of opportunity, by selecting a threshold on the training labels to convert
# the scoring scenario to a selection scenario

def testEqualityOfOpportunity(scores, demo_att, blind_labels, blind_labels_test, config, p = 75):
    
    threshold = np.percentile(blind_labels, p)
    
    blind_labels_test[blind_labels_test < threshold] = 0
    blind_labels_test[blind_labels_test >= threshold] = 1
    scores[scores < threshold] = 0
    scores[scores > threshold] = 1
    
    if config in ['gender', 'agnostic gender', 'neutral gender']:
        male = (demo_att == 0)
        female = (demo_att == 1)
        
        labels_male = (blind_labels_test[male] == 1)
        labels_female = (blind_labels_test[female] == 1)
        scores_male = scores[male]
        scores_female = scores[female]
        
        tpr_male = np.sum(scores_male[labels_male])/np.sum(labels_male)
        tpr_female = np.sum(scores_female[labels_female])/np.sum(labels_female)
    
        print('TPR Male: {:.2f}%'.format(tpr_male * 100))
        print('TPR Female: {:.2f}%'.format(tpr_female * 100))
        print('TPR Difference: {:.2f}%'.format((tpr_male - tpr_female)*100))
        
    elif config in ['ethnicity', 'agnostic ethnicity', 'neutral ethnicity']:
        G1 = (demo_att == 0)
        G2 = (demo_att == 1)
        G3 = (demo_att == 2)
        
        labels_g1 = (blind_labels_test[G1] == 1)
        labels_g2 = (blind_labels_test[G2] == 1)
        labels_g3 = (blind_labels_test[G3] == 1)
        scores_g1 = scores[G1]
        scores_g2 = scores[G2]
        scores_g3 = scores[G3]
        
        tpr_g1 = np.sum(scores_g1[labels_g1])/np.sum(labels_g1)
        tpr_g2 = np.sum(scores_g2[labels_g2])/np.sum(labels_g2)
        tpr_g3 = np.sum(scores_g3[labels_g3])/np.sum(labels_g3)
        
        print('TPR G1: {:.2f}%'.format(tpr_g1 * 100))
        print('TPR G2: {:.2f}%'.format(tpr_g2 * 100))
        print('TPR G2: {:.2f}%'.format(tpr_g3 * 100))
        print('TPR Difference(G1-G2): {:.2f}%'.format((tpr_g1 - tpr_g2)*100))
        print('TPR Difference(G1-G3): {:.2f}%'.format((tpr_g1 - tpr_g3)*100))
        print('TPR Difference(G2-G3): {:.2f}%'.format((tpr_g2 - tpr_g3)*100))
        
def analyzeResults(data_path, database_file, config = 'neutral'):
    
    results_file = 'FairCVtest ' + config + '.npy'
    results = np.load(os.path.join(data_path, results_file), allow_pickle = True).item()
    scores = results['Scores']
    
    labels_train, labels_test, _, profiles_test, _, _ = loadDataset(data_path, database_file, config = 'evaluation')
    
    ethnicity_labels = profiles_test[:,0]
    gender_labels = profiles_test[:,1]
    suitability_labels = profiles_test[:,3]
    
    # Gender distributions
    save_file = 'Gender distribution ' + config + '.png'
    
    male = (gender_labels == 0)
    female = (gender_labels == 1)
    
    KL = computeKL(scores[male], scores[female])
        
    plt.figure(figsize = (7.3,5.2))
    ax = plt.axes()
    ax.set_facecolor('white')
    

    plt.title('Hiring Results by Gender, KL(P||Q) = {:.3f}'.format(KL))
    sns.distplot(scores[male], hist = False, kde_kws = {'shade' : True}, color = 'r', label = 'Male')
    sns.distplot(scores[female], hist = False, kde_kws = {'shade' : True}, color = 'c', label = 'Female')
    plt.legend(loc='upper right')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True,  linestyle=':',color = 'lightgrey')
    plt.savefig(os.path.join(data_path, save_file), dpi = 1000)
    plt.show()
    
    # Ethnicity distributions
    save_file = 'Ethnicity distributions ' + config + '.png'
    
    E1 = (ethnicity_labels == 0)
    E2 = (ethnicity_labels == 1)
    E3 = (ethnicity_labels == 2)
    
    KL1 = computeKL(scores[E1], scores[E2])
    KL2 = computeKL(scores[E1], scores[E3])
    KL3 = computeKL(scores[E2], scores[E3])
    KL = (KL1 + KL2 + KL3)/3
            
    plt.figure(figsize = (7.3,5.2))
    ax = plt.axes()
    ax.set_facecolor('white')

    plt.title('Hiring Results by Ethnicity, KL(P||Q) = {:.3f}'.format(KL))
    sns.distplot(scores[E1], hist = False, kde_kws = {'shade' : True}, color = 'y', label = 'Group 1')
    sns.distplot(scores[E2], hist = False, kde_kws = {'shade' : True}, color = 'g', label = 'Group 2')
    sns.distplot(scores[E3], hist = False, kde_kws = {'shade' : True}, color = 'b', label = 'Group 3')
    plt.legend(loc='upper right')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True,  linestyle=':',color = 'lightgrey')
    plt.savefig(os.path.join(data_path, save_file), dpi = 1000)
    plt.show()
    
    # Occupation distributions
    save_file = 'Occupation distributions ' + config + '.png'
    
    O1 = (suitability_labels == 0.25)
    O2 = (suitability_labels == 0.5)
    O3 = (suitability_labels == 0.75)
    O4 = (suitability_labels == 1)
            
    plt.figure(figsize = (7.3,5.2))
    ax = plt.axes()
    ax.set_facecolor('white')

    plt.title('Hiring Results by Occupation Group')
    sns.distplot(scores[O1], hist = False, kde_kws = {'shade' : True}, color = 'r', label = 'AV') 
    sns.distplot(scores[O2], hist = False, kde_kws = {'shade' : True}, color = 'g', label = 'JA')
    sns.distplot(scores[O3], hist = False, kde_kws = {'shade' : True}, color = 'b', label = 'HC')
    sns.distplot(scores[O4], hist = False, kde_kws = {'shade' : True}, color = 'y', label = 'EN')
    plt.legend(loc='upper right')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True,  linestyle=':',color = 'lightgrey')
    plt.savefig(os.path.join(data_path, save_file), dpi = 1000)
    plt.show()
    
    # Evaluate different fairness metrics
    if config in ['gender', 'agnostic gender']:
        computeTopScore(scores, gender_labels, config)
        testDemographicParity(scores, gender_labels, config)
        testEqualityOfOpportunity(scores, gender_labels, labels_train[:,0],
                                  labels_test[:,0], config)
        
    elif config in ['ethnicity', 'agnostic ethnicity']:
        computeTopScore(scores, ethnicity_labels, config)
        testDemographicParity(scores, ethnicity_labels, config)
        testEqualityOfOpportunity(scores, ethnicity_labels, labels_train[:,0],
                                  labels_test[:,0], config)
        
    elif config == 'neutral':
        computeTopScore(scores, gender_labels, 'neutral gender')
        computeTopScore(scores, ethnicity_labels, 'neutral ethnicity')
        testDemographicParity(scores, gender_labels, 'neutral gender')
        testDemographicParity(scores, ethnicity_labels, 'neutral ethnicity')
        testEqualityOfOpportunity(scores, gender_labels, labels_train[:,0],
                                  labels_test[:,0], 'neutral gender')
        testEqualityOfOpportunity(scores, ethnicity_labels, labels_train[:,0],
                                  labels_test[:,0], 'neutral ethnicity')
 
    
# FairCV main function
def fairCVtest(data_path, database_file, word_emb_file, config = 'neutral'):
    
    keras.backend.clear_session()
    
    labels_train, labels_test, profiles_train, profiles_test, bios_train, bios_test = loadDataset(data_path, database_file, config)
    labels_train = labels_train[:,0]
    labels_test = labels_test[:,0]
    
    bios_train, t, max_len = processBios(bios_train)
    bios_test, _, _ = processBios(bios_test, t, max_len)
    
    model = baseNetwork(profiles_train.shape[1], max_len, data_path, word_emb_file, t)
    model, history = trainHiringTool(model, profiles_train, bios_train, labels_train)
    scores = evaluateHiringTool(model, profiles_test, bios_test, labels_test)
    
    saveResults(data_path, config, model, t, history, scores, max_len)
    analyzeResults(data_path, database_file, config)


if __name__ == '__main__':
    
    os.chdir('data')
    database_file = 'FairCVdb.npy'
    word_emb_file = 'crawl-300d-2M.vec'
    config = 'neutral'
    
    fairCVtest(data_path, database_file, word_emb_file, config)
    evaluateDemographics(data_path, database_file, config)






