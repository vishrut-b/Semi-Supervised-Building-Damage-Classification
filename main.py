#!/usr/bin/env python3

import argparse
import os
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from dataloader import load_task
from processdata import split_data
from pseudolabeling import PseudoCallback
from model import build_model
from train import train_pseudo, train_model
from evaluate import plot_multiclass_roc, plot_performance, score_model, pred_confidence
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)


def main():
    
    parser = argparse.ArgumentParser(description = 'Keras Pseudo-Label Training')
    parser.add_argument('--task', 
                        default = 1,
                        type = int,
                        choices = [1, 2], 
                        help = 'Task 1: Scene Level (Pixel, Object, Structure), Task 2: Damage State (Damaged, Undamaged)')
    parser.add_argument('--semisupervised',
                        default = False,
                        action='store_true',
                        help = 'True/False use unlabeled images for pseudo-label training')
    parser.add_argument('--path', 
                        default = '/home/ubuntu', 
                        type = str,
                        help = 'Location of folders containing datasets for each task')
    parser.add_argument('--val_split',
                        default = 0.1,
                        type = float,
                        help = 'Proportion of (labeled) training set to use as validation')
    parser.add_argument('--batch_size',
                        default = 32,
                        type = int,
                        help = 'Number of examples per training batch')
    parser.add_argument('--epochs',
                        default = 1,
                        type = int,
                        help = 'Number of epochs to train from data')
    parser.add_argument('--lr',
                        default = 1e-4,
                        type = float,
                        help = 'Adam optimizer learning rate')
    parser.add_argument('--alpha_range',
                        nargs = 2,
                        type = int,
                        default = [2,5],
                        help = 'List of length two defining the epoch to start\
                                including pseudo-labels in loss function, and\
                                the epoch to end the ramp-up of loss weighting')
    parser.add_argument('--crop_dataset',
                        default = False,
                        action='store_true',
                        help = 'Decrease the size of the training dataset for debugging speed')
    
    args = parser.parse_args()    
    
    # Load Data
    X_train, X_test, y_train, y_test = load_task(args.task,args.path)
    
    # if args.semisupervised:
    X_train_unlabeled, X_test_unlabeled, y_test_unlabeled_1, y_test_unlabeled_2 = load_task(0,args.path)
    if args.task == 1:
        y_test_unlabeled = y_test_unlabeled_1
    elif args.task == 2:
        y_test_unlabeled = y_test_unlabeled_2    
    
    # Train Model
    
    n_class = len(y_train[0])
    
    model = build_model(n_class)
    
    shuffle = True
    y_stratify = True
    seed = 0

    X_train, X_val, y_train, y_val = split_data(X_train, y_train, args.val_split, shuffle, y_stratify, seed)
    
    
    if args.crop_dataset:
        # Shuffle and shorten
        indices = np.random.randint(0,10000,(320,))
        X_train = X_train.take(indices,axis=0)
        y_train = y_train.take(indices,axis=0)
        indices = np.random.randint(0,1000,(64,))
        X_val = X_test.take(indices,axis=0)
        y_val = y_test.take(indices,axis=0)
        if args.semisupervised:
            indices = np.random.randint(0,4000,(64,))
            X_train_unlabeled = X_train_unlabeled.take(indices,axis=0)
    
    
    if args.semisupervised:
        
        round1epochs = args.alpha_range[0]-1
        
        print(round1epochs)
        
        hist = train_model(model, X_train, y_train, X_val, y_val, lr = args.lr, batch_size = args.batch_size, epochs = round1epochs)
        
        alpha_range = np.array(args.alpha_range) - min(args.alpha_range) + 1
        
        print(alpha_range)
        round2epochs = args.epochs - round1epochs
        
        pseudo = PseudoCallback(model, X_train, y_train, X_train_unlabeled,
                         X_val, y_val, args.batch_size, alpha_range)
                
        hist2 = train_pseudo(model, pseudo, X_val, y_val, lr = args.lr, batch_size = args.batch_size, epochs = round2epochs)
        
        hist2.history['accuracy'] = hist2.history['labeled_accuracy']
        
        for key in hist.history.keys():
            hist.history[key].extend(hist2.history[key])
        
    
    else:
        
        hist = train_model(model, X_train, y_train, X_val, y_val, lr = args.lr, batch_size = args.batch_size, epochs = args.epochs)
  
        
    # Evaluate Model 
    
    n_class = len(y_test[0])
    
    if not os.path.exists("Results"):
        os.mkdir("Results")
    
    plot_performance(hist, save = 'Results/Task'+str(args.task)+'history.png')
    
    # Test set predictions (labeled)
    y_pred_probs = model.predict(X_test)   # Softmax class probabilities from model
    y_pred = np.argmax(y_pred_probs, axis = 1)
    y_pred_oh = to_categorical(y_pred, n_class)
    
    if args.task == 1:
        title = 'Task 1: Scene Level'    
    elif args.task == 2:
        title = 'Task 2: Damage State'
    
    
    plot_multiclass_roc(y_pred_probs, y_pred, X_test, y_test, n_class, title, 
                        figsize=(9.5,5), flag=False, 
                        save= 'Results/Task'+str(args.task)+'ROClabeltest.png')
    
    score_model(y_test, y_pred_oh, y_pred_probs, save = 'Results/Task'+str(args.task)+'scoreslabeltest.csv')
    
        
    y_pred_probs = model.predict(X_test_unlabeled)   # Softmax class probabilities from model
    y_pred = np.argmax(y_pred_probs, axis = 1)
    y_pred_oh = to_categorical(y_pred, n_class)
    
    plot_multiclass_roc(y_pred_probs, y_pred, X_test_unlabeled, y_test_unlabeled, n_class, title, 
                        figsize=(9.5,5), flag=False, 
                        save= 'Results/Task'+str(args.task)+'ROCunlabeltest.png')
    
    score_model(y_test_unlabeled, y_pred_oh, y_pred_probs, save = 'Results/Task'+str(args.task)+'scoresunlabeltest.csv')
    
if __name__ == '__main__':
    
    main()