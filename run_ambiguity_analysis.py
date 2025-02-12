import pandas as pd
import numpy as np
import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
from os.path import join, exists
from os import makedirs
from datasets import load_dataset
from xai.xai_llama import override_llama_xai_layers
import matplotlib.pyplot as plt
import numpy as np
from utils import set_up_dir
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def linear_probe_cv(data, labels, cv=5, max_iter=1000):
    """
    Performs a linear probe using logistic regression with cross-validation.

    Parameters:
    - data: NumPy array of shape (num_samples, hidden_dim), hidden representations
    - labels: NumPy array of shape (num_samples,), corresponding class labels
    - cv: Number of cross-validation folds (default=5)
    - max_iter: Maximum iterations for logistic regression (default=1000)

    Returns:
    - mean_accuracy: Mean accuracy over all folds
    - std_accuracy: Standard deviation of accuracy across folds
    """
    clf = LogisticRegression(max_iter=max_iter)

    # Stratified K-Fold ensures each fold has balanced class distribution
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(clf, data, labels, cv=skf, scoring='accuracy')

    # Return mean and std accuracy
    return scores.mean(), scores.std()

# Example usage
# mean_acc, std_acc = linear_probe_cv(data, labels, cv=5)
# print(f"Mean Accuracy: {mean_acc:.4f}, Std Dev: {std_acc:.4f}")


def linear_probe(data, labels, test_size=0.2, max_iter=1000, random_state = 42):
    """
    Performs a linear probe using logistic regression on given hidden states.

    Parameters:
    - data: NumPy array of shape (num_samples, hidden_dim), hidden representations
    - labels: NumPy array of shape (num_samples,), corresponding class labels
    - test_size: Fraction of data used for testing (default=0.2)
    - max_iter: Maximum iterations for logistic regression (default=1000)

    Returns:
    - accuracy: Accuracy score of the linear classifier
    - model: Trained logistic regression model
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)

    # Train logistic regression
    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    f1 =  f1_score(y_test, y_pred)
    return accuracy, f1

import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform


def get_projection(embeddings, method, n_components = 2,  perplexity=50, fontsize=20, n_neighbors=15, min_dist=0.1, random_state=42):
    
    # Perform dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)

    elif method == "umap":

        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric='euclidean', init="random")
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    return reduced_embeddings


def decode(x):
    txt = tokenizer.decode(x.squeeze(), skip_special_tokens=False)
    return txt


data_all = pd.read_csv('llm_ambig_syntactic_ambiguity.csv')


# apptainer run --nv /home/oeberle/requirements/foundrationales/container_cogeval.sif python run_ambiguity_analysis.py




all_change_types = ['subj_random_rhyme',
                     'impossible',
                     'subj_random',
                     'all_random',
                     'pp_synonym',
                     'original',
                     'obj_synonym',
                     'inverse_object',
                     'subj_synonym',
                     'pp_random',
                     'all_random_rhyme',
                     'all_synonym',
                     'pp_random_rhyme',
                     'obj_random',
                     'obj_random_rhyme']

color_change_type = {k:np.random.rand(3) for k in all_change_types}


color_change_type['obj_random_rhyme'] = color_change_type['obj_random'] 
color_change_type['subj_random_rhyme'] = color_change_type['subj_random'] 
color_change_type['all_random_rhyme'] = color_change_type['all_random'] 
color_change_type['pp_random_rhyme'] = color_change_type['pp_random'] 


ignore_list = ['impossible']
random_state_ = 42


# Contains all 142 samples for the "The man saw the woman with the telescope." 
files = [
        'results/xai/llama3_saw_results.p', 
         'results/xai/gemma_saw_results.p']

res_dir_main = 'results/xai/11022025'
set_up_dir(res_dir_main)

f, axs = plt.subplots(1,2, figsize = (8,4))

f2, ax2 = plt.subplots(1,1)

focus_verb = 'saw'

method = 'pca'


for file_ in files:

    eval_data = pickle.load(open(file_, 'rb'))
    
    print(len(eval_data))

    
    if 'gemma' in file_:
        n_layers = 28
        model_name_short = 'gemma'
    
    elif 'llama' in file_:
        n_layers = 32
        model_name_short = 'llama3'


    res_dir = os.path.join(res_dir_main, model_name_short)
    set_up_dir(res_dir)
    
    probe_acc = {}
    probe_f1 = {}
    
    for type_ in ['eos', 'VERB_1', 'NOUN_1', 'NOUN*PROPN_1', 'NOUN_last', 'PUNCT_last']:
    
        Yt = []
        Yp = []
        sentences = []

        
        yes_logs = []
        scores_all = []
        change_types_list = []
        
        C = []
        C_type = []
        
        As = {i:[] for i in range(n_layers)}
        Bs =  {i:[] for i in range(n_layers)}
        EOS = {i:[] for i in range(n_layers)}
        
        verbs = []
        probe_acc[type_] = []
        probe_f1[type_] = []
        
        for k,v in eval_data.items():#      

            data = v['data']
    
            sentence = data.sentence
            change_type = data.change_type
            is_ambig = data.is_ambiguous
        
            # cosine analysis
            H_all = v['H_all']
            tagged_words = v['tagged_words']
            tagged_dict = {p: [(word, ix) for word, pos, ix in tagged_words if pos == p] for _, p, _ in tagged_words}
    
    
            if change_type in ignore_list:
                continue
        
            if False:
                change_types = ['original', 'pp_synonym', 'pp_random', 'subj_synonym', 'subj_random', 'inverse_object', 'obj_synonym', 'obj_random']
                
                if change_type not in change_types :
                    continue
            
            verb = tagged_dict['VERB_1'][0][0]
    
            if focus_verb != 'all':
                if verb != focus_verb:
                    continue
        
            if type_ not in tagged_dict:
     
                if type_ == 'PUNCT_last':
                    if (np.array([True if 'PUNCT' in t_ else False for t_ in tagged_dict.keys()])==True).any() ==False:
                        print(type_, sentence)
                        continue
    
                elif type_ == 'NOUN*PROPN_1':
                    # determine which one is first
                    k1 = 'NOUN_1'
                    k2 = 'PROPN_1'
    
                    if (k1 in tagged_dict) and (k2 in tagged_dict):
                        hilf1 = tagged_dict[k1]
                        hilf2 = tagged_dict[k2]
                        assert len(hilf1) == 1
                        assert len(hilf2) == 1
    
                        ix_1 = [h_[1] for h_ in  hilf1]
                        ix_2 = [h_[1] for h_ in  hilf2]
    
                        if ix_1 < ix_2:
                            type_used = k1
                        else:
                            type_used = k2
    
                    elif k1 in tagged_dict:
                        type_used = k1
    
                    elif k2 in tagged_dict:
                        type_used = k2
                    else:
                        continue
                                   
                
                elif type_ not in ['eos', 'NOUN_last']:
                    continue
                      
            model_probs = v['probs']
            p_yes = np.array(model_probs['yes'])
            p_no = np.array(model_probs['no'])
        
            model_logits = v['logits']
            logit_yes = np.array(model_logits['yes'])
            logit_no = np.array(model_logits['no'])
        
        
            y_model = 1 if p_yes[-1] >= p_no[-1] else 0
            y_true = is_ambig
                
            if is_ambig:
                line1,  = axs[0].plot(logit_yes, label = 'Decision: Yes', color='green')
                line2, = axs[0].plot(logit_no, label = 'Decision: No', color='red')
        
                line3 = ax2.scatter(p_no[-1], p_yes[-1], color = 'green', label = 'Ambig')
            
            else:
                axs[1].plot(p_yes, label = 'Decision: Yes', color='green')
                axs[1].plot(p_no, label = 'Decision: No', color='red')
                
                line4 = ax2.scatter(p_no[-1], p_yes[-1], color = 'red', label = 'Unambig.')
            
            ax2.text(p_no[-1], p_yes[-1], s= str(k) + ' ' + sentence)
        
          #  print(y_true, y_model, '\t' , p_yes[-1].round(2),  p_no[-1].round(2), '\t'+ change_type, '\t'+  sentence)
            
            h_dict = {type_ : []}
    
            if type_ == 'NOUN_last':
                type_used = sorted([k for k in tagged_dict.keys() if 'NOUN' in k])[-1]
                print(type_used)
            elif type_ == 'PUNCT_last':
                type_used = sorted([k for k in tagged_dict.keys() if 'PUNCT' in k])[-1]
                print(type_used)     
            elif type_ == 'NOUN*PROPN_1':
                type_used = type_used
                # see above
    
            elif type_ == 'eos':
                type_used = 'VERB_1'
                # dummy
            else:
                type_used = type_
        
            w1, ix1 = tagged_dict[type_used][0]
        
            if (np.array(['{}_{}'.format(w1,ix1) in x for x in v['L_all']])==True).any()==False:
                continue
                
        
            for i,x in enumerate(v['L_all']):
        
                layer = int(x.split(' ')[-1])
            
                if '{}_{}'.format(w1,ix1) in x:
                    h_dict[type_].append(H_all[i])
                    As[layer].append(H_all[i])
                
            for i in range(n_layers):
                EOS[i].append(v['EOS_all'][i])
    
    
            yes_logs.append(p_yes[-1])
          
            C.append('green' if is_ambig else 'red')
            C_type.append(color_change_type[change_type])
    
            
            sentences.append(sentence)
            Yt.append(y_true)
            Yp.append(y_model)
            change_types_list.append(change_type)
        
        
            if change_type == 'original':
                verbs.append('*' +' | ' + sentence + '*')
            else:
                verbs.append( change_type +' | ' + sentence )
            
            
        Yp, Yt = np.array(Yp), np.array(Yt)
        match = Yp==Yt
    
        print(match.sum()/len(match))


        np.random.seed(random_state_)
            
        for i in range(n_layers):
        
        
            # ax3 = axs3[i]
        
            A = np.array(As[i])
        
            if type_ == 'eos':
                AB = EOS[i]
            else:
                AB = A
        
            mask_rand = np.arange(0,len(AB))
            np.random.shuffle(mask_rand)


            f3 = plt.figure(figsize=(6, 6))

            n_components = 2
            reduced = get_projection(np.array(AB), method=method, n_components=n_components, perplexity=25, n_neighbors=10, min_dist=0.1, random_state = random_state_)

            
            if n_components == 2:
                ax3 = f3.add_subplot(111)  # Set up 3D plot

                Xs, Ys = reduced[mask_rand,0], reduced[mask_rand,1]
                Cs = np.array(C)[mask_rand]
                    

                hilf = np.array(change_types_list)[mask_rand] 
                
                ax3.scatter(Xs[hilf!='original'], Ys[hilf!='original'], c=Cs[hilf!='original'], alpha=0.5, s = 120)
                
                ax3.scatter(Xs[hilf=='original'], Ys[hilf=='original'], c=Cs[hilf=='original'], alpha=0.5, s = 150, marker = '*', edgecolor= 'black')
                
                for j,_ in enumerate(Xs):

                    ax3.text(Xs[j], Ys[j]+np.random.normal(0,0.02), np.array(verbs)[mask_rand][j], fontsize=1, color=np.array(C)[mask_rand][j])

            
            probe_data = AB
            probe_labels = np.array(Yt)
        
            acc_ = []
            f1_ = []
            for seed in [42, 81, 95, 56, 23]:
                acc, f1 = linear_probe(probe_data, probe_labels, test_size=0.4, max_iter=1000, random_state=seed)
                acc_.append(acc)
                f1_.append(f1)

        
            acc_mean = np.mean(acc_)
            f1_mean = np.mean(f1_)
            print(i, acc_mean, f1_mean)
        
            probe_acc[type_].append(acc_mean)
            probe_f1[type_].append(f1_mean)

        
            ax3.set_title('probe acc: {:0.2f} | f1:  {:0.2f} '.format(acc_mean, f1_mean))
        
            res_dir_type = os.path.join(res_dir, type_ + '_' + focus_verb)
            set_up_dir(res_dir_type)
            
            f3.savefig(os.path.join(res_dir_type, '{}_l{}.pdf'.format(method,i)), dpi=300,  bbox_inches = "tight")
            f3.show()
    
            plt.close()
    
            if focus_verb != 'all':
                try:
                    original = np.array(sentences)[np.array(change_types_list)=='original']
                    assert len(original) ==1 
                    original = original[0]

                except:
                    import pdb;pdb.set_trace()
            else:
                original = 'all'
    
    f, ax = plt.subplots(1,1, figsize=(5,4))
    
    for k in probe_acc.keys():
    
        ax.plot(range(n_layers), probe_acc[k], label=k)
        print(np.sum(probe_acc[k]))
    
    #ax.set_title('Type: ' + original) 
    
    llm_acc = (Yt==Yp).mean()
    ax.hlines(llm_acc, xmin=0, xmax=n_layers, color='black', linestyle='--')
    
    ax.text(0.7*(n_layers/2), llm_acc+0.02, model_name_short)
    
    ax.set_ylim([0.28,1.0])
    
    ax.set_ylabel('probe accuracy')
    ax.set_xlabel('layers')
    ax.spines[['right', 'top']].set_visible(False)
    
    plt.legend(fontsize=9, ncol=3, bbox_to_anchor=(1.01, 1.2))
    
    f.savefig(os.path.join(res_dir, 'probe_acc_{}_{}.pdf'.format(focus_verb, model_name_short)), dpi=300,  bbox_inches = "tight")
    plt.show()
