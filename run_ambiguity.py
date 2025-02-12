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
import torch.nn as nn
from utils import get_model, get_syntax_model, fix_syntax, StopOnAnyTokenSequence, find_subsequence, set_up_dir, simplified_forward
from transformers.generation import StoppingCriteria, StoppingCriteriaList
import itertools
import os
import itertools
import matplotlib.cm as cm

import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def heat2hex(heat, transparency=95):
    # Define your range of values from -1 to 1
    min_value = -1
    max_value = 1
    
    # Create a colormap using 'bwr'
    cmap = plt.get_cmap('bwr')
    
    # Normalize your values to the [0, 1] range
    norm = Normalize(vmin=min_value, vmax=max_value)
    
    # Create a ScalarMappable to map values to colors
    sm = ScalarMappable(cmap=cmap, norm=norm)
    
    # Get the RGBA color for your value
    rgba_color = sm.to_rgba(heat)
    
    # Convert the RGBA color to HEX
    hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255), transparency)
    return hex_color

def get_projection(embeddings, method, labels, ax, c = None, perplexity=50, fontsize=20, s=100,  n_neighbors=15, min_dist=0.1, random_state=42, plot=False): # {0:'red', 31:'blue'}):
    
    # Perform dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        reduced_embeddings = reducer.fit_transform(embeddings)

    elif method == "umap":

        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state, metric='euclidean', init="random")
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    # Plot the reduced embeddings using Matplotlib

    if plot:
        for i, label in enumerate(labels):
            x, y = reduced_embeddings[i]
            #c = color_dict[label]
            ax.scatter(x, y, s=s, alpha=0.7, color= c[i])
            ax.text(x + np.random.normal(0,0.01), y, label, fontsize=fontsize, alpha=0.9)    

    return reduced_embeddings


def get_last_token_embedded(model, input_ids, next_token):

    input_ids =  torch.cat((input_ids.squeeze(), next_token)).unsqueeze(0)
    hidden_states = model(input_ids, output_hidden_states = True)
    hidden_state_next_token = hidden_states[1][-1][0, -1, :].detach().cpu().numpy().squeeze()
    return hidden_state_next_token



def get_layerwise_representations(model, tokenizer, input_text, pooling="mean"):
    """
    Computes sentence embeddings from each layer of a model.
    
    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer corresponding to the model.
        input_text (str): The text to encode.
        pooling (str): Pooling strategy - 'mean' or 'last'.
    
    Returns:
        dict: A dictionary where keys are layer indices and values are sentence representations.
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].cuda()

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of tensors (one per layer)

    layerwise_representations = {}
    
    for layer_idx, hidden_state in enumerate(hidden_states):
        if pooling == "mean":
            sentence_embedding = hidden_state.mean(dim=1)  # Mean over all tokens
        elif pooling == "last":
            sentence_embedding = hidden_state[:, -1, :]  # Last token's hidden state
        elif pooling =="token":
            sentence_embedding = hidden_state
        else:
            raise ValueError("Invalid pooling method. Use 'mean' or 'last'.")
        
        layerwise_representations[layer_idx] = sentence_embedding.squeeze().cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(), skip_special_tokens=False)
    tokens = [token.replace('Ġ', '') for token in tokens]

    return layerwise_representations, tokens
    

import spacy
from collections import defaultdict

def pos_tag_sentence_enumerate(sentence, tokenizer):
    nlp = spacy.load("en_core_web_sm")  # Load the English model
    doc = nlp(sentence)

    toks = tokenizer.tokenize(' ' + sentence) # add space to avoid "J enny"
    subtokens = iter(toks)

    pos_counts = defaultdict(int)  # Dictionary to count occurrences of each POS
    tagged_words = []
    
    words = {i: [] for i in range(len(doc))}
    count = 0
    for i, token in enumerate(doc):
        pos_counts[token.pos_] += 1
        tagged_words.append((f"{token.text}", token.pos_ + '_' + str(pos_counts[token.pos_]), i))

      #  print(subtokens)
        try:
            subt = next(subtokens)
        except:
            import pdb;pdb.set_trace()
        sub_ = [subt.replace("Ġ", "").replace('▁','')]

        
        while token.text.lower() != ''.join(sub_).lower():
         #   subt = next(subtokens)

            try:
                subt = next(subtokens)
            except:
                print('Matching failed', sentence)
                return None, None, None
                
            sub_.append(subt)
        
        for j in range(len(sub_)):
            words[i].append(count)
            count+=1

    try:
        assert count== len(toks)
    except:
        import pdb;pdb.set_trace()
    return tagged_words, words, toks



def get_input_summary(model, tokenizer, input_text, pooling="mean"):
    """
    Computes a sentence embedding from a model's hidden states.
    
    Args:
        model: The pre-trained language model.
        tokenizer: The tokenizer corresponding to the model.
        input_text (str): The text to encode.
        pooling (str): Pooling strategy - 'mean' or 'last'.
    
    Returns:
        torch.Tensor: The sentence representation.
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Get last layer's hidden states
    
    if pooling == "mean":
        sentence_embedding = hidden_states.mean(dim=1)  # Mean over all tokens
    elif pooling == "last":
        sentence_embedding = hidden_states[:, -1, :]  # Last token's hidden state
    else:
        raise ValueError("Invalid pooling method. Use 'mean' or 'last'.")

    return sentence_embedding.squeeze().cpu().numpy()



def get_token_pair(token_explained):
    if token_explained == 'yes':
        contrast_token = 'no'
    elif  token_explained == 'no':
        contrast_token = 'yes'
    if token_explained == 'Yes':
        contrast_token = 'No'
    elif  token_explained == 'No':
        contrast_token = 'Yes'
    else:
        raise
    return contrast_token


def decode(x):
    txt = tokenizer.decode(x.squeeze(), skip_special_tokens=False)
    return txt


def hex_to_rgb(hex_color):
    """Convert a hex color string to an RGB array normalized to [0, 1]."""
    return np.array(mcolors.to_rgb(hex_color))



def get_color_transparent(rgba_color, transparency=100):
    hex_color = "#{:02x}{:02x}{:02x}{:02x}".format(int(rgba_color[0] * 255), int(rgba_color[1] * 255), int(rgba_color[2] * 255), transparency)
    return hex_color


import re

def extract_isolated_capitals(input_string):
    # Use regex to find single capital letters surrounded by non-word characters or spaces
    return re.findall(r'\b[A-Z]\b', input_string)


from transformers import AutoTokenizer, AutoModelForCausalLM

#model_name =  "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "google/gemma-7b-it"


model_name_short = {"meta-llama/Meta-Llama-3.1-8B-Instruct": 'llama3',
                    "google/gemma-7b-it": "gemma"
                   }


if 'llama' in model_name:
    xai = True
elif 'gemma' in model_name:
    xai = False



model_config = AutoConfig.from_pretrained(
    model_name
)
model_config.use_cache = False
model_config._attn_implementation = 'sdpa' 

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    config=model_config,
    quantization_config=None,
    device_map='auto',
    torch_dtype=torch.float16)

_ = model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)



if xai:

    test_ids = tokenizer("Paris is the capital of", return_tensors='pt').to(model.device)
    test1 = model(**test_ids).logits
        
    # Does not change the forward predictions, only the gradient computation for lrp
    # only implemented for llama3 so far
    model = override_llama_xai_layers(model, model_config)
    # model = model.to(torch.float16)
    test2 = model(**test_ids).logits
    # assert torch.isclose(test1, test2)
    print(torch.isclose(test1, test2))
    print(((test1 - test2) ** 2).sum())

tokenizer = AutoTokenizer.from_pretrained(model_name)

embedding_module = model.model.embed_tokens

if 'llama' in model_name:    
    answer_tokens = ['yes', 'no', 'Yes', 'No', 'Ġno', 'Ġyes'] 
    yes_ids = [9891, 9642]
    no_ids = [2201, 2822]

elif 'gemma' in model_name:
    answer_tokens = ['yes', 'no', 'Yes', 'No', '▁no', '▁yes', '▁Yes', '▁No'] 
    yes_tokens = ['yes','Yes', '▁yes', '▁Yes']
    no_tokens = [ 'no', 'No', '▁no', '▁No'] 
    yes_ids = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in yes_tokens]
    no_ids = [tokenizer(token, add_special_tokens=False).input_ids[0] for token in no_tokens]

stop_token_ids = [tokenizer(token, add_special_tokens=False).input_ids for token in answer_tokens]



stop_token_ids_decoded = [tokenizer.decode(token) for token in stop_token_ids]
stop_token_ids_decoded_split = [tokenizer.convert_ids_to_tokens(token) for token in stop_token_ids]


special_string = ''


request_template =    ['Given a sentence, determine if it is ambiguous based on real-world knowledge. \n'
              'Provide a "Yes" or "No" answer along with a justification. \n'
              'The sentence is: {} \n'
              'Answer: ']

data_all = pd.read_csv('llm_ambig_syntactic_ambiguity.csv')


html_all = ''

transparency=95
proj_method = 'tsne'


res_dir = 'results/xai/08022025'
set_up_dir(res_dir)


res_dir_ambig = os.path.join(res_dir, 'ambig')
res_dir_umambig = os.path.join(res_dir, 'unambig')
set_up_dir(res_dir_ambig)
set_up_dir(res_dir_umambig)


# Set seed for generation
SEED = 1


premises = False

eval_data = {}


sentence_case = 'saw'
inds = list(range(1, 143))


sentences = []


for k in inds:

    data  =  data_all.loc[0]

    res_dir_ = res_dir_ambig if data.is_ambiguous else res_dir_umambig
    
    sentence = data.sentence.strip()

    assert sentence_case in sentence

    tagged_words, subtoken_dict, subtokens = pos_tag_sentence_enumerate(sentence, tokenizer)
    if tagged_words is None:
        continue
    tagged_dict = {p: [(word, ix) for word, pos, ix in tagged_words if pos == p] for _, p, _ in tagged_words}
    
    request = request_template[0].format(sentence)

    premise1 = data.premise1
    premise2 = data.premise2

    pooling = 'token'


    if premises:
        
        try:
            if isinstance(premise1, str) & isinstance(premise2, str):
           #    premise_vec1 = get_input_summary(model, tokenizer, premise1, pooling="last")
           #     premise_vec2 = get_input_summary(model, tokenizer, premise2, pooling="last")
    
                try:
                    premise_vec1, premise_toks1 = get_layerwise_representations(model, tokenizer, premise1, pooling=pooling)
                    premise_vec2, premise_toks2 = get_layerwise_representations(model, tokenizer, premise2, pooling=pooling)
    
                except:
                    import pdb;pdb.set_trace()
    
            elif np.isnan(premise1) & np.isnan(premise2) :
                premise_vec1 = premise_vec2 = None
            else:
                import pdb;pdb.set_trace()
    
        except:
            import pdb;pdb.set_trace()
    
    change_type = data.change_type
    is_ambig = data.is_ambiguous
    
    messages = [
        {
            "role": "user",
            "content": request,
        },
    ]
    

    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_tensors="pt").to(model.device)
    
    prompt_length = tokenized_chat.shape[1]
    
    # Create a stopping criteria list    
    stopping_criteria = StoppingCriteriaList([StopOnAnyTokenSequence(stop_token_ids)])
    
    # Regenerate for explainability but with a stopping criterion, requires seed to be set again
    torch.manual_seed(SEED)
    
    # Generate with a stopping criteria and from input_ids (to find prompt length until answer)
    output_until_answer = model.generate(tokenized_chat,
                                         max_new_tokens=5,
                                         eos_token_id=tokenizer.eos_token_id,
                                         pad_token_id=tokenizer.eos_token_id,
                                         repetition_penalty= 1, #config.repetition_penalty,
                                         do_sample=False,
                                         num_return_sequences=1,
                                         stopping_criteria=stopping_criteria,
                                         use_cache=False
                                        )
    
    print(decode(output_until_answer[0]))
    
    responses = tokenizer.decode(output_until_answer[0][prompt_length:], skip_special_tokens=True)
    
    tokenized_chat_until_answer = output_until_answer[:, :-1]
    
    embeddings_out = embedding_module(tokenized_chat_until_answer)
    embeddings_ = embeddings_out.detach().requires_grad_(True)
    
    torch.manual_seed(SEED)
    output_xai = model.generate.__wrapped__(model,
                                            inputs_embeds=embeddings_,
                                            max_new_tokens=1,  # just generate one next token
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.eos_token_id,
                                            repetition_penalty= 1., #config.repetition_penalty,
                                            do_sample=False,
                                            num_return_sequences=1,
                                            stopping_criteria=stopping_criteria,
                                            return_dict_in_generate=True,
                                            output_scores=True,
                                            output_logits=True,
                                            use_cache=False
                                            )
    
    
    responses = tokenizer.decode(output_until_answer[0].squeeze(), skip_special_tokens=True)

    # Take the logits of the last generated token (which should be the answer token)
    xai_scores = output_xai.scores[-1]
    xai_logits = output_xai.logits[-1]
    
    xai_generated_ids = output_xai.sequences[0]
    xai_output_ids = torch.cat([tokenized_chat_until_answer, xai_generated_ids[None, :]], dim=-1)
    
    
    xai_output_ids_ = xai_output_ids[0].tolist()
    output_words = tokenizer.convert_ids_to_tokens(xai_output_ids_, skip_special_tokens=False)
    output_words = [token.replace('Ġ', '').replace('▁','') for token in output_words]
    
    # print(output_words)
    
    assert xai_scores.argmax() == xai_logits.argmax()
    
    next_token = xai_logits.argmax()
    token_explained = tokenizer.decode([next_token]).strip()
    
    orig_logit = xai_logits[:, next_token]
    

    try:
        if token_explained not in list(itertools.chain(*stop_token_ids_decoded_split)):
            print('not in list', token_explained)
            example['relevance_{}'.format(config.xai)] = 'N/T'
            compute_xai = False
            raise
    except:
        import pdb;pdb.set_trace()
    
    responses_xai = tokenizer.decode(xai_output_ids[prompt_length:], skip_special_tokens=False)

    # contrastive
    if False:
        contrast_token = get_token_pair(token_explained)
        
        contrast_id = tokenizer(contrast_token, add_special_tokens=False).input_ids
        selected_logit = xai_logits[:, next_token] - xai_logits[:, contrast_id]
    
    else:
        selected_logit = xai_logits[:, next_token]
    
    # Compute explanation

    print('\n')
   
    output_xai = model(inputs_embeds=embeddings_, output_hidden_states = True)

    hidden_states = output_xai.hidden_states
    embedding_output = hidden_states[0]
    attention_hidden_states = hidden_states[1:]

    
    selected_logit.backward()
    print(selected_logit)
    
    gradient = embeddings_.grad
    relevance_ = gradient * embeddings_
    relevance = relevance_.sum(2).detach().cpu().numpy().squeeze()

    
    
    relevance_with_next = np.hstack([relevance, np.array(0)])
    relevance_with_next_normalized = relevance_with_next / np.max(np.abs(relevance_with_next))
    

    context_ids = tokenizer(f' {sentence.strip()}', add_special_tokens=False).input_ids
    output_words = tokenizer.convert_ids_to_tokens(xai_output_ids.squeeze(), skip_special_tokens=False)
    output_words = [token.replace('Ġ', '').replace('▁','') for token in output_words]
    
    xai_output_ids = xai_output_ids[0].detach().cpu().numpy().tolist()
    start_idx, end_idx = find_subsequence(xai_output_ids, context_ids)
    
    
    output_words_context = output_words[start_idx:end_idx + 1]
    relevance_context = relevance_with_next[start_idx:end_idx + 1]
    relevance_context_normalized = relevance_context / np.max(np.abs(relevance_context))

    
    ######### also do it for each layer  #########

    all_relevance = {}
    attention_mask = torch.ones_like(tokenized_chat_until_answer)
    
    start_ = 0

    end_ = model.config.num_hidden_layers
    for l in range(start_, end_):


        rotary_embed = model.model.rotary_emb if 'llama' in model_name else None

        outputs, o2, relevance, hidden_states_xai, answer_logs, answer_probs = simplified_forward(model, tokenizer, model.model.layers, model.model.norm, rotary_embed, model.lm_head, embeddings_, attention_mask, model_name, yes_ids=yes_ids, no_ids=no_ids,  l_lrp=(l, next_token), xai=xai)
    
        r_with_next = np.hstack([relevance, np.array(0)])


        relevance_context = r_with_next[start_idx:end_idx + 1]
        relevance_context_normalized = relevance_context / np.max(np.abs(relevance_context))
            
        all_relevance[l] = (relevance, hidden_states_xai)

    assert l+1==end_
    
    all_relevance[l+1] = (np.zeros_like(relevance),  outputs[l+1][:,-1,:].detach().cpu().numpy().squeeze())


    contrast_token = get_token_pair(token_explained)

    contrast_id = torch.tensor(tokenizer(contrast_token, add_special_tokens=False).input_ids).cuda()
    class_id = torch.tensor(tokenizer(token_explained, add_special_tokens=False).input_ids).cuda()

    contrast_vec0 = model.lm_head.weight[contrast_id].detach().cpu().numpy().squeeze()
    class_vec0 = model.lm_head.weight[class_id].detach().cpu().numpy().squeeze()


    if False:
    
        H_all = [contrast_vec0, class_vec0]
        L_all = [ contrast_token.lower(), token_explained.lower() ]
        R_all = [-1, 1.]
        C_all = ['pink', 'green']
        filter_mask = [-1,-1]

    else:
        H_all = []
        L_all = []
        R_all = []
        C_all = []
        EOS_all = []
        filter_mask = []


    
    
    if premises:
        if premise_vec1 is not None:
        #    import pdb;pdb.set_trace()
            if False:
                H_all += [premise_vec1, premise_vec2]
                L_all += [premise1, premise2]
                R_all += [1, 1.]
                C_all += ['magenta', 'lime']
    
            keys_ = premise_vec1.keys()
    
    
            if pooling == 'token':
             for jj in sorted(keys_):
                    H_all += [premise_vec1[jj][itok] for itok in range(len(premise_toks1))] + [premise_vec2[jj][itok] for itok in range(len(premise_toks2))]
    
    
                    if jj == max(keys_):
    
                        L_all += [tok + ' ' + str(jj) for itok, tok in enumerate(premise_toks1)] +  [tok + ' ' + str(jj) for itok, tok in enumerate(premise_toks2)]
    
                    else:
                         L_all += [str(jj) for itok, tok in enumerate(premise_toks1)] +  [str(jj) for itok, tok in enumerate(premise_toks2)]
                        
                    
                    R_all += [1.]*len(premise_toks1) + [1.]*len(premise_toks2)                 
                    C_all += ['magenta']*len(premise_toks1) + ['lime']*len(premise_toks2)
    
            else:
                
                for jj in sorted(keys_):
                    H_all += [premise_vec1[jj], premise_vec2[jj]]
        
                    if jj == max(keys_):
                        L_all += [premise1 + ' ' + str(jj), '\n\n ' +  premise2 + '\n' + str(jj)]
                    else:
                        L_all += [str(jj), str(jj)]
                    R_all += [1, 1.]
                    C_all += ['magenta', 'lime']




    n0 = len(H_all)

    keys_ = list(range(start_, end_))

    if end_-1 not in keys_:
        keys_ += [end_-1]


    seq = {}
    all_last = []

    all_dists = {'euclidean':[], 'cosine':[] }
    for k_ in  keys_:

        r,h = all_relevance[k_]

        if False:

            # Take all
            relevance_norm = r / np.max(np.abs(r))
            h_rep = h
            
            L = output_words[:-1]
            C = [heat2hex(val, transparency) for i,val in enumerate(relevance_norm)]


        if True:
            r_with_next = np.hstack([r, np.array(0)])
            relevance_context = r_with_next[start_idx:end_idx + 1]
            relevance_context_normalized = relevance_context / np.max(np.abs(relevance_context))

            h_rep = h[start_idx:end_idx + 1]

            do_dist = True # if len(h_rep) == len(tagged_words) else False     

            if do_dist:
                H_ = {w_[0] + '_' + str(w_[2]): [] for w_ in  tagged_words}
                W_ = []
                T_ = []

                try:

                    focus_tags = [pos_ for pos_ in tagged_dict.keys() if pos_.startswith('NOUN')]
                    focus_tags += [pos_ for pos_ in tagged_dict.keys() if pos_.startswith('VERB')]
                    focus_tags += [pos_ for pos_ in tagged_dict.keys() if pos_.startswith('PUNCT')]
                    
                    for t in focus_tags:
                        word_, ix_ = tagged_dict[t][0][0], tagged_dict[t][0][1]#, tagged_dict[t][0][2]   

                        if len(subtoken_dict[ix_])==1:
                            H_[word_+'_' + str(ix_)].append(h_rep[subtoken_dict[ix_][0]])
                            W_.append(word_ + '_' + str(ix_))
                            T_.append(t)

                        else:
                            # take the last
                            H_[word_+'_' + str(ix_)].append(h_rep[subtoken_dict[ix_][-1]])
                            W_.append(word_ + '_' + str(ix_))
                            T_.append(t)
                            

                        if False:
                            for sub_ix_ in subtoken_dict[ix_]:
                                H_[word_+'_' + str(ix_)].append(h_rep[ix_])
                                W_.append(word_ + '_' + str(ix_))
                                T_.append(t)

                except:
                   import pdb;pdb.set_trace()


            
                H_focus = []
                _ = [H_focus.extend(H_[w_]) for w_ in W_]           
                
               # dist_matrix = squareform(pdist(H_focus, metric=lambda u, v: np.dot(u,v)))        
                dist_matrix = squareform(pdist(H_focus, metric='euclidean'))
                cosine_matrix = 1. - squareform(pdist(H_focus, metric='cosine'))

                all_dists['euclidean'].append(dist_matrix)
                all_dists['cosine'].append(cosine_matrix)

            L = [o_ + '_' +str(ii) for ii, o_ in enumerate(output_words_context)]
            C = [heat2hex(val, transparency) for i,val in enumerate(relevance_context_normalized)]


            if k_ == 0:
                seq = {n_+ '_' + str(ii) :[] for ii, n_ in enumerate(output_words_context)}

            for ii, n_ in enumerate(output_words_context):
                seq[n_+ '_' + str(ii)].append(len(L_all)+ii)
    
        all_last.append(all_relevance[end_-1][1])
        
    
        H_all.extend(h_rep)
        L_all.extend([l_ + ' ' + str(k_) for l_ in L])
    #    R_all.extend(relevance_context_normalized)
        C_all.extend(C)

        EOS_all.append(h[-1])

        filter_mask.extend([k_]*len(L))

        assert len(H_all) == len(C_all)


    ########################################################################
    
    score_class = (model.lm_head.weight[class_id]*outputs[l+1][:,-1,:]).sum().detach().cpu().numpy()
    score_contrast = (model.lm_head.weight[contrast_id]*outputs[l+1][:,-1,:]).sum().detach().cpu().numpy()


    dist_class = torch.nn.functional.pairwise_distance(model.lm_head.weight[class_id],outputs[l+1][:,-1,:]).detach().cpu().numpy()
    dist_contrast = torch.nn.functional.pairwise_distance(model.lm_head.weight[contrast_id],outputs[l+1][:,-1,:]).detach().cpu().numpy()

    print(class_id, score_class, contrast_id, score_contrast)
    print('*****')


    import matplotlib.gridspec as gridspec

    f = plt.figure(figsize=(9, 11))

    if True:
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1,1,1])  # Main plot is 3x the height of the smaller one
        
        ax = plt.subplot(gs[0])  # Main plot
        ax2 = plt.subplot(gs[1])  # Smaller plot below
        ax3 = plt.subplot(gs[2])  # Smaller plot below
        ax4 = plt.subplot(gs[3])  # Smaller plot below


    layer_filt = 0

    layer_mask = np.ones(end_)
    layer_mask[layer_filt] = 0.
    
    mask_ = np.array(filter_mask)!=layer_filt
    L_all_filt = np.array(L_all)[mask_]
    H_all_filt = np.array(H_all)[mask_]
    C_all_filt = np.array(C_all)[mask_]

    
    reduced_embeddings = get_projection(H_all_filt, method=proj_method, labels=L_all, ax=None, c=C_all,
                                perplexity=20, fontsize=5, s=70,
                                n_neighbors=32,
                                min_dist=.6)
    
    for i, label in enumerate(L_all_filt):
        x, y = reduced_embeddings[i]
        ax.scatter(x, y, s=70, color= C_all_filt[i])

        fontsize = 7

        if False:
            # premises 
            if i <= n0 and premises:
                if i %4==0 or i+1 == n0:
                    ax.text(x + np.random.normal(0,0.03), y + np.random.normal(0,0.03), label, fontsize=fontsize, alpha=1.)
            if i >= len(L_all)-3:
                ax.text(x + np.random.normal(0,0.03), y + np.random.normal(0,0.03), label, fontsize=fontsize, alpha=1.)
            elif i %4==0:
                ax.text(x + np.random.normal(0,0.03), y + np.random.normal(0,0.03), label, fontsize=fontsize, alpha=1.)
        
        else:

            ax.text(x + np.random.normal(0,0.02), y + np.random.normal(0,0.02), label, fontsize=fontsize, alpha=1.)

    if True:
        try:
            for ii, n_ in enumerate(output_words_context):
                ixs = np.array(seq[n_+'_'+str(ii)])[layer_mask==1.]  - 1*(len(H_all)-len(H_all_filt))      
                ax.plot(np.array(reduced_embeddings)[ixs,0], np.array(reduced_embeddings)[ixs,1], alpha=0.4, color='#a0a0a0') 

        except:
            import pdb;pdb.set_trace()

    
    ax.spines[['right', 'top']].set_visible(False)

    ax.set_xlabel(proj_method + ' 1', fontsize=14)
    ax.set_ylabel(proj_method + ' 2',  fontsize=14)

    str_ = ''
    for kk,vv in answer_probs.items():
        try:
            str_ += kk + ' {:0.3f} '.format(vv[-1]) 
        except:
            import pdb;pdb.set_trace()
    
    title_str = sentence + '\n' + ', '.join(['Model: ' + token_explained, str_,  ' | ' + change_type, 'is_ambig: '+str(is_ambig)  ])

    ax.set_title(title_str, fontsize=11)


    for kk,v in answer_logs.items():
        c = 'green' if kk=='yes' else 'red'    
        xs = np.arange(len(v))
        p = answer_probs[kk][-1]
        ax2.plot(xs[1:], v[1:], label=kk, marker='x', linestyle='-', color =c)
        ax2.text(xs[-1], v[-1], s = 'p=' + str(p.round(3)), color =c)
    ax2.set_ylabel('logits')
    ax2.set_xlabel('layers')


    if do_dist:
        
        ixs = list(zip(*np.triu_indices(len(W_),k=1)))

        keys_filt = keys_[1:]
        for ix,iy in  ixs:
    
            l2dists = [all_dists['euclidean'][k][ix,iy] for k in keys_filt]
            cosdists = [all_dists['cosine'][k][ix,iy] for k in keys_filt]
                
            ax3.plot(keys_filt, l2dists, label=W_[ix] + '_' + W_[iy])
            ax4.plot(keys_filt, cosdists, label = W_[ix] + '_' + W_[iy])
    
            
         #   ax3.text(keys_[0], l2dists[0], s = W_[ix] + '_' + W_[iy], label=)
         #   ax4.text(keys_[0], cosdists[0], s = W_[ix] + '_' + W_[iy])
    
            ax3.set_title('L2')
            ax4.set_title('Cosine')


    ax4.set_ylim([0, 1.])

    plt.legend(fontsize=7, ncol=5)

    f.tight_layout()
    f.savefig(os.path.join(res_dir_, str(k).zfill(3) +'_' +special_string +'_' + proj_method + '.pdf'), dpi=300, transparent=True)
    plt.show()


    eval_data[k-1] = {'data': data, 'probs': answer_probs, 'logits': answer_logs, 'L_all':L_all, 'H_all':H_all, 'C_all':C_all , 'R_all': R_all, 'context': (output_words_context, relevance_context), 'token_explained': token_explained,  'tagged_words':tagged_words, 'EOS_all': EOS_all}

  
pickle.dump(eval_data, open(os.path.join(res_dir,  model_name_short[model_name] + '_' + sentence_case +  '_results.p'), 'wb'))

