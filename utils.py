import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.generation import StoppingCriteria, StoppingCriteriaList
import transformers
import torch

import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn as nn


def simplified_forward(model, tokenizer, layers, norm, rotary_emb, lm_head, inputs_embeds, attention_mask, modelname, yes_ids, no_ids,  l_lrp=(None, None), xai=True):


    if 'llama' in modelname:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
    
        hidden_states = inputs_embeds
        position_embeddings = rotary_emb(hidden_states, position_ids)

    elif 'gemma' in modelname:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        hidden_states = inputs_embeds
        normalizer = torch.tensor(model.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        
    
    # decoder layers
    all_hidden_states = {}
    all_self_attns = () 
    next_decoder_cache = None

    causal_mask = model.model._update_causal_mask(
        attention_mask, inputs_embeds, None, None, False
    )

    answer_probs = {'yes': [] , 'no': []}
    answer_logs = {'yes': [] , 'no': []}


    last_ = model.config.num_hidden_layers - 1

    
    for l, decoder_layer in enumerate(layers):
        
        all_hidden_states[l] = hidden_states

        if l_lrp[0] is not None:
            if l_lrp[0] == l:
                hidden_states_ = hidden_states.detach().requires_grad_(True)

                layer_outputs = decoder_layer(
                hidden_states_,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings if 'llama' in modelname else None,
            )

            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=position_embeddings if 'llama' in modelname else None,
                )
        
        else:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings if 'llama' in modelname else None,
            )


        if l_lrp[0] == last_:
            # each layer classification (only do this when calling the function for the last layer)
            output_scores = lm_head(hidden_states)
            probs = nn.functional.softmax(output_scores[:,-1,:], dim=-1)
            
            pred_intermediate = output_scores[:, -1, :][0].argmax()
            p = nn.functional.softmax(output_scores[:,-1,:], dim=-1)[:,int(pred_intermediate)].squeeze().detach().cpu().numpy()

            

            str_ = ''
            for tok_, ids_ in [('yes', yes_ids), ('no', no_ids)]:

               
                logs_= []
                ps_ = []
                for id_ in ids_:
                
                    logs_.append(output_scores[:,-1,:][:,int(id_)].squeeze().detach().cpu().numpy())
                    ps_.append(probs[:,int(id_)].squeeze().detach().cpu().numpy())
                    
                str_ +=  tok_ + ' '.join(['{:0.3f}'.format(p_) for p_ in ps_]) + '\t'
    
                answer_probs[tok_].append(np.max(ps_))
                answer_logs[tok_].append(np.max(logs_))
    
                
            
            print(l, int(pred_intermediate), tokenizer.convert_ids_to_tokens([int(pred_intermediate)]), p)
            print(str_)

        hidden_states = layer_outputs[0]

    hidden_states = norm(hidden_states)

    
    all_hidden_states[l+1] = hidden_states


    output_scores = lm_head(hidden_states)

    pred_intermediate = output_scores[:, -1, :][0].argmax()
    probs = nn.functional.softmax(output_scores[:,-1,:], dim=-1)
    p = probs[:,int(pred_intermediate)].squeeze().detach().cpu().numpy()

    
    if l_lrp[0] == last_:
        str_ = ''
        for tok_, ids_ in [('yes', yes_ids), ('no', no_ids)]:

            logs_= []
            ps_ = []
            for id_ in ids_:
            
                logs_.append(output_scores[:,-1,:][:,int(id_)].squeeze().detach().cpu().numpy())
                ps_.append(probs[:,int(id_)].squeeze().detach().cpu().numpy())
                
            str_ +=  tok_ + ' '.join(['{:0.3f}'.format(p_) for p_ in ps_]) + '\t'

            answer_probs[tok_].append(np.max(ps_))
            answer_logs[tok_].append(np.max(logs_))


        print('final', l, int(pred_intermediate), tokenizer.convert_ids_to_tokens([int(pred_intermediate)]), p)
        print(str_)

    
    if l_lrp[0] is not None and xai == True:
        next_token = l_lrp[1]

        assert int(pred_intermediate) == int(next_token)

        selected_logit = output_scores[:, -1, next_token]
        selected_logit.backward()
        
        gradient = hidden_states_.grad
        relevance_ = gradient * hidden_states_
        relevance = relevance_.sum(2).detach().cpu().numpy().squeeze()

    else:
        relevance = np.ones(inputs_embeds.squeeze().shape[0])


    return all_hidden_states, output_scores, relevance, hidden_states_.detach().cpu().numpy().squeeze(), answer_logs, answer_probs



def simplified_forward_gemma(model, layers, norm, rotary_emb, lm_head, inputs_embeds, attention_mask, modelname, yes_ids, no_ids,  l_lrp=(None, None), xai=True):



    # create position_ids on the fly for batch generation
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    hidden_states = inputs_embeds
    normalizer = torch.tensor(model.config.hidden_size**0.5, dtype=hidden_states.dtype)
    hidden_states = hidden_states * normalizer



    
    # decoder layers
    all_hidden_states = {}
    all_self_attns = () 
    next_decoder_cache = None

    causal_mask = model.model._update_causal_mask(
        attention_mask, inputs_embeds, None, None, False
    )

    answer_probs = {'yes': [] , 'no': []}
    answer_logs = {'yes': [] , 'no': []}

    for l, decoder_layer in enumerate(layers):
        
        all_hidden_states[l] = hidden_states

        if l_lrp[0] is not None:
            if l_lrp[0] == l:
                hidden_states_ = hidden_states.detach().requires_grad_(True)

                layer_outputs = decoder_layer(
                hidden_states_,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )

            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                )
        
        else:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                position_embeddings=position_embeddings,
            )


        if l_lrp[0] == 31:
            # each layer classification (only do this when calling the function for the last layer)
            output_scores = lm_head(hidden_states)
            pred_intermediate = output_scores[:, -1, :][0].argmax()
            p = nn.functional.softmax(output_scores[:,-1,:], dim=-1)[:,int(pred_intermediate)].squeeze().detach().cpu().numpy()


            str_ = ''
            for tok_, id_ in [('yes', yes_ids), ('no', no_ids)]:

                log1 = output_scores[:,-1,:][:,int(id_[0])].squeeze().detach().cpu().numpy()
                log2 = output_scores[:,-1,:][:,int(id_[1])].squeeze().detach().cpu().numpy()
            
                
                p1 = nn.functional.softmax(output_scores[:,-1,:], dim=-1)[:,int(id_[0])].squeeze().detach().cpu().numpy()
                p2 = nn.functional.softmax(output_scores[:,-1,:], dim=-1)[:,int(id_[1])].squeeze().detach().cpu().numpy()
                str_ +=  tok_ + ' {:0.3f} {:0.3f} \t'.format(p1, p2)

                answer_probs[tok_].append(np.max([p1, p2]))
                answer_logs[tok_].append(np.max([log1, log2]))

            
            
            print(l, int(pred_intermediate), tokenizer.convert_ids_to_tokens([int(pred_intermediate)]), p)
            print(str_)

        hidden_states = layer_outputs[0]

    hidden_states = norm(hidden_states)

    
    all_hidden_states[l+1] = hidden_states


    output_scores = lm_head(hidden_states)

    pred_intermediate = output_scores[:, -1, :][0].argmax()
    probs = nn.functional.softmax(output_scores[:,-1,:], dim=-1)
    p = probs[:,int(pred_intermediate)].squeeze().detach().cpu().numpy()


    if l_lrp[0] == 31:

        str_ = ''
        for tok_, id_ in [('yes', yes_ids), ('no', no_ids)]:

            log1 = output_scores[:,-1,:][:,int(id_[0])].squeeze().detach().cpu().numpy()
            log2 = output_scores[:,-1,:][:,int(id_[1])].squeeze().detach().cpu().numpy()
            
            p1 = probs[:,int(id_[0])].squeeze().detach().cpu().numpy()
            p2 = probs[:,int(id_[1])].squeeze().detach().cpu().numpy()
            str_ +=  tok_ + ' {:0.3f} {:0.3f} \t'.format(p1, p2)


            answer_probs[tok_].append(np.max([p1, p2]))
            answer_logs[tok_].append(np.max([log1, log2]))


        print('final', l, int(pred_intermediate), tokenizer.convert_ids_to_tokens([int(pred_intermediate)]), p)
        print(str_)

    
    if l_lrp[0] is not None and xai == True:
        next_token = l_lrp[1]

        assert int(pred_intermediate) == int(next_token)

        selected_logit = output_scores[:, -1, next_token]
        selected_logit.backward()
        
        gradient = hidden_states_.grad
        relevance_ = gradient * hidden_states_
        relevance = relevance_.sum(2).detach().cpu().numpy().squeeze()

    else:
        relevance = np.ones(inputs_embeds.squeeze().shape[0])


    return all_hidden_states, output_scores, relevance, hidden_states_.detach().cpu().numpy().squeeze(), answer_logs, answer_probs




# Function to extract embeddings from a specific layer
def get_layer_embeddings(tokenized_chat, model, layer_index):
    with torch.no_grad():
        outputs = model(tokenized_chat, output_hidden_states=True)
    # Ensure the layer index is within bounds
    if layer_index < 0 or layer_index >= len(outputs.hidden_states):
        raise ValueError(f"Invalid layer_index: {layer_index}. Must be between 0 and {len(outputs.hidden_states) - 1}.")
    # Extract the embeddings from the specified layer
    layer_embeddings = outputs.hidden_states[layer_index].squeeze(0)  # (seq_len, hidden_size)
    return layer_embeddings




def visualize_embeddings(pickle_file, method="pca"):
    """
    Load embeddings from a pickle file, reduce dimensions, and plot them in 2D using Matplotlib.

    Args:
        pickle_file (str): Path to the pickle file containing embeddings.
        method (str): Dimensionality reduction method ('pca' or 'tsne').
        average_sequence (bool): Whether to average embeddings across sequence length.
    """
    # Load embeddings from the pickle file
    with open(pickle_file, "rb") as f:
        embeddings_dict = pickle.load(f)

    # Prepare embeddings and labels
    embeddings = []
    labels = []
    for key, v in embeddings_dict.items():

        embedding = v['embeddings']
        
        embeddings.extend(embedding)  # Average across sequence

        
        labels.extend([key]*len(embedding))

    embeddings = np.array(embeddings)

    # Perform dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        reduced_embeddings = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
    else:
        raise ValueError("Unsupported method. Use 'pca' or 'tsne'.")

    # Plot the reduced embeddings using Matplotlib
    plt.figure(figsize=(8, 6))


    color_dict = {0:'red', 31:'blue'}
    
    for i, label in enumerate(labels):
        x, y = reduced_embeddings[i]

        c = color_dict[label]
        
        plt.scatter(x, y, s=20, alpha=0.7, color= c)
        plt.text(x + 0.1, y, label, fontsize=6, alpha=0.9)
        
    
    plt.title(f"Embeddings Visualized with {method.upper()}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
 #   plt.legend() #bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.show()




def fix_syntax(pipeline, tokenizer, model_response, config):
    print("fixing syntax")

    annotation_request = [
        {
            "role": "user",
            "content": f"Please check the syntax of the following json file and output the correct syntax: {model_response}"
                       "The json file should contain a list of json entries with each entry having one field 'rationales'"
                       "Output only the json file and no other text.",
        },
    ]

    responses = pipeline(
        annotation_request,
        do_sample=True,
        num_return_sequences=1,
        return_full_text=False,
        max_new_tokens=1000,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        repetition_penalty=config.repetition_penalty,
    )

    return responses[0]["generated_text"].strip()


# Define the custom stopping criteria
class StopOnAnyTokenSequence(StoppingCriteria):
    def __init__(self, stop_sequences_ids):
        self.stop_sequences_ids = stop_sequences_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq in self.stop_sequences_ids:
            if input_ids.shape[1] >= len(seq):
                if torch.equal(input_ids[0, -len(seq):], torch.tensor(seq, device=input_ids.device)):
                    print('Stop generation', seq)
                    return True
        return False


# Function to find the start and end indices of the specific sentence in input_ids
def find_subsequence(input_ids, sentence_ids):
    for i in range(len(input_ids) - len(sentence_ids) + 1):
        if input_ids[i:i + len(sentence_ids)] == sentence_ids:
            return i, i + len(sentence_ids) - 1
    return None, None


def get_syntax_model(bnb_config):
    model_config_syntax = AutoConfig.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    model_syntax = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        config=model_config_syntax,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )

    tokenizer_syntax = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")

    return model_config_syntax, tokenizer_syntax


def get_model(config):
    # Load tokenizer and model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    if config.quant:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            # use_flash_attention=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb_config = None

    model_config = AutoConfig.from_pretrained(
        config.model_name
    )

    print(f"load model: {config.model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )
    _ = model.eval()

    return model, tokenizer, bnb_config




def get_base_str(task, m, lang, sparsity, seed, xai_method):
    root = "model_responses/eval_results/{}/{}/{}/{}/{}/".format(task, m, sparsity, seed, xai_method)
    if m == 'mixtral':
        base_str = root + "{}_{}{}_quant_seed_{}_sparsity_{}".format(task, m, lang, seed,
                                                                     sparsity) if sparsity != 'full' \
            else root + "{}_{}{}_quant_seed_{}".format(task, m, lang, seed)
    else:
        base_str = root + "{}_{}{}_seed_{}_sparsity_{}".format(task, m, lang, seed, sparsity) if sparsity != 'full' \
            else root + "{}_{}{}_seed_{}".format(task, m, lang, seed)
    return base_str



def init_logger(file_name, experiment_name):
    # logging stuff
    logging_datetime_format = '%Y%m%d__%H%M%S'
    logging_time_format = '%H:%M:%S'
    exp_start_time = datetime.datetime.now().strftime(logging_datetime_format)
    log_level = logging.DEBUG
    log_format = '%(asctime)10s,%(msecs)-3d %(module)-30s %(levelname)s %(message)s'
    
    logging.basicConfig(datefmt=logging_time_format)

    logger = logging.getLogger(name=experiment_name)
    logger.setLevel(log_level)
    fh = logging.FileHandler(file_name, mode="w")
    fh.setFormatter(logging.Formatter(log_format))

    logger.addHandler(fh)
    return logger

def set_up_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def plot_generation(R, tokens, column_idx = 0, fax=None, fontsize=11):

    if len(tokens) == 2:
    
        tokens_x = tokens_y = tokens
        
    elif len(tokens) == 1:
        tokens_x = tokens_y = tokens[0]

    if fax is None:
        f,ax = plt.subplots(1,1, figsize=(8,6))
    else:
        f,ax = fax
    h = sns.heatmap(R[:, column_idx:], annot=R[:, column_idx:], vmin=-1, vmax=1,
                cmap='bwr',
                ax=ax, 
                fmt="0.1f",
                annot_kws={"size": fontsize})

    # Adjust colorbar (legend) ticks and font size
    cbar = h.collections[0].colorbar
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])  # Set the ticks you want on the colorbar
    cbar.set_ticklabels([-1.0, -0.5, 0.0, 0.5, 1.0])  # Set the tick labels accordingly

    # Set the width of the colorbar
    cbar.ax.figure.canvas.draw_idle()  # Ensure the canvas is drawn before setting the width
    cbar.ax.set_aspect(22) 
    
    # Set font size for colorbar ticks
    cbar.ax.tick_params(labelsize=10)
    
        

    ax.xaxis.tick_top()
    ax.set_xticks(np.array(range(len(tokens_x)-column_idx))+0.5)
    ax.set_yticks(np.array(range(len(tokens_y)))+0.5)

    ax.set_xticklabels(tokens_x[column_idx:], rotation=90, fontsize=14)#, va='center')
    ax.set_yticklabels(tokens_y, rotation=0, fontsize=14) #, va='center')

    if fax is None:
        plt.show()
    