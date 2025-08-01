import pathlib
import torch
import ankh
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel, EsmModel, EsmTokenizer, BertModel, BertTokenizer

class ProteinLanguageModel:
    """
    A wrapper class for various Protein Language Models
    Supports: ESM2, ProtBert, ProtT5, and Ankh
    """
    def __init__(self, 
                 model_name: str,
                 max_length: int = 1024,
                 device: str = "cuda" if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the PLM wrapper
        
        Args:
            model_name (str): Name/type of the model to use
                            ('esm2', 'protbert', 'prott5', etc.)
            max_length (int): Maximum sequence length
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.model_name = model_name.lower()
        self.max_length = max_length
        self.device = device

        # init model and tokenizer
        self.model, self.tokenizer = self._initialize_model()
        self.model.to(self.device)
        if device == torch.device("cpu"):
            self.model.to(torch.float32)
        self.model.eval() #set evaluation mode by default
    
    def _initialize_model(self):
        if 'esm2' in self.model_name:
            model_name = "facebook/esm2_t12_35M_UR50D"
            tokenizer = EsmTokenizer.from_pretrained(model_name)
            model = EsmModel.from_pretrained(model_name)
        elif 'bert' in self.model_name:
            tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
            model = BertModel.from_pretrained("Rostlab/prot_bert")
        elif 't5' in self.model_name:
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
            tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        elif 'ankh' in self.model_name:
            model, tokenizer = ankh.load_large_model()
        else:
            raise ValueError(f'Unsupported model type : {self.model_name}')
        
        return model, tokenizer
    
    def encode_seq(self,
                   sequence: str):
        if 'ankh' in self.model_name:
            sequence = list(sequence)
            # example: 'V', 'M', 'K', 'A', 'N'
            encoded = self.tokenizer.batch_encode_plus([sequence],
                                                       padding=True,
                                                       is_split_into_words=True,
                                                       return_tensors= 'pt',
                                                       truncation=True,
                                                       max_length = self.max_length,
                                                       add_special_tokens=True)
        else:
            if 'bert' in self.model_name or 't5' in self.model_name:
                sequence = ' '.join(sequence)
                #example: V M K A N V T
            
            encoded = self.tokenizer(sequence,
                                    max_length = self.max_length,
                                    padding='max_length',
                                    truncation=True,
                                    return_tensors = 'pt')
        
        return({k: v.to(self.device) for k, v in encoded.items()})
    
    def get_embeddings(
            self,
            sequence: str,
            layer: int = -1,
            pool_method: str = 'mean'):
        with torch.no_grad():
            # encode sequence
            inputs = self.encode_seq(sequence)
            # get model outputs
            outputs = self.model(**inputs, output_hidden_states=True)
            # get hidden states from specified layer
            if isinstance(outputs, tuple):
                hidden_states = outputs.last_hidden_state[layer]
            else:
                hidden_states = outputs.last_hidden_state
        
        # pool embeddings
        if pool_method == 'mean':
            mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (hidden_states * mask).sum(1) / mask.sum(1)
        else: #'cls'
            embeddings = hidden_states[:, 0, :]

        return embeddings
