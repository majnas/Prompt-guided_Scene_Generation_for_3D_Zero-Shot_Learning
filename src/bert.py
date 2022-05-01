import os
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertModel
import scipy.io


"""
    text_embeddings = {"text1": embedding1,
                       "text2": embedding2,
                       ...
                       "textN": embeddingN,
                       } 
"""

class BERT():
    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model.eval()
        self.texts = []

    def load_texts(self, texts_path: str) -> None:
        """ read class names """
        fh = open(texts_path, 'r')
        # lines = fh.readlines()
        lines=[c.rstrip() for c in fh.readlines()]
        fh.close()
        self.texts = [l.split(", ") for l in lines]


    def process(self):
        #### word vector intial
        text_embeddings = {}
        for i, text in enumerate(self.texts):
            if text in list(text_embeddings.keys()):
                continue
            listToStr = ' '.join([str(elem) for elem in text])    
            text=listToStr    
            marked_text = "[CLS] " + text + " [SEP]"
            print(f"{i}/{len(self.texts)}: {marked_text} ... ")
            tokenized_text = self.tokenizer.tokenize(marked_text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])

            ### forward pass
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
            token_vecs = hidden_states[-2][0]

            #### word embedding
            word_embed = torch.mean(token_vecs, dim=0)

            # store embedding in dictionary
            text_embeddings[text] = word_embed.cpu().detach().numpy()

        save_path = args.texts_path.replace(".txt", ".mat")
        scipy.io.savemat(save_path, mdict=text_embeddings)


def main(args):
    bert_mdl = BERT()
    bert_mdl.load_texts(texts_path=args.texts_path)
    bert_mdl.process()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--texts_path', type=str, default='./data/ModelNet/bert_txt_embedding_modelnet40.txt')
    args = parser.parse_args()

    main(args)

