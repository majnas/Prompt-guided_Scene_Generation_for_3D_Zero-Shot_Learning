classes = [
"airplane"
,"bathtub"
,"bed"
,"bench"
,"bookshelf"
,"bottle"
,"bowl"
,"car"
,"chair"
,"cone"
,"cup"
,"curtain"
,"desk"
,"door"
,"dresser"
,"flower_pot"
,"glass_box"
,"guitar"
,"keyboard"
,"lamp"
,"laptop"
,"mantel"
,"monitor"
,"night_stand"
,"person"
,"piano"
,"plant"
,"radio"
,"range_hood"
,"sink"
,"sofa"
,"stairs"
,"stool"
,"table"
,"tent"
,"toilet"
,"tv_stand"
,"vase"
,"wardrobe"
,"xbox"
]
print(len(classes))

import scipy.io as sio

"""
    text_embeddings = {"text1": embedding1,
                       "text2": embedding2,
                       ...
                       "textN": embeddingN,
                       } 
"""


# data = sio.loadmat("ModelNet40_w2v.mat")
# data_word = data['word']
# print(data_word.shape)
# text_embeddings = {}
# for c, v in zip(classes, data_word):
#     text_embeddings[c] = v
# for a in classes:
#     for b in classes:
#         text_embeddings[f"{a} {b}"] = (text_embeddings[a] + text_embeddings[b]) / 2
# print(len(text_embeddings.keys()))
# sio.savemat("w2v_text_embedding.mat", mdict=text_embeddings)


data = sio.loadmat("ModelNet40_glove.mat")
data_word = data['word']
print(data_word.shape)
text_embeddings = {}
for c, v in zip(classes, data_word):
    text_embeddings[c] = v
for a in classes:
    for b in classes:
        text_embeddings[f"{a} {b}"] = (text_embeddings[a] + text_embeddings[b]) / 2
print(len(text_embeddings.keys()))
sio.savemat("glove_text_embedding.mat", mdict=text_embeddings)
