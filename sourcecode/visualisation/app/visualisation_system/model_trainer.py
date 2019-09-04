#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:11:47 2019

@author: majiga
Source: https://www.depends-on-the-definition.com/attention-lstm-relation-classification/
"""


with open("SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT") as f:
    train_file = f.readlines()
    
with open("SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT") as f:
    test_file = f.readlines()
    
def prepare_dataset(raw):
    sentences, relations = [], []
    to_replace = [("\"", ""), ("\n", ""), ("<", " <"), (">", "> ")]
    last_was_sentence = False
    for line in raw:
        sl = line.split("\t")
        if last_was_sentence:
            relations.append(sl[0].split("(")[0].replace("\n", ""))
            last_was_sentence = False
        if sl[0].isdigit():
            sent = sl[1]
            for rp in to_replace:
                sent = sent.replace(rp[0], rp[1])
            sentences.append(sent)
            last_was_sentence = True
    print("Found {} sentences".format(len(sentences)))
    return sentences, relations


sentences, relations = prepare_dataset(train_file)
sentences_test, relations_test = prepare_dataset(test_file)

sentences = sentences_test + sentences
relations = relations_test + relations

print(sentences[156])
print(relations[156])

n_relations = len(set(relations))
print("\nFound {} types of relations\n".format(n_relations))
print("Relations:\n{}".format(list(set(relations))))


from models import KerasTextClassifier
import numpy as np
from sklearn.model_selection import train_test_split


kclf = KerasTextClassifier(input_length=50, n_classes=n_relations, max_words=15000)

tr_sent, te_sent, tr_rel, te_rel = train_test_split(sentences, relations, test_size=0.1, random_state=42)
kclf.fit(X=tr_sent, y=tr_rel, X_val=te_sent, y_val=te_rel, batch_size=10, lr=0.001, epochs=30)
#kclf.fit(X=sentences, y=relations, X_val=sentences_test, y_val=relations_test, batch_size=10, lr=0.001, epochs=40)

#label_idx_to_use = [i for i, c in enumerate(list(kclf.encoder.classes_)) if  c !="Other"]
label_idx_to_use = [i for i, c in enumerate(list(kclf.encoder.classes_))]
label_to_use = list(kclf.encoder.classes_)
#label_to_use.remove("Other")
for i in range(len(label_to_use)):
    print(label_idx_to_use[i], ' = ', label_to_use[i])
    


"""
def call(self, h, mask=None):
    h_shape = K.shape(h)
    d_w, T = h_shape[0], h_shape[1]
    
    logits = K.dot(h, self.w)  # w^T h
    logits = K.reshape(logits, (d_w, T))
    alpha = K.exp(logits - K.max(logits, axis=-1, keepdims=True))  # exp
    alpha = alpha / K.sum(alpha, axis=1, keepdims=True)  # softmax
    r = K.sum(h * K.expand_dims(alpha), axis=1)  # r = h*alpha^T
    h_star = K.tanh(r)  # h^* = tanh(r)
    return h_star


import matplotlib.pyplot as plt

y_pred = kclf.predict(sentences_test)
y_attn = kclf._get_attention_map(sentences_test)

i = 354
activation_map = np.expand_dims(y_attn[i][:len(te_sent[i].split())], axis=1)

f = plt.figure(figsize=(8, 8))
ax = f.add_subplot(1, 1, 1)

img = ax.imshow(activation_map, interpolation='none', cmap='gray')

plt.xlim([0,0.5])
ax.set_aspect(0.1)
ax.set_yticks(range(len(te_sent[i].split())))
ax.set_yticklabels(te_sent[i].split());
ax.grid()
plt.title("Attention map of sample {}\nTrue relation: {}\nPredicted relation: {}"
          .format(i, te_rel[i], kclf.encoder.classes_[y_pred[i]]));

# add colorbar
cbaxes = f.add_axes([0.2, 0, 0.6, 0.03]);
cbar = f.colorbar(img, cax=cbaxes, orientation='horizontal');
cbar.ax.set_xlabel('Probability', labelpad=2);
"""

from sklearn.metrics import f1_score, classification_report, accuracy_score

#y_test_pred = kclf.predict(sentences_test)
#print("F1-Score: {:.1%}".format(f1_score(kclf.encoder.transform(relations_test), y_test_pred, average="macro", labels=label_idx_to_use)))
#print(classification_report(kclf.encoder.transform(relations_test), y_test_pred, target_names=label_to_use, labels=label_idx_to_use))

y_test_pred = kclf.predict(te_sent)

print("F1-Score: {:.1%}".format(f1_score(kclf.encoder.transform(te_rel), y_test_pred, average="macro", labels=label_idx_to_use)))
print(classification_report(kclf.encoder.transform(te_rel), y_test_pred, target_names=label_to_use, labels=label_idx_to_use))

kclf.save('SemEval_relation_model')  # creates a HDF5 file 'my_model.h5'
#del kclf  # deletes the existing model



from models import KerasTextClassifier
model = KerasTextClassifier(input_length=50, n_classes=10, max_words=15000)
# returns a compiled model
# identical to the previous one
model.load('SemEval_relation_model')

label_idx_to_use = [i for i, c in enumerate(list(model.encoder.classes_))]
label_to_use = list(model.encoder.classes_)

y_icdm_triples = ["<e1>Song Pro SUV<e1> at <e2>Shanghai International Automobile Industry Exhibition<e2>",
"<e1>Song Pro SUV<e1> alongside <e2>all-new e-series models<e2>"]

y_icdm_triples_prediction = model.predict(y_icdm_triples)
print(y_icdm_triples_prediction)

for a,b in zip(y_icdm_triples, y_icdm_triples_prediction):
    print(a, '\n - ', label_to_use[b])
