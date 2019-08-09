
import fasttext
import io
import argparse 
import json
import csv
from numpy import arange



def main():


  with open("preprocessed_3grams_models_performance_pretrainedvec", "w") as file:
    #for _lr in arange(0.05, 0.4, 0.05):
    #  for _ws in range(2,9):
    #    for _epochs in range(2,5):
  #_lr = 0.3
  #_ws = 8
  #_epochs = 2
    configs = [(3, 5, 0.050),(3, 2, 0.100),(4, 3, 0.100),(3, 2, 0.250),(2, 8, 0.300),(2, 3, 0.350),(3, 3, 0.350),(2, 4, 0.350)]
    for _epochs, _ws, _lr in configs:
      model = fasttext.train_supervised(input='fasttext_train_data.preprocessed.txt', epoch=_epochs, lr=_lr, ws=_ws, loss="softmax", wordNgrams=3, dim=300, pretrainedVectors="../fasttext_vectors/crawl-300d-2M.vec",)
      (N, p, r) = model.test('fasttext_dev_data.preprocessed.txt')
      save_data(file,_epochs,_ws,_lr,p,r)
#         # print_results(N,p,r)

  #fres=open("testRes.txt","w")
  #f=open("test.preprocessed.txt","r")
  #sentences = f.read().split("\n")
  #for sentence in sentences:
  #    prediction = model.predict(sentence)
  #    fres.write(prediction[0][0]+"\n")

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def save_data(_file,epochs,ws,lr,p,r):
  f1score = (p*r)/(p+r)
  print("epochs: {} ".format(epochs) + " windsize: {} ".format(ws) + " lr: {:.3f} ".format(lr) + " P@{}\t{:.3f}".format(1, p) + " R@{}\t{:.3f} ".format(1, r) + " F1 {:.3f} ".format(f1score), file=_file, flush=True)



main()