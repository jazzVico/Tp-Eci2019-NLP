
import fasttext
import io
import argparse 
import json
import csv
from numpy import arange



def main():


#  with open("models_performance_3grams_preprocessed", "w") as file:
    #for _lr in arange(0.05, 0.2, 0.05):
     # for _ws in range(2,9):
      #  for _epochs in range(2,5):
  _lr = 0.2
  _ws = 6
  _epochs = 3
  model = fasttext.train_supervised(input='fasttext_train_data.preprocessed.txt', epoch=_epochs, lr=_lr, ws=_ws, loss="softmax", wordNgrams=3)# dim=300, pretrainedVectors="../fasttext_vectors/crawl-300d-2M.vec",)
  (N, p, r) =  model.test('fasttext_dev_data.preprocessed.txt')
  print_results(N,p,r)
    
    #save_data(file,_epochs,_ws,_lr,p,r)
  #print(" ")
  
#      f=open("test.txt","r")
#      sentences = f.read().split("\n")
#      for sentence in sentences:
#          prediction = model.predict(sentence)
#          fres.write(prediction[0][0]+"\n")

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def save_data(_file,epochs,ws,lr,p,r):
  f1score = (p*r)/(p+r)
  print("epochs: {} ".format(epochs) + " windsize: {} ".format(ws) + " lr: {} ".format(lr) + " P@{}\t{:.3f}".format(1, p) + " R@{}\t{:.3f} ".format(1, r) + " F1 {:.3f} ".format(f1score), file=_file, flush=True)



main()