
import fasttext
import io
import argparse 
import json
import csv



def main():

    model = fasttext.train_supervised('pepe.txt', epoch=50, lr=0.4, ws=10)
    print_results(*model.test('validacion.txt'))

    fres=open("testRes.txt","w")
    f=open("test.txt","r")
    sentences = f.read().split("\n")
    for sentence in sentences:
        prediction = model.predict(sentence)
        fres.write(prediction[0][0]+"\n")
        
        

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))





main()