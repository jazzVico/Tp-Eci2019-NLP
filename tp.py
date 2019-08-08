
import fasttext
import io
import argparse 



def main():

    model = fasttext.train_supervised('pepe.txt', epoch=50, lr=0.4, ws=10)
    print_results(*model.test('validacion.txt'))

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))



main()