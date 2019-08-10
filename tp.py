import fasttext

def main():
  _epochs = 4
  _ws = 3
  _lr = 0.025
  model = fasttext.train_supervised(input='fasttext_train_data.preprocessed.txt', epoch=_epochs, lr=_lr, ws=_ws, loss="softmax", wordNgrams=3, dim=300, pretrainedVectors="../fasttext_vectors/crawl-300d-2M.vec")
  fres=open("testRes.txt","w")
  f=open("test.preprocessed.txt","r")
  sentences = f.read().split("\n")
  for sentence in sentences:
    prediction = model.predict(sentence)
    fres.write(prediction[0][0]+"\n")

main()