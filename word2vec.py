from nltk.corpus import brown
from gensim.models.word2vec import Word2Vec
from pathlib import Path

def generate_brown_w2vec():
    sentences_brown = brown.sents()
    print ('training....')
    model_brown = Word2Vec(sentences_brown, size=100, window=10, min_count=10, workers=100)
    print ('done!')
    model_brown.save('./data/model-brown-vectors.bin')
    print ('saved!')
    return model_brown

def load_brown_w2vec():
    return  Word2Vec.load('./data/model-brown-vectors.bin')

if __name__ == '__main__':
    model_brown = None
    if ( Path('./data/model-brown-vectors').is_file() ):
        model_brown = generate_brown_w2vec()
    else:
        model_brown = load_brown_w2vec()

    print (model_brown['Power'])