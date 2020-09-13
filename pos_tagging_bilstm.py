'''
Bidirectional LSTM for POS Tagging in TensorFlow 2

Features implemented
- Densely connected LSTM layers
- Pretrained word embeddings with Gensim's Word2Vec
- Character-level convolutional network

Results
Achieved F-scores of 93% and 91% for entity and sentiment POS tagging of an English corpus respectively.

References
- General model: https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
- Usage of pretrained Gensim embeddings: https://www.depends-on-the-definition.com/guide-to-word-vectors-with-gensim-and-keras/
- Usage of character-level convolutional network: https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10
- Character pre-processing: https://www.depends-on-the-definition.com/lstm-with-char-embeddings-for-ner/
'''

import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec

class Tagger:
    def __init__(self, train_text_path, min_freq=3, emb_size=50, units=50, max_len_char=5, show_data_info=True):
        """
        Initialise text, weights and model.
        """
        assert type(emb_size) == int
        assert type(min_freq) == int
        assert type(units) == int
        assert train_text_path is not None

        self.X_word, self.X_char, self.y, self.words, self.chars, self.max_input_seq_length, self.max_target_seq_length, self.num_words, self.num_tags, self.num_chars, self.word2idx, self.tag2idx, self.char2idx, self.input_seq = self.process_text(train_text_path, 			min_freq, max_len_char, show_data_info)
        self.embedding_matrix = self.get_embeddings(self.input_seq, emb_size, min_freq)
        self.max_len_char = max_len_char

        # input for words
        word_in = tf.keras.layers.Input(shape=(self.max_input_seq_length,))
        
        # use gensim embeddings
        word_emb = tf.keras.layers.Embedding(input_dim=self.num_words + 1, output_dim=emb_size, weights=[self.embedding_matrix],
                  input_length=self.max_input_seq_length, mask_zero=True)(word_in)

        # input and embeddings for characters
        char_in = tf.keras.layers.Input(shape=(self.max_input_seq_length, self.max_len_char,))
        char_emb = tf.keras.layers.TimeDistributed(tf.keras.layers.Embedding(input_dim=self.num_chars + 2, output_dim=emb_size,
                                  input_length=self.max_len_char, mask_zero=False))(char_in)

        # convolutional network for character embeddings (alternatively, revert to mask_zero=True)
        char_conv = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu', kernel_initializer='glorot_normal'))(char_emb)
        char_drop = tf.keras.layers.Dropout(0.3)(char_conv)
        char_max_pool = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool1D(2))(char_drop)
        # character LSTM to get word encodings by characters
        char_enc = tf.keras.layers.TimeDistributed(tf.keras.layers.LSTM(units=units, return_sequences=False,
                                        recurrent_dropout=0.2))(char_max_pool)

        # main LSTM
        combined = tf.keras.layers.Concatenate()([word_emb, char_enc])
        dropout = tf.keras.layers.SpatialDropout1D(0.3)(combined)
        lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, recurrent_dropout=0.1))(dropout)
        lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, recurrent_dropout=0.1))(lstm_1)
        # dense connections
        concat12 = tf.keras.layers.Concatenate()([lstm_1, lstm_2])
        lstm_3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=True, recurrent_dropout=0.1))(concat12)
        concat123 = tf.keras.layers.Concatenate()([concat12, lstm_3])
        output = tf.keras.layers.Dropout(0.4)(concat123)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_tags, activation="softmax"))(output)

        model = tf.keras.models.Model([word_in, char_in], output)

        sgd = tf.keras.optimizers.SGD(lr=0.1, nesterov=True, momentum=0.9)
        model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
        
        self.model = model
        
        
    def process_text(self, train_text_path, min_freq, max_len_char, show_data_info):
        """
        Process train text dataset.
        """
        assert train_text_path is not None
        assert type(max_len_char) == int

        with open(train_text_path) as file:
            train_data = file.readlines()
        file.close()
      
        input_seq = []
        target_seq = []
        temp_input = []
        temp_target = []
        
        # get unique characters for input
        tags = set()
        tags.add("__PAD__")

        # handle uncommon words in train dataset with character frequency dictionary
        freq = {}
        freq['UNKNOWN'] = min_freq
        
        # get sequences
        for i in range(len(train_data)):
            train_data[i] = train_data[i].strip('\n')
            if train_data[i] != '':
                data = train_data[i].split(' ')
                # convert input data to lowercase
                lower_input = data[0].lower()
                temp_input.append(lower_input)
                temp_target.append(data[1])
                try:
                  freq[lower_input] += 1
                except:
                  freq[lower_input] = 1
                if data[1] not in tags:
                    tags.add(data[1])
            else:
                input_seq.append(temp_input)
                target_seq.append(temp_target)
                temp_input = []
                temp_target = []

        words = [x for x in freq.keys() if freq[x] >= min_freq]
            
        # organise necessary data
        num_words = len(words)
        num_tags = len(tags)
        max_input_seq_length = max([len(txt) for txt in input_seq])
        max_target_seq_length = max([len(txt) for txt in target_seq])
        max_word_length = max([len(w) for w in words])

        # note that we increased the index of the words by one to use zero as a padding value
        # this is done because we want to use the mask_zero parameter of the embedding layer to ignore inputs with value zero
        word2idx = dict(
            [(token, i + 1) for i, token in enumerate(words)])
        word2idx["__PAD__"] = 0
        tag2idx = dict(
            [(token, i) for i, token in enumerate(tags)])
        
        # character processing
        chars = set()
        for token in words:
            for c in token:
                if c not in chars:
                    chars.add(c)
        num_chars = len(chars)
        char2idx = {c: i + 2 for i, c in enumerate(chars)}
        char2idx["UNKNOWN"] = 1
        char2idx["__PAD__"] = 0
        temp = []
        X_char = []
        for seq in input_seq:
            temp_seq = []
            for i in range(max_input_seq_length):
                temp_word_seq = []
                for j in range(max_len_char):
                    try:
                        if seq[i][j].lower() in chars:
                            temp_word_seq.append(char2idx.get(seq[i][j].lower()))
                        else:
                            temp_word_seq.append(char2idx.get('UNKNOWN'))
                    except:
                        temp_word_seq.append(char2idx.get("__PAD__"))
                temp_seq.append(temp_word_seq)
            X_char.append(np.array(temp_seq))
        X_char = np.asarray(X_char)
        
        # convert tokens to vectors
        X_word = []
        y = []
        for seq in input_seq:
            temp = []
            for token in seq:
                if token not in words:
                    temp.append(word2idx['UNKNOWN'])
                else:
                    temp.append(word2idx[token])
            X_word.append(temp)
        X_word = tf.keras.preprocessing.sequence.pad_sequences(maxlen=max_input_seq_length, sequences=X_word, padding="post", value=word2idx['__PAD__'])
        for seq in target_seq:
            temp = []
            for token in seq:
                temp.append(tag2idx[token])
            y.append(temp)
        y = tf.keras.preprocessing.sequence.pad_sequences(maxlen=max_target_seq_length, sequences=y, padding="post", value=tag2idx['__PAD__'])
        
        # show data
        if show_data_info:
            print('\n')
            print('Number of samples:', len(input_seq))
            print('Number of unique words:', num_words)
            print('Number of unique characters:', num_chars)
            print('Number of unique tags:', num_tags)
            print('Max sequence length for inputs:', max_input_seq_length)
            print('Max sequence length for outputs:', max_target_seq_length)
            print('Max word length:', max_word_length)
            print('\n')

        return X_word, X_char, y, words, chars, max_input_seq_length, max_target_seq_length, num_words, num_tags, num_chars, word2idx, tag2idx, char2idx, input_seq
    

    def get_embeddings(self, sequences, emb_size, min_freq):
        """
        Get pretrained word embeddings.
        """
        model = Word2Vec(sequences, size=emb_size, window=10, min_count=min_freq, workers=16, sg=1, iter=10, negative=5)
        word_vectors = model.wv
        del model

        embedding_matrix = (np.random.rand(self.num_words + 1, emb_size) - 0.5) / 5.0
        for word, i in self.word2idx.items():
            try:
                embedding_vector = word_vectors[word]
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            except:
                pass

        return embedding_matrix


    def train(self, batch_size=32, epochs=100, split=0.2, patience=5, save_path=None):
        """
        Train model.
        """
        assert type(batch_size) == int
        assert type(epochs) == int
        assert type(split) == float
        assert type(patience) == int
        assert type(save_model) == bool
        
        self.model.summary()

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit([self.X_word, self.X_char], self.y, batch_size=batch_size, callbacks=[callback], epochs=epochs, validation_split=split)
        
        # save model
        if save_path is not None:
            self.model.save(save_path)

        
    def get_test_outputs(self, test_text_path, output_path, load_path=None):
        """
        Get test outputs.
        """
        assert test_text_path is not None
        assert output_path is not None

        # load model
        if load_path is not None:
          self.model = tf.keras.models.load_model(load_path)

        assert self.model is not None
        
        # reverse-lookup token index to decode sequences back to something readable
        idx2word = dict(
            (i, char) for char, i in self.word2idx.items())
        idx2tag = dict(
            (i, char) for char, i in self.tag2idx.items())

        with open(test_text_path) as file:
            test_data = file.readlines()
        file.close()

        # get test sequences
        test_seq = []
        temp_input = []
        for i in range(len(test_data)):
            test_data[i] = test_data[i].strip('\n')
            if test_data[i] != '':
                temp_input.append(test_data[i])
            else:
                test_seq.append(temp_input)
                temp_input = []

        output = []
        temp_seq = []
        count = 0
        show_freq = 100

        # get model predictions
        print("\n")
        for seq in test_seq:
            count += 1
            seq_words = []
            seq_chars = []
            # process words
            for i in range(self.max_input_seq_length):
                try:
                    if seq[i].lower() not in self.words:
                        seq_words.append(self.word2idx['UNKNOWN'])
                    else:
                        seq_words.append(self.word2idx[seq[i].lower()])
                except:
                    seq_words.append(self.word2idx['__PAD__'])

            # process characters
            for i in range(self.max_input_seq_length):
                temp_word_seq = []
                for j in range(self.max_len_char):
                    try:
                        if seq[i][j].lower() in self.chars:
                            temp_word_seq.append(self.char2idx.get(seq[i][j].lower()))
                        else:
                            temp_word_seq.append(self.char2idx.get('UNKNOWN'))
                    except:
                        temp_word_seq.append(self.char2idx.get("__PAD__"))
                seq_chars.append(temp_word_seq)
            
            pred = self.model.predict([np.asarray([seq_words]),np.asarray([seq_chars])])
            pred = np.squeeze(pred)
            
            temp_seq = []
            for i in range(len(seq)):
                max_idx = np.argmax(pred[i])
                temp_seq.append(seq[i] + ' ' + idx2tag[max_idx] + '\n')
            temp_seq.append('\n')

            output = output + temp_seq

            if count % show_freq == 0:
                print("Prediction done for " + str(count) + " test sequences...")

        # write model predictions as outputs
        with open(output_path, 'w') as f:
            for item in output:
                f.write(item)
        f.close()
        
        print('\nTest results written to ' + str(output_path) + '!')

TRAIN_DATASET_PATH = ''
TEST_DATA_PATH = ''
TEST_OUTPUT_PATH = ''
SAVE_MODEL_PATH = ''
LOAD_MODEL_PATH = ''

tagger = Tagger(TRAIN_DATASET_PATH, emb_size=128, units=96, max_len_char=15)
tagger.train(batch_size=128, epochs=1000, patience=5, save_path=SAVE_MODEL_PATH)
tagger.get_test_outputs(TEST_DATA_PATH,TEST_OUTPUT_PATH, LOAD_MODEL_PATH)
