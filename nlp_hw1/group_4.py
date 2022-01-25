import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import random
from collections import Counter
from nltk.corpus import stopwords
nltk.download('stopwords')
from tokenizers import BertWordPieceTokenizer

class my_corpus():
    def __init__(self, params):
        super().__init__()

        self.params = params
        print('setting parameters')

    def read_data(self):
        with open(self.params['filename']) as f:
            contents = f.read()
        return contents
    
    def text_tokens(self, texts):
        sent_token = []
        for text in texts:
          text = text.lower()
          text = re.sub('\n','',text)
          text = nltk.word_tokenize(text)
          sent_token.append(text)
        return sent_token
    
    def preprocess(self, corpus):
        sentences = sent_tokenize(corpus)
        word_token = self.text_tokens(sentences)
        return word_token
        
    def num_tokens(self, s):
        lst_months = ['jan','january','feb','february','mar','march','apr','april','may','june','july','aug','august','oct','october','nov','november','dec','december']
        for i in range(len(s)):
            s[i] = re.sub('1\d{3}|20\d{2}','<year>',s[i])
            s[i] = re.sub('\d+\.\d+','<decimal>',s[i])
        for i in range(len(s)):
            try:
                if (s[i+1] in lst_months and s[i].isnumeric()) or (s[i-1] in lst_months and s[i].isnumeric()):
                    s[i] = '<days>'
            except:
                pass
        for i in range(len(s)):
            s[i] = re.sub(r'\b\d+\b','<integer>',s[i])
            s[i] = re.sub('\d+','<other>',s[i])
        return s
    
    def preprocess_num(self, word_tkn):
        num_edits_tokens = []
        for k in range(len(word_tkn)):
            num_edits_tokens.append(self.num_tokens(word_tkn[k]))
        return num_edits_tokens
        
    def train_test_split(self, num_edits_tokens):
        random.seed(123)
        random.shuffle(num_edits_tokens)

        train_size = int(len(num_edits_tokens) * 0.8)
        train_data_part = num_edits_tokens[0:train_size]
        train_data = [item for sublist in train_data_part for item in sublist]
        val_data_part = num_edits_tokens[train_size:train_size+(len(num_edits_tokens)-train_size)//2]
        val_data = [item for sublist in val_data_part for item in sublist]
        test_data_part = num_edits_tokens[train_size+(len(num_edits_tokens)-train_size)//2:]
        test_data = [item for sublist in test_data_part for item in sublist]
        return train_data, val_data, test_data
        
    def stat(self, train, val, test):
        cnt_tokens_train = Counter(x for x in train)
        cnt_tokens_valid = Counter(x for x in val)
        cnt_tokens_test = Counter(x for x in test)
        train_dict = {k:v for k,v in cnt_tokens_train.items() if v >= 3}
        val_dict = {k:v for k,v in cnt_tokens_valid.items() if v >= 3}
        test_dict = {k:v for k,v in cnt_tokens_test.items() if v >= 3}
        # number of tokens in each split
        train_tokens = 0
        val_tokens = 0
        test_tokens = 0
        for k, v in train_dict.items():
            train_tokens += v
        for k, v in val_dict.items():
            val_tokens += v
        for k, v in test_dict.items():
            test_tokens += v
        print("Number of tokens in training data:", train_tokens)
        print("Number of tokens in validation data:", val_tokens)
        print("Number of tokens in test_data:", test_tokens)
        # vocabulary size
        print("Vocabulary size in training data:", len(train_dict))
        print("Vocabulary size in validation data:", len(val_dict))
        print("Vocabulary size in test_data:", len(test_dict))
        #<unk> token calculation
        unk_val_tkn = 0
        unk_test_tkn = 0
        oov_tkn_train = 0
        oov_tkn_val = 0
        oov_tkn_test = 0
        for tkn, val in val_dict.items():
            if tkn not in train_dict:
                unk_val_tkn += val
                val_dict[tkn] = '<unk>'
        for tkn, val in test_dict.items():
            if tkn not in train_dict:
                unk_test_tkn += val
                test_dict[tkn] = '<unk>'
        print("Number of <unk> tokens in validation data:", unk_val_tkn)
        print("Number of <unk> tokens in test data:", unk_test_tkn)
        #oov words calculation
        for tkn, val in train_dict.items():
            if tkn not in cnt_tokens_train:
                oov_tkn_train += 1
        for tkn, val in val_dict.items():
            if tkn not in cnt_tokens_train:
                oov_tkn_val += 1
        for tkn, val in test_dict.items():
            if tkn not in cnt_tokens_train:
                oov_tkn_test += 1
        print("Number of oov words in train data:", oov_tkn_train)
        print("Number of oov words in validation data:", oov_tkn_val)
        print("Number of oov words in test data:", oov_tkn_test)
        #unk types calculation
        sum_cnt_unk_val = sum(x == '<unk>' for x in val_dict.values())
        sum_cnt_unk_test = sum(x == '<unk>' for x in test_dict.values())
        print("Number of <unk> types in validation data:", sum_cnt_unk_val)
        print("Number of <unk> types in test data:", sum_cnt_unk_test)
        #stopwords calculation
        stpwrds_train = []
        stpwrds_val = []
        stpwrds_test = []
        for k, v in train_dict.items():
            if k in stopwords.words():
                stpwrds_train.append(k)

        for k, v in val_dict.items():
            if k in stopwords.words() and v != '<unk>':
                stpwrds_val.append(k)

        for k, v in test_dict.items():
            if k in stopwords.words() and v != '<unk>':
                stpwrds_test.append(k)
        print("Number of stopwords in train data:", len(stpwrds_train))
        print("Number of stopwords in validation data:", len(stpwrds_val))
        print("Number of stopwords in test data:", len(stpwrds_test))
        #custom metric calculation
        #(i) number of punctuation tokens in each split
        cnt_train_punc = 0
        train_punct = []
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for k, v in train_dict.items():
            if k in punctuations:
                cnt_train_punc += v
                train_punct.append(k)
        cnt_val_punc = 0
        val_punct = []
        for k, v in val_dict.items():
            if k in punctuations and val_dict[k] != '<unk>':
                cnt_val_punc += v
                val_punct.append(k)
        cnt_test_punc = 0
        test_punct = []
        for k, v in test_dict.items():
            if k in punctuations and test_dict[k] != '<unk>':
                cnt_test_punc += v
                test_punct.append(k)
        print("Number of punctuation tokens in training data:", cnt_train_punc)
        print("Number of punctuation tokens in validation data:", cnt_val_punc)
        print("Number of punctuation tokens in test data:", cnt_test_punc)
        #(ii) number of punctuations in vocabulary
        print("Count of type of punctuations in training data:", len(train_punct))
        print("Count of type of punctuations in validation data:", len(val_punct))
        print("Count of type of punctuations in test data:", len(test_punct))
        
        
    
    def encode_as_ints(self, sequence):

        int_represent = []

        print('encode this sequence: %s' % sequence)
        print('as a list of integers.')
        tokens = sequence.split()
        self.token_to_integer_mapping = {}
        self.integer_to_token_mapping = {}
        unique_token_id = 0
        for cur_token in tokens:
            if cur_token not in self.token_to_integer_mapping:
                unique_token_id += 1
                self.integer_to_token_mapping[unique_token_id] = cur_token
                self.token_to_integer_mapping[cur_token] = unique_token_id
                int_represent.append(unique_token_id)
            else:
                int_represent.append(self.token_to_integer_mapping[cur_token])

        return (int_represent)

    def encode_as_text(self, int_represent):

        text = ''
        tokens = []
        print('encode this list', int_represent)
        print('as a text sequence.')
        for cur_integer in int_represent:
            tokens.append(self.integer_to_token_mapping[cur_integer])

        text = ' '.join(tokens)
        return (text)
        
    def wpt(self, paths):
        tokenizer = BertWordPieceTokenizer(
            clean_text=True,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=False
            )
        tokenizer.train(files=paths, vocab_size=5000, min_frequency=3,
                special_tokens=['<s>', '<pad>', '</s>', '<unk>', '<mask>'])
        return tokenizer


def main():
    params = {
        'filename': 'source_text.txt'
    }
    corpus = my_corpus(params)
    data = corpus.read_data()
    wrd_tokens = corpus.preprocess(data)
    final_tokens = corpus.preprocess_num(wrd_tokens)
    train, val, test = corpus.train_test_split(final_tokens)
    corpus.stat(train, val, test)
    
    
    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ', ints)

    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)
    
    train_ip = ' '.join(map(str, train))
    val_ip = ' '.join(map(str, val))
    test_ip = ' '.join(map(str, test))
    with open(f'./text_train_corpus.txt', 'w', encoding='utf-8') as fp:
        fp.write(' '.join(train_ip))
    with open(f'./text_val_corpus.txt', 'w', encoding='utf-8') as fp:
        fp.write(' '.join(val_ip))
    with open(f'./text_test_corpus.txt', 'w', encoding='utf-8') as fp:
        fp.write(' '.join(test_ip))
    
    path = [str(params['filename'])]
    new_tokenizer = corpus.wpt(path)
    new_tokens = new_tokenizer(data)
    

if __name__ == "__main__":
    main()
