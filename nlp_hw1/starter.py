class my_corpus():
    def __init__(self, params):
        super().__init__() 
        
        self.params = params
        print('setting parameters')
    
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

        return(int_represent)
    
    def encode_as_text(self,int_represent):

        text = ''
        tokens = []
        print('encode this list', int_represent)
        print('as a text sequence.')
        for cur_integer in int_represent:
            tokens.append(self.integer_to_token_mapping[cur_integer])

        text = ' '.join(tokens)
        return(text)
    
def main():
    corpus = my_corpus(None)
    
    text = input('Please enter a test sequence to encode and recover: ')
    print(' ')
    ints = corpus.encode_as_ints(text)
    print(' ')
    print('integer encodeing: ',ints)
    
    print(' ')
    text = corpus.encode_as_text(ints)
    print(' ')
    print('this is the encoded text: %s' % text)
    
if __name__ == "__main__":
    main()
        
    
    
              