import os
batch_size=1
max_decoding_length = 200
num_epochs=10000
display=5


data_root_dir="data/data_v11_d2/"
#data_root_dir="data/data_v11/"
#data_root_dir="data/data_v01_d4/"
#data_root_dir="data/data_v01/"


# train

#train_src_file='weibo_78w_v00_d0.src'
#train_src_vocab='weibo_78w_v00_d0.2wdict'
#train_tgt_file='weibo_78w_v00_d0.tgt'
#train_src_file='reddit_spo.basic.test.src'
#train_tgt_file='reddit_spo.basic.test.tgt'
#train_src_vocab='vocab.txt'
#train_src_file='src10'
#train_tgt_file='tgt10'
#train_src_vocab='weibo_78w_v00_d0.2wdict'
#train_src_file='train.src'
#train_tgt_file='train.tgt'
#train_file_prefix='weibo_78w_12kdict'
#train_file_prefix='weibo_78wu_v11_d1'
train_file_prefix='weibo_78wu_v11_d2'
train_src_file=train_file_prefix
train_tgt_file=train_file_prefix
train_src_vocab='dict.txt'
train_tgt_vocab=train_src_vocab

train_data_params={
    'datasets':[
        {'files': os.path.join(data_root_dir, train_src_file), 
            'vocab_file': os.path.join(data_root_dir, train_src_vocab), 
            'data_name': 'src'},
        {'files': os.path.join(data_root_dir, train_tgt_file), 
            'vocab_file': os.path.join(data_root_dir, train_tgt_vocab), 
            'data_name': 'tgt'},
    ],
    'batch_size': batch_size,
    'total_sample_num_of_all_dataset': 34600000,
}

# test

test_src_file='weibo_78w_v01_d2_12kdict.test1k.src'


test_data_params={
    'datasets':[
        {'files': os.path.join(data_root_dir, test_src_file), 
            'vocab_file': os.path.join(data_root_dir, train_src_vocab), 
            'data_name': 'src'},
    ],
    'batch_size': 64
}
