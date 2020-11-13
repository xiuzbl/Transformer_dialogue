import os
batch_size=128
max_decoding_length = 200
num_epochs=10000
display=5

#data_root_dir="data/data_v15_d1g10_for_base_transductive_ori/"
data_root_dir="data/data_v15_d1g10_for_base/release_0614/"

# train
#train_file_prefix='weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_h125ktrain.ori'
train_file_prefix='weibo_78wu_v12_d1filtered_nodatauser.f_v2.wholegraph15wf_wsim_0.2.f_0.6_train_0_to_34_byword_v2.train.ori'

train_src_file=train_file_prefix
train_tgt_file=train_file_prefix
#train_src_vocab='dict.txt'
train_src_vocab='dict.txt_1col'
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

#test_src_file='weibo_78wu_v15_d1g10f3_s10p3_0p35_0p25_1k_train_0_to_34_for_base.shuffled_t12869t_wotrain_t9515_t3000.ori.src'
test_src_file='weibo_78wu_v12_d1filtered_nodatauser.f_v2.wholegraph15wf_wsim_0.2.f_0.6_train_0_to_34_byword_v2.test.ori.src'


test_data_params={
    'datasets':[
        {'files': os.path.join(data_root_dir, test_src_file), 
            'vocab_file': os.path.join(data_root_dir, train_src_vocab), 
            'data_name': 'src'},
    ],
    'batch_size': 64,
    'shuffle': False
}
