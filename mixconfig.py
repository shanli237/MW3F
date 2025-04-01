fix_seed = 2024
seq_len=10000
batch_size=64
lr=0.0005
num_epoches=300
model_path='MW3F'
feature_type='MIXT'
cuda_num=3
loss_name='AsymmetricLoss'
dataset_path='closed_5tab'
step_size = 30
gamma = 0.74
threshold = 0.5 # find best precision and recall
train_file = 'train'#trainname
valid_file = 'valid'#validname
test_file = 'test'#testname

