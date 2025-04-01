import os
import torch
import random
import numpy as np
from utils import *
from models import *
from torch import nn
from sklearn.metrics import precision_score,average_precision_score,recall_score,f1_score
from mixconfig import fix_seed,seq_len,batch_size,model_path,feature_type,cuda_num,test_file,num_epoches,loss_name,lr,dataset_path

random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
#cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", cuda_num)
print (f'Device: {device}')

#dataset
in_path = os.path.join("your_path", dataset_path)

# Load training and validation data

test_X, test_y = load_data(os.path.join(in_path, f"{test_file}.npz"), feature_type,seq_len)
num_classes = len(test_y[0])

# Print dataset information
# print(f"Test: X={test_X.shape}, y={test_y.shape}")
print(f"num_classes: {num_classes}")

# Load data into iterators

test_iter = load_iter(test_X, test_y, batch_size, feature_type,is_train=False)

model = eval(model_path)(num_classes)
checkpoints_path = os.path.join('./checkpoints', dataset_path , model_path)
status = model.load_state_dict(torch.load(f'{checkpoints_path}/1.pth', map_location="cpu"))
model.to(device)

probs = []
one_hot = []
# y_pred_score = np.zeros((0, num_classes))
with torch.no_grad():
    model.eval()
    y_pred = []
    y_true = []

    for index, cur_data in enumerate(test_iter):


        inputs ,labels = cur_data
        
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = input1.to(device)
        input2 = input2.to(device)
        inputs = (input1,input2)
        labels = labels.to(device)

        outs = model(inputs)
        outs = nn.functional.sigmoid(outs)
        probs.append(outs.data.cpu().numpy())
        one_hot.append(labels.data.cpu().numpy())
        # y_pred_score = np.append(y_pred_score, outs.cpu().numpy(), axis=0)
# y_true = test_y.numpy()
prob_matrix = np.concatenate(probs, axis=0)
one_hot_matrix = np.concatenate(one_hot, axis=0)


thresholds = np.linspace(0, 1, 100)  # 从0到1生成100个阈值
best_macro_f1 = 0
best_threshold = 0

# # 遍历每个阈值
for th in thresholds:
    # 使用当前阈值生成二进制预测
    prob_matrix_1 = (prob_matrix >= th).astype(int)
    
    # 计算该阈值下的 macro-F1 分数
    macro_f1 = f1_score(one_hot_matrix, prob_matrix_1, average='macro')
    
    # 如果当前 macro-F1 更高，则更新最佳阈值和分数
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_threshold = th

print(f"Best Threshold: {best_threshold:.2f}")
print(f"Best Macro-F1 Score: {best_macro_f1:.2f}")

# map
mAP = average_precision_score(one_hot_matrix, prob_matrix,average='macro')
# Macro-Averaged Precision
macro_precision = precision_score(one_hot_matrix, prob_matrix>best_threshold,average='macro')
macro_recall = recall_score(one_hot_matrix, prob_matrix>best_threshold,average='macro')
macro_f1 = f1_score(one_hot_matrix, prob_matrix>best_threshold,average='macro')
information = f'seq_len={seq_len}\nbatch_size={batch_size}\nlr={lr}\nnum_epoches={num_epoches}\nmodel_path=\'{model_path}\'\nfeature_type=\'{feature_type}\'\ncuda_num={cuda_num}\nloss_name=\'{loss_name}\'\n'
results = f'macro_precision:{round(macro_precision,4)},macro_recall:{round(macro_recall,4)},macro_f1:{round(macro_f1,4)},mAP:{round(mAP,4)}\n'
print(results)

result_path = os.path.join('results',dataset_path,model_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)
outfile = os.path.join(result_path,'test.txt')
with open(outfile, "a+") as fp:
    fp.write(information)
    fp.write(results)
fp.close()