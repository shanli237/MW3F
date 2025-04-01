import os
import torch
import random
import numpy as np
import torch.optim as optim
from torch import nn
from utils import *
from models import *
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from asyloss import AsymmetricLoss
from mixconfig import fix_seed,seq_len,batch_size,lr,num_epoches,model_path,feature_type,cuda_num,loss_name,train_file,valid_file,step_size,gamma,dataset_path
# Set a fixed seed for reproducibility
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
information = f'seq_len={seq_len}\nbatch_size={batch_size}\nlr={lr}\nnum_epoches={num_epoches}\nmodel_path=\'{model_path}\'\nfeature_type=\'{feature_type}\'\ncuda_num={cuda_num}\nloss_name=\'{loss_name}\'\ndataset_path=\'{dataset_path}\'\n'
print(information)
#cuda
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu", cuda_num)
print (f'Device: {device}')

#dataset

in_path = os.path.join("your_path", dataset_path)

# Load training and validation data
train_X, train_y = load_data(os.path.join(in_path, f"{train_file}.npz"), feature_type,seq_len)
valid_X, valid_y = load_data(os.path.join(in_path, f"{valid_file}.npz"), feature_type,seq_len)

num_classes = len(train_y[0])
# print(f"Train: X={train_X.shape}, y={train_y.shape}")
# print(f"Valid: X={valid_X.shape}, y={valid_y.shape}")
print(f"Train_num_classes: {num_classes}")

# Load data into iterators
train_iter = load_iter(train_X, train_y, batch_size, feature_type,is_train=True)
valid_iter = load_iter(valid_X, valid_y, batch_size, feature_type,is_train=False)

model =  eval(f"{model_path}")(num_classes).to(device)
if loss_name == "CrossEntropyLoss":
    criterion = nn.CrossEntropyLoss()
elif loss_name == "BCEWithLogitsLoss":
    criterion = nn.BCEWithLogitsLoss()
elif loss_name == "AsymmetricLoss":
    criterion = AsymmetricLoss()
elif loss_name == "MultiLabelSoftMarginLoss":#可以不用转换
    criterion = nn.MultiLabelSoftMarginLoss()
else:
    raise ValueError(f"Loss function {loss_name} is not matched.")
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.005)#
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)#StepLR用于每经过多少个epoches，更新学习率
logs_path = os.path.join('logs', dataset_path , model_path)
if not os.path.exists(logs_path):
    os.makedirs(logs_path)# 如果不存在，则创建文件夹
writer = SummaryWriter(logs_path)

#train
checkpoints_path = os.path.join('checkpoints', dataset_path , model_path)
metric_best_value = 0 #最好的验证结果
# 简单的训练循环
for epoch in range(num_epoches):
    model.train()
    sum_loss = 0
    sum_count = 0
    for batch_idx,td in enumerate(train_iter):
        inputs ,labels = td
        input1 = inputs[0]
        input2 = inputs[1]
        input1 = input1.to(device)
        input2 = input2.to(device)
        inputs = (input1,input2)
        # with torch.no_grad():
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if (batch_idx+1) % 100 == 0:
            print(f'batch_idx {batch_idx+1}, Loss: {loss.item()}')
            writer.add_scalar('training loss',loss.item(),epoch * len(train_iter) + batch_idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss += loss.data.cpu().numpy() * outputs.shape[0]
        sum_count += outputs.shape[0]
    train_loss = round(sum_loss / sum_count, 5)
    print(f"epoch {epoch}: train_loss = {train_loss}")
    
    probs = []
    one_hot = []
    with torch.no_grad():
        model.eval()
        val_loss = 0
        val_count = 0
        for index, cur_data in enumerate(valid_iter):
            inputs ,labels = cur_data
            
            input1 = inputs[0]
            input2 = inputs[1]
            input1 = input1.to(device)
            input2 = input2.to(device)
            inputs = (input1,input2)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            outputs = nn.functional.sigmoid(outputs)
            probs.append(outputs.data.cpu().numpy())
            one_hot.append(labels.data.cpu().numpy())
            val_loss += loss.data.cpu().numpy() * outputs.shape[0]
            val_count += outputs.shape[0]
    prob_matrix = np.concatenate(probs, axis=0)
    one_hot_matrix = np.concatenate(one_hot, axis=0)
    # 计算平均损失和总体准确率
    avg_loss = val_loss / val_count
    mAP = average_precision_score(one_hot_matrix, prob_matrix,average='macro')
    print(f'Valid Loss: {avg_loss:.4f}, mAP: {mAP:.2f}')
    scheduler.step()
    if mAP > metric_best_value:
        if not os.path.exists(checkpoints_path ):
            os.makedirs(checkpoints_path)# 如果不存在，则创建文件夹
        torch.save(model.state_dict(), f'{checkpoints_path}/1.pth')
        metric_best_value = mAP