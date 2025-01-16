#import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import numpy as np
import pandas as pd
from 模型.mymodel import DAR,LSTM,CNNGRU,GRU,TCN
import matplotlib.pyplot as plt




def paint(filename):
    data = np.load('result11/pre&loss/'+filename)
    loss = np.load('result11/loss/'+filename)
    test_actual = data[0,:,-1]
    test_predicted = data[1,:,-1]


    # 均方误差
    sme = mean_squared_error(test_actual, test_predicted)
    # 平均绝对百分比误差
    mape = np.mean(np.abs((test_actual - test_predicted) / test_actual))
    # 对称平均绝对百分比误差
    smape = np.mean(np.abs(test_actual - test_predicted) / np.abs(
        (test_actual + test_predicted) / 2))
    r_squared = r2_score(test_actual, test_predicted)
    print(sme)
    # print(srme)
    # print(mae)
    print(mape.item())      
    # print(smape)
    print(r_squared)

    plt.rcParams['font.sans-serif'] = ['SimSun']  
    plt.rcParams['axes.unicode_minus'] = False

    sns.set_theme(style="white", palette="tab10", font_scale=2.5, rc={"lines.linewidth": 1.5}, font='SimSun')
    
    fig, axs = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.2, top=0.9)


    # 如果有+1     axs[0].set_title(f"{filename[3:-10]}")
    axs.set_title(f"{filename[:-10]}")
    sns.lineplot(data=test_actual, label='真实值', ax=axs)
    sns.lineplot(data=test_predicted, label='预测值', ax=axs)
    axs.set_xlabel("时间轮次t")
    axs.set_ylabel("地下水位/m")

    plt.savefig('result11/picc/%s %.3f %.3f %.3f.png' % (filename[:-10], sme, mape.item(), r_squared))




    fig, axs = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.2, top=0.9)


    train_losses = loss[0]
    val_losses = loss[1]
    test_losses = loss[2]
    axs.set_title(f"损失函数值({filename[:-10]})")
    sns.lineplot(data=train_losses, label='训练集损失', ax=axs)
    sns.lineplot(data=val_losses, label='验证集损失', ax=axs)
    sns.lineplot(data=test_losses, label='测试集损失', ax=axs)
    min_train_loss = min(train_losses)
    min_train_loss_idx = np.argmin(train_losses)
    min_val_loss = min(val_losses)
    min_val_loss_idx = np.argmin(val_losses)

    axs.scatter(min_train_loss_idx, min_train_loss, color='r', marker='o',
                    label=f'训练集最小损失 ({min_train_loss:.4f})')
    axs.scatter(min_val_loss_idx, min_val_loss, color='r', marker='o',
                    label=f'验证集最小损失 ({min_val_loss:.4f})')
                    
    axs.text(min_val_loss_idx-12, min_val_loss, f'验证集最小损失 ({min_val_loss:.4f})',
                fontsize=16, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
    axs.set_xlabel("训练周期/轮")

    fig.subplots_adjust(hspace=0.5)




    plt.savefig('result11/picc/%s %.3f %.3f %.3f loss.png' % (filename[:-10], sme, mape.item(), r_squared))




if not os.path.exists('result11/picc'):
    os.makedirs('result11/picc')
    print(f"文件夹 '{'result11/picc'}' 不存在，已成功创建。")
else:
    print(f"文件夹 '{'result11/picc'}' 已存在。")


model_savepath='result11/pre&loss'
model_name = []
model_namepath = os.listdir(model_savepath)

for name in model_namepath:
    model_name.append(name)


for filename in model_name:
    print(filename)
    paint(filename)



