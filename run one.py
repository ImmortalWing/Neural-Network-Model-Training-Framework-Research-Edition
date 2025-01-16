#import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import numpy as np
import pandas as pd
from modelcode.mymodel import DAR,LSTM,CNNGRU,GRU,TCN
import matplotlib.pyplot as plt




def paint(filepath):
    data_list=[]
    loss_list=[]
    datat = np.load('result11/pre&loss/'+filepath[0])
    test_actual = datat[0,:,-1]

    for filename in filepath:

        datat = np.load('result11/pre&loss/'+filename)
        
        test_predicted = datat[1,:,-1]
        data_list.append(test_predicted)  
        losst = np.load('result11/loss/'+filename)
        loss_list.append(losst)

    data_list=np.array(data_list).T
    loss_list=np.array(loss_list)
    labels = ['CNN-GRU', 'DAR', 'GRU', 'LSTM', 'TCN']  
    data_list = pd.DataFrame(data_list, columns=labels) 


    sns.set_theme(style="white", palette="tab10", font_scale=2, rc={"lines.linewidth": 1.5})
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 20})

    fig, axs = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    axs.set_title('result')
    sns.lineplot(data=data_list)
    sns.lineplot(data=test_actual, label='Actual')
    axs.set_xlabel("Test sample")
    axs.set_ylabel("Groundwater level/m")
    axs.legend()

    plt.savefig('result11/allone/1.png')

#5,3,80
    sns.set_theme(style="white", palette="tab10", font_scale=2, rc={"lines.linewidth": 1.5})
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 20})

    fig, axs = plt.subplots(5, 1, figsize=(10, 25))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)
    for i in range(5):
        axs[i-1].set_title(labels[i-1])

        sns.lineplot(data=loss_list[i-1,0,:], label='Train Loss', ax=axs[i-1])
        sns.lineplot(data=loss_list[i-1,1,:], label='Validation Loss', ax=axs[i-1])
        sns.lineplot(data=loss_list[i-1,2,:], label='Test Loss', ax=axs[i-1])

        min_train_loss = min(loss_list[i-1,0,:])
        min_train_loss_idx = np.argmin(loss_list[i-1,0,:])
        min_val_loss = min(loss_list[i-1,2,:])
        min_val_loss_idx = np.argmin(loss_list[i-1,2,:])
        axs[i-1].scatter(min_train_loss_idx, min_train_loss, color='r', marker='o',
                        label=f'Min Train Loss ({min_train_loss:.4f})')
        axs[i-1].scatter(min_val_loss_idx, min_val_loss, color='r', marker='o',
                        label=f'Min Validation Loss ({min_val_loss:.4f})')
        axs[i-1].text(min_val_loss_idx-2, min_val_loss, f'Min Val Loss ({min_val_loss:.4f})',
                    fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
        axs[i-1].set_xlabel("epochs")
        fig.subplots_adjust(hspace=0.5)

    plt.savefig('result11/allone/2.png')


    print(data_list)
    print(data_list.shape)
    sns.set_theme(style="white", palette="tab10", font_scale=2, rc={"lines.linewidth": 1.5})

    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axs = plt.subplots(5, 2, figsize=(20, 25))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)

    for i in range(5):

        axs[i-1,0].set_title(labels[i-1])
        sns.lineplot(data=data_list.iloc[:,i-1], label='Predicted', ax=axs[i-1,0])
        sns.lineplot(data=test_actual, label='actrual', ax=axs[i-1,0])
        axs[i-1,0].set_xlabel("Test sample")

        axs[i-1,1].set_title(labels[i-1])

        sns.lineplot(data=loss_list[i-1,0,:], label='Train Loss', ax=axs[i-1,1])
        sns.lineplot(data=loss_list[i-1,1,:], label='Validation Loss', ax=axs[i-1,1])
        sns.lineplot(data=loss_list[i-1,2,:], label='Test Loss', ax=axs[i-1,1])

        min_train_loss = min(loss_list[i-1,0,:])
        min_train_loss_idx = np.argmin(loss_list[i-1,0,:])
        min_val_loss = min(loss_list[i-1,2,:])
        min_val_loss_idx = np.argmin(loss_list[i-1,2,:])
        axs[i-1,1].scatter(min_train_loss_idx, min_train_loss, color='r', marker='o',
                        label=f'Min Train Loss ({min_train_loss:.4f})')
        axs[i-1,1].scatter(min_val_loss_idx, min_val_loss, color='r', marker='o',
                        label=f'Min Validation Loss ({min_val_loss:.4f})')
        axs[i-1,1].text(min_val_loss_idx-2, min_val_loss, f'Min Val Loss ({min_val_loss:.4f})',
                    fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))
        axs[i-1,1].set_xlabel("epochs")
        fig.subplots_adjust(hspace=0.5)


    plt.savefig('result11/allone/4.png')


    sns.set_theme(style="white", palette="tab10", font_scale=2, rc={"lines.linewidth": 1.5})
    plt.rcParams['font.family'] = 'Times New Roman'

    # 创建子图
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    # 修改图片边距

    test_actual1 = pd.DataFrame(test_actual,columns=['Actual'])

    #sns.lineplot(data=test_actual1.iloc[::5])
    aa= (data_list.sub(test_actual1['Actual'], axis=0)).abs()

    sns.lineplot(data=data_list.iloc[::5,:],dashes=False,ax = axs[0])
    sns.lineplot(data=test_actual1[::5],dashes=False,ax = axs[0])

    #axs[0].set_xlabel("Test sample")
    axs[0].set_ylabel("Groundwater level/m")

    sns.boxenplot(data=aa,ax = axs[1])
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.95)
    fig.subplots_adjust(hspace=0.5)

    plt.savefig('result11/allone/5.png')
    print(aa.mean())



if not os.path.exists('result11/allone'):
    os.makedirs('result11/allone')
    print(f"文件夹 '{'result11/allone'}' 不存在，已成功创建。")
else:
    print(f"文件夹 '{'result11/allone'}' 已存在。")


model_savepath='result11/bestnp'
model_name = []
model_namepath = os.listdir(model_savepath)

for name in model_namepath:
    model_name.append(name)


paint(model_name)
