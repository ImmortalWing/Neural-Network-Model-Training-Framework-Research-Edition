#import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import numpy as np
import pandas as pd
from 模型.mymodel import DAR,LSTM,CNNGRU,GRU,TCN
import matplotlib.pyplot as plt
import seaborn as sns

#本文件的任务是
def disparity(filename):
    data = np.load('result11/bestnp/'+filename)
    test_actual = data[0,:,-1]
    test_predicted = data[1,:,-1]
    test_disparity = test_actual-test_predicted

    # 均方误差
    sme = mean_squared_error(test_actual, test_predicted, squared=False)
    # 平均绝对百分比误差
    mape = np.mean(np.abs((test_actual - test_predicted) / test_actual))
    # 对称平均绝对百分比误差
    smape = np.mean(np.abs(test_actual - test_predicted) / np.abs(
        (test_actual + test_predicted) / 2))
    r_squared = r2_score(test_actual, test_predicted)

    filepath = 'result11/disparity/%s %.3f %.3f %.3f.png' % (filename[:-10], sme, mape.item(), r_squared)
    name = filename[:-10]
    return test_disparity,name,filepath


if not os.path.exists('result11/disparity'):
    os.makedirs('result11/disparity')
    print(f"文件夹 '{'result11/disparity'}' 不存在，已成功创建。")
else:
    print(f"文件夹 '{'result11/disparity'}' 已存在。")


model_savepath='result11/bestnp'
model_name = []
model_namepath = os.listdir(model_savepath)


for name in model_namepath:
    model_name.append(name)


disparitylist = []
modelnamelist = []
for filename in model_name:
    disparitylist.append(disparity(filename)[0]) #
    modelnamelist.append(disparity(filename)[1])


# 使用Seaborn绘制曲线  
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.5, rc={"lines.linewidth": 1.5})
#plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(18, 5))
#plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9)


for i in range(3):  
    
    plt.subplot(1, 3, i + 1)  
    sns.lineplot(data=disparitylist[i], label=modelnamelist[i], color='red')  # 将颜色设置为红色  
    plt.title(f'Subplot {i + 1}')  
    plt.xlabel("Test sample")  
    plt.ylabel("disparity/m")  
    plt.ylim(-1, 1)
    plt.tight_layout()    
  
plt.savefig('result11/disparity/1.png') 

# 创建新的图形以绘制第二行子图  
plt.figure(figsize=(12, 5))  
  
# 第二行两张子图  

for i in range(3, 5):  # 从第三个模型开始，绘制两个子图  
    plt.subplot(1, 2, i - 2)  # 由于子图是从1开始编号的，这里需要调整编号  
    sns.lineplot(data=disparitylist[i], label=modelnamelist[i], color='red')  # 将颜色设置为红色  
    plt.title(f'Subplot {i - 1}')  # 标题编号也相应调整  
    plt.xlabel("Test sample")  
    plt.ylabel("disparity/m")  
    plt.ylim(-1, 1)

    plt.tight_layout()    
plt.savefig('result11/disparity/2.png')
  

