# # 两种方法都能打开
# import pickle
# import numpy as np

# f = open("/root/autodl-tmp/ad_wxf/osad/experiment/general/10/AITEX_20241211_103750/model.pkl","rb")
# data = pickle.load(f)
# print(data)

# # img_path = './train_data.pkl'
# # img_data = np.load(img_path)
# # print(img_data)

import torch

filePath = "/root/autodl-tmp/ad_wxf/osad/experiment/general/10/AITEX_20241211_103750/model.pkl"
f = open(filePath,'rb')
data = torch.load(f,map_location='cpu')#可使用cpu或gpu
print(data)


