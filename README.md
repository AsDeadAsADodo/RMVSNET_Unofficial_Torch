# RMVSNET_Unofficial_Torch
## Warning
**Unfinished** ! training loss high , possibly something went wrong with the code.

Currently using models.convgru for cost regularization

Only implement forward cost regularization

### Update 
2023/4/14 training loss goes down , tested within 1 epoch

2023/5/10 finish eval code , enable refineNet , tested full 16 epoches (No refineNet ,No backward regularization)

![_cgi-bin_mmwebwx-bin_webwxgetmsgimg  MsgID=2680283510007717649 skey=@crypt_b958ddbf_1ce47d463743668958258de22af31e17 mmweb_appid=wx_webfilehelper](https://github.com/AsDeadAsADodo/RMVSNET_Unofficial_Torch/assets/38915818/40c10b55-d5fb-4830-885b-e154c381b603)


## Description
Unofficial RMVSNET implementation .

Code mostly borrowed from  
1. [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch)
2. [RMVSNet-Pytorch](https://github.com/leejaeyong7/RMVSNet-Pytorch)
3. loss function from  [AA-RMVSNet](https://github.com/QT-Zhu/AA-RMVSNet)  

## File Tree
```
root
  |
  |___datasets
  |
  |___evaluations
  |
  |___lists
  |
  |___models
 ```
 
 ## Train
 Modify data root and other parameters in train.sh
 
 run `sh train.sh`
 
 ## Eval
 run `sh eval.sh`
 
 for visulaize depth image , modify pfm.py `line29`
 run `python pfm.py`
