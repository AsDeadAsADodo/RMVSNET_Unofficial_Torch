# RMVSNET_Unofficial_Torch
## Warning
**Unfinished** ! training loss high , possibly something wrong with the code.

Currently using models.convgru for cost regularization

Only implement forward cost regularization

If uncomment convgru.py variable code , could run roughly 32 batch on a 16GB GPU 

### Update 2023/4/14 training loss go down

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
 
 ## Run
 Modify data root and other parameters in train.sh
 
 run `sh train.sh`
