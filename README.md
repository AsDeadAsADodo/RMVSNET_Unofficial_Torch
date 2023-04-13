# RMVSNET_Unofficial_Torch
## Warning
**Unfinished** ! training loss high , possibly something wrong with the code.

Currently using models.convgru for cost regularization

Only implement forward cost regularization

If uncomment convgru variable code , able to run around 32 batch on a 16GB GPU 

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
