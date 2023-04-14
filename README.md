# RMVSNET_Unofficial_Torch
## Warning
**Unfinished** ! training loss high , possibly something went wrong with the code.

Currently using models.convgru for cost regularization

Only implement forward cost regularization

### Update 
2023/4/14 training loss goes down , tested within 1 epoch

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
