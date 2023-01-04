# 决策树


[决策树.html](http://xuaii.github.io/决策树.html)
### 导入相关库
###### `pandas`：决策树的构建涉及到数据集的一些操作，利用`pandas的DataFrame`数据结构可以很好方便的完成
###### `copy` :在递归时浅拷贝会导致问题,使用`copy.deepcopy()`进行深拷贝
###### `matplot.pyplot`:绘制决策树的划分图像


```python
import pandas as pd
import copy
import matplotlib.pyplot as plt
import numpy as np
from math import fabs
```

### 导入数据


```python
input = [[0.697, 0.460, 1],
[0.774, 0.376, 1],
[0.634, 0.264, 1],
[0.608, 0.318, 1],
[0.556, 0.215, 1],
[0.403, 0.237, 1],
[0.481, 0.149, 1],
[0.437, 0.211, 1],
[0.666, 0.091, 0],
[0.243, 0.267, 0],
[0.245, 0.057, 0],
[0.343, 0.099, 0],
[0.639, 0.161, 0],
[0.657, 0.198, 0],
[0.360, 0.370, 0],
[0.593, 0.042, 0],
[0.719, 0.103, 0]]
```

### 定义回归树的节点类Node
###### `attrList      节点剩下的属性列表`
###### `Dataset       节点划分到的数据集`
###### `left/right    左右子树`
###### `c             叶节点的预测值`
###### `description   该节点的描述（可选）`
###### `attr          该节点划分属性`
###### `s             划分属性的值`

**考虑到使用非二叉树，在每次寻找最优化分的时候算法复杂度太高，所以此时使用二叉树就OK**


```python
class Node:
    def __init__(self, description="", c = -1, Dataset=pd.DataFrame(), attrList=[]):
        self.attrList = attrList
        self.Dataset = Dataset
        self.left = None
        self.right = None
        self.c = c
        self.attr = ""
        self.s = -1
        self.desciption = description

```

### 计算损失
$$\ell = \sum_{x_i \in R_1(j,s)}(y_i-c_1)^2+\sum_{x_i \in R_2(j,s)}(y_i-c_2)^2$$ 



```python
def loss(attr, s, data):
    D1 = data[data[attr] <= s]
    D1_mean = D1['label'].std() * D1.size
    D2 = data[data[attr] > s]
    D2_mean = D2['label'].std() * D2.size
    return D1_mean + D2_mean
```


```python

```

### 最小化损失
$$ \min_{j,s} ( \min_{c_1} \sum_{x_i \in R_1(j,s)}(y_i-c_1)^2 + \min_{c_2} \sum_{x_i \in R_2(j,s)}(y_i-c_2)^2   ) $$


**此处数据集不大 且 为了实现起来简单就是用了遍历所有属性的值找到最小的损失，应该还有一些方法可以优化找到最小值的过程**


```python
def findOptDiv(root):
    losses = []
    for attr in root.attrList:
        for s in root.Dataset[attr]:
            losses.append((loss(attr, s, root.Dataset), attr, s))
    minLoss = min(losses)
    return minLoss
```

### 二叉树的构建
**在以下情况返回IF**

` len(attrList) == 0 `：此时所有属性已经划分完毕， 就以该集合所有样本的`label`的均值作预测值
` Dataset.size == 1 `：此时该节点的样本仅有一个 就 以该样本的`label`值做预测值

**ELSE**
*将样本按最优划分划分为两个集合D1，D2，并分别构建`subTree`*


```python
def buildTree(root):
    # if root.Dataset.size() <= 1:
    #     description = "leaf node"
    #     c_p = root.Dataset['label'].mean()
    #     leaf = Node(description=description, c = c_p) 
    
    # 如果样本集合中只有一个样本那么该节点为叶节点，该叶节点的预测值是该唯一样本的label
    if root.Dataset.size == 1:
        root.c = root.Dataset['label']
        return
    
    # 如果已经将属性分完了，那么该节点为叶节点，剩下的样本集中label的期望为该叶节点的预测值
    elif len(root.attrList) == 0:
        root.description = "leaf node"
        root.c = root.Dataset['label'].mean()
        return 
    
    else:
        # 找到最优化分
        (_, attr, s) = findOptDiv(root)
        
        # 将节点的划分属性设为找到的attr
        root.attr = copy.deepcopy(attr)
        
        # 将按属性attr划分该节点值划分值s设为最优的s
        root.s  = copy.deepcopy(s)
        
        # 将样本集合按照找到的最优化分划分为D1， D2
        D1 = copy.deepcopy(root.Dataset[root.Dataset[attr] <= s])
        D2 = copy.deepcopy(root.Dataset[root.Dataset[attr] > s])
        
        # 将划分该节点属性从属性集合中删除
        list_notremoved = copy.deepcopy(root.attrList)
        root.attrList.remove(attr)
        list_removed =  copy.deepcopy(root.attrList)
        root.attrList = copy.deepcopy(list_notremoved)

        # 构建左子树和右子树
        root.left = Node(Dataset = D1, attrList=copy.deepcopy(list_removed))
        root.right = Node(Dataset = D2, attrList=copy.deepcopy(list_removed))
        buildTree(root.left)
        buildTree(root.right)
    return root
```

### 预测函数


```python
def predict(x, root):
    while(len(root.attrList) != 0):
        if x[root.attr] < root.s:
            root = root.left
        else:
            root = root.right
    return root.c
        
```

### 评估函数


```python

def evaluate(data_test, root):
    accuracy = 0
    for i in range(len(data_test)):
        res = predict(data_test.loc[i], root)
        # 将回归问题转为分类问题
        res = .5 if res > .5 else 0
        accuracy += fabs(res - data_test.loc[i]["label"])
    return 1 - accuracy / len(data_test)
        
```


```python
data = pd.DataFrame(input, columns=['密度','含糖率',"label"])
root = Node(Dataset=data, attrList = ['密度','含糖率'])
root = buildTree(root)
```

### 可以大致看出决策过程
* 先看含糖率：
	* 小于.13
		* 小于.666 坏瓜
		* 大于.666 好瓜
	* 大于.13
		* 小于.697 0.6的概率是好瓜
		* 大于.697 1的概率是好瓜


```python
print(root.attr,root.s)
print(root.left.attr,root.left.s,root.left.left.c,root.left.right.c)
print(root.right.attr,root.right.s,root.right.left.c,root.right.right.c)
```

    含糖率 0.103
    密度 0.666 0.0 0.0
    密度 0.697 0.6363636363636364 1.0


## 可视化和评估模型表现


```python
s1 = root.s
s21 = root.left.s
s22 = root.right.s
plt.plot([.2,.8],[s1,s1],'r-')
plt.plot([s21,s21],[0,s1],'r-')
plt.plot([s22,s22],[s1,.5],'r-')
for plot in input:
    if plot[2] == 0:
        c = 'red'
    else:
        c = 'green'    
    plt.scatter(plot[0],plot[1],c=c)
print("THE ACCURACY OF REGRESSION TREE IS {}".format(evaluate(data, root)))

```

    THE ACCURACY OF REGRESSION TREE IS 0.6176470588235294



![png](C:/Users/herrn/Downloads/1185491/output_23_1.png)



```python

```


```python

```

