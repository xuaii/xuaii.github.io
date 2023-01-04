# 线性模型


[线性模型.html](http://xuaii.github.io/线性模型.html)
```python
import numpy as np
import matplotlib.pyplot as plt
import copy
from IPython import display
np.random.seed(0)
data = [[0.697, 0.460, 1],
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


```python
def load_data():
    X = np.array(data).T
    label = copy.deepcopy(X[2,:].reshape(1,17))
    X[2,:] = 1
    assert X.shape == (3,17)
    assert label.shape == (1,17)
    return X, label
```


```python

```

$$
l(\beta) = \sum_{i=1}^{m}(-y_i \beta^Tx_i+ln(1+e^{\beta^T x_i}))
$$
$$
= \sum_{i=1}^{m}-y_i \beta^Tx_i+\sum_{i=1}^{m}ln(1+e^{\beta^T x_i})
$$
$$
= (\beta^TX)Y + np.sum(ln(1+e^{Y_i}))
$$


使用numpy实现如下：

```python
part_1 = np.dot(Y, label.T)
part_2 = np.sum(np.ln(1+np.exp(Y)))
```


```python
def loss(label, Y):
    part_1 = -np.dot(Y, label.T)
    part_2 = np.sum(np.log(1+np.exp(Y)))
    return part_1 + part_2
```



### 前向传播

$input = W,X$

$output = W^{'T} X^{'} = W^T X+b$


```python
def forward_propagation(W,X):
    Z = np.dot(W,X)
    A = 1/(1+np.exp(-Z))
    return A
```

### 初始化参数:
* 使用随机初始化
* n ------ 特征值数量为
* m ------ 样本数量

$w^{'} = (w,b)$

$w.shape == （1,n+1）$



```python
def initialization(n):
    return np.random.randn(1,n+1)
#     return np.zeros((1,n+1))
```

### 反向传播
$$
dw = -\sum_{i=1}^{m}x_i(y_i-p_1(x_i;\beta))
$$

$$
dw = -(label-Y)X^T
$$



```python
def back_propagation(X,Y,label):
    return -np.dot((label-Y),X.T)
```

### 训练循环
* 使用梯度下降法
* 学习率 = 0.5
* 迭代次数 10000


```python
def train(epoch=100,n = 2, learning_rate = 0.5, detial = True):
    X, label = load_data()
    W = initialization(n)
    losses = []
    W_list = []
    for i in range(0,epoch):
        Y = forward_propagation(W,X)
        l = loss(label, Y)
        dw = back_propagation(X,Y,label)
        losses.append(float(l))
        W += learning_rate * dw
        if  i % 100 == 0:
            W_list.append(copy.deepcopy(W))
        if i % 1000 == 0 and detial ==True:
            print("This is {}th epoch , loss = {}".format(i, l))
    return W_list, losses 
```


```python
def draw(W):
    W = W[0]
    x = np.linspace(0.2,0.8,100)
    y = -(W[0] * x + W[2])/W[1]
    plt.plot(x,y)
    for plot in data:
        if plot[2] == 0:
            c = 'red'
        else:
            c = 'green'    
        plt.scatter(plot[0],plot[1],c=c)
```




```python
def show(W_list):
    for W in W_list:
        plt.clf()
        display.clear_output(wait=True)
        draw(W)
        
        plt.pause(0.001)
```


```python
W_list,losses = train(epoch=10000,n=2,learning_rate=.01,detial=False)
```


```python
show(W_list[:100])
```


![png](C:/Users/herrn/Downloads/1185477/output_23_0.png)



```python
plt.plot(losses)
```




    [<matplotlib.lines.Line2D at 0x7f1b3a14aa50>]




![png](C:/Users/herrn/Downloads/1185477/output_24_1.png)



```python

```


```python

```


```python

```


```python

```


```python

```


```python

```



