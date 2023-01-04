# Neural Network


[L层神经网络.html](http://xuaii.github.io//L层神经网络.html)
# Neural Network
## 实现方式：采用BP算法，使用梯度下降来优化W，b
## 需要实现的函数如下：
`initial()`:根据用户自定义的层`layer`->[ $n_1, n_2,n_3,...,n_m$ ]来初始化参数W，b

`sigmoid()`:输入$X \in R^{m x n}$ ，返回$A = \frac{1}{1+e^{-WX + b}}$

`liner_propagation()`:实现一层的前向传播，返回 A，并且缓存中间量Z 

`L_layers_propagation()`:实现L层的前向传播，并且缓存所有中间层的A，Z

`predict()`:使用训练好的模型Ws，bs预测某一输入特征向量对应的预测值

`back_propagation()`:实现一层的反向传播

`L_back_propagation()`:实现L层链式反向传播

`update_parameters()`:每一个epoch更新参数

`evaulate()`:评估模型

`model()`:主循环，BP算法梯度下降的循环

`load_data()`:加载数据

**在整体的设计中并没有loss函数的出现，是因为，在反向传播过过中，dloss/dW的计算并不涉及loss的值，dloss/dW的表达式中仅有y_truth 和 y_pred**


# 链式偏微分 推导（这里是根据西瓜书上原始公式推导，所以是累计神经网络）
## 由于在布置编程作业之前就自己实现了一下神经网络，所以实现的是任意层数和任意数量神经元的神经网络，所以就改下参数交作业了
* 由于神经网络层数可能会非常多，所以在反向传播时loss对每一层的W，b求导会重复很多中间步骤
 $$设：\beta^i \in R^{m \cdot n}$$
  $$ A^{i-1} \in R^{n \cdot s}$$
  $$ Z^{i} \in R^{m \cdot s}$$


$$Z^i = W^i A^{i-1} + b^i = \beta^i A^{i-1}, $$ 

$$A^i = sigmoid(Z^i);$$


$$\frac{\partial E}{\partial \beta^i}  = \frac{\partial Z^i}{\partial \beta^i} \cdot  \underbrace{ \frac{\partial A^i}{\partial Z^i} \cdot  \overbrace {\frac{\partial E}{\partial A^i}}^{\partial A^i} }_{\partial Z^i}  ， \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad  （1.1）$$


$$ \frac{\partial E}{\partial A^{i-1}}  = \frac{\partial Z^i}{\partial A^{i-1}}  \cdot \underbrace {\frac{\partial A^i}{\partial Z^i}  \cdot  \overbrace{  \frac{\partial E}{\partial A^i} }^{\partial A^i} }_{\partial Z^i}, \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad  （1.2）$$


其中： 
$$\partial A^i由上一层反向传播提供,$$ 


**现在考虑每一层反向传播的计算**

`已知:`
$$\frac{\partial E}{\partial A^k}, \frac{\partial A^k}{\partial Z^k}, \qquad其中：k是层数$$

`求解:`
$$\frac{\partial E}{\partial {Z^k}_{i,j}}  \qquad其中： i,j \in \{ i,j | i \in (1,Z^k.shape[0]， j \in (1,Z^k.shape[1])\}$$


`分析：`

***由于 （暂时忽略掉上标，仅在需要的时候添加）***
$$Z = \beta A$$
$$Z_{i,j} = \sum_{k=1}^{n}\beta_{i,k}A_{k,j}$$
**该式子表明**
$$Z_{i,j}的值与同行\beta{i,k}$$

$$即每一个\beta与Z的第a行相关$$
$$那么要计算\frac{\partial E}{\partial \beta_{a,b}}$$

$$即需要\frac{\partial E}{\partial \beta_{a,b}} = \sum_{k=1}^{s}\frac{\partial E}{\partial Z_{a,k}} \frac{\partial Z_{a,k}}{\partial \beta_{a,b}}$$
$$而：\frac{\partial Z_{a,k}}{\partial \beta_{a,b}} = A_{b,k}$$
$$那么：\frac{\partial E}{\partial \beta_{a,b}} = \sum_{k=1}^{s}\frac{\partial E}{\partial Z_{a,k}} \cdot A_{b,k}$$
$$\frac{\partial E}{\partial \beta_{a,b}} = \sum_{k=1}^{s} {(\frac{\partial E}{\partial Z})}_{a,k} \cdot A_{b,k}$$
$$ = \sum_{k=1}^{s}(\frac{\partial E}{\partial Z})_{a,k}(A^T)_{k,b}$$
$$所以有：\frac{\partial E}{\beta} = dZ A^T$$





**由于推导的迭代式子中包含A, Z, 等前向传播产生的中间结果， 所以在前向传播时需要将它们缓存下来**


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import copy
from IPython import display
import time
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report, roc_auc_score
# 设置随机种子，避免每次的结果不一样
np.random.seed(1)
%matplotlib inline

```

## 初始化函数`initial(layers)`
```
input: layers -> [layer_1, layer_2, layer_3, ..., layer_m]
```

```python
output: paramters -> dict()  
    parameters["w"] = temp_w -> list()
    parameters['layers'] = layers -> list()
```


```python

def initial(layers):
    parameters = dict()
    temp_w = list()
    for i in range(1, len(layers)):
        temp_w.append(np.random.randn(layers[i], layers[i-1]))
    parameters["w"] = temp_w
    parameters['layers'] = layers
    return parameters
```

## 激活函数 Sigmoid()
**激活函数种类：**

`sigmoid()`:"S"形函数$f(x) = \frac{1}{1+e^{-WX + b}}$$

`ReLU()`:线性修正单元$f(x) = max(0,x)$

`tanh()`:双曲正切函数$f(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}$

`ELU`:

![](https://img-blog.csdn.net/20180104121237935?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

`PReLU`:

![](https://img-blog.csdn.net/20180104115618817?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQva2FuZ3lpNDEx/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


`LReLU`:x负半轴斜率比较小的PReLU


```python
def sigmoid(A):
    return 1/(1+np.exp(-A))
```

## 前向线性传播

$Z = WX + b$

$A = \frac{1}{1+e^-Z}$


```python
def liner_propagation(w, A):
    Z = np.dot(w, A)
    A = sigmoid(Z)
    return Z, A

```

## L层的前向传播
* 是L次`linear_propagation`的叠加
* 由于反向传播的需要，这里的L层前向传播需要缓存每一层的A，Z到`A_cache` and `Z_chache`


```python
def L_layers_propagation(parameters, x):
    A_cache = list()
    Z_cache = list()
    cache = dict()
    A_cache.append(x)
    Z_cache.append(x)
    w = parameters["w"]
    layers = parameters['layers']
    for i in range(0, len(layers) - 1):
        Z, A = liner_propagation(w[i], A_cache[i])
        assert Z.shape == A.shape
        Z_cache.append(Z)
        A_cache.append(A)
        
    cache["A"] = A_cache
    cache["Z"] = Z_cache
    
    return cache
```

### Loss函数（均方误差）
* 如果要实现任意loss函数，只会影响链式传播哦的最后一项，而不会影响链式传播的中间过程
* 由于最近DDL多，所以任意loss函数的实现，在寒假实现


```python
def lossFunc(Y_pre, Y_true, method = "MSE"):
    temp = Y_pre - Y_true
    return .5 * np.dot(temp, temp.T)
```

## 预测函数
`input = x`

`output = y`

$y = f(x)$

$y>.5$          =>      $y=1$

$y\le .5$       =>      $y= 0$


```python
def predict(parameters, x, draw = False):
    Y = L_layers_propagation(parameters, x)['A'][-1]
    if draw == False:    
        Y[Y > .5] = 1
        Y[Y <= .5] = 0
    return Y
```

## 反向传播函数

`back_propagation`

**实现反向传播的难点在于求梯度**

**求梯度的方法如下：**


1. $dx = \frac{f(x) - f(x-\Delta x)}{\Delta x}$
    * 老师上课提到的这种方法会导致精确度问题
    * 如当在梯度下降算法中设定$\Delta x= \eta$时该算法退化为 `W := W - loss(x-learning_rate)` 


2.利用求偏导法则，一层一层的求偏导
    
* 使用类似 tensorflow 的 Autograd 的计算图，将每一个参数变量加入`计算图`中，然后可以找到变量之间的关联然后求导(实现过于复杂，对于简单的全链接网络没必要这样做)
  
* 使用高等数学中的多变量求导，结合线性代数的矩阵变换进行 实数 对 矩阵 的  链式求导

**参考**

[矩阵求导术（下）](https://zhuanlan.zhihu.com/p/24863977)

[矩阵求导术（上）](https://zhuanlan.zhihu.com/p/24709748)

*引入克罗内克积实现矩阵的链式求导，在注释代码中是没经过花间的Kron积，运算量极大，经过化简得到简单的矩阵表达式*


```python
def back_propagation(W, A_prev, dZ):
    # dZA_prev = np.diag(   (A_prev * (1 - A_prev)).T.flatten()  )

    # I = np.eye(A_prev.shape[1])


    # dZ_prev = np.dot(np.dot(dZA_prev,np.kron(I, W.T)), dZ)

    # I = np.eye(W.shape[0])

    # dW = np.dot(np.kron(A_prev, I),dZ)
    # print(A_prev.shape, W.shape, dZ.shape)
    
    dZ_prev = np.multiply( A_prev * (1 - A_prev), np.dot(W.T,dZ))
    
    dW = np.dot(dZ,A_prev.T)

    return dZ_prev, dW
```


```python
def L_back_propagation(cache, parameters, y):
    w = parameters["w"]
    layers = parameters["layers"]
    L = len(layers) - 1
    A = cache["A"]
    dW_list = list()
    dZ_prev = A[L] * (1 - A[L]) * (A[L] - y)

    for i in reversed(range(0, L)):
        dZ_prev, dW = back_propagation(w[i], A[i], dZ_prev)
        dW_list.append(dW)
    cache["dW"] = dW_list
    return cache
```

## 更新参数
pass


```python
def update_parameters(cache, parameters, learning_rate):
    w = parameters["w"]
    layers = parameters["layers"]
    L = len(layers) - 1
    dW = cache["dW"]
    for i in range(L):
        w[i] -= dW[L-i-1] * learning_rate
        # b[i] = b[i] - db[len(b)-i-1] * learning_rate
    parameters["w"] = w
    return parameters
```

## evaulate
**使用准确度作为评估标准**


```python
def evaluate(Z, Y):
    bools = Z == Y
    accuracy = np.sum(np.reshape(bools,bools.size))/Y.shape[1]
    return accuracy
```


```python
def evaluate_detial(y_t, y_p):
    
    y_test = copy.deepcopy(y_t)
    y_pred = copy.deepcopy(y_p)
    
    print("The roc_auc_score is {}".format(roc_auc_score(y_test, y_pred)))    
    tpr,fpr,thresholds = roc_curve(y_test,y_pred)
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC curve for diabetes classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    
    # mean = np.sum(y_pred)/y_pred.shape[0]
    y_pred[(y_pred <  0.5)] = 0
    y_pred[(y_pred >= 0.5)] = 1
    # plt.scatter(x_test[0, :], x_test[1, :], c = y_pred[:, 0])
    print("The accuracy is {}".format(accuracy_score(y_test[:,0], y_pred[:,0])))
    
    # 计算召回， 查全率， 查准率 。。。。
    target_names = ['class0','class1']
    print(classification_report(y_test,y_pred,target_names = target_names))
```


```python

## 绘图

```


```python
def Plot(X, label, parameters):

    x_min, x_max = X[0,:].min() - .5, X[0,:].max() + .5
    y_min, y_max = X[1,:].min() - .5, X[1,:].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = predict(parameters, np.c_[xx.ravel(), yy.ravel(), np.ones((yy.ravel().shape))].T)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0,:],X[1,:],c=label[0,:])

```

## load_data
***加载数据的管道***


```python
def load_data(shape = "circle"):
    X = np.random.rand(2,200)
    one = np.ones((1,200))
    X = np.vstack((X,one))
    C = []
    for x in X.T:
        if shape == "circle":
            # 同心圆
            if (x[0]-.5)**2 + (x[1]-.5)**2 < .03:
                C.append(1)
            elif (x[0]-.5)**2 + (x[1]-.5)**2 < .13:
                C.append(0)
            else:
                C.append(1)
        elif shape == "xor":
            # 四分
            if (x[0] - .5)*(x[1] - .5) >=0:
                C.append(1)
            else:
                C.append(0)
                
    x_min, x_max = X[0,:].min() - .05, X[0,:].max() + .05
    y_min, y_max = X[1,:].min() - .05, X[1,:].max() + .05
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    test = np.c_[xx.ravel(), yy.ravel(), np.ones((yy.ravel().shape))].T

    return X, np.array(C).reshape(1,-1), test, (xx,yy)
```

## Back Propagation(BP算法的整体框架)
* 使用梯度下降的优化方法
* 其他可用的优化方法
    * SGD 随机梯度下降
    * MBGD （Mini Batch Gradient Descent）用于样本容量大，内存/现存不够的情况
    * Momentum 动量梯度下降
    * Nesterov NAG
    * Adagrad
    * Adaelta
    * Adam



```python
def model(X, Y, test,canvs, layers, learning_rate = 0.01, epoch = 1000, detial = True, draw = False):
    parameters = initial(layers)
    draw_param = []
    for i in range(epoch):
        cache = L_layers_propagation(parameters, X)
        cache = L_back_propagation(cache, parameters, Y)
        parameters = update_parameters(cache, parameters, learning_rate)
        if i % 500 == 0 and draw == True:
            # draw_param.append(copy.deepcopy(parameters))
            yield predict(parameters, test, draw = True).reshape(canvs[0].shape)
        if i % 1000 == 0 and detial == True:
            print("this is {}th epoch.".format(i))
    Y_predict = predict(parameters, X, draw=False)
    accuracy = evaluate(Y, Y_predict)
    print("accuracy is {}".format(accuracy))
    if draw == False:
        return parameters
```

### 实现累计BP算法


```python
def model_caculate(X, Y, X_test, Y_test, layers, learning_rate = 0.01, epoch = 1000, interval = 50):
    parameters = initial(layers)
    Loss = []
    loss = None
    Acc = []
    accuracy = None
    start = time.time()
    Y_predict = None
    for i in range(epoch):
        cache = L_layers_propagation(parameters, X)
        cache = L_back_propagation(cache, parameters, Y)
        loss = lossFunc(Y, cache["A"][-1])
        
        parameters = update_parameters(cache, parameters, learning_rate)
        if i % interval == 0:
            Loss.append(loss[0,0])
            Y_predict = predict(parameters, X_test, draw=False)
            accuracy = evaluate(Y_test, Y_predict)
            Acc.append(accuracy)

            if i % (interval * 50) == 0:
                now = time.time()
                print("Calculate Dense Net:~$ {}th epoch,  accuracy: {},   loss:{},  time:{}-".format(i, accuracy, loss[0,0], now - start))
                start = time.time()
    
    return Loss, Acc, Y_predict
```


```python
X, label, test, canvs = load_data()

layers = [3,6,9,9,6,1]
epoch = 25000
Loss, accuracy, Y_pred = model_caculate(X, label, X, label, layers, learning_rate=0.01, epoch= epoch)


#开始画图
x = np.linspace(0, 25000, 500)
plt.plot(x, np.array(Loss)/(max(Loss) - min(Loss)), color='green', label='training loss')
plt.plot(x, np.array(accuracy)/(max(accuracy) - min(accuracy)), color='red', label='training accuracy')
# 显示图例
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()
evaluate_detial(label.T, Y_pred.T)


```

    Calculate Dense Net:~$ 0th epoch,  accuracy: 0.725,   loss:21.40100100371333,  time:0.0010008811950683594-
    Calculate Dense Net:~$ 2500th epoch,  accuracy: 0.725,   loss:19.748958341789084,  time:0.7027204036712646-
    Calculate Dense Net:~$ 5000th epoch,  accuracy: 0.725,   loss:18.484585855571503,  time:0.7009453773498535-
    Calculate Dense Net:~$ 7500th epoch,  accuracy: 0.705,   loss:16.377806924887945,  time:0.6996634006500244-
    Calculate Dense Net:~$ 10000th epoch,  accuracy: 0.87,   loss:10.950215467935237,  time:0.7041025161743164-
    Calculate Dense Net:~$ 12500th epoch,  accuracy: 0.88,   loss:9.160948492962692,  time:0.701648473739624-
    Calculate Dense Net:~$ 15000th epoch,  accuracy: 0.885,   loss:8.84026852930535,  time:0.6946816444396973-
    Calculate Dense Net:~$ 17500th epoch,  accuracy: 0.965,   loss:6.105461022050368,  time:0.685279369354248-
    Calculate Dense Net:~$ 20000th epoch,  accuracy: 0.99,   loss:2.3057686501449832,  time:0.7090015411376953-
    Calculate Dense Net:~$ 22500th epoch,  accuracy: 0.99,   loss:0.958385284049958,  time:0.6965689659118652-



![png](C:/Users/herrn/Downloads/1185500/output_36_1.png)


    The roc_auc_score is 1.0
    The accuracy is 1.0
                  precision    recall  f1-score   support
    
          class0       1.00      1.00      1.00        55
          class1       1.00      1.00      1.00       145
    
       micro avg       1.00      1.00      1.00       200
       macro avg       1.00      1.00      1.00       200
    weighted avg       1.00      1.00      1.00       200




![png](C:/Users/herrn/Downloads/1185500/output_36_3.png)



```python

```

### 实现标准BP算法（也就是Mini-batch = 1，这样的话在python下使用for循环喂数据会非常慢）


```python
def model_std(X, Y, X_test, Y_test, layers, learning_rate = 0.01, epoch = 1000, interval = 50):
    parameters = initial(layers)
    Loss = []
    Acc = []
    accuracy = None
    loss = None
    Y_pred = None
    start = time.time()
    for i in range(epoch):
        for j in range(0, X.shape[1]):
            cache = L_layers_propagation(parameters, X[:, j].reshape(3, -1))
            cache = L_back_propagation(cache, parameters, Y[:, j].reshape(1, -1))
            parameters = update_parameters(cache, parameters, learning_rate)
        if i % interval == 0:
            Y_pred = predict(parameters, X_test, draw = False)
            
            loss = lossFunc(Y_test, Y_pred)[0,0]
            Loss.append(loss)
            accuracy = evaluate(Y_test, Y_pred)
            Acc.append(accuracy)
            if i % (interval * 20) == 0:
                now = time.time()
                print("Stander Dense Net:~$  {}th epoch,  accuracy: {},   loss:{},  time:{}".format(i, accuracy, loss, now - start))
                start = time.time()
    return Loss, Acc, Y_pred

```


```python

```


```python
X, label, test, canvs = load_data()
layers = [3,6,9,9,6,1]
epoch = 33300
Loss,accuracy, Y_pred = model_std(X, label, X, label, layers, learning_rate=0.01, epoch= epoch)

#开始画图
x = np.linspace(0, 25000, epoch/50)
plt.plot(x, np.array(Loss)/(max(Loss) - min(Loss)), color='green', label='training loss')
plt.plot(x, np.array(accuracy)/(max(accuracy) - min(accuracy)), color='red', label='training accuracy')
# 显示图例
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()

evaluate_detial(label.T, Y_pred.T)
```

    Stander Dense Net:~$  0th epoch,  accuracy: 0.65,   loss:35.0,  time:0.018254756927490234



```python

```

## 动态可视化训练过程


```python
def main(epoch= 50000, detial=True, draw=False):   
    X, label, test, canvs = load_data()
    layers = [3,6,9,9,6,1]
    if draw == True:
        for im in model(X, label, test, canvs, layers, learning_rate=0.01, epoch= epoch, detial = detial, draw = True):
            plt.clf()
            display.clear_output(wait=True)
            plt.contourf(canvs[0] ,canvs[1], im, cmap=plt.cm.Spectral)
            plt.scatter(X[0,:], X[1,:],c=label[0,:])
            plt.pause(0.01)
    else:
        parameters = model(X, label, test, canvs, layers, learning_rate=0.01, epoch= 30000, detial = detial, draw = draw)
        Plot(X, label, parameters)
```


```python
main(epoch= 20000, detial=False, draw=True)
```

# 西瓜数据集3.0$\alpha$上的分类



### 导入数据


```python
import numpy as np
import copy
def load(): 
    data = np.array([[0.697, 0.460, 1],
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
    [0.719, 0.103, 0]])
    
    Y = copy.deepcopy(data[:, 2].reshape(-1, 1).T)
    data[:, 2] = 1
    X = copy.deepcopy(data.T)
    
    X_test = copy.deepcopy(np.c_[X[:, 0:2],X[:, -3:-1]])
    Y_test = copy.deepcopy(np.c_[Y[:, 0:2],Y[:, -3:-1]])
    
    X_train = copy.deepcopy(X[:, 2:-2])
    Y_train = copy.deepcopy(Y[:, 2:-2])

    return X_train, Y_train, X_test, Y_test, X, Y
X_train, Y_train, X_test, Y_test, X, Y = load()
print(Y.shape)
```

### 累计BP


```python
X_train, Y_train, X_test, Y_test, X, Y = load()
```


```python
X_train.shape, Y_train.shape, X_test.shape, Y_test.shape
```


```python
layers = [3,5,1]
epoch = 5500
interval = 1
Loss, accuracy, Y_pred = model_caculate(X_train, Y_train, X_test, Y_test, layers, learning_rate=0.1, epoch= epoch, interval = interval)


#开始画图
x = np.linspace(0, 25000, epoch/interval)
plt.plot(x, np.array(Loss)/(max(Loss) - min(Loss)), color='green', label='training loss')
plt.plot(x, np.array(accuracy)/(max(accuracy) - min(accuracy)), color='red', label='training accuracy')
# 显示图例
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()
evaluate_detial(Y_test.T, Y_pred.T)
```


```python
X, label, test, canvs = load_data()
```


```python
X.shape, label.shape
```

### 标准BP算法
**由于实现了累计误差的算法， 所以可以将样样本拆分得到mini-batch的神经网络就是标准BP算法**


```python
layers = [3,5,1]
epoch = 5500
interval = 1
Loss,accuracy, Y_pred = model_std(X_train, Y_train, X_test, Y_test, layers, learning_rate=0.01, epoch= epoch)

#开始画图
x = np.linspace(0, 25000, epoch/50)
plt.plot(x, np.array(Loss)/(max(Loss) - min(Loss)), color='green', label='training loss')
plt.plot(x, np.array(accuracy)/(max(accuracy) - min(accuracy)), color='red', label='training accuracy')
# 显示图例
plt.legend()
plt.xlabel('iteration times')
plt.ylabel('rate')
plt.show()

evaluate_detial(Y_test.T, Y_pred.T)
```

# 模型比较
* 标准BP算法收敛更快
* 累计BP在python可以通过矩阵运算向量化加速,在迭代过程中震荡变小
* 累计BP 更占用内存，而标准BP可以一边读一边运算，收敛速度变慢但是梯度方向更准确
* Minibatch = n，可以通过调节n的大小使其达到一个合适的值，其收敛速度和迭代的准确性都能提高！

### 神经网络的优势
* 相对于决策树，线性模型，神经网络可以通过增加层数/增加神经元的方式拟合任意非线性函数
* 在之前的同心圆数据集中，神经网络的表达能力远远优于其他模型

\begin{split}
& C(F , G) = \frac{F \cdot G+(1 - F\otimes G)}{2}\\
& F\cdot G = \vee_U(\mu_F(u_i)\land\mu_G(u_i)) \\
& F\otimes G = \vee_U(\mu_F(u_i)\vee\mu_G(u_i))
\end{split}
