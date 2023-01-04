# 支持向量机



[svm.html](http://xuaii.github.io/svm.html)

# 线性核SVM 和 高斯核SVM




# SMO 算法解析


#### 最优化目标
$$\max_{\alpha} =\sum_{i=1}^{m}\alpha_i -  \sum_{i=1}^{m} \sum_{j=1}^{m}\alpha_i\alpha_jy_iy_jx_i^Tx_j \tag{1}$$
$$\text{ s.t. } \qquad \alpha_i \ge 0,\quad \sum_{i=1}^{N}\alpha_iy_i = 1 \tag{2}$$

#### 约束条件

$$ $$



#### 算法原理
**我们可以先确定 两个$\alpha_i,\alpha_j$.**
**在例子中设$i=1 , \quad j=2$.**


`此时最大化目标`：
$$arg\max_{\alpha_1, \alpha_2}W(\alpha_1, \alpha_2) = \alpha_1 + \alpha_2 - \frac{1}{2}K_{1,1}y_1^2\alpha_1^2 - \frac{1}{2}K_{2,2}y_2^2\alpha_2^2 - K_{1,2}y_1y_2\alpha_1\alpha_2 - y_1\alpha_1\sum_{i=3}^{m}\alpha_iy_iK_{i,1} - y_2\alpha_2\sum_{i=3}^{m}\alpha_iy_iK_{i,2} + C \tag{3}$$


`根据(2)`
$$\alpha_1y_1 + \alpha_2y_2 = -\sum_{i=3}^{N}\alpha_iy_i = \eta$$

`两边同时乘以`$y_1,由于y_i^2 = 1$
$$\alpha_1 = \eta y_1 - \alpha_2y_1y_2 \tag{4}$$



`令：`
$$v_1 = \sum^{N}_{i=3}\alpha_iy_iK_{i,1}, \quad v_2 = \sum^{N}_{i=3}\alpha_iy_iK_{i,2} \tag{5}$$
`将 (4), (5) 带入(3)`
$$ W(\alpha_2) = - \frac{1}{2}K_{1,1}(\eta - \alpha_2y_2)^2 - \frac{1}{2}K_{2,2}\alpha_2^2 - K_{1,2}y_2\alpha_2(\eta - \alpha_2y_2) - v_1(\eta - \alpha_2y_2) - v_2y_2\alpha_2 + \alpha_1 + \alpha_2 + C \tag{6}$$


`由于需要更新`$\alpha_2$`所以令`$\frac{dW}{d\alpha} = 0$
$$\frac{\partial W(\alpha_2)}{\partial \alpha_2} = -\alpha_2(K_{1,1} + K_{2,2} - 2K_{1,2}) + K_{1,1}\eta y_2 - K_{1,2}\eta y_2 + v_1y_2 - v_2y_2 - y_1y_2 + y_2^2 = 0 \tag{7}$$


<hr>


`对(7)式变形，使得`$\alpha_2^new$`能被`$\alpha_2^old$`表示(而不是用不方便的\eta)：`
`SVM预测值如下(该式子不需要保留所有的x，因为很多无关的x的alpha都为0，因此alpha需要初始化为0)：`
$$f(x) = \sum_{i=1}^{N}\alpha_iy_iK(x_i,x) + b \tag{8}$$

`则v可以表示为：`
$$ v_1 = \sum^{N}_{i=3}\alpha_iy_iK_{1,i} = f(x_1) - \alpha_1y_1K_{1,1} - \alpha_2y_2K_{1,2} - b $$

$$v_2 = \sum^{N}_{i=3}\alpha_iy_iK_{2,i} = f(x_2) - \alpha_1y_1K_{1,2} - \alpha_2y_2K_{2,2} - b $$

`已知：`
$$\alpha_1= (\eta - \alpha_2y_2)y_2$$
`可得到：`
$$v_1 - v_2 = f(x_1) - f(x_2) - K_{1,1}\eta + K_{1,2}\eta + \alpha_2y_2(K_{1,1}+K_{2,2}-2K_{1,2}) \tag{9}$$
`将(9)带入(7):`
$$\frac{\partial W(\alpha_2)}{\partial \alpha_2} = -(K_{1,1} + K_{2,2} - 2K_{1,2})\alpha_2^{new} + (K_{1,1} + K_{2,2}- 2K_{1,2})\alpha_2^{old} + y_2(y_2 - y_1 + f(x_1) - f(x_2))$$

<hr>


`记误差项`$E_i = f(x_i) - y_i$

`令`$\theta = K_{1,1,}+K_{2,2}-2K_{1,2}$
`可以得到最终表达式：`
$$\frac{\partial W(\alpha_2)}{\partial \alpha_2} = -\theta\alpha_2^{new}+ \theta\alpha_2^{old}+y2(E_1-E_2) = 0$$

`得到：`
$$\alpha_2^{new} = \alpha_2^{old}  + \frac{y_2(E_1- E_2)}{\theta} \tag{10}$$


**到了这里，如果要更新参数，仅需计算E，和\theta**
**然后计算\alpha2**

#### 原始解的修剪

**现在考虑约束条件**
**上面通过对一元函数求极值的方式更新了参数得到了**$\alpha_2^{new,unclipped}$ 
**现在通过对原始解的修正得到**$\alpha_2^{new,cilpped}$


$$\alpha_2^{new,unclipped} = \alpha_2^{old}  + \frac{y_2(E_1- E_2)}{\theta} \tag{10}$$


`约束条件（画图分析）：`

$if \quad y_1 \ne y_2 :$

$$ 上界：\qquad L = max(0, \quad \alpha_2^{old} - \alpha_1^{old})$$

$$ 下界：\qquad H = max(C, \quad C+\alpha_2^{old} - \alpha_1^{old})$$

$elif \quad y_1 = y_2 :$

$$ 上界：\qquad L = max(0, \quad \alpha_2^{old} + \alpha_1^{old} -C)$$

$$ 下界：\qquad H = max(C, \quad \alpha_2^{old} + \alpha_1^{old})$$


`更新参数都可以计算：`$\alpha_1^{new}$

`由：`
$$\alpha_1^{old}y_1 + \alpha_2^{old}y_2 = \alpha_1^{new}y_1 + \alpha_2^{new}y_2 $$

`得到：`

$$\alpha_1^{new} = \alpha_1^{old} + y_1y_2(\alpha_2^{old} - \alpha_2^{new})$$


`最后由式(6.17)->(6.18)可得到：`

$$b = \frac{1}{m}\sum_{s=1}^{m}[1/y_s - \sum_{m}^{i=1}\alpha_iy_ix_i^Tx_s]$$






#### 3.启发式选择变量
&nbsp;&nbsp;&nbsp;&nbsp;上述分析是在从N个变量中已经选出两个变量进行优化的方法，下面分析如何高效地选择两个变量进行优化，使得目标函数下降的最快。

##### 第一个变量的选择
&nbsp;&nbsp;&nbsp;&nbsp;第一个变量的选择称为外循环，首先遍历整个样本集，选择违反KKT条件的$\alpha_i$作为第一个变量
&nbsp;&nbsp;&nbsp;&nbsp;接着依据相关规则选择第二个变量(见下面分析),对这两个变量采用上述方法进行优化。
&nbsp;&nbsp;&nbsp;&nbsp;当遍历完整个样本集后，遍历非边界样本$(0<α_i<C)$中违反KKT的$\alpha_i$作为第一个变量，同样依据相关规则选择第二个变量，对此两个变量进行优化。当遍历完非边界样本集后，再次回到遍历整个样本集中寻找，即在整个样本集与非边界样本集上来回切换，寻找违反KKT条件的αiαi作为第一个变量。直到遍历整个样本集后，没有违反KKT条件$\alpha_i$，然后退出。
&nbsp;&nbsp;&nbsp;&nbsp;边界上的样本对应的$\alpha_i = 0$或者$\alpha_i = C$，在优化过程中很难变化，然而非边界样本$0<α_i<C$会随着对其他变量的优化会有大的变化。

<hr>
<center>KTT条件</center>

$$\alpha_i = 0 \Longleftrightarrow y^{(i)}(w^Tx^{(i)}+b)\ge1 $$
$$\alpha_i = C \Longleftrightarrow y^{(i)}(w^Tx^{(i)}+b)\le 1 $$
$$0 \lt \alpha_i \lt C \Longleftrightarrow y^{(i)}(w^Tx^{(i)}+b) =1 $$


<hr>


##### 第二个变量的选择
&nbsp;&nbsp;&nbsp;&nbsp;SMO称第二个变量的选择过程为内循环，假设在外循环中找个第一个变量记为$\alpha_1$，第二个变量的选择希望能使$\alpha_2$有较大的变化，由于$\alpha_2$是依赖于$|E1−E2|$,当$E1$为正时，那么选择最小的$Ei$作为$E2$,如果$E1$为负，选择最大$Ei$作为$E2$，通常为每个样本的$Ei$保存在一个列表中，选择最大的$|E1−E2|$来近似最大化步长。
&nbsp;&nbsp;&nbsp;&nbsp;有时按照上述的启发式选择第二个变量，不能够使得函数值有足够的下降，这时按下述步骤:

    1.首先在非边界集上选择能够使函数值足够下降的样本作为第二个变量，
    2.如果非边界集上没有，则在整个样本集上选择第二个变量，
    3.如果整个样本集依然不存在，则重新选择第一个变量。

**参考**


[机器学习算法实践-SVM中的SMO算法](https://zhuanlan.zhihu.com/p/29212107)

*详细推导的SMO更新参数和修剪结果的公式， 优化算法效率：例如只需最后计算b。。。。*

[浅谈最优化问题的KKT条件](https://zhuanlan.zhihu.com/p/26514613)



```python
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report, roc_auc_score
np.random.seed(1)
```

### 定义训练集

$$X_{train} \in \R^{n \times m}$$

$$x_i \in \R^n,即每一个样本有n个特征$$

$$label \in \{-1,1\},二分类问题$$


```python

```

## 多种核函数


```python

def LinearKernel(x_i, x_j):
    if x_i.shape[1] != 1:
        raise Exception("x_i is wrong shape!")
    if x_j.shape[1] != 1:
        raise Exception("x_j is wrong shape!")
    return np.dot(x_i.T, x_j)


def GaussKernel(x_i, x_j, sigmoid):
    if x_i.shape[1] != 1:
        raise Exception("x_i is wrong shape!")
    if x_j.shape[1] != 1:
        raise Exception("x_j is wrong shape!")
    if sigmoid <= 0:
        raise Exception("sigmoid must be a positive number")
    return np.exp(-np.dot((x_i - x_j).T, (x_i - x_j))/(2 * sigmoid**2))
def PolyKernel(x_i, x_j, d):
    if x_i.shape[1] != 1:
        raise Exception("x_i is wrong shape!")
    if x_j.shape[1] != 1:
        raise Exception("x_j is wrong shape!")
    if d < 0:
        raise Exception("d must be a Semi-positive number")
    return LinearKernel(x_i, x_j)**d


def LaplaceKernel(x_i, x_j, s):
    if x_i.shape[1] != 1:
        raise Exception("x_i is wrong shape!")
    if x_j.shape[1] != 1:
        raise Exception("x_j is wrong shape!")
    if s <= 0:
        raise Exception("sigmoid must be a positive number")
    # print(np.exp(-np.sqrt(np.dot((x_i - x_j).T, (x_i - x_j))) / sigmoid))
    return np.exp(-np.sqrt(np.dot((x_i - x_j).T, (x_i - x_j))) / s)[0][0]

def SigmoidKernel(x_i, x_j, beta, theta):
    if x_i.shape[1] != 1:
        raise Exception("x_i is wrong shape!")
    if x_j.shape[1] != 1:
        raise Exception("x_j is wrong shape!")
    if theta >= 0:
        raise Exception("theta must be a negetive number")
    if beta <= 0:
        raise Exception("theta must be a positive number")
    return np.tanh(beta * LinearKernel(x_i, x_j) + theta)

def K(x_i, x_j, kernel = "linear", s = .5, beta = 1, theta = -1, d = 2):
    # print(x_i.shape, x_j.shape)
    kernel = kernel.lower()
    try:
        if kernel == 'linear':
            return LinearKernel(x_i, x_j)
        elif kernel == 'gauss':
            return GaussKernel(x_i, x_j, s)
        elif kernel == 'poly':
            return PolyKernel(x_i, x_j, d)
        elif kernel == 'laplace':
            return LaplaceKernel(x_i, x_j, s)
        elif kernel == 'sigmoid':
            return SigmoidKernel(x_i, x_j, beta, theta)
    except Exception as err:
        print('An exception happened: ' + str(err))

if __name__ == "__main__":
    x_i = np.ones((10,1))
    x_j = np.random.randn(10,1)
    print(np.exp(-np.sqrt(np.dot((x_i - x_j).T, (x_i - x_j))) / 1))
    print(K(x_i,x_j,kernel="Laplace", s=1))
    
```

    [[0.005973]]
    0.005973003723489288


### 生成同心圆数据


```python
import numpy as np
def load(type = "circle"):
    
    x = np.random.uniform(-1,1,(100,1))
    y = np.random.uniform(-1,1,(100,1))
    x_train = np.c_[x,y].T
    print(x_train.shape) 
    label_train = np.ones((100,1))
    mask = (x**2 + y**2 ) < .5
    label_train[mask] = -1
    
    
    x1 = np.random.uniform(-1,1,(100,1))
    y1 = np.random.uniform(-1,1,(100,1))
    x_test = np.c_[x1,y1].T
    print(x_test.shape) 
    label_test = np.ones((100,1))
    mask = (x1**2 + y1**2 ) < .5
    label_test[mask] = -1
    
    return x_train, label_train, x_test, label_test

if __name__ == "__main__":
    x_train, label_train, x_test, label_test = load()
    print(x_train.shape, label_train.shape, x_test.shape, label_test.shape)

```

    (2, 100)
    (2, 100)
    (2, 100) (100, 1) (2, 100) (100, 1)


### 参照第一个SMO推导笔记需要实现以下功能：
* 初始化参数（主要是\alpha）
* 由于采用缓存（用一个列表记录）所有的$E_i = y_i - f(x_i)$
	* 所以每一次更新参数$\alpha$后都需要更新$E_i, E_j$
* 因为由KKT条件约束，所以需要对最后的结果进行修剪 所以需要实现 clip()
* $f(index_i)用于实现对训练集的某一个特征向量计算预测值$
* $predict(x_i)用于预测任意给定的特征向量x_i$
* select()函数的内循环和外循环用于选择最不满足KKT条件的并且能够使得更新效果最明显的$\alpha_i \quad \alpha_j$
* loss()用于计算E
* update_a2()用于更新$\alpha_2$
* update_a1()使用$\alpha_2$和$\alpha_1^{old}$更新$\alpha_1$
* SMO()是smo算法的主循环


**以上函数与SMO算法紧密相关，为了方便参数，数据集，的传输将他们放到一个SMO类中实现**


```python
class SMO:
    def __init__(self, x, y, kernel = "Laplace", C=10):
        self.x = copy.deepcopy(x)
        self.y = copy.deepcopy(y)
        self.m = x.shape[1]
        self.n = x.shape[0]
        self.kernel = kernel
        self.C = C
    
    def init(self):
        # self.w = np.random.randn(self.n, 1)
        self.b = np.zeros((1, 1))
        self.a = np.zeros((self.m, 1))
        E = []
        for i in range(0, self.m):
            E.append(self.loss(i))
        self.E = E


    # 每次更新参数后需要更新E
    def updateE(self, i, j):
        self.E[i] = self.loss(i)
        self.E[j] = self.loss(j)

 
    def clip(self, index_1, index_2, old_1, old_2):
        # get H, L
        alpha = self.a[index_2,:][0]
        if self.y[index_1, :] != self.y[index_2, :]:
            L = max(0.0, old_2[0] - old_1[0])
            H = max(self.C, self.C + old_2[0] - old_1[0])
        else:
            # print("----------" , self.C, old_2[0], self.C + old_2[0],'-------------')
            L = max(0.0, old_2[0] + old_1[0] - self.C)
            H = max(self.C, old_2[0] + old_1[0])
        
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha 

    def f(self, x_index):
        k = np.zeros(self.y.shape)
        for i in range(0, self.m):
            k[i,:] = copy.deepcopy(K(self.x[:,i].reshape(-1,1), self.x[:,x_index].reshape(-1,1), kernel = self.kernel))
        return np.sum(self.a * self.y * k)+self.b
    
    # 此处是选择第二个变量，第一个变量仅需要无脑便利就好了
    # 需要传入第一个变量的index以计算E_1
    # 是否可以将select写成一个生成器？
    def select(self):
        for i in range(0, self.m):
            # label用于标志是否违反KKT
            label = False

            temp = self.y[i, :] * self.f(i)
            if self.a[i, :] == 0:
                if temp < 1 :
                    label = True
            elif 0 < self.a[i, :] and self.a[i, :] < self.C:
                if temp > 1:
                    label = True
            else:
                if temp != 1:
                    label = True

            # 如果违反KKT条件，进入内循环选择第二个a
            if label == True and self.E[i] <= 0:
                j = np.argmax(np.array(self.E))
                yield i,j
            if label == True and self.E[i] >0:
                j = np.argmin(np.array(self.E))
                yield i,j
            


        # 需要得到边界变量的下标值
        boolen = (self.a > 0) & (self.a < self.C)
        edgeVar = []
        for i in range(0, len(boolen)):
            if boolen[i] == True:
                edgeVar.append(i)

        for i in edgeVar:
            # label用于标志是否违反KKT
            label = False

            temp = self.y[i, :] * self.f(i)
            if self.a[i, :] == 0:
                if temp < 1 :
                    label = True
            elif 0 < self.a[i, :] and self.a[i, :] < self.C:
                if temp > 1:
                    label = True
            else:
                if temp != 1:
                    label = True

            # 如果违反KKT条件，进入内循环选择第二个a
            if label == True and self.E[i] <= 0:
                j = np.argmax(np.array(self.E))
                yield i,j
            if label == True and self.E[i] >0:
                j = np.argmin(np.array(self.E))
                yield i,j
            


    # E_i = f(x_i) - y_i
    def loss(self, index):
        return self.f(index) - self.y[index, :]

    def update_a2(self, alpha2_old, index_1, index_2):
        theta = K(self.x[:,index_1].reshape(-1,1), self.x[:,index_1].reshape(-1,1), kernel = self.kernel) + K(self.x[:,index_2].reshape(-1,1), self.x[:,index_2].reshape(-1,1), kernel = self.kernel) - 2*K(self.x[:,index_1].reshape(-1,1), self.x[:,index_2].reshape(-1,1), kernel = self.kernel)
        if theta < 0.001 and  theta >= 0:
             theta = 0.001
        if theta > -0.001 and theta <0:
            theta = 0.001
            
        dE = self.E[index_1] - self.E[index_2]
        # print("in update_a2 theta:{}, dE:{}, y2:{}".format(theta, dE, self.y[index_2]))
        return alpha2_old + (self.y[index_2] * dE) / theta 

    def update_a1(self, alpha1_old, alpha2_old, alpha2_new, index_1, index_2):
        return alpha1_old + self.y[index_1] * self.y[index_2] * (alpha2_old - alpha2_new)

    def SMO(self, maxtimes):
        for i in range(0, maxtimes):
            if i % 10 == 0:
                print("this is {}th loop.".format(i))
            for i, j in self.select():
                temp = copy.deepcopy(self.a[j])
                self.a[j] = self.update_a2(self.a[j], i, j)

                self.a[j] = self.clip(i, j, self.a[i], temp)

                self.a[i] = self.update_a1(self.a[i], temp, self.a[j], i, j)

                self.updateE(i, j)
        # 更新b
        self.b = np.sum(1/self.y)/self.m
        for i in range(0, self.m):
            self.b -= (self.f(i)/self.m)
        print(self.b)    
            

    def predict(self, x):
        k = np.zeros(self.y.shape)
        for i in range(0, self.m):
            k[i,:] = copy.deepcopy(K(self.x[:,i].reshape(-1,1), x.reshape(-1,1), kernel = self.kernel))
        # return k
        return (np.sum(self.a * self.y * k) + self.b)
        
        
    def predictAll(self, x_test):
        # get prected y
        y_pred = np.zeros(y_test.shape)
        for i in range(0, x_test.shape[1]):
            y_pred[i, :] = self.predict(x_test[:, i].reshape(2, -1))
        return y_pred

```


```python
def evaluate(y_t, y_p):
    
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
    y_pred[(y_pred <  0)] = -1
    y_pred[(y_pred >= 0)] =  1
    # plt.scatter(x_test[0, :], x_test[1, :], c = y_pred[:, 0])
    print("The accuracy is {}".format(accuracy_score(y_test[:,0], y_pred[:,0])))
    
    # 计算召回， 查全率， 查准率 。。。。
    target_names = ['class0','class1']
    print(classification_report(y_test,y_pred,target_names = target_names))
```


```python
def draw_svm(smo):
    xx, yy = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
    zz = np.zeros((200, 200))
    for i in range(0, 200):
        for j in range(0, 200):
            temp = np.array([xx[i, j], yy[i, j]]).reshape(2, -1)
            res = smo.predict(temp)
            # print(res)
            zz[i, j] = res
            # print(zz[i, j])
    zz[(zz < 0)] = -1
    zz[(zz >= 0)] = 1
    plt.subplot(1,2,2)
    plt.contourf(xx, yy, zz)
    # # print(x[0,:].shape, x[1,:].shape, y.shape)
    plt.scatter(smo.x[0,:].reshape(1, -1), smo.x[1,:].reshape(1, -1), c = 0 - smo.y.T)
    plt.show()
```

### 导入数据
* 此处生成同心圆数据，以区别线性核和非线性核之间的区别


```python
x_train, y_train, x_test, y_test = load()
# plt.scatter(x_train[0, :], x_train[1, :], c = y_train[:, 0])

```

    (2, 100)
    (2, 100)


### 使用高斯核的SVM 
* 此处$\sigma = 0.5$,取值为1时，准确率就很低了，不知道怎么回事（先补一下核函数更多的技术了，  狗头.jpg）


```python
smo = SMO(x_train, y_train, C = 2, kernel = "gauss")
smo.init()
smo.SMO(110)
y_pred = smo.predictAll(x_test)
evaluate(y_test, y_pred)
draw_svm(smo)
```

    this is 0th loop.
    this is 10th loop.
    this is 20th loop.
    this is 30th loop.
    this is 40th loop.
    this is 50th loop.
    this is 60th loop.
    this is 70th loop.
    this is 80th loop.
    this is 90th loop.
    this is 100th loop.
    1.0542578147143928
    The roc_auc_score is 0.982048143614851
    The accuracy is 0.89
                  precision    recall  f1-score   support
    
          class0       0.80      1.00      0.89        43
          class1       1.00      0.81      0.89        57
    
        accuracy                           0.89       100
       macro avg       0.90      0.90      0.89       100
    weighted avg       0.91      0.89      0.89       100




![png](C:/Users/herrn/Downloads/183918/output_20_1.png)



```python

```

### 使用线性核的SVM


```python
smo = SMO(x_train, y_train, C = 2, kernel = "linear")
smo.init()
smo.SMO(110)
y_pred = smo.predictAll(x_test)
evaluate(y_test, y_pred)
draw_svm(smo)
```

    this is 0th loop.
    this is 10th loop.
    this is 20th loop.
    this is 30th loop.
    this is 40th loop.
    this is 50th loop.
    this is 60th loop.
    this is 70th loop.
    this is 80th loop.
    this is 90th loop.
    this is 100th loop.
    0.27041462716608683
    The roc_auc_score is 0.4618523051815585
    The accuracy is 0.48
                  precision    recall  f1-score   support
    
          class0       0.41      0.49      0.45        43
          class1       0.55      0.47      0.51        57
    
        accuracy                           0.48       100
       macro avg       0.48      0.48      0.48       100
    weighted avg       0.49      0.48      0.48       100




![png](C:/Users/herrn/Downloads/183918/output_23_1.png)


### 调用之前写过的决策树C4.5


```python
from dt import buildTree, Node, predictAll, predict, draw_tree
import pandas as pd
```


```python
X_train = np.r_[x_train, y_train.T]
X_train = pd.DataFrame(X_train.T, columns=["x",'y',"label"])
X_test = np.r_[x_test, y_test.T]
X_test = pd.DataFrame(X_test.T, columns=["x",'y',"label"])

Y_test = np.array(list(X_test["label"])).reshape(-1,1)
```


```python
root = Node(Dataset=X_train, attrList = ['x','y'])
root = buildTree(root)
```


```python
Y_pred = predictAll(root ,X_test)
evaluate(Y_test, Y_pred)
draw_tree(root)
plt.scatter(X_train["x"], X_train["y"], c = 0 - X_train["label"])
```

    The roc_auc_score is 0.7105263157894737
    The accuracy is 0.67
                  precision    recall  f1-score   support
    
          class0       0.57      1.00      0.72        43
          class1       1.00      0.42      0.59        57
    
        accuracy                           0.67       100
       macro avg       0.78      0.71      0.66       100
    weighted avg       0.81      0.67      0.65       100




![png](C:/Users/herrn/Downloads/183918/output_28_1.png)





    <matplotlib.collections.PathCollection at 0x7f523e323310>




![png](C:/Users/herrn/Downloads/183918/output_28_3.png)


### 分析
* 对于同心圆这种线性不可分的数据，使用linear kernel只能得到约50/100的准确度
* 使用决策树，这里由于决策变量只有x，y所以，这里的决策树虽多两层，对于特征向量较小的数据集不友好
* 使用拉普拉斯核（测试时用的拉普拉斯核，准确率略高于高斯核）   或者     高斯核在该种线性不可分的数据集中起到了很好的升维作用
* 使用神经网络可以得到很好的效果（在神经网络实现的时候，测试过该数据集，随着层数的增加，能达到99/100的准确率）


```python
import tensorflow==2.0.0


```

    Cannot run import tensorflow because of system compatibility. AI Studio prepared an entire environment based on PaddlePaddle already. Please use PaddlePaddle to build your own model or application.



```python

```


```python

```


```python

```


```python

```



