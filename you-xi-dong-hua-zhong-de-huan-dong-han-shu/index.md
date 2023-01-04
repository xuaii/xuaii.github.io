# 游戏动画中的缓动函数


## 情景引入
考虑以下情形，采集用户水平输入 `float HorizontalInput` 来控制 player 的水平位移
```c#
void Update(float delta)
{
    velocity.x = HorizontalInput * WalkSpeed * delta;
}
``` 
这样写有两个缺点：
1. 当影响水平速度的因素不只是水平输入时，不同因素的叠加不方便。不应该是 `=` 而应该是 `+=`;
2. 移动完全与输入一直，看起来很僵硬；

## 解决方案

所以需要有这么一个函数(缓动函数)，它对输入的响应由可控的延迟，但最终会与输入保持一致
// 添加示例函数1.1

为了使缓动曲线更好的模拟现实世界中由于力产生加速度的运动，这里使用一个二阶系统来描述：
$$
y + k_1\cdot y^{'} + k_2\cdot y^{''} = x + k_3 \cdot x^{'}
$$

等价的写法：
$$
y + k_1\cdot \frac{\mathrm{d} y}{\mathrm{d} t} + k_2\cdot \frac{\mathrm{d} ^2 y}{\mathrm{d}^2 t} = x + k_3 \cdot \frac{\mathrm{d} x}{\mathrm{d} t}
$$

通过调整 $k_1, k_2,k_3$ 可以改变图像的形状，
// 插入示例图像1.2
现在添加三个变量：
$$
f = \frac{1}{2\pi \sqrt{k_2}}, \quad
\theta = \frac{k_1}{2\sqrt{k_2}}, \quad
r = \frac{2k_3}{k_1}
$$
解方程组可得：
$$
k_1 = \frac{\theta}{\pi f}, \quad k_2 = \frac{1}{(2\pi f)^2}, \quad k_3 = \frac{r \theta}{2\pi f}
$$

原二阶系统的微分方程变为：
$$
y +\frac{\theta}{\pi f} \cdot y^{'} + \frac{1}{(2\pi f)^2} \cdot y^{''} = x + \frac{r \theta}{2\pi f} \cdot x^{'}
$$
 这里 $f, \theta, r$ 都具有现实意义了
 * $f$ 以 $hz$ 为单位，代表系统**固有频率**，它描述系统对输入变化的响应速度
 * $\theta$ 代表系统的**阻尼系数**，描述了系统如何最终趋于稳定
 * $r$ 控制系统的**初始响应**
当 $r = 0$ 时系统需要花费一点时间才能从禁止开始加速
当 $r > 0$ 时系统立刻对变化做出反应
当 $r > 1$ 时系统变化将冲过目标
当 $r < 0$ 时系统会有抬手运动（先反向运动）
一般为机械链接设置 $r = 2$

现在只剩下最后一个问题了，如何解二阶系统？

## 二阶微分方程数值解
### 1. 半隐式欧拉法
该方法在该问题下与复杂的 **Verlet 积分法** 有着相同的精度，
首先计算 x 变化率
$$
x^{'}_{n+1} = \frac{x_{n+1} - x_n}{T} \\
$$
然后计算 y 的变化
$$
\begin{cases}
y_{n + 1} = y_{n} + Ty^{'}_{n} \\
\quad\\
y^{'}_{n+1} = y^{'}_{n} + Ty^{''} \\
 \end{cases} 
$$
由于:
$$
y^{''} = \frac{x + k_3 x^{'} - y - k_1y^{'}}{k_2}
$$
所以：
$$
y^{'}_{n+1} = y^{'}_{n} + T\cdot  \frac{x_{n+1} + k_3 x^{'}_{n+1} - y_{n+1} - k_1y^{'}_{n}}{k_2}
$$

这存在一个问题，如果频率 $f$ 远大于帧率，系统将变得不稳定，会产生无穷大的值：
1. 可以简单地设置 f 的取值范围
2. 通过数学方法确保不发生极端情况

为了提供更多的鲁棒性，采用方案二
**分析**
不稳定产生的原因，该系统的本质是**反馈系统**，他的迭代输出将被反馈到后续的迭代用于计算，当**帧间时间步长**和**参数**相比太大时，随着时间增加，误差将逐渐累积，当超过某个**临界值**时，误差会开始滚雪球，迅速导致灾难性后果，为了计算该临界值，引入线性代数方法：
$$
\begin{cases}
y_{n + 1} = y_{n} + Ty^{'}_{n} \\
\quad \\
y^{'}_{n+1} = y^{'}_{n} + T\cdot  \frac{x_{n+1} + k_3 x^{'}_{n+1} -(y_{n} + Ty^{'}_{n}) - k_1y^{'}_{n}}{k_2}
 \end{cases} 
$$
展开合并同类项：

$$
\begin{cases} 
y_{n + 1} = y_{n} + Ty^{'}_{n} \\
\quad \\
y^{'}_{n+1} = \frac{-T}{k_2} y_n + \frac{k_2 - T^2 - Tk_1}{k_2}y^{'}_{n} + \frac{T}{k_2}x_{n+1} + \frac{Tk_3}{k_2}x^{'}_{n+1}\\
 \end{cases} 
$$

矩阵表示如下：
$$
\begin{bmatrix}
y\\
y^{'}
\end{bmatrix}_{n+1} 

= 

\begin{bmatrix}
1 & T\\
-\frac{T}{k_2} & \frac{k_2 - T^2 - Tk_1}{k_2}
\end{bmatrix} 

\cdot 

\begin{bmatrix}
y\\
y^{'}
\end{bmatrix}_{n}

+ 

\begin{bmatrix}
0 & 0\\
\frac{T}{k_2} & \frac{Tk_3}{k_2}
\end{bmatrix} 

\cdot

\begin{bmatrix}
x\\
x^{'}
\end{bmatrix}_{n+1}
$$
简写如下
$$
Y_{n+1} = A\cdot Y_n + B\cdot X_{n+1}
$$
分析：
$A$ 称为状态转移矩阵，他表示迭代是如何影响状态变量的，直观来说如果 $A$ 矩阵不导致状态变量 $Y$ 的增长，那该反馈是稳定的；考虑$A$ 作为值而非变量，利用特征值理论，记 $A$ 的特征值$\lambda$
如果 $\lambda_i < 1$, Y 将逐渐减小趋于稳定
如果 $\lambda_i > 1$, Y 迅速增大，很快变得无法控制

计算特征值如下：
$$
det(A - \lambda I) = 0
$$
展开后
$$
k_2 \lambda^2 + (T^2 + Tk_1 - 2k_2)\lambda (k_2-Tk_1) = 0
$$
解关于 $\lambda$ 的二次方程
$$
\lambda = \frac{ -b \pm \sqrt{b^2-4ac}}{2a}
$$

令 $|\lambda| < 1$:
得:
$$
T < \sqrt{4k_2 + k_1^2} - k_1
$$
 这里 如果时间步长大于临界值, 将拆分为多个了迭代来计算
 ```c#
 public classs SecondOrderDynamics
 {
     private float T_crit; // critical stable time step
     public SecondOrderDynamics(float f, floatz, float r, Vector x0)
     {
         // update compute constants
         k1 = z / (PI * f);
         k2 = 1 / ((2 * PI * f) * (2 * PI * f));
         k3 = r * z / (2 * PI * f);
         T_crit = 0.8f * (sqtr(4 * k2 + k1 * k1));
         xp = x0;
         y = x0;
         yd = 0;
     }
     public Vector Update(float delta, Vector x, Vector xd = null)
     {
         if(xd == null)
         {
             xd = (x - xp) / delta;
             xp = x;
         }
         int interations = (int)Ceil(delta / T_crit); // take extra iterations if delta > T_crit
         delta = delta / iterations;
         for(int i = 0; i < iterations; i++)
         {
             y = y + delta * yd;
             yd = yd + delta * (x + k3 * xd - y - k1*yd) / k2;
         }
         return y;
     }
 }
 ```
 此外，如果想避免迭代次数过多，可以限制 $k_2$ 的值（减缓运动）
 $$
 k_2 > \frac{T^2}{4} + \frac{Tk_1}{2}
 $$
 ```c#
 float k2_stable = Max(k2, 1.1f * (T*T/4 + T*k1/2));
 ```
### 2.零极点匹配法（高精度）没懂
```c#
public classs SecondOrderDynamics
{
    private Vector xp;
    private Vector y, yd;
    private float _w, _z, _d, k1, k2, k3;
    public SecondOrderDynamics(float f, float z, float r, Vetor x0)
    {
        _w = 2 * PI * f;
        _z = z;
        _d = _w * sqrt(Abs(z*2-1));
        k1 = z / (PI * f);
        k2 = 1 / (_w * _w);
        k3 = r * z / _w;
        xp = x0;
        y = x0;
        yd = 0;
    }
    public Vector Update(float delta, Vector x, Vector xd = null)
    {
        if(xd == null)
        {
            float k1_stable, k2_stable;
            if(_w * T < _z)
            {
                k1_stable = k1;
                k2_stable = Max(k2, T * T / 2+ T * k1 / 2, T*k1);
            }
            else
            {
                float t1 = Exp(-_z * _w * T);
                float alpha = 2 * t1 * (_z <= 1 ? cos(T * _d) : cosh(T * _d));
                float beta = t1 * t1;
                float t2 = T / (1 + beta -alpha);
                k1_stable = (1 - beta) * t2;
                k2_stable = T * t2;
            }
            y = y + T * yd;
            yd = yd + T * (x + k3*xd -y - k1*yd) / k2_stable;
            return y;
        }
    }
}
```

### 3. 其他数值解方法（略）

参考:
t3ssel8r:[Giving Personality to Procedural Animations using Math](https://www.youtube.com/watch?v=KPoeNZZ6H4s)



