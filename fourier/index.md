# 傅里叶变换


### 1. Dirichlet Conditions

* 一个周期内, **连续**或者有**有限个第一类间断点**
* 一个周期内, 有限个极值点
* 一个周期内可积

### 2. Trangle Transform

1. 假设
$$ 
f(t) = c_0+\sum_{n=1}^{\infty}c_n\cos(n \omega t + \varphi)
=c_0+\sum_{n=1}^{\infty}[c_n\cos\varphi\cos(n\omega t)- c_n\sin\varphi \sin(n\omega t)]
$$
   
2. 令 $a_n = c_n \cos\varphi,\qquad b_n = -c_n\sin\varphi$
$$
f(t) = c_0+\sum_{n=1}^{\infty}[a_n\cos(n\omega t) + b_n \sin(n\omega t)]\\
\int_0^T f(t)\sin(k\omega t) dt= \int_0^Tc_0\sin(k\omega t) dt + \int_0^T\sin(k\omega t)\sum_{n=1}^{\infty}[a_n\cos(n\omega t) + b_n \sin(n\omega t)] dt\\
\int_0^T f(t)\sin(n\omega t) dt= 0 + b_n \frac{T}{2}\\
b_n = \frac{2}{T} \int_0^T f(t)\sin(n\omega t) dt
$$
   
3. 与 2 相似的可以计算得到 $c_n, a_n, \varphi, c_0$

### 3. Fourier 级数

1. 根据欧拉公式:
$$
e^{jx} = \cos x + j\sin x\\
\cos(n\omega t) = \frac{e^{jn\omega t} + e^{-jn\omega t}}{2}\\
\sin(n\omega t) = \frac{e^{jn\omega t} - e^{-jn\omega t}}{2j}\\
$$
   
2. (8) (9)代入(3)
$$
f(t) = c_0+\sum_{n=1}^{\infty}[a_n \frac{e^{jn\omega t} + e^{-jn\omega t}}{2} + b_n \frac{e^{jn\omega t} - e^{-jn\omega t}}{2j}]\\
$$

3. 由于 :
$$
a_n = \frac{2}{T} \int_0^T f(t)\cos(-n\omega t)dt = a_{-n}\\
同理:-b_n = b_{-n}
$$
   
4. 整理得到:
$$
f(t) = c_0 + \sum_{n=1}^{\infty}[\frac{a_n-jb_n}{2}e^{jn\omega t} + \ \frac{a_{-n} - jb_{-n}}{2}e^{-jn\omega t}]\\
f(t) = c_0 + \sum_{n=1}^{\infty}\frac{a_n-jb_n}{2}e^{jn\omega t} + \sum_{n = -\infty}^{-1} \frac{a_n - jb_n}{2}e^{jn\omega t}\\
合并得到:\\
\qquad\qquad f(t) = \sum_{n=-\infty}^{\infty}\frac{a_n - jb_n}{2}e^{jn\omega t}\\
令:\\
\qquad\qquad A_n = \frac{a_n - jb_n}{2}\\
f(t) = \sum_{n=-\infty}^{\infty}A_ne^{jn\omega t}\\
$$

5. 在 4. 中得到 Fourier 级数, 再两边同时 乘以 $e^{-jk\omega t}$  并在一个周期内积分得到:
$$
\int_{0}^{T}f(t)e^{-jn\omega t}dt = \int_{0}^{T}\sum^{+\infty}_{n = -\infty}A_ne^{j(n-k)\omega t}dt\\
\int_0^Tf(t)e^{-jn\omega t}dt = A_nT\\
A_n = \frac{1}{T}\int_0^Tf(t)e^{-jn\omega t}dt
$$
   

### 4. Fourier Transform

1. 1
$$
A_n = \frac{1}{T}\int_0^Tf(t)e^{-jn\omega t}dt\\
F(\omega) = \lim_{T\rightarrow \infty}A_nT = \int_0^\infty f(t)e^{-jn\omega t}dt\qquad  (Fourier Transform + )\\
\lim_{T\rightarrow \infty}A_n = \lim_{T\rightarrow \infty}\frac{F(\omega)}{T} = \lim_{T\rightarrow \infty}\frac{F(\omega)\cdot \omega}{2\pi}\\
$$
   
2. 结合 (19)(25) 得到
$$
f(t) = \lim_{T\rightarrow \infty}\sum^\infty_{n=-\infty}A_n e^{jn\omega t}
=\lim_{T\rightarrow \infty}\sum^\infty_{n=-\infty}\frac{F(\omega)\cdot e^{jn\omega t}}{2\pi} d\omega
=\frac{1}{2\pi}\int^\infty_{-\infty}F(\omega)\cdot e^{jn\omega t}d\omega \qquad  (Fourier Transform -)\\
$$
   

### 5. Convolution Theorem

1. 定理描述
$$
设:f_1(t) 的Fourier变换为F_1(\omega), f_2(t) 的Fourier变换为F_2(\omega), \\
那么:\\
时域:\\
\qquad\qquad F[f_1(t)\otimes f_2(t)] = F_1(\omega)\cdot F_2(\omega)\\
频域:\\
\qquad\qquad F[f_1(t)\cdot f_2(t)] = \frac{1}{2\pi}F_1(\omega)\otimes F_2(\omega)\\
$$
   



### 6. Fourier 时移性质

$$
F[f(t)] = F(\omega) \qquad则: F[f(t - \tau)] = F(\omega)e^{-jn\omega \tau}
$$

**证明:**
$$
F[f(t - \tau)] = \int^{+\infty}_{-\infty}f(t-\tau)e^{-jn\omega t}dt\\
$$
**令** $x= t - \tau:$
$$
F[f(t - \tau)] = \int^{+\infty}_{-\infty}f(x)e^{-jn\omega (x+\tau)}dx
=e^{jn\omega\tau}\int^{+\infty}_{-\infty}f(x)e^{-jn\omega x}dx
= F(\omega)\cdot e^{jn\omega\tau}
$$


### 7. Convolution Theorem 证明

1. 定义卷积运算
$$
f_1(t)\otimes f_1(t) = \int_{-\infty}^{+\infty}f_1(\tau)f_2(t - \tau)d\tau\\
$$

2. 将 (35) 带入(24)
$$
F[f_1(t)\otimes f_1(t)] = \int_{-\infty}^{+\infty}[\int_{-\infty}^{+\infty}f_1(\tau)f_2(t - \tau)d\tau]e^{-jn\omega t}dt\\
= \int_{-\infty}^{+\infty}f_1(\tau)[\int_{-\infty}^{+\infty}f_2(t - \tau)e^{-jn\omega t}dt]d\tau \qquad(调换积分顺序)\\
= \int_{-\infty}^{+\infty}f_1(\tau)F_2(\omega)e^{-jn\omega \tau}d\tau\\
= F_2(\omega)\cdot F_1(\omega)\\
$$
