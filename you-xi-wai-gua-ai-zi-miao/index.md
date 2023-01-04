# [游戏外挂] 辅助瞄准系统


**参考列表**
[Python embedding](http://study.yali.edu.cn/pythonhelp/extending/embedding.html)

[c++多线程调用python](http://t.zoukankan.com/watermoon-p-4367528.html)

[yolov7-ncnn](https://github.com/Baiyuetribe/ncnn-models/tree/main/object_dection/yolov7)

[Yolo-FastestV2](https://github.com/dog-qiuqiu/Yolo-FastestV2)

[ncnn教程](https://blog.csdn.net/u012483097/article/details/109069388)

[ncnn 环境](https://zhuanlan.zhihu.com/p/391609325)

[opencv之Mat格式数据转换成onnxruntime的输入tensor处理的c++写法](https://blog.csdn.net/znsoft/article/details/117128781)

[Tracker - Norfair](https://tryolabs.github.io/norfair/reference/tracker/#norfair.tracker.Tracker)

[FPS游戏的鼠标灵敏度换算方法](https://www.bilibili.com/read/cv11146846)

[phoboslab/jsmpeg-vnc](https://github.com/phoboslab/jsmpeg-vnc)

[桌面复制 API - Win32 apps | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/win32/direct3ddxgi/desktop-dup-api)[桌面复制 API - Win32 apps | Microsoft Learn](https://learn.microsoft.com/zh-cn/windows/win32/direct3ddxgi/desktop-dup-api)

最近看到有些 FPS 游戏主播被锤外挂，不禁有些感叹 人菜逼事多。所以产生了研究AI自瞄原理的想法。首先做AI自瞄不是想开挂，而是对于反外挂而言 只有了解Hacker才能Anti-Hacker。

## 总体思路

传统的 FPS 游戏外挂需要**读内存**、**改内存**，因此很容易被检测。而 AI 自瞄只读取游戏画面(可以通过**屏幕捕获**、**外置摄像头**)，通过**目标检测方法** 准确的识别出游戏中的目标的类型和位置，然后通过各种**移动准星**的方法将准星移动到目标位置，AI 自瞄系统应该仅提供基础设施（识别系统、脚本系统、配置系统），玩家可以自定义视频输入方式，通过脚本系统定义 "如何瞄准敌人"（添加抖动等），定义识别系统的参数。

// 为了高效运行，外挂的运行时使 C/C++，脚本系统使用 Python / C#

但是，将屏幕捕获交给玩家会加重识别系统的的任务（识别敌人 + 识别屏幕），再加上坐标变换后降低了准确度，为了使识别准确度尽可能提高，将采用捕获屏幕作为输入。

github [项目地址](https://github.com/xuaii/apex-auto-aim)

## **读取游戏画面**

[windows 屏幕抓取技术总结](https://blog.csdn.net/tuan8888888/article/details/120761111) 总结了windows 下各个平台的屏幕抓取性能开销对比，但是和我实测的数据有些偏差，在我实测中 DXGI 几乎不消耗 CPU/GPU 资源，GDI方案消耗大量的 CPU 资源。但是在启动了 APEX 的情况下 DXGI 仅能达到 90fps， GDI 能达到 200fps，所以我们的方案采用 GDI 方法，GDI 抓屏参考(几乎是copy)了 [phoboslab/jsmpeg-vnc: A low latency, high framerate screen sharing server for Windows and client for browsers](https://github.com/phoboslab/jsmpeg-vnc) 的grabber方法，并且取得很高的效率。

```c
#ifndef GRABBER_H
#define GRABBER_H

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

typedef struct {
    int x, y, width, height;
} grabber_crop_area_t;

typedef struct {
    HWND window;

    HDC windowDC;
    HDC memoryDC;
    HBITMAP bitmap;
    BITMAPINFOHEADER bitmapInfo;

    int width;
    int height;

    void *pixels;
    grabber_crop_area_t crop;
} grabber_t;

grabber_t *grabber_create(HWND window, grabber_crop_area_t crop);
void grabber_destroy(grabber_t *self);
void *grabber_grab(grabber_t *self);

#endif
```

该实现需要先获取 windows 窗口句柄，然后初始化 grabber 

```c++
#include <Windows.h> 
...
HWND handle = FindWindow(NULL, TEXT("Apex Legends"));
if(!handle) return -1;
grabber_crop_area_t crop { 0, 0, 0, 0 };
grabber_t * grabber = grabber_create(handle, crop);
...
while(1)
{
    void* data = grabber_grab(grabber);
    ...
}
```

这里踩的坑是 void* 是一个字节数组，也就是 uchar* 数组而在  grabber.c 中设置了 

```c++
bitmapInfo.biBitCount = 32;
bitmapInfo.biCompression = BI_RGB;
```

 我们不需要 A 通道数据，所以设置 

```c++
bitmapInfo.biBitCount = 24
```

这意味着要将 grabber 捕获到的数据是 RGB 格式的用 opencv 读取需要格式 CV_8U3C:

```c++
void* pixel = grabber_grab(grabber);
// 8U3C -> 3 通道每通道 8 位
cv::Mat frame = cv::Mat(cv::Size{ grabber->width, grabber->height }, CV_8UC3);
frame.data = (uchar*)pixel;
```

由于opencv 的格式是 BGR，神经网络输入格式是 RGB；将 cv::Mat 转为 ncnn::Mat 时需要进行格式转换：

```c++
ncnn::Mat in = ncnn::Mat::from_pixels_resize(croped.data, ncnn::Mat::PIXEL_BGR2RGB, croped.cols, croped.rows, target_size, target_size);
```

这里可以直接从 void* 转换到 ncnn::Mat , 为了测试方便还是先转换成 cv::Mat 方便使用 cv::imshow() 显示到窗口。

## **识别目标**

**识别目标** 是最重要的一环，也是踩坑最多的一环。神经网络**训练时框架**太笨重，不适合用于软件嵌入，现有的推理框架：

1. NCNN：腾讯的产品，号称 "**0 依赖**"，运行时确实不需要 .dll 开发时需要 **protobuf**，**vulkan**。在移动平台等边缘设备优化是最好的，很适合用于 嵌入应用.

2. MNN、MACE、TF-lite、Paddle-lite 这些都类似 NCNN 主打移动端推理

3. TensorRT：Nvidia 的框架，一般来说 Nvidia 下用该框架时最快的依赖 **cuda**,**cudann**
   
   [TensorRT-For-YOLO-Series/main.cpp at main · Linaom1214/TensorRT-For-YOLO-Series (github.com)](https://github.com/Linaom1214/TensorRT-For-YOLO-Series/blob/main/cpp/end2end/main.cpp)

4. OpenVINO：Intel家的，缺点很明显不支持 AMD CPU

对于 AI 自瞄来说最好的选择是 NCNN 或者 TensorRT（但是开始这个项目之前没怎么了解过，先后使用了libtorch、OnnxRuntime、NCNN、TensorRT）

为了方便后续添加功能，例如移动端的实现，采用 **Tensorrt** 和 **NCNN** 两种实现。

模型使用 最新的 Yolov7， 准确率高，速度快，在不启动游戏的情况下 RTX2060 可以有120fps 的推理速度。

[GitHub - WongKinYiu/yolov7: Implementation of paper - YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://github.com/WongKinYiu/yolov7) 给出了模型训练和模型导出的方法，NCNN/TensorRT 的使用可以在 Github 找到大量的Demo，拿到Demo搞清楚输入输出即可。

这一环最麻烦的是数据格式的转换

* BGR -> RGB : 可以由 `cv::Mat.convertTo`, `ncnn::Mat.resize` 等方法

* NHWC -> NCHW: `cv::dnn::blobFromImage`, `transposeND`, 三层  for 循环硬转，tensorflow/pytorch 的轴旋转方法
  
  

## **调用track脚本**

这里使用 Python 脚本来处理神经网络的输出和鼠标移动，由于不同的用户需求不同，例如 鼠标平滑，定位精度，拉枪速度，鼠标吸附等，process 方法将在每一个推理帧的最后被调用。

```python
def process(targets) -> (float, float):
    for target in target:
        print(target)
    return (1.1, 2.0)
```

脚本需要和c++交互，process 方法需要被c++调用并捕获输出值，c++ 调用python需要include Python.h 头文件，需要添加 <python_dir>/libs 库文件， 最后程序打包时需要将 python39.dll 添加到 .exe 文件目录，可以使用更小的 embed 版本来将 python 嵌入到应用（只占用10mb）

```c++
#define PY_SSIZE_T_CLEAN
#include <Python.h>
int main() 
{ 
    Py_SetPythonHome(std::wstring(config.pythonHome.begin(), config.pythonHome.end()).c_str());

Py_Initialize();//初始化python 

    PyObject *pModule = NULL, *pFunc = NULL, *pArg = NULL; 

    pModule = PyImport_ImportModule("core");//引入模块 

    pFunc = PyObject_GetAttrString(pModule, "process");//直接获取模块中的函数 

    PyObject* list = PyList_New(0);
    Py_INCREF(list);

    for (auto& obj : boxes)
    {
        PyList_Append(list, Py_BuildValue("(f,f,f,f,f)", (obj.x1 + obj.x2) / 2, (obj.y1 + obj.y2) / 2, obj.x2 - obj.x1, obj.y2 - obj.y1, obj.score));
    }
    PyDict_SetItemString(dict, "target_list", list);
    Py_DECREF(list);
    PyDict_SetItemString(dict, "mouse_left_button", Py_BuildValue("b", KEY_DOWN(VK_LBUTTON)));
    PyDict_SetItemString(dict, "mouse_middle_button", Py_BuildValue("b", KEY_DOWN(VK_MBUTTON)));
    PyDict_SetItemString(dict, "mouse_right_button", Py_BuildValue("b", KEY_DOWN(VK_RBUTTON)));
    PyDict_SetItemString(dict, "mouse_ctrl_button", Py_BuildValue("b", KEY_DOWN(VK_CONTROL)));

    PyObject* args = PyTuple_New(1);
    PyTuple_SetItem(args, 0, dict);

    PyObject* pRet = PyObject_CallObject(pFunc, args);
    if (!pRet) return;
    if (PyErr_Occurred())
    {
        PyErr_Print();
    }

    Py_Finalize(); //释放python 

    return 0; 
} 
```

python 端需要获取非激活窗口的键盘或者鼠标状态需要使用pyhook， 但是在 c++ 可以很简单的获得：

```c++
#define KEY_DOWN(VK_NONAME) ((GetAsyncKeyState(VK_NONAME) & 0x8000)


KEY_DOWN(VK_LBUTTON);
KEY_DOWN(VK_MBUTTON);
KEY_DOWN(VK_RBUTTON);
KEY_DOWN(VK_CONTROL);
```

并通过 c++ 传递给 python

## **更新准星位置**

参考了这位老哥的鼠标定位方法 [FPS游戏（AI自瞄原理） - 哔哩哔哩 (bilibili.com)](https://www.bilibili.com/read/cv17317767?from=search&spm_id_from=333.337.0.0)

但是这里给出的计算方式是有偏差的, 但是利用 **角度替代像素** 是正确的，游戏引擎中相机的角度是使用[欧拉角](https://baike.baidu.com/item/%E6%AC%A7%E6%8B%89%E8%A7%92/1626212?fr=aladdin)来计算的, 一般第一人称游戏使用鼠标的 **水平输出**(HorizontalInput) 和

**竖直输出**(VerticalInput) 来控制摄像机转动，这一位置，所以对于不同的游戏 XInput 的大小和相机转动的欧拉角可能不一样，也就是说我们定义一个单位:

$$
像素角(fa) = 转动一弧度移动的像素(pixel/rad)
$$

像素角的计算和测量

$$
fa = \frac{游戏水平转动一周需要的像素\times 2 \times \pi}{360} \times 游戏内灵敏度  \times ADS
$$

所以只需要测量游戏内水平转动一周需要移动的像素即可(游戏内瞄准一个点不断地改变pixel 的值，直到相机画面不会抖一下)，不同倍镜下的 fa 值是不同的需要单独测量。

```python
import win32api
import win32con
pixel = 10909
for i in range(pixel)
{
    win32api.mouse_event(xxx, 1, 0, 0, 0)
}
```

现在有了 fa 的值之后，再来计算如何将鼠标移动到频幕上的点 Point(x, y), 首先计算目标点和屏幕某点 P 的偏移向量 offset(x, y)但是，真实的移动是发生在游戏空间的 3D 世界。

![](https://xuaii.github.io/post-images/1664242533685.png)
也就是需要将视角方向从OB 调整到OG方向，借鉴 [水平像素转角度](https://www.bilibili.com/read/cv17317767?from=search&spm_id_from=333.337.0.0)方法可以先将视线 OB 调整到 OC，再从OC调整到OG，这个过程中 相机到视平面的距离是不变的。
![](https://xuaii.github.io/post-images/1664244877946.png)
 为了保持距离不变，所以是先从 OB -> OI -> OK，所以将三维的转动过程转化为两次二维的转动
 
 $$
 dis2screen = AB = \frac{half\_screen\_width}{\tan{\frac{fov}{2}}} \\
 \quad\\
  \quad\\

 \theta_x = \angle{BAI} =  \arctan{\frac{BC}{AB}} \arctan{\frac{offset_x}{dis2screen}}
  \quad\\
  \quad\\
  \quad\\
  \quad\\
\theta_y = \angle{CAG} = \arctan{\frac{CG}{AC}} = \arctan{\frac{CG}{\sqrt{AB^2+BC^2}}} = \arctan{\frac{offset_y}{\sqrt{dis2screen^2 + offset_x^2}}}
 $$
这样就得到了两个方向的偏转角$(\theta_x, \theta_y)$，所以根据之前测量得到的 fa 值，可以得到鼠标的像素偏移值：
$$
pixel_i = \theta_i \times \frac{fa}{游戏内灵敏度\times FOV}
$$
所以就可以由屏幕偏移量 offset 得到鼠标偏移量 pixel， 此外需要能够移动鼠标的方法，在 python 环境下，常规的自动化库方法都是无效的，鼠标驱动层的移动应该是有效的但是麻烦，win32api 也是有效的，不过需要以管理员权限启动程序。
```python
import win32api
import win32con
win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(pixel_x), int(pixel_y), 0, 0)
```
## 使用 Json 读取配置

使用简单又方便的 [nlohmann/json: JSON for Modern C++ (github.com)](https://github.com/nlohmann/json)，只有 .hpp 头文件

```c
std::cout << "# init config ..." << std::endl;
// ----------------------- config -----------------------
std::ifstream f = std::ifstream();
json json_data;;
Config config;
// 需要根据当前运行环境构造文件全名
try {
    f.open("./config.json");
    json_data = json::parse(f);
    std::cout << "[init] load config from: " << "config.json" << std::endl;
    config.windowName = json_data["windowName"];
    config.classNamesPath = json_data["classNamesPath"];        config.pythonHome = json_data["pythonHome"];
    config.debug = json_data["debug"];
    config.detectorName = json_data["detectorName"];
    config.paramPath = json_data["paramPath"];
    config.binPath = json_data["binPath"];
    config.boxThreshold = json_data["boxThreshold"];
    config.nmsThreshold = json_data["nmsThreshold"];
    config.useGPU = true;
    config.mouseMovementDelay = json_data["mouseMovementDelay"];
    config.receptiveField = json_data["receptiveField"];
}
catch (std::exception& e) {
    std::cout << "配置文件读取失败" << std::endl;
    return -1;
}
```


