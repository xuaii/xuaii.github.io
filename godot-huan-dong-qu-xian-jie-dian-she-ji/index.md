# [Godot] 缓动曲线节点设计




### 动机
Tween 节点可以用于创建补间动画，但是由于只能指定固定缓动函数，并且不能使用 value = f(time) 的形式计算值，所以需要设计一个新的缓动函数类，并实现对应的`Inspector插件`

通常的缓动曲线都是选用 三角函数，或者使用贝塞尔曲线绘制的自定义曲线，为了逼近现实的运动感，这里实现一种自动控制系统里常用的[二阶曲线](../you-xi-dong-hua-zhong-de-huan-dong-han-shu/)

 ### `SecondOrderCurves` 设计
 首先时二阶系统需要的三个参数，在 Godot 中使用 `Vector3` 表示， 以及运算过程的中间变量
 ```c#
 // 使用插件修复 Godot 不识别自定义类型的 Bug
[RegisteredTypeAttribute(nameof(SecondOrderCurves), "", nameof(Godot.Object))]
public class SecondOrderCurves : Godot.Object
{
    [Export] public Vector3 Parameters = Vector3.One;
    private Vector2 xp;
    private Vector2 y, yd;
    private float k1, k2, k3;
    private float T_crit; // 用于处理单时间片无法完成的模拟
}
 ```
 二阶系统的参数只有三个变量，但是我们需要一个输入函数 `X(y)`, 和得到一个响应函数`Y(y)`，其中输入函数是必须的，它决定以某一时刻系统的输入, 而响应函数可以在游戏运行时实时模拟。

 ```c#
// 输入函数由一些点组成，是折线函数
[Export] public Array<Rect2> points = new Array<Rect2>()
{
    new Rect2(new Vector2(0, 300), new Vector2(10, 10)),
    new Rect2(new Vector2(200, 300), new Vector2(10, 10)),

    new Rect2(new Vector2(200, 150), new Vector2(10, 10)),
    new Rect2(new Vector2(400, 150), new Vector2(10, 10)),
};
```
<img src="https://cdn.staticaly.com/gh/xuaii/xuaii.github.io/1.4/static_easing_curve/x_t_curve.png" alt="" title="x(t)">
点集是 `X(t)` 的离散表示，对于连续值得输入，需要在两个点之间进行插值，这里简单的采用线性插值

```c#
public Vector2 Interpolate(float x)
{
    for(int i = 0; i < points.Count - 1; i++)
    {
        if(points[i].GetCenter().x <= x && x <= points[i+1].GetCenter().x)
        {
            float y1 = points[i].GetCenter().y;
            float y2 = points[i+1].GetCenter().y;
            float x1 = points[i].GetCenter().x;
            float x2 = points[i+1].GetCenter().x;
            return new Vector2(x, (y1 - y2) / (x1 - x2) * (x - x2) + y2);

        }
    }
    return -1 * Vector2.One;
}
```

对于某时刻 `t` 一个 `X(t)` 对应一个X(t); 为了计算 `Y(t)`，这里采用逐帧计算，

```c#
// 输入 X(t), 和 delta， 得到当前t时刻的Y(t)
public Vector2 Interpolate(Vector2 x, float delta)
{
    Vector2 xd = (x - xp) / delta;
    xp = x;
    int iterations = Mathf.CeilToInt(delta / T_crit); // take extra iterations if delta > T_crit
    delta = delta / iterations;
    for(int i = 0; i < iterations; i++)
    {
        y = y + delta * yd;
        yd = yd + delta * (x + k3 * xd - y - k1*yd) / k2;
    }
    return y;
}
```

### 编辑器插件设计
可以参考 `Godot.Curve` 的实现，但是没有开放 `gdscript/c#` 接口，只能使用c++拓展，比较繁琐，就自行实现显示和交互逻辑。
#### 编辑器绘图
为了能在编辑器中看到效果，需要在编辑器环境中预览响应函数 `Y(t)` 的形状。所以使用 `point_set` 存储曲线值， 表现效果如下图所示：
<img src="https://cdn.staticaly.com/gh/xuaii/xuaii.github.io/1.4/static_easing_curve/y_t_curve_1.png" alt="" title="y(t)_1">
<img src="https://cdn.staticaly.com/gh/xuaii/xuaii.github.io/1.4/static_easing_curve/y_t_curve_2.png" alt="" title="y(t)_2">

```c#
//  只有标记为 Tool 的脚本才会在编辑器环境被加载
[Tool]
public class CurveCanvas : Control
{
    Array<Vector2> point_set = new Array<Vector2>();
    Vector2 point_size = Vector2.One * 10;
}
```
模拟每帧的`Y(t)` 计算：
```c#
public void ReDraw(SecondOrderCurves curve = null)
{
    // 缓存 cached
    if(curve != null)
    {
        cachedCurve = curve;
    }
    else
    {
        return;
    }

    point_set.Clear();
    Vector2 start = Transform(cachedCurve.StartPoint);
    Vector2 end = Transform(cachedCurve.EndPoint);
    if(start.x == -1 || end.x == -1 || cachedCurve == null)
    {
        return;
    }
    // points.Sort(new PointComp()); // Godot.Collection.Array 不支持排序，需要实现一下
    cachedCurve.Init(start);
    int iter = 200;
    for(int i = 0; i < iter; i++)
    {
        Vector2 x = Transform(cachedCurve.Interpolate(cachedCurve.StartPoint.x + i* (cachedCurve.EndPoint.x - cachedCurve.StartPoint.x) / iter));
        Vector2 point = cachedCurve.Interpolate(new Vector2(x.y, x.y), 0.01f);
        point_set.Add(new Vector2(x.x * RectSize.x, (1 - point.x) * RectSize.y));
    }
    Update();
}
```
由于 Godot 是以右下为正，所以需要对坐标进行转换
```c#
private Vector2 Transform(Vector2 point)
{
    return new Vector2(point.x / RectSize.x, (RectSize.y - point.y) / RectSize.y);
}
```
使用 Godot 的绘图函数
```c#
public override void _Draw()
{
    // 具体就不展示了，是一些繁琐的代码
    DrawMesh();
    DrawPoints();
    DrawCurve();
    DrawInteractive();
}
```
#### 编辑器交互
为了能够添加/移动/删除图像上的点， 需要实现以下功能：
1. 单击鼠标左键添加点
2. 长按鼠标左键移动点
3. 单机鼠标右键删除点
```c#
public override void _Process(float delta)
{
    cover_index = cachedCurve.FindPoint(GetLocalMousePosition());
    SafeDragingArray = new Rect2(GetRect().Position, GetRect().Size - Vector2.One * 10);
    // 退出拖动的调节
    if(Dragging && IsMouseIn && 0 <=  DraggingIndex&& DraggingIndex < cachedCurve.points.Count && SafeDragingArray.HasPoint(GetLocalMousePosition()))
    {
        (cachedCurve.points[DraggingIndex]) = new Rect2(GetLocalMousePosition(), point_size);
    }
    Update();
}
```
监听鼠标输入，在Godot 中连接 CurveCanvas 的鼠标信号
```c#
void _on_CurveCanvas_gui_input(InputEvent @event)
{
    if(!IsMouseIn)
    {
        return;
    }
    if(@event is InputEventMouseButton mouse)
    {
        if(mouse.ButtonIndex == (int)ButtonList.Left)
        {
            // 1. 创建节点
            if(cover_index == -1 && !mouse.Pressed && !Dragging)
            {
                cachedCurve.points.Add(new Rect2(GetLocalMousePosition(), point_size));
            }
            // 2. 开始拖动
            if(cover_index != -1 && mouse.Pressed)
            {
                Dragging = true;
                DraggingIndex = cover_index;
            }
            // 3. 结束拖动
            if(!mouse.Pressed)
            {
                if(Dragging) ReDraw();
                Dragging = false;
                DraggingIndex = -1;
            }
            ReDraw();
        }
        if(mouse.ButtonIndex == (int)ButtonList.Right)
        {
            // 删除当前 cover 的节点
            if(cover_index != -1 && !mouse.Pressed)
            {
                cachedCurve.points.RemoveAt(cover_index);
            }
            ReDraw();
        }
    }
}
```

#### 编辑器插件（踩坑）
Godot 的编辑器插件功能非常多，可以添加Dock，Inspector，主屏幕，等插件。几乎可以实现编辑器阶段的的所有拓展；这里我们主要使用 Inspector 插件来编辑 `SecondOrderCurves.parameters`.
##### `plugin.cs`
首先在 `plugin.cs` 中添加加载/删除 `CurveEditorInspector` 的代码
tips: 千万不能忘记添加`[Tool]`，第一次 Build 时会报错，然后再Build一次就Ok了，原因是 C# 是需要编译的而gdscript可以热更新，所以干儿子者不如亲儿子。
```c#
#if TOOLS
using Godot;
using System;
[Tool]
public class plugin : EditorPlugin
{
    CurveEditorInspector inspector;

    public override void _EnterTree()
    {
        inspector = GD.Load<CSharpScript>("res://addons/curve_editor/CurveEditorInspector.cs").New() as CurveEditorInspector;
        AddInspectorPlugin(inspector);
    }
    public override void _ExitTree()
    {
        RemoveInspectorPlugin(inspector);
    }
}
#endif  
```
##### `EditorInspectorPlugin.cs`
EditorInspectorPlugin 是一个大坑，它有两种主要的运行模式：
1. 修改单属性
2. 修改多属性

起初我不明白 `AddPropertyEditor` 和 `AddPropertyEditorForMultipleProperties` 有什么区别，他们似乎具有相似的行为。后来理解了，在 `AddPropertyEditorForMultipleProperties` 中指定的**属性名列表**和在`AddPropertyEditor` 中指定属性名，这样Godot就会知道需要序列化哪些字段，并且在来回切换窗口或者充气Godot后，已经被编辑的属性能够不被重置。
```c#
using Godot;
using System;
#if TOOLS
public class CurveEditorInspector : EditorInspectorPlugin
{
    public override bool CanHandle(Godot.Object @object)
    {
        if(@object is SecondOrderCurves)
        {
            AddPropertyEditorForMultipleProperties("", new string [] {"F", "Z", "R"}, new CurveEditorProperty());
            return true;
        }
        return false;
    }
    public override bool ParseProperty(Godot.Object @object, int type, string path, int hint, string hintText, int usage)
    {
        return false;
    }
}
#endif
```

##### `CurveEditorProperty.cs`
这里需要指出的是，Godot规定修改属性值不能直接修改
只能通过 `EmitChanged(property_name, property value);` 来修改，该信号会回调`UpdateProperty()`方法，这里可以访问到被修改的对象(`GetEditedObject()`)和被修改对象的属名(`GetEditedProperty()`);
值得注意的是，Godot 会在每一此打开`Inspector `时调用`UpdateProperty()`,这之前会重新创建并且加载`Inspector` 插件对象，所以需要通过被编辑对象来恢复编辑器预览参数，并且使用`IsInit` 保证仅调用一次;
```c#
using Godot;
using System;
#if TOOLS
public class CurveEditorProperty : EditorProperty
{
    Control editor;
    SecondOrderCurves CacheCurve = null;
    bool IsInit = false;
    // 缓存被编辑对象
    public override void _Ready()
    {
        editor = GD.Load<PackedScene>("res://addons/curve_editor/CurveEditor.tscn").Instance<Control>();
        AddChild(editor);
        AddFocusable(editor);
        SetBottomEditor(editor);
        editor.Connect("PorpertyChanged", this, "OnPorpertyChanged");
        editor.Call("Load");
    }
    void OnPorpertyChanged(Vector3 _parameters)
    {
        EmitChanged("parameters", _parameters);
    }

    public override void UpdateProperty()
    {
        if(!IsInit)
        {
            editor.Call("Load", GetEditedObject().Get("parameters"));
            IsInit = true;
        }
        editor.Call("Refresh", GetEditedObject());
    }
}
#endif
```
##### `CurveEditor.cs`
该脚本用于总结三个`HSlider`的变化,并且生成`PorpertyChanged` 事件
```c#
using Godot;
using System;
// 这里的tool不能缺少
[Tool]
public class CurveEditor : VBoxContainer
{
    [Signal] public delegate void PorpertyChanged(Vector3 _parameters);
    // move point
    Vector3 parameters;
    public void Load(Vector3 parameters)
    {
        (GetNode("F/FHSlider") as HSlider).Value = (double)parameters.x;
        (GetNode("Z/ZHSlider") as HSlider).Value = (double)parameters.y;
        (GetNode("R/RHSlider") as HSlider).Value = (double)parameters.z;
    }
    public void Refresh(SecondOrderCurves curve)
    {
        CurveCanvas canvas = GetNode<CurveCanvas>("CurveCanvas");
        canvas.ReDraw(curve);
    }
    public void _on_FHSlider_value_changed(float value)
    {
        parameters.x = value;
        EmitSignal("PorpertyChanged", parameters);
    }
    public void _on_ZHSlider_value_changed(float value)
    {
        parameters.y = value;
        EmitSignal("PorpertyChanged", parameters);
    }
    public void _on_RHSlider_value_changed(float value)
    {
        parameters.z = value;
        EmitSignal("PorpertyChanged", parameters);
    }
}
```
### Show Show Way
最后展示一下运行效果

<img src="https://cdn.staticaly.com/gh/xuaii/xuaii.github.io/1.4/static_easing_curve/usage_demo.gif" alt="" title="usage demo">

<img src="https://cdn.staticaly.com/gh/xuaii/xuaii.github.io/1.4/static_easing_curve/anima_demo.gif" alt="" title="anima demo">

