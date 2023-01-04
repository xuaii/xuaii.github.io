# [游戏机制] 窗体互动



使用 Godot 对[WindowFrame](https://www.bilibili.com/video/BV1gi4y1D7Vx?spm_id_from=333.337.search-card.all.click) 进行拙劣的模仿，观察视频中的窗体有以下性质：
1. 窗体既是 *UI* 也是可以与玩家互动的 *场景物体*
2. 窗口的位置可以由场景物理改变也可以由鼠标控制改变
3. 玩家可以发射子弹，子弹碰到的边进入锁定状态，一段时间后子弹消失，并且解除锁定
考虑一下几种设计：
## 方案一
存在一个窗体对象window和场景对象rect
![](https://xuaii.github.io/post-images/1659391754573.png)
1. 每一帧将rect大小和位置经过MPV变换后同步到窗口大小
2. 每一个rect 对象有一个默认的初始大小
3. rect 对象持有四条边的对象 line，每个line对象有 follow/move/lock/idle状态
    Follow 状态的边跟随 target(player) 运动
    Move 状态的边能被鼠标拖动
    Lock 状态的边在玩家移动过程中充当 **墙** 或 **地板**
    Idle 状态，是场景中没有 target 时的状态
4. rect 对象实进入场景树的时候生成 windows 对象，并持有，rect退出场景树的时候回收 windows 对象
5. 各个状态都有对应的 Physics Collision Layer / Mask，例如锁定状态的边会阻挡某些攻击，拖动时和锁定时与玩家的碰撞是不一样的。
```c#
using Godot;
using System;

public class InteractiveAnchor : RigidBody2D
{
    public Vector2 velocity;
    public bool IsStop = false;
    public Vector2 StopPosition = Vector2.Zero;
    ColorRect rect;
    RandomNumberGenerator _random = new RandomNumberGenerator();
    public delegate void InteractiveAnchorCallBack(InteractiveAnchor anchor);
    public InteractiveAnchorCallBack callback;
    // cache 一旦锁定目标，不可更改！！
    InteractiveLine AnotherCache = null;
    public override void _Ready();
    public void Init(Vector2 _velocity, InteractiveAnchorCallBack _callback);
    public override void _ExitTree();

//  // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _PhysicsProcess(float delta);
    // 添加抖动将要删除时 
    public void _on_DeleteEffect_timeout();
    void _on_Area2D_body_entered(Node body);
    void _on_Area2D_body_exited(Node body);
    void CacheCurrentState(Vector2 origin_global_position);
}
public class InteractiveArea : Area2D
{
    public override void _Ready();
    Node Scene;
    public override void _EnterTree();
    public override void _PhysicsProcess(float delta);
    void _on_Area2D_body_entered(Node body);
    void _on_Area2D_body_exited(Node body);
}

public class InteractiveBorder : Node2D
{
    // 显示区域大小
    public Rect2 window;
    [Export] public Rect2 DefalutRect;
    [Export] public float BorderWidth;
    public InteractiveLine left;
    public InteractiveLine right;
    public InteractiveLine top;
    public InteractiveLine down;
    public InteractiveWindow interactiveWindow;
    public Vector2 Start;
    public Vector2 End;
    public Rect2 UIRect;
    public override void _Ready();
    public override void _EnterTree();
    public override void _Draw();
    public override void _Process(float delta);
    public override void _PhysicsProcess(float delta);
    public void Reset();
    public void SetRect(Rect2 _window);
    public void SetCollision(bool flag);
}
public class InteractiveLine : KinematicBody2D
{
    public bool IsLocked;
    static public string IsSlideName = "";
    public SegmentShape2D shape;
    public InteractiveWindow window;
    public Vector2 velocity = Vector2.Zero;
    public int ColllisionCount = 0;
    [Export] public Vector2 DefaultBias;

    public override void _Ready();
    public override void _ExitTree();
    public override void _PhysicsProcess(float delta);
    public void SetCollision(uint layer, uint mask);
    public void Reset();
    public float DistanceToLine(Vector2 P, Vector2 A, Vector2 B);
}
public class InteractiveLineFollow : StateNode<InteractiveLine>
{
    public override void Enter()
    {
        target.SetCollision(target.window.FollowLayer, target.window.FollowMask);
    }
    public override void _PhysicsUpdate(float delta)
    {
        /// 如果没有目标，就转移到Lock？
        if(!target.window.IsLockTarget)
        {
            _machine.Transition<InteractiveLineIdle>();
            return;
        }
        if(target.window.Target.IsInsideTree())
        {
            target.GlobalPosition = target.window.Target.GlobalPosition + target.DefaultBias;
        }

        if(target.ColllisionCount != 0)
        {
            _machine.Transition<InteractiveLineLock>();
            return;
        }
    }
public class InteractiveLineIdle : StateNode<InteractiveLine>
{
    public override void _PhysicsUpdate(float delta)
    {
        if(target.window.IsLockTarget)
        {
            _machine.Transition<InteractiveLineFollow>();
            return;
        }
    }
}
public class InteractiveLineLock : StateNode<InteractiveLine>
{
    Vector2 cachePosition = Vector2.Zero;
    public override void Enter()
    {
        target.IsLocked = true;
        cachePosition = target.Position;
        target.SetCollision(target.window.LockLayer, target.window.LockMask);
    }
    public override void Exit()
    {
        target.IsLocked = false;
    }
    public override void _PhysicsUpdate(float delta)
    {
        // TODO:这里是否可以不移动锁定的目标不移动
        if(cachePosition != null) target.Position = cachePosition;

        if(target.ColllisionCount == 0)
        {
            _machine.Transition<InteractiveLineFollow>();
            return;
        }
        if(Input.IsActionJustPressed("mouse_right") && target.NormalToLine(target.GetGlobalMousePosition(), target.GlobalPosition + target.shape.A, target.GlobalPosition + target.shape.B).Length() < 14f)
        {
            // 进入拖动状态
            _machine.Transition<InteractiveLineMove>();
            return;
        }
    }
}
public class InteractiveLineMove : StateNode<InteractiveLine>
{
    public override void Enter()
    {
        target.SetCollision(target.window.MoveLayer, target.window.MoveMask);
    }
    public float PlayerMargin = 50f;
    private float MoveSpeed = 50f;

    public override void _PhysicsUpdate(float delta)
    {
        if(!target.window.IsLockTarget)
        {
            _machine.Transition<InteractiveLineIdle>();
            return;
        }
        Vector2 line_movement = target.NormalToLine(target.GetGlobalMousePosition(), target.GlobalPosition + target.shape.A, target.GlobalPosition + target.shape.B);
        Vector2 player_movement = target.NormalToLine(target.window.Target.GlobalPosition, target.GlobalPosition + target.shape.A, target.GlobalPosition + target.shape.B);

        if(line_movement.Dot(player_movement) < 0f 
        || !target.window.Target.TestMove(target.window.Target.GlobalTransform, line_movement * delta)
        || player_movement.Length() > PlayerMargin)
        {
            // 鼠标位置与Player在line异侧不限制速度，player与line距离足够大不限制速度；其余情况限制速度
            if(line_movement.Dot(player_movement) < 0f || player_movement.Length() > PlayerMargin)
            {
                target.GlobalPosition += line_movement;
            }
            else
            {
                target.GlobalPosition += line_movement.Normalized() * MoveSpeed * delta;
            }
        }

        if(Input.IsActionJustReleased("mouse_right"))
        {
            _machine.Transition<InteractiveLineLock>();
            return;
        }
    }
}

public class InteractiveWindow : Node2D
{
	// window local position
	[Export] PackedScene bullet;
    [Export(PropertyHint.Layers2dPhysics)] public uint MoveMask;
    [Export(PropertyHint.Layers2dPhysics)] public uint MoveLayer;
    [Export(PropertyHint.Layers2dPhysics)] public uint FollowMask;
    [Export(PropertyHint.Layers2dPhysics)] public uint FollowLayer;
    [Export(PropertyHint.Layers2dPhysics)] public uint LockMask;
    [Export(PropertyHint.Layers2dPhysics)] public uint LockLayer;

	ShaderMaterial shader;
	public InteractiveBorder border;
    Main screen;
    // canvas + window related
    WindowDialog dialog;
    CanvasLayer canvas;
    // screen effects
    Tween tween;
    RandomNumberGenerator _random = new RandomNumberGenerator();
    Queue<InteractiveAnchor> queue = new Queue<InteractiveAnchor>();
    // mouse related
    // physics's disabled area
    Area2D area;
    RectangleShape2D shape;
    [Export] Vector2 SoftMargin = new Vector2(5, 5);

    // target lock related
    public bool IsLockTarget = false;
    public KinematicBody2D Target = null;
    public Console _wrapper;
	public override void _Ready();
    public override void _EnterTree();
    public override void _ExitTree();
    void _DeferredPrograce()
    {
        screen.RemoveChild(canvas);
        screen.ResetScreen();
        this.AddChild(canvas);
    }
	
	public override void _PhysicsProcess(float delta)
	{
		// 1. window 更新 + 可视区剪裁
		screen.SetDisplayRect(border.Start, border.End);
        // 2. windowdialog 对齐
        dialog.Popup_(border.UIRect.Clip(screen.viewport.GetVisibleRect()));
        // 3. 物理开启区域对齐
        area.Position = border.window.Position - SoftMargin;
        shape.Extents = border.window.Size +  SoftMargin * 2;

	}
    public void OnWindowGuiInput(InputEvent inputEvent)
    {
        if (inputEvent is InputEventMouseButton mouseEvent && mouseEvent.Pressed)
		{

			switch ((ButtonList)mouseEvent.ButtonIndex)
			{
				case ButtonList.Left:
				{
                    if(IsLockTarget)
                    {
                        InteractiveAnchor anchor = bullet.Instance<InteractiveAnchor>();
                        anchor.GlobalPosition = GetGlobalTransform().AffineInverse() * GetViewportTransform().AffineInverse() * (border.Start + mouseEvent.Position);
                        this.CallDeferred("add_child", anchor);
                        // anchor.CallDeferred("Init", border.Start + mouseEvent.Position - BasicFollowCamera.Target.ScreenPosition, (object)EffectCallback);
                        anchor.Init((border.Start + mouseEvent.Position - BasicFollowCamera.Target.ScreenPosition), EffectCallback);
                    }		
					break;
				}    
				case ButtonList.WheelUp:
					break;
			}
		}
        if (inputEvent is InputEventKey keyEvent && keyEvent.Pressed)
		{

			switch ((int)keyEvent.Scancode)
			{
				case (int)KeyList.L:
				{
                    break;
                }
			}
		}
    }
    // delete callback
    public void EffectCallback(InteractiveAnchor anchor);
    void disturb_offset(float _strength)
    {
        dialog.RectPosition += new Vector2(_random.RandfRange(-_strength, _strength), _random.RandfRange(-_strength, _strength));
        screen.GetNode<TextureRect>("CanvasLayer/_ScreenTexture").RectPosition += new Vector2(_random.RandfRange(-_strength, _strength), _random.RandfRange(-_strength, _strength));
    }
    public void LockTarget(KinematicBody2D target);
    public void UnlockTarget();
    public void EnableBackground();
    public void DisableBackground();
    public void SetRect(Rect2 _window);
    public void Reset();
    // CommandLine
    private void LockPlayer();
    private void _on_ActivateArea_body_entered(Node body);
}

```
总结：这样的设计有以下缺陷
1. 拖动 Line 时不能直接修改 velocity，因为拖动时Line有可能会与Player发生碰撞（需要KinematicBody2D 的SlideAndCollide方法更新速度），所以只能设计一种有鼠标位置计算 Line.velocity 的方法。
2. 窗口Window对象在场景的UI节点下，借此将场景相机的画面绘制到窗口相应的位置，这样设计与物体外的结构耦合。
3. 如果想实现窗口外的一切对象停止更新比较麻烦（仅仅 Update 相机内的对象），应该有引擎自带的方法比较方便。

## 方案二
不使用 UI 对象，将交互和物理全部集中到一个 Rect 对象中。
![](https://xuaii.github.io/post-images/1659395615675.png)
1. 交互窗体包含四条边，每条边包含碰撞设置和鼠标靠近检查（锁定状态）
2. 每条边拥有相似的几种状态，特别的在拖动时，依然使用velocity来移动每条边
当鼠标左键按下 && 鼠标位置在边的拖动区域时 ==> 进入拖动状态
松开鼠标左键时 ==> 退出拖动状态
拖动状态下，鼠标位置，边位置，边速度的关系如下
$$
velocity_L = \frac{(pos_M - pos_L) \cdot normal_L}{||normal_L||}
$$
这样，窗口移动始终比鼠标位置延后一帧，但是移动过程中可以进行物理检测
3. 窗口根据三条边的位置绘制窗口（场景中绘制），并且生成一个mask覆盖整个屏幕（不使用方案一中生成相机纹理在windows中显示）
4. 如果某条边的移动会使得 player 与其他物体碰撞，那么该移动应该被修正:
会与player碰撞的物体有两种可能：
a. 场景中的物体          -> 丢弃当前的移动
b.窗体中的自由对边   -> 保持当前移动
c.窗体中的锁定对边   -> 丢弃当前移动
为了使移动边感知到玩家的碰撞，需要为边增加一个Push状态（不用增加，写在Move里即可），该状态下每次移动时应该对玩家进行移动测试，如果通过测试则移动否则不移动。
5. 在边的Move状态下，应该停止相机对 Player的跟踪
6. 2D 引擎不能如3D一样剔除视锥外的物体，不渲染/不更新窗口外的物体只能通过碰撞来实现
a. 所有进入窗体的对象将恢复更新，退出窗体的对象停止更新
b. 场景对象持有所有场景物体的引用，所以当玩家进入窗口时可以直接调用场景节点的Stop方法，这将停止所有除（player，window，obj_in_window）的对象。



