# [游戏机制] 里世界-优化




在之前的文章里提到过 [里世界机制](https://www.bilibili.com/video/BV1Er4y1t7NK?spm_id_from=333.999.0.0&vd_source=f793394b2f7b2b82cc3f09ecd0d4cd91) 游戏机制，但是最初的版本是需要手动用右摇杆移动 "透镜" 来观察里世界的物体，这样的设计有以下几个**缺陷**：

1. 需要控制左右摇杆和跳跃键来控制人物和移动 "透镜"
2. 透镜 **自由度过高**，限制了关卡设计的可能性
3. 仅仅实现了遮挡 / 显示 物体，并没有修改物体的**碰撞体积**

后来玩到 [《塞尔达传说三角力量 2》](https://zelda.fandom.com/zh/wiki/%E5%A1%9E%E5%B0%94%E8%BE%BE%E4%BC%A0%E8%AF%B4%EF%BC%9A%E4%BC%97%E7%A5%9E%E7%9A%84%E4%B8%89%E8%A7%92%E5%8A%9B%E9%87%8F2) ，其中林克化身壁画在墙壁游走的能力给了我很大的启发："壁画林克只能在一条水平线上移动，而作者利用地形的高低差设计了很多意想不到的关卡", 这背后的思考是 "通过限制玩家的能力，来增加关卡设计的可能性"，回到里世界机制上，我们可以通过限制"透镜"的移动和玩家进入的方式来增加关卡设计的可能性：
![](https://xuaii.github.io/post-images/1662048022127.gif)
于是，我对机制进行如下修正:

1. 通过**推箱子**的形式移动透镜，而不是用**右摇杆**
2. 限制透镜的**进入点**，而不是**全开放**

这样的修正可以以某种方式设计关卡，让玩家思考以何种顺序移动各个箱子能到达想要去的目标，它的本质是**推箱子**， 只是换上了一层 **里世界** 的皮

下面来构思整个机制的实现：

1. 实现 "透镜" 内的物体不显示, 之外的物体显示 -> 使用光照剔除
2. 实现 "透镜" 内的物体与玩家碰撞，之外的物体不碰撞 -> 使用 PolygenShape动态修改
3. 实现 推箱子 机制 -> 箱子内和箱子都能推动（避免死锁）

### 光照剔除

在之前的实现里，将里世界的所有物体放入一个单独的 CanvasLayer，再应用一个 CanvasModulate 将不透明度设置为 0， 然后用一一个 Light2D 来为里世界物体添加 透明度，这样实现玩家不能与里世界物体发生物理交互所以采用了很复杂的实现方式。

Godot Light2D 有一个 Mask 模式专门用于剔除物体，被剔除的物体需要确保

1. 层号在 (layer_min, layer_max) 之间
2. 光照层在 Item Cull Mask 之中

使用一个 Texture 与透镜大小相同的 Light2D 节点， 选择 Add / Mix 模式位被剔除的物体赋予颜色，同样需要确保光照层和画布层设置正确

此外，可以通过 WorldEnviroment 节点为里世界的物体增加荧光，或者设置边缘光 Shader 增加区分度。

### 动态碰撞体

动态碰撞体很简单，只需要每帧取碰撞体矩形 DefaultRect 透镜碰撞体矩形 LensRect，计算两个 Rect2 的 ∩ 即可



```c#
public class LensRect : Area2D
{
    public Dictionary<string, Godot.Object> NearbyLens = new Dictionary<string, Godot.Object>();
    Vector2 extent;
    CollisionShape2D shape;
    public override void _Ready()
    {
        shape = GetNode<CollisionShape2D>(nameof(CollisionShape2D));
        extent = (shape.Shape as RectangleShape2D).Extents;
    }

    // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _PhysicsProcess(float delta)
    {
        foreach(DynamicPolygon polygon in NearbyLens.Values)
        {
            polygon.Clips(ToGlobal(shape.Position - extent), ToGlobal(shape.Position + extent));
        }
    }
    void _on_LensRect_area_entered(Area2D body)
    {
        Node parent = body.GetParent();
        if(parent is DynamicPolygon)
        {
            NearbyLens[parent.Name] = parent;
        }
    }
    void _on_LensRect_area_exited(Area2D body)
    {
        Node parent = body.GetParent();
        if(parent is DynamicPolygon)
        {
            NearbyLens.Remove(parent.Name);
        }
    }
}
```

Lens 需要使用一个 Area2D 记录附近可能相交的**里世界物体**，在每一个物理帧调用 `polygon.Clips()` 更新附近**里世界物体**的多边形点集



对于每一个里世界物体， 只需实现 Clips 方法， 并保证碰撞体附近没有 Lens 时，碰撞体保持休眠状态

```c#
public class DynamicPolygon : KinematicBody2D
{
	private CollisionPolygon2D shape;
	private Rect2 rect;
	public override void _Ready()
	{
		Sprite sprite = GetNode<Sprite>(nameof(Sprite));
		shape = GetNode<CollisionPolygon2D>(nameof(CollisionPolygon2D));
		rect = sprite.GetRect(); ;
	}
	// Called every frame. 'delta' is the elapsed time since the previous frame.
	public void Clips(Vector2 p1, Vector2 p2)
	{
		Rect2 _rect = new Rect2(ToLocal(p1), ToLocal(p2) - ToLocal(p1));
        Rect2 inter = rect.Clip(_rect);

		Vector2 [] array = new Vector2[4];
		Vector2 local = inter.Position;
		
		array[0] = local;
		array[1] = local + Vector2.Right * inter.Size.x;
		array[2] = local + inter.Size;
		array[3] = local + Vector2.Down * inter.Size.y;
		shape.Polygon = array;
	}
	void _on_Area2D_area_entered(Area2D area)
	{
		if (area is LensRect magic)
		{
			shape.SetDeferred("disabled", false);
        }
	}
    void _on_Area2D_area_exited(Area2D area)
    {
        if (area is LensRect magic)
        {
            shape.SetDeferred("disabled", true);
        }
    }
}
```

这个地方有很多可以优化的点（下次一定），例如：

1. 目前仅支持 矩形 Lens 和 矩形里世界物体，可以考虑支持任意多边形

2. 可以优化同时对 **多个Lens**， **多个 里世界物体** 更新（例如使用 ECS， 做一个 System 统一更新，效率更高，也可以做一些向量优化，maybe）

3. 实现 其他 PhysicsBody 的里世界物体

4. 添加 [Tool]编辑器代码，方便 Debug

### 推箱子

首先实现**箱子类**，他是一个可推动接口，实现 Push 和Stop方法, 受到重力影响

```c#
public interface IPushable
{
    void Push(Vector2 direction);
    void Stop();
}
public class TestMovableCube : KinematicBody2D, IPushable
{
    private Vector2 _velocity = Vector2.Zero;
    [Export] private float _pushSpeed = 100f;
    public override void _PhysicsProcess(float delta)
    {
        _velocity += Vector2.Down * 100f;
        _velocity = MoveAndSlide(_velocity);
    }
    public void Push(Vector2 direction)
    { 
        _velocity += _pushSpeed * direction;
    }
    public void Stop()
    {
        _velocity = Vector2.Zero;
    }
}
```

其次是 玩家/怪物的Push状态, 这里在每一帧计算**正在推动的物体列表**， 然后每帧调用IPushable.Push() 方法，每帧开始时和退出状态时需要将所有正在推动的物体停下来（否则将做匀速直线运动）

```c#
/*
    状态有四个生命周期, 进入, 更新， 物理更新， 退出
    有 Init 接口函数用于初始化，有 Exit() 方法用于退出当前状态
*/

public class Push : StateBase
{
    private Player target;
    private List<IPushable> cacheCubes = new List<IPushable>();
    public override void OnEnter() 
    {
        target = agent as Player;
        GD.Print("Enter ", StateName);    
    }
    public override void Update(float delta) { }
    public override void PhysicsUpdate(float delta) 
    {
        target.GravityHandler(target.DefaultGravity.JumpGravity, delta);
        target.GestureHandler();
        target.HorizontalHandler();
        target.SnapHandler();

        foreach (var cube in cacheCubes)
        {
            cube.Stop();
        }
        cacheCubes.Clear();
        for (int i = 0; i < target.GetSlideCount(); i++)
        {
            // tips: Godot.Collections.Array dont support the c# interface
            var collision = target.GetSlideCollision(i);

            if (collision.Collider is IPushable pushable)
            {
                GD.Print(collision.Normal);

                cacheCubes.Add(pushable);
                pushable.Push(-collision.Normal);
            }
        }

        if ((!Input.IsActionPressed("ui_left") && !Input.IsActionPressed("ui_right"))
            || target.JumpRequest()
            || !target.IsOnFloor()
            )
        {
            Exit();
        }
    }
    public override void OnExit() 
    {
        foreach (var cube in cacheCubes)
        {
            cube.Stop();
        }
        GD.Print("Exit ", StateName);    
    }
}
```

Tips: 这里**为什么不让物体自己动**? 而是要推动者维护一个推动列表呢？如果 IPushable 自己动，即使 **Player 紧贴着 IPushable推动**，IPushable仍然**不能在每帧检测到有 Player 正在推它**，这会导致 IPushable 移动时断断续续的（不清楚这是bug还是特性)

### 美化 - 添加进出口特效

由于这本来是一个原型项目，所以只实现见简单特效即可（主要是花里胡哨的我也不会），Lens的进出口使用激光门（就是两束激光）

**激光实现**

```c#
- RayCast2D
    - Position2D
        - Sprite
    - CPUParticals2D
```

```c#
[Tool]
public class Laser : RayCast2D
{
    Sprite sprite;
    Position2D anchor;
    float default_length = 580f;
    Vector2 target = Vector2.Right*10;
    CPUParticles2D articles;
    public override void _Ready()
    {
        anchor = GetNode<Position2D>(nameof(Position2D));
        sprite = anchor.GetNode<Sprite>(nameof(Sprite));
        Enabled = true;
        articles = GetNode<CPUParticles2D>("LaserPartical");
        Scale = Vector2.One;
    }
    //  // Called every frame. 'delta' is the elapsed time since the previous frame.
    public override void _PhysicsProcess(float delta)
    {
        anchor.Rotation = anchor.Position.AngleToPoint(CastTo) + Mathf.Pi;
        float dis;
        if (!Engine.EditorHint)
        {
            if (IsColliding())
            {
                target = GetCollisionPoint();
                articles.GlobalPosition = GetCollisionPoint();
                articles.Direction = GetCollisionNormal();
                articles.Emitting = true;
            }
            else
            { 
                articles.Emitting = false;
            }

            dis = ToLocal(target).Length();
        }
        else
        {
             dis = CastTo.Length();

        }

        anchor.Scale = new Vector2(dis / default_length, anchor.Scale.y);
    }
}
```

激光使用了[Energy Beams - Godot Shaders](https://godotshaders.com/shader/energy-beams/) 的实现， 使用一个 Texture 表示一个激光，这种做法可以使用Shader 计算扰动，但是对于本题这种点到点激光，并且可能发生转动的情况下，还是很不方便，在计算**起始/结束点**和激光 Sprite 的Scale 的关系时不太好计算（主要是 Sprite 只能通过 Scale 调整大小而不便指定 Rect 的 Size， 好像也可以,emmm）， 激光命中的粒子就随便弄一弄就好了。
### 一些思考核问题
在实现过程中遇到很多问题：
1. 被推动的箱子时主动移动还是被动移动?
2. 箱子使用 KinamicBody 还是RigidBody（能够模拟下落，抛，旋转）?
3.  箱子应该从内部移动还是外部移动?
4.  是否需要设计里世界物体Mask？（例如，红箱子只能显示红色物体，绿箱子只能显示绿物体，其实应该有Mask，这样可以增加关卡设计的多样性）
5.  是否应该设置里世界物体的显示时间？（可以增加该机制）
6.  Godot.Collections.Array 不能存放 C# 接口，会报错，不知道是 Bug还是什么？
7.  是否应该有其他多边形状的 Lens 和 里世界物体？ （应该有的）
8.  Lens 的矩形是否应该有很多个进出口？（为了方便机制设计，应该有，所以一个Lens有很多个 PolyShape）
9.  是否里世界物体应该高亮（应该的，为了区分里外物体，但是还没想好高亮应该怎么做？边缘发光，荧光，虚线还是整体发光？）
10. 是否应该区分Lens 的可推动部分和不可推动部分？（应该，有利于复杂关卡设计）

