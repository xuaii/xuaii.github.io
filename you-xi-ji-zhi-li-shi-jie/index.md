# [游戏机制] 里世界



[里世界机制](https://www.bilibili.com/video/BV1Er4y1t7NK?spm_id_from=333.999.0.0&vd_source=f793394b2f7b2b82cc3f09ecd0d4cd91)
机制概念简述：存在两个不同的世界，里世界和表世界
1. 他们有大致相同的场景整体形状，不同的场景风格（例如，崭新<->老旧，清洁<->污染）
2. 有一个透镜能在表世界看见里世界的物体并且互动
3. 只有玩家能在里世界和表世界间穿梭，这意味着玩家可以通过切换世界来躲避伤害
   所有的外部物体（除玩家外），与透镜发生碰撞时可以
   1. 以一种特效的的方式穿过，暗示对玩家没有伤害
   2. 直接与透镜碰撞然后消失
4. 通过对透镜的形状限制来设计关卡
![](https://xuaii.github.io/post-images/1659399769652.JPG)

![](https://xuaii.github.io/post-images/1659399587154.JPG)





实现：表示世界和里世界将在一个场景节点中表示
1. 透镜物体是一个Light2D， 通过光照Mask显示/剔除里世界的物体的Sprite
2. 透镜物体是一个Area2D，通过记录进入区域中的里世界物体，并且实时修改里世界物体的碰撞形状，如果是表世界（那么根据简述3来处理）
```c#
public class SuperLens : Area2D
{
    public Dictionary<string, InnerObject> data = new Dictionary<string, InnerObject>();
    public void OnObjectExit(Node node)
    {
        if(node is InnerObject innerObject && data.ContainsKey(innerObject.Name))
        {
            data.remove(innerObject.Name);
        }
    }
    public void OnObjectEnter(Node node)
    {
        if(node is InnerObject innerObject )
        {
            data[innerObject.Name] = innerObject;
        }
        if(node is OuterObject outerObject )
        {
            ProcessOuterObject(outerObject);
        }
    }
    public override void _PhysicsProcess(float delta)
    {
        foreach(var inner in data.Values)
        {
            // 更改碰撞体积
            inner.Rect = inner.DefaultRect.overlap(this.Rect);
        }
    }
}

``` 



