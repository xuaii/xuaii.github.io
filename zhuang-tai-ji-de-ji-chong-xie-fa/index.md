# 状态机的几种写法


在各个Godot初学者教程中，实现第一个人物控制器都是使用的状态机，常见的有以下写法：
## 写法一 全在 Update 中
```c#
using Godot;
using System;
public class ScriptName : Node
{
	// fields
   [Export] float Speed；
   [Export] float JumpHeight;
   private State state = State.Idle;
   public enum State : int
   {
   		Run, Jump, Idle, Fall
   }
	public void Update
   {
    	switch(state)
        {
        	case State.Idel:
            // code
           case State.Run:
           	// code
           case State.Jump:
           	// code
           case State.Fall:
           	// code
           default:
           	// code
        }

   }
} 
```
**优点** ： 至少能实现功能，
**缺点**： 不利于维护，不方便拓展 

## 写法二 写成状态类和状态机类 
```c#
public interface IState
{
    void Enter();
    void Exit();
    void Update(float delta);
    void PhysicsUpdate(float delta);
}
public class StateBase<T> : IState
{
    private StateMachine _machine;
    private T agent;
    public virtual void Enter() { }
    public virtual void Exit() { }
    public virtual void Update(float delta) { }
    public virtual void PhysicsUpdate(float delta) { }
}
public class StateMachine : Node
{
    Dictionary<string, IState> StateInfo;
    IState CurrentState;
    public void ChangeTo<T>() where T : IState { ... }
    public override void _Process(float delta){ CurrentState.Update(delta); }
    public override void _Process(float delta) { CurrentState.PhysicsUpdate(delta); }
}

class RunState : StateBase<T> { ... }
class JumpState : StateBase<T> { ... }
class IdleState : StateBase<T>
{
    if(Mathf.Abs(horizontalInput) > 0.01)
    {
        _machine.ChangeTo<RunState>();
    }
    if(verticalInput > 0.1)
    {
        _machine.ChangeTo<Jump>();
    }
}
```
**优点**:比写法一便于维护
**缺点**：如果要实现一个 Player 的状态机 和一个Enemy的状态机，他们都有相似的移动逻辑，但是Player状态机多出一些技能相关的状态；如果采用上述写法，ChangeTo()是硬编码在状态内部的，这样就不方便添加边和状态以拓展状态机

## 写法三 ：基于 GraphEditor的状态机
[Godot Asset Store](https://godotengine.org/asset-library/asset) 有很多基于 GraphEdit 的状态机实现，通过：
1. 为状态绑定状态脚本
2. 为转移边绑定脚本或者添加转换条件

减少状态 和 转移边 的耦合。
它们都有一个共同的缺点：不支持 c# 或者c#版本bug多
所以能不能实现一个足够简单的，C#版本的，解耦状态和转化边的状态机呢？
## 写法四
### 状态类
```c#
using Godot;
using Godot.Collections;
namespace StateMachine.Base
{
    public interface IState
    {
        string StateName {get;}
        void Init(Object agent, Dictionary<string, object> blackboard, IStateMachine machine);
        void OnEnter();
        void OnExit();
        void Update(float delta);
        void PhysicsUpdate(float delta);
        void Exit();
    }

    public class StateBase : Object, IState
    {
        // state 只持有 agent 对象不持有状态机对象
        protected string _StateName = "";
        public string StateName => _StateName;
        public Object agent;
        public IStateMachine machine;
        private Dictionary<string, object> blackboard;

        public virtual void Init(Object _agent, Dictionary<string, object> _blackboard,  IStateMachine _machine)
        {
            agent = _agent;
            blackboard = _blackboard;
            machine = _machine;
        }
        public virtual void OnEnter() { }
        public virtual void Update(float delta) { }
        public virtual void PhysicsUpdate(float delta) { }
        public virtual void OnExit() { }
        public virtual void Exit()
        {
            machine.Transition(StateName);
        }

    }
}
```
   状态类内部不能调用StateMachine的ChangeTo()方法，只能实现状态自身的更新逻辑 
### 工具类
IStateMachine 接口，Condition 是用于转换边的方法（返回true就转换，否则就不转换）
满足AssertTo.condition 则转换到AssertTo.nextState
Transition的属性用于标志该方法是判断哪个状态转移到哪个状态的
   ```c#
    public interface IStateMachine
    {
        void Transition(string current);
        void Start(string name);
        void Exit();
    }
    public delegate bool Condition();

    public class AssertTo : Godot.Object
    {
        public Condition condition;
        public string nextState;
        public AssertTo(Condition _condition, string _nextState)
        {
            condition = _condition;
            nextState = _nextState;
        }
    }
    [System.AttributeUsage(System.AttributeTargets.Method)]
    public class Transition : System.Attribute
    {
        public string from;
        public string to;
        public Transition(string _from, string _to)
        {
            from = _from;
            to = _to;
        }
    } 
   ```
### 状态机类
状态机类继承自 StateBase，因此状态机是可嵌套的
```c#
public class StateMachineBase : StateBase, IStateMachine
{

    public IState CurrentState = null;
    protected Dictionary<string, Array<AssertTo>> TransitionMap = new Dictionary<string, Array<AssertTo>>();
    public Dictionary<string, Godot.Object> Name2State = new Dictionary<string, Godot.Object>();

    // 状态机黑板
    protected Dictionary<string, object> blackboard;

    public override void Init(Godot.Object _agent, Dictionary<string, object> _blackboard = null, IStateMachine _machine = null)
    {
        base.Init(_agent, _blackboard, _machine);
        if(blackboard == null)
        {
            blackboard = new Dictionary<string, object>();
        }
        if(_machine == null)
        {

        }
    // 初始化状态列表 && 初始化状态名
    foreach(var info in this.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
    {
        if(info.ReflectedType.IsSubclassOf(typeof(StateMachine.Base.StateBase)))
        {
            if(Name2State.ContainsKey(info.Name))
            {
                GD.Print("[FSM runtime] duplicate state ", info.Name);
            }
            else
            {
                IState state = info.GetValue(this) as IState;
                if(state == null)
                {
                    // current state is null before start
                    continue;
                }
                Name2State[info.Name] = state as Godot.Object;
                state.Init(agent, blackboard, this as IStateMachine);
                (state as Godot.Object).Set("_StateName", info.Name);

            }
        }
    }

        // 初始化转换边列表
        foreach(var info in this.GetType().GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
        {
            if(info.GetCustomAttribute<Transition>() is Transition transition)
            {
                if(!Name2State.ContainsKey(transition.from) || !Name2State.ContainsKey(transition.to))
                {
                    return;
                }
                if(!TransitionMap.ContainsKey(transition.from))
                {
                    TransitionMap[transition.from] = new Array<AssertTo>();
                }
                TransitionMap[transition.from].Add(new AssertTo((Condition)Delegate.CreateDelegate(typeof(Condition), this, info), transition.to));
            }
        }
    }
    #region life cycle

    public override void OnEnter()
    {
        Start("entry");
    }
    public override void Update(float delta)
    {
        if(CurrentState == null) return;
        CurrentState.Update(delta);
    }
    public override void PhysicsUpdate(float delta)
    {
        if(CurrentState == null) return;
        CurrentState.PhysicsUpdate(delta);
    }
    public override void Exit()
    {
    if(machine == null)
    {
        CurrentState = null;
    }
    else
    {
        base.Exit();
    }
    }

    // API
    public void Transition(string current)
    {
        if(CurrentState == null) return;
        if(CurrentState is ExitState)
        {
            CurrentState.OnExit();
            Exit();
            return;
        }
        if(!Name2State.ContainsKey(current))
        {
            return;
        }
        if(TransitionMap == null)
        {
            return;
        }
        if(!TransitionMap.ContainsKey(current))
        {
            return;
        }
        foreach(AssertTo edge in TransitionMap[current])
        {
            if(edge.condition.Invoke())
            {
                // 添加信号
                CurrentState.OnExit();
                CurrentState = Name2State[edge.nextState] as IState;
                CurrentState.OnEnter();
                return;
            }
        }
    }
    public void Start(string name)
    {
        if(Name2State.ContainsKey(name))
        {
            CurrentState = Name2State[name] as IState;
            CurrentState.OnEnter();
        }
    }

}
```
### Demo
```c#
// 状态定义
public class Idle : StateBase
{
    public override void OnEnter()
    {
        PlayAnimation("Idle");
    }
    public override void PhysicsUpdate(float delta)
    {
        if(InputCache != null)
        {
            Exit();
        }
    }
}
public class Run : StateBase
{
    public override void OnEnter()
    {
        PlayAnimation("Run");
    }
    public override void PhysicsUpdate(float delta)
    {
        Position += velocity * delta;
    }
}
public class PlayerStateMachine : StateMachineBase
{
    // 绑定状态名和状态类型
    EntryState entry = new EntryState();
    Run run = new Run();
    Idle idle = new Idle();
    ExitState exit = new ExitState();

    // 定义转换 
    [Transition("entry", "idle")]
    public bool transition_1() { return true;  }

    [Transition("idle", "run")]
    public bool transition_2()
    {
        if(horizontalInput != 0)
        {
            return true;
        }
        return false;
    }

    [Transition("run", "idle")]
    public bool transition_3()
    {
        if(agent.IsOnGround() && velocity == agent.GetGroundVelocity())
        {
            return true;
        }
        return false;
    }
}
```


这样写就把状态 和 状态转移 分离开了，另外还可以实现一些编辑器工具来生成状态机的文件，以及为指定节点绑定状态机以简化操作

Todo:
每个状态机绑定的对象类型不固定，但是状态机继承应该与 agent 对象继承具有相似的层级，所以每写一个新的状态机就强制转换agent类型过于繁琐，尝试找到优化方法！


