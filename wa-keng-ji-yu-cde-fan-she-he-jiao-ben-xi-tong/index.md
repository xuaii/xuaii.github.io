# [挖坑] 反射和脚本系统


c++ 反射实现起来好像很麻烦，所以就干脆学习和借鉴已有的方案[CPP-Reflection-code](https://github.com/AustinBrunkHorst/CPP-Reflection)

## 环境搭建
1. 安装 LLVM
    * windows 下直接下载 .exe 版本安装即可 [LLVM](https://releases.llvm.org/download.html)
2. 安装 Boost
    * 下载 && 解压 压缩包 [Boost](https://www.boost.org/users/download/)
    * 打开 x64 Native Tools Command Prompt for VS2022 切换到解压目录 && 运行 `bootstrap.bat`
    * 按输出提示 操作
3. 将 LLVM-ROOT 和 BOOST-ROOT 添加到环境变量 && 重启生效
4. 按照 [README](https://github.com/AustinBrunkhorst/CPP-Reflection/README.md) 编译创建 vs 项目(vs2022 生成器参数`"Visual Studio 17 2022"`) 并且编译 
5. 打开 Examples 项目，选择一个测试项目作为启动项，并且运行测试！

*tips 1*:更换`CPP-REFLECTION` 项目的中 `json11.cpp` 和 `json11.hpp` [json11](https://gitee.com/ClickHouse-Build/json11/blob/master/json11.cpp)， 不明白有代码有bug为什么能跑起来？
*tips2*:使用 Runtime 需要将 `Source\Runtime\Common` 和 `Source\Common\Lib` 目录添加到**附加包含目录**, 需要添加到 `Runtime` 的引用
## CPP-REFLECTION 目录结构
### Parser
1. 各种语言元素类型（类，构造函数，枚举，External？，字段Field，函数，Global，Invokeable， Method），这是用于 **解释AST节点** 和  **构造代码渲染数据** 类型。
2. 模块 Module（是一个代码集合，其中包含 classes ， globals， globalfuncs，enums
3. 一个词法分析器？意义不明
4. lang-c前端语法树模型
    1. CursorType AST语法节点类型（例如 定义 声明 表达式 等
    2. MetaDataConfig 配置文件
    3. MetaDataManager 类型 用于 GetProperty GetFlag GetNativeString 
    4. MetaUtils 工具类，处理字符串等C++不好处理的内特容
    5. NameSpace 列表 用于处理嵌套名字空间
    6. 预编译 -> include 列表
    7. ReflectionOptions 反射参数
    8. ReflectionParser 用于分析代码 提取需要反射的类，最后渲染反射代码
    9. 渲染模板路径 常量
### Runtime
Todo

### 使用方法
1. 自定义属性
```c++
enum class SliderType
{
    Horizontal,
    Vertical
} Meta(Enable);

struct Slider : ursine::meta::MetaProperty
{
    META_OBJECT;

    SliderType type;

    Slider(SliderType type)
        : type(type) { }
} Meta(Enable);

struct Range : ursine::meta::MetaProperty
{
    META_OBJECT;

    float min, max;

    Range(float min, float max)
        : min(min)
        , max(max) { }
} Meta(Enable);
```

2. 自定义被反射类
```c++
#pragma once

#include <Meta.h>

#include "TestProperties.h"

#include <string>
#include <vector>

#include <Array.h>

enum TestEnum
{
    One,
    Two,
    Three,
    Four,
    Five,
    Eighty = 80
} Meta(Enable);

struct SoundEffect
{
    Meta(Range(0.0f, 100.0f), Slider(SliderType::Horizontal))
    float volume;

    void Load(const std::string &filename)
    {
        std::cout << "Loaded sound effect \"" << filename << "\"." << std::endl;
    }
} Meta(Enable);

struct ComplexType
{
    std::string stringValue;
    int intValue;
    float floatValue;
    double doubleValue;
    
    SoundEffect soundEffect;

    ursine::Array<int> arrayValue;

    TestEnum enumValue;

    ComplexType(void) = default;
} Meta(Enable);
```
3. 调用方法
```c++
#include "TestReflectionModule.h"

#include "TestTypes.h"
#include "TestProperties.h"

#include "TypeCreator.h"

using namespace ursine::meta;

int main(void)
{
    MetaInitialize( UsingModule( TestModule ) );

    Type soundEffectType = typeof( SoundEffect );
    Field volumeField = soundEffectType.GetField( "volume" );

    // the runtime supports overloading, but by default returns the first overload
    Method loadMethod = soundEffectType.GetMethod( "Load" );

    // creates an instance of a sound effect
    Variant effect = TypeCreator::Create( soundEffectType );

    // effect.volume is now 85
    volumeField.SetValue( effect, 85.0f );

    // 85 -- can also use GetValue<float>( )
    float volumeValue = volumeField.GetValue( effect ).ToFloat( );

    std::cout << "SoundEffect.volume: " << volumeValue << std::endl;

    // effect.Load is called
    loadMethod.Invoke( effect, std::string { "Explosion.wav" } );

    return 0;
}
```

## Reference
[CPP-Reflection-code](https://github.com/AustinBrunkHorst/CPP-Reflection)
[CPP-Reflection-doc](https://austinbrunkhorst.com/cpp-reflection-part-1/)


