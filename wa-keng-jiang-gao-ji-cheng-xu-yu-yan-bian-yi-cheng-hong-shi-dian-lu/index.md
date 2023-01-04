# [挖坑] 将高级语言编译为红石电路


既然我的世界是图灵完备的，那么一定可以将c/c++等静态语言编译成红石电路，只用提供一些基础的模块就能（输入输出，各种库）编译出确定的红石电路。

1. 要通过代码控制我的世界，就要有 Minecraft 的API接口
   [Mine Player API](https://github.com/PrismarineJS/mineflayer) 用于创建游戏机器人，通过对机器人的操作可以按照编译出的指令执行创建/设置方块，并且可以在游戏中录制建造过程，API功能如下：
    * Entity knowledge and tracking.
    * Block knowledge. You can query the world around you. Milliseconds to find any block.
    * Physics and movement - handle all bounding boxes
    * Attacking entities and using vehicles.
    * Inventory management.
    * Crafting, chests, dispensers, enchantment tables.
    * Digging and building.
    * Miscellaneous stuff such as knowing your health and whether it is raining.
    * Activating blocks and using items.
    * Chat.
2. 使用 [ANTLR4](https://www.bilibili.com/read/cv17459807?spm_id_from=333.999.0.0) 将目标语言翻译成 AST
3. 解析 AST 并生成 **机器人行动指令**  (RAC)
4. 根据行动指令在我的世界中创建机器

要实现上述功能需要以下模块：
1. 机器人控制器
    * 该模块使用 Mine Player API 控制机器人操作方块
    * 当接收到某个指令，需要指挥机器人行动
    * 维护一个 世界对象，需要计算 创建/销毁 的位置
2. AST  解析器
    * 遍历 AST 生成指令（这里的指令是位置无关的，例如创建一个变量，但是不需要知道在什么位置创建，需要 机器人控制器来计算合理的位置） 
3. 设计一个合适的 **中间语言(RAC)** 来表示机器人行为（位置无关）


