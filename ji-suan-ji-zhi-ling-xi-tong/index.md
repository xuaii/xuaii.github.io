# 计算机指令系统

## 指令

计算机指令是机器码，是汇编码，是硬件软件交互界面；指令长度是CPU指令长度（例如：32
位/64位）
常规的指令形式是 OP + Address * n;
指令字长和机器字长没有固定关系，通常有半指令字长，单指令字长，双指令字长，此外指令长度可以是固定的也可以是不定的，通常来说定长指令更易于处理

1. 零地址指令
   
   * 不需要操作数的指令(空操作，停机，关中断指令)，
   * 堆栈计算机指令

2. 一地址指令
   
   * $OP(A_1)\rightarrow A_1$(自增,自减,求反求补)
   * $(ACC)OP(A-1)\rightarrow ACC$, 隐含约定地址， 计算寻址范围

3. 二地址指令
   算术指令，逻辑运算指令
   指令含义 $(A_1)OP(A_2)\rightarrow A_1$

4. 三地址指令
   $(A_1)OP(A_2)\rightarrow A_3$, 如果地址均为主存地址， 完成该类指令需要四次访存(取指令 * 1， 取操作数*2， 存放结果*1)

5. 四地址指令
   最后一个地址 $A_4$ 是下一条将要执行的指令

## 指令扩展

虽然现代处理器采用大都采用定长指令，但是为了在此基础上保留指令的丰富类别，采用可变长操作码的定长指令；这意味着操作码位数长度不变，且分散在指令的不同位置，这将增加指令的译码难度，增加控制器复杂度。

## 指令操作类型

**数据传送**

1. 寄存器之间的传送(MOV)
2. 从内存单元读取数据到CPU寄存器(LOAD)
3. 从CPU寄存器写数据到内存单元(STORE)

**算术和逻辑**
加(ADD)、 减(SUB)、 乘(MUL)、 除(DIV)、 比较(CMP)、 加1(INC)、 减1(DEC)、 与(AND)、或(OR)、 取反(NOT)、 亦或(XOR)

**移位操作**
算法移位、逻辑移位、循环移位

**转移操作**

1. 无条件转移(JMP)
2. 条件转移(BRANCH)
3. 调用(CALL)
4. 返回(RET)
5. 陷阱(TRAP)

**输入输出操作**
完成 CPU 与外部设备交换数据或传送控制命令及状态信息

## 指令寻址方式

 '' **寻址方式**是指寻找指令或指令操作数有效地址的方式 ''
 寻址方式分为**指令寻址**和**数据寻址**两类

 指令中数据字段代表的不是真实地址而是**形式地址**， 形式地址结合**寻址方式**可以计算真实地址
 如果 A 表示**寄存器编号**或者**内存地址**， 那么 (A) 表示其中保存的值

### 指令寻址

1. 顺序寻址: 程序计数器PC++，自动形成下一条指令地址
2. 跳跃寻址:
   跳跃寻址的地址依然是由 PC 寄存器指出的， 指令跳跃收到 **状态寄存器** 和 **操作数**的控制，跳跃地址可以是直接由标记符得到，也可以由当前指令偏移得到；即 **绝对地址** 和 **相对地址**

### 数据寻址

本质是如何在指令中 **表示** 操作数的地址，如何用这种表示计算出 **操作数地址**

1. 隐含寻址
   例如单地址指令中隐含的地址一样，操作数地址隐含在操作码的定义里

2. 立即数寻址
   补码表示操作数直接在指令中给出，例如 ADD 1 1
    **优点** 是过程中不用访存
    **缺点** 立即数长度限制了立即数范围

3. 直接寻址
   直接在指令中指出**操作数地址**的真实值
    **优点** 是简单仅访问一次内存
    **缺点** 是指令长度决定了地址的上限，操作数的地址修改不容易！

4. 间接寻址
   指令地址给出**操作数地址**所在**储存单元的地址**， 也就是需要一次间接寻址，间接寻址也可以是多次间接寻址，通过主存中取得的字的**第一位**的取值判断是否是取得操作数地址：
   
   * 取 0， 表示当前地址是操作数地址
   * 取 1， 表示当前地址是操作数间址
     **优点** 扩大寻址范围，方便子程序返回
     **缺点** 多次访存， 一般扩大寻址地址的方法是寄存器间接寻址

5. 寄存器寻址
   直接在指令字中给出操作数所在寄存器编号
   
    **优点**  不访问主存，执行速度快， 支持向量矩阵运算
    **缺点** 寄存器价格贵，数量有限

6. 寄存器间接寻址
   在指令字中给出寄存器编号，该寄存器内存储操作数的地址
    **优点** 速度更快，用于扩展寻址范围
    **缺点** 只能一次间址，需要访问主存

7. 相对寻址
   相对寻址是相对于 PC 上加上指令偏移量 A 而形成的有效地址，A 可正可负，补码表示
    **优点** 操作数地址不固定，广泛用于**转移指令**
    **缺点** A 的位数决定寻址范围
    **注意** PC 寄存器取下一条指令后自增，然后再加上偏移量

8. 基址寻址
   实际地址 EA = (BR) + A，即基址 + 形式地址。基址寄存器是**面向操作系统**的，主要解决程序逻辑空间与储存器物理空间无关
    **优点** 扩大寻址范围，利于多道程序设计，利于编写非线性程序（浮动程序），用于各种页表，段表实现
    **缺点** 形式地址位数较短

9. 变址寻址
   变址寄存器寻址是 形式地址 A + 变址寄存器 IX，这里的变址寄存器区别于基址寄存器，是**面向用户**的，其中 IX 的位数足以表示整个存储空间
   **优点** 扩大寻址范围，便于编制循环程序，用于实现数组，因为指令中的 A 是**固定的**， 而IX是可以由用户设定的
   **缺点** A 不可变？硬找借口？ 

10. 堆栈寻址
    堆栈是存储器(或专用寄存器)中一块特定的，后进先出的(LIFO) 原则管理的存储区，该存储区中 读/写单元的地址使用一个特定**寄存器SP**给出的，硬堆栈/软堆栈。
    **硬堆栈** 是寄存器堆栈，速度快，成本高，不适合做大容量堆栈
    **软堆栈** 是主存中划分的一段区域，速度稍慢，成本低，适合大容量

| 寻址方式      | 有效地址      | 访存次数 |
|:---------:|:---------:|:----:|
| 隐含寻址      | 程序指定      | 0    |
| 立即寻址      | A是操作数     | 0    |
| 直接寻址      | EA=A      | 1    |
| 一次间接寻址    | EA=(A)    | 2    |
| 寄存器寻址     | EA=R_i    | 0    |
| 寄存器一次间接寻址 | EA=(R_i)  | 1    |
| 相对寻址      | EA=(PC)+A | 1    |
| 基址寻址      | EA=(BR)+A | 1    |
| 变址寻址      | EA=(IX)+A | 1    |

## 程序的机器码表示 - 汇编

x86 处理器中包含8个32位通用寄存器
| 寄存器名称 | 位数 | 说明 |
| :----:| :----: | :----: |
| EAX | 32 | 累加器(Accumulator) |
| EBX | 32 | 基址寄存器(Base Register) |
| ECX | 32 | 计数寄存器(Count Register) |
| EDX | 32 | 数据寄存器(Data Register) |
| ESI | 32 | 变址寄存器(Index Register) |
| EDI | 32 | 变址寄存器(Index Register) |
| EBP | 32 | 堆栈基指针(Base Pointer) |
| ESP | 32 | 堆栈顶指针(Stack Pointer) |

**说明** 前四个寄存器 EAX, EBX, ECX, EDX 都是32位寄存器，但是可以仅仅访问其中的半字例如：($EAX = AH + AL$), 除EBP和ESP外，其余寄存器的使用是比较任意的。

x64 汇编增加了几个寄存器和引入新的指令格式，在此仅学习 x86汇编。

### 环境配置

vscode + MASM/TASM 插件
测试代码如下：

```nasm
DATA SEGMENT
    MESG DB "This is an Assembly Language Programe.", 0DH, 0AH, "$"
    SPACE DB " ", "$"
DATA ENDS
CODE SEGMENT
    ASSUME CS:CODE, DS:DATA
    START:
        MOV AX, DATA
        MOV DS, AX              ; 将 DATA 段的段首址存入 DS
        MOV BX, 01H             ; BX 初始值设为 1
    AGAIN:
        MOV DX, OFFSET MESG     ; 取欲显示的字符偏移量赋给 DX
        MOV AH, 09H             ; 调用 9号(显示)DOS功能子程序
        INT 21H
        MOV CX, BX              ; 将 BX 的值赋给 CX
        INC BX
    NEXT:
        MOV DX, OFFSET SPACE    ; 取空格字符偏移量赋给 DX
        MOV AH, 09H
        INT 21H
        LOOP NEXT               ; 继续显示空格字符，直到 CX 为 0
        CMP BX, 10              ; BX 与 10 比较
        JBE AGAIN               ; 没显示 10 次，转移到 AGAIN 继续执行程序
        MOV AH, 4CH
        INT 21H                 ; 返回 DOS
CODE ENDS
END START
```

右键 运行/调试

### 汇编指令格式

AT&T格式和Intel格式区别如下
| 项目/区别 | AT&T | Intel |
| :----:| :----: | :----: |
| 大小写 | 小写 | 大小写不敏感 |
| 操作数 | 第一个为源操作数,第二个为目的操作数 | 第一个位目的操作数，第一个为源操作数 |
| 前缀 | 寄存器前缀%，立即数前缀$ | 不需要前缀 |
| 寻址 | 使用"("和")" | 使用"[" 和 "]" |
| 复杂寻址 | disp(base, index, scale), 表示偏移量，基址寄存器， 变址寄存器，比例因子;如 8(%edx, %eax, 2) | [edx+eax*2+8] |
| 数据长度 | 在操作码后边一个字符表示操作数大小 b->byte, w->word, l->long | 显式的注明byte ptr, word ptr, dword ptr |

**注意** 由于32位或64位体系结构都是由16位扩展而来，因此用 word(字) 表示16位 

### 常用机器指令

汇编指令通常可以分为 **数据传送指令**， **逻辑计算指令**， **控制流指令**
以下以** Intel 格式**为例介绍重要指令:
**约定** 
\<reg32\> 表示eax,ebx,edx,....
\<reg16\> 表示ax, bx, dx
\<reg8\>   表示ah, al, bh,bl,...
\<mem\>  表示内存地址，如\[eax\]、\[var+4\]、dword ptr \[eax+ebx\]
\<con\>   表示8位、16位或32位常数

**mov 指令**
将 **第二个操作数**(寄存器，内存，常数内容) 复制到 **第一个操作数**(寄存器或内存)，不能用于内存到内存的复制

```nasm
mov     <reg>,<reg>
mov     <reg>,<mem>
mov     <mem>,<reg>
mov     <reg>,<con>
mov     <mem>,<con>
```

**push 指令**
将操作数压入内存的栈，常用于函数调用。ESP 是栈顶指针，压栈前 ESP-=4，栈增长方向与内存地址增长方向相反，然后将操作数压入 ESP 指示的地址

```nasm
push     <reg32>
push     <mem>
push     <con32>
// 栈元素固定 32 位 eg:
push     eax
push     [var]
```

**pop 指令**
将栈顶元素送出，pop指令将 ESP 地址内容出栈，再将 ESP 值加 4

```nasm
pop     edi
pop     [ebx]
```

**add/sub 指令**
将两个操作数相加/减，结果保存到第一个操作数中

```nasm
add     <reg>,<reg> / sub <reg>,<reg>
add     <reg>,<mem> / sub <reg>,<mem>
add     <mem>,<reg> / sub <mem>,<reg>
add     <reg>,<con> / sub <reg>,<con>
add     <mem>,<con> / sub <mem>,<con>
```

**inc/dec 指令**
操作数自增自减指令

```nasm
inc     <reg> / dec <reg>
inc     <mem> / dec <mem>
// eg:
dec eax
inc     dword ptr [var]
```

**imul 指令**
带符号整数乘法指令，有两种形式 1. 两个操作数相加结果保存到**第一个操作数** 2. 三个操作数后两个数相加结果保存到**第一个操作数**

```nasm
imul     <reg32>,<reg32>
imul     <reg32>,<mem>
imul     <reg32>,<reg32>,<con>
imul     <reg32>,<mem>,<con>
// tag: 这里显然不能存储到 <mem> 所以该过程最多只有一次非读指令访存
//        如果乘法操作可能溢出，则编译器溢出标志 OF = 1，是CPU调溢出处理程序
```

**idiv 指令**
带符号整数除法指令，他只有一个操作数，而被除数为 edx:eax中的内容（是拼接的64位整数），操作结果有两部分：商和余数，商->eax, 余数->edx

```nasm
// (edx:eax) / op_1 = edx:eax
idiv     <reg32>
idiv     <mem>
// eg:
idiv     ebx
idiv     dword ptr [var]
```

**and/or/xor 指令**
结果放在第一个操作数里 

```nasm
and     <reg>,<reg> / or <reg>,<reg> / xor <reg>,<reg>
and     <reg>,<mem> / or <reg>,<mem> / xor <reg>,<mem>
and     <mem>,<reg> / or <mem>,<reg> / xor <mem>,<reg>
and     <reg>,<con> / or <reg>,<con> / xor <reg>,<con>
and     <mem>,<con> / or <mem>,<con> / xor <mem>,<con>
// eg:
and     eax, 0fH
xor     edx, edx
```

**not 指令**
位反转指令，将操作数每一位反转

```nasm
not     <reg>
not     <mem>
// example:
not     byte ptr [var]
```

**neg 指令**
取负指令

```nasm
neg     <reg>
neg     <mem>
// eg:
neg     eax
```

**shl/shr 指令**
逻辑移位指令， l 为左， r 为右，第一个操作数是被移位，第二个操作数是移位位数

```nasm
shl     <reg>,<con8> / shr <reg>,<con8>
shl     <mem>,<con8> / shr <mem>,<con8>
shl     <reg>,<cl> / shr <reg>,<cl>
shl     <mem>,<cl> / shr <mem>,<cl>
// cl 是 8 位寄存器
// eg：
shl     eax, 1
shr     ebx cl
```

**标签**
IP 寄存器是 CPU 硬件结构，其值是不能直接指定的，只能通过控制流指令来更改，在 x86 汇编中使用标签来标记代码地址

```nasm
            movesi, [ebp+8]
begin:      xor ecx, ecx
            mov eax, [esi]
```

**jump 指令**
直接跳转类似 goto 语句

```nasm
jmp <label>
// eg:
jmp begin
```

**j\<condition\> 指令**
条件转移指令

```nasm
je      <label> (jump when equal)
jne     <label> (jump when not equal)
jz      <label> (jump when last result was zero)
jg      <label> (jump when greater than)
jge     <label> (jump when greater than or equal to)
jl      <label> (jump when less than)
jle     <label> (jump when less than or equal to)
// eg:
cmp eax, ebx
jle done
// tips:这里的跳转指令并不依赖某个寄存器的值，而是依赖 由cmp和test指令确定的 CPU 的状态值
```

**cmp/test 指令**
cmp 用于比较两个操作数的值，test 对两个数进行逐位与运算，他们不保存操作结果，仅根据运算结果设置 CPU 状态字中的条件码

```nasm
cmp     <reg>,<reg> / test <reg>,<reg>
cmp     <mem>,<reg> / test <mem>,<reg>
cmp     <reg>,<mem> / test <reg>,<mem>
cmp     <reg>,<con> / test <reg>,<con>
// tips:cmp,test和jcondotion指令搭配使用，举例：
cmp dword ptr [var], 10     // 比较 4 字节
jne loop                              // 相等继续执行，否则跳转到loop 
test eax, eax                       // 测试eax 是否为0
jz xxxx                                 // 为0则设置标志位 ZF 位1，跳转到 xxxx 处执行                               
```

**call/ret 指令**
用于实现子程序（过程，函数等）的调用和返回， call 指令将当前执行的指令地址**入栈**，然后**无条件转移**到有标签指示的指令。call指令还会保存调用之前的信息。 ret 实现了子程序的返回机制，ret 指令**弹出栈中保存的指令地址**，然后无条件**转移到保存的指令地址**。

```nasm
call <label>
ret
```

### 过程调用机器表示

使用 call/ret 指令实现过程调用，假定过程**P**(调用者) 和 过程**Q**(被调用者)，过程调用步骤如下：

* P 将入口参数放在 Q 能访问到的地方
* P 将返回地址存到特定的地方，然后将控制权转移到 Q (IP寄存器指向)
* Q 保存 P 的现场(**通用寄存器**的内容)，并为自己的**非静态局部变量**分配空间
* 执行过程 Q
* Q 回复 P 的现场，将结果放到 P 能访问到的地方，并释放局部变量所占空间
* Q 取出返回地址，将控制转移到 P

**注意   ：** 上述步骤中需要为**入口参数**，**返回地址**，**过程P现场**，**过程Q局部变量**，**返回结果** 找到存放空间，而用户可见的寄存器数量是有限的，因此需要在内存中用一个栈来存放数据，EAX、ECX、EDX是**调用者保存寄存器**，其保存和恢复过程由P负责。EBX、ESI、EDI 是**被调用则保存寄存器**，每一个过程都有其自己的栈区，成为**栈帧(Stack Frame)**。寄存器 **EBP指示栈底**，**ESP 指向栈顶**，栈从高地址向低地址增长。

**例如**

```c
// 被调用函数/过程
int add(int x, int y){
    return x + y;
}
// 调用者过程
int caller(){
    int temp1 = 125;
    int temp2 = 80;
    int sum = add(temp1, temp2);
    return sum;
}
```

经过 GCC 编译后 caller 过程的汇编码如下 ：

```nasm
caller:
    # stage 1
    pushl       %ebp
    movl        %esp, %ebp        # 系统栈 && 用户栈
                                  # ebp 和 esp 是 用户栈 指针
                                  # 而 push 指令是对系统栈的操作
                                  # 1. 将当前用户栈基址压入系统栈
                                  # 2. 新用户栈基址设为栈顶指针

    subl        $24, %esp         # 为当前过程开辟 24B 内存空间(栈帧)
                                  # 此时 esp = ebp - 24，所以栈顶指针更小
                                  # 这里 只用了 28B，是因为 GCC 规定栈帧必须是16B的整数倍

    # stage 2 定义变量
    movl        $125, -12(%ebp)   # 这里的变量是由 esp 向 ebp 方向依次放置
    movl        $80, -8(%ebp)

    # stage 3 传入函数参数
    movl        -8(%ebp), %eax  
    mov         %eax, 4(%esp)     # temp2 先入栈
    movl        -12(%ebp), %eax    
    movl        %eax, (%esp)      # temp1 后入栈

    # stage 4 调用函数
    call        add

    # stage 5 取出函数返回值
    movl        %eax, -4(%ebp)    # 将返回值取出到变量 sum
    movl        -4(%ebp), %eax    # 将 sum 作为返回值
                                  # 返回值是通过 eax 寄存器返回的，所以E(A|C|D)X寄存器是由调用者返回的

    # stage 6 离开 add 函数
    leave                         # 等价于：
                                  # movl     %ebp, %esp
                                  # popl     %ebp

    # stage 7 退出当前 caller 过程
    ret
```

add 过程对应过程的汇编码如下

```nasm
    push       %ebp
    mov        %esp, %ebp
    mov        0xc(%ebp), %eax          # 将参数2放入寄存器
                                        # 这里的栈帧是有高位向低位
                                        # 函数参数是从右往左入栈

    mov        0x8(%ebp), %edx          # 将参数1放入寄存器

    lea        (%edx, %eax, 1), %eax    # 将add_1传入add_2
                                        # 这里是直接通过地址计算
                                        # 使用 lea 指令是因为不占用 ALU，比较快
                                        # 使用单指令代替了多条指令
    pop        %ebp
    ret
```

**Question**：传入参数时是将参数的临时变量拷贝到了 4(%esp) 和 (%esp)，但是为什么在 add 过程中取参数时是取的 0xc(%ebp) 和 0x8(%ebp)？？

**Answer**： 因为 call 指令会改变 esp 寄存器的值，会将函数返回地址入栈，然后 esp += 4

### 选择语句机器表示

以 c 语言为例，选择语句主要是 **if-then**, **if-then-else**, **switch**, 编译器条件码(标志位)设置指令各类转移指令来实现选择结构语句

1. **条件码（标志位）**
   
   条件码是CPU维护的状态寄存器，他们描述算术逻辑操作的**属性**，可以检测这些寄存器来执行分支指令：
   
   **CF** 进(借)位标志，适用于最近的无符号整数加减运算，有进(借)位 CF = 1
   
   **ZF** 零标志，标志最近操作结果是否为0
   
   **SF** 符号标志，标志最近的带符号数运算结果，负 :SF = 1
   
   **OF** 溢出标志，标志最近带符号数运算结果是否溢出
   
   由于OF，SF对无符号数无意义， CF对带符号数无意义，常见的算术逻辑运算指令都会设置条件码

2. **IF 语句**
   
   通用形式如下
   
   ```c
   if(test_expr)
       then_statement
   else
       else_statement
   ```
   
   翻译为 goto 语句形式
   
   ```c
   t = test_expr;
   if(!t)
       goto false;
   then_statement
   goto done;
   
   false:
       else_statement
   done:
   ```
   
   对于下面的 c 语言函数
   
   ```c
   int get_count(int *p1, int *p2)
   {
       if(p1 > p2)
           return *p2;
       else
           return *p1;
   }
   ```
   
   将得到以下汇编码
   
   ```nasm
   movl     8(%ebp), %eax
   movl     12(%ebp), %edx
   cmpl     %edx %eax
   jbe      .L1
   movl     (%edx), %eax
   jmp      .L2
   .L1:
   movl     (%eax), %eax
   .L2:
   ```

3. **SWITCH 语句**
   
   对于 **小范围** 和 **小量** 的选择语句，使用 **跳转表**，对于**大范围**或者**大量**的选择语句编译器还是会和 **if-then-else** 一样的方式来处理
   
   跳转表即：
   
   ```nasm
   .L12:
       .long    .L3
       .long    .L4
       .long    .L5
       .long    .L6
       .long    .L7
       .long    .L8
   
   ```

这样就可以通过 相对于 .L12 的**偏移量**来**计算真实的跳转目的地了**，这样的跳转只需一个指令来找到跳转目的，而不是多层的 if-else 嵌套，有点字典内味了。这也解释了为什么switch-case语句需要break来退出case，因为所有的case在**地址上是连续**的。

### 循环语句机器表示

常规的循环语句 有**while**,**for**, **do-while** 大多数编译器都将上述三种形式的循环语句转化为**do-while** 语句

**DO-WHILE 循环**

```c
do
    body_statement
    while(text_expr)
```

goto 语句形式

```c
loop:
    body_statement
    t = test_expr;
    if(t)
        goto loop;
```

**WHILE循环**

```c
while(test_expr)
    body_statement
```

在第一次执行循环体之前先执行一次

```c
t = test_expr;
if(!t)
    goto done;
do
    body_statement
    while(text_expr);
done:
```

```c
t = test_expr
if(!t)
    goto done;
loop:
    body_statement
    t = test_expr;
    if(t)
        goto loop;
done:
```

**FOR 循环**

一般形式

```c
for(init_expr; test_expr; update_expr)
    body_statement
```

转化为 while 循环

```c
init_expr;
while(test_expr){
    body_statement
    update_expr;
}
```

转化为 goto 语句

```c
init_expr;
t = test_expr;
if(!t)
    goto done;

loop:
    body_statement
    update_expr;
    t = test_expr;
    if(t)
        goto loop;
done:   
```

```c
int nsum_for(int n){
    int i;
    int result = 0;
    for(i = 1; i<=n; i++)
        result += i;
    return result;
}
==== 编译 ====>

movl        8(%ebp), %ecx
movl        $0, %eax
movl        $1, %edx
cmp         %edx, %ecx
jg          .L2
.L1:
addl        %edx, %eax
addl        $1, %edx
cmpl        %ecx, %edx
jle         .L1
.L2:
```

直接按照模板即可


