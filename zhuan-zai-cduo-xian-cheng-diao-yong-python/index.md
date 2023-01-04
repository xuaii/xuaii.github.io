# [转载] c 多线程调用 python


脚本语言是快速编写富有弹性的代码的重要方法之一，在 Unix 系统自动化管理中已经应用了多种脚本语言。现在，在许多应用开发中，也提供了脚本层，这大大方便用户实现通用任务自动处理或者编写应用扩展，许多成功的应用，诸如 GIMP、Emacs、MS Office、PhotoShop、AutoCAD 等都应用了脚本技术。在某种意义上，一切皆可脚本化。

在另一篇文章中，我们已经介绍了如何在 C 应用中嵌入 Python 语言，通过这项技术，可以让应用的高级用户来修改或定制化他们的程序，你可以充分利用 Python 的语言能力而不用自己去实现嵌入语言。Python 是一个不错的的选择，因为它提供了干净直观的 C 语言 API。关于如何在 C 应用中嵌入 Python 解释器，你可以参考：让Python成为嵌入式语言一文。

现在我们来更深入地探讨一些问题。 鉴于许多复杂的应用都会利用多线程技术，本文将着重介绍如何创建线程安全的界面来调用Python解释器。

这里的所有例子都是用 Python 2.7.2，所有的 Python 函数都以extern “C”定义，因此对于 C 和 C++，其使用是别无二致的。
## Python C 和线程

在C程序中创建执行线程是很简单的。在 Linux 中，通常的做法是使用 POSIX 线程（pthread) API 并调用 pthread_create 函数。关于如何使用 pthreads，你可以参考 Felix Garcia 和Javier Fernandez 著的 “POSIX Thread Libraries”一文。为了支持多线程， Python 使用了互斥使访问内部数据结构串行化。这种互斥即 “全局解释器锁 – global interpreter lock”，当某个线程想使用 Python 的C API的时候，它必须获得 全局解释器锁，这避免了会导致解析器状态崩溃的竞争条件（race condition)。

互斥的锁定和释放是通过 PyEval_AcquireLock 和 Eval_ReleaseLock 来描述的。调用了 PyEval_AcquireLock 之后，可以安全地假定你的线程已经持有了锁，其他相关线程不是被阻塞就是在执行与 Python 解析器无关的代码。现在你可以任意调用 Python 函数了。一旦取得了锁，你必须确保调用 PyEval_ReleaseLock 来释放它，否则就会导致线程死锁并冻结其他 Python 线程。

更复杂的情况是，每个运行 Python 的线程维护着自己的状态信息。这些和特定线程相关的数据存储在称为 PyThreadState 的对象中。当在多线程应用中用 C 语言调用 Python API 函数时，你必须维护自己的 PyThreadState 对象以便能安全地执行并发的 Python 代码。

如果你对开发多线程应用相当有经验，你可能会发现全局解释器锁的概念相当不方便。不过，现在它已经不像首次出现时那样糟糕了。当 Python 对脚本进行解释时，它会定期切换出当前 PyThreadState 对象并释放全局解释器锁，从而将控制权释放给其他线程。之前被阻塞的线程可以试图锁定全局解释器锁从而被运行。有些时候，原来的线程会再次获得全局解释器锁再次切回解释器。

这意味着当调用 PyEval_SimpleString 时，即使你持有全局解释器锁，其他线程仍有机会被执行，这样的副作用无可避免。另外，当你调用以 C 语言写就的 Python 模块（包括许多内置模块） 存在着将控制权释放给其他线程的可能性。基于这个原因，当你用两个 C 线程来执行计算密集的 Python 脚本，它们确实能分享 CPU 时间并发运行，但由于全局解释器锁的存在，在多处理器的计算机上，Python 无法通过线程充分计算机的 CPU 处理能力。

## 启用线程支持
在多线程的 C 程序使用 Python API 之前，必须调用一些初始化例程。如果编译解释器库时启用了多线程支持（通常情况如此），你就有了一个是否启用线程的运行时选项。除非你计划使用线程，否则不建议启用该选项。未启用该选项，Python 可以避免因互斥锁定其内部数据结构而产生的系统开销。但是如果你打算用 Python 来扩展多线程应用，你就需要在初始化解释器的时候启用线程支持。我个人建议，应该在主线程执行时就初始化 Python，最好是在应用程序启动的时候，就调用下面两行代码：

```python
// initialize Python
Py_Initialize();
// initialize thread support
PyEval_InitThreads();
``` 

这两个函数都返回 void，所以无需检查错误代码。现在，我们可以假定 Python 解释器已准备好执行 Python 代码。Py_Initialize 分配解释器库使用的全局资源。调用PyEval_InitThreads 则启用运行时线程支持。这导致 Python 启用其内部的互斥锁机制，用于解释器内代码关键部分的系列化访问。此函数的另一个作用是锁定全局解释器锁。该函数完成后，需要由用户负责释放该锁。不过，在释放锁之前, 你应该捕获当前 PyThreadState 对象的指针。后续创建新的 Python 线程以及结束使用 Python 时要正确关闭解释器，都需要用到该对象。下面这段代码用来捕获 PyThreadState 对象指针:

```python
PyThreadState * mainThreadState = NULL;
// save a pointer to the main PyThreadState object
mainThreadState = PyThreadState_Get();
// release the lock
PyEval_ReleaseLock();
```

## 创建新的执行线程

在 Python 里，每个执行 Python 代码的线程都需要一个 PyThreadState 对象。解释器使用此对象来管理每个线程独立的数据空间。理论上，这意味着一个线程中的动作不会牵涉到另一个线程的状态。例如，你在一个线程中抛出异常，其他 Python 代码片段仍会继续运行，就好象什么事情都没有发生一样。你必须帮助 Python 管理每个线程的数据。为此，你需要为每个执行 Python 代码的 C 线程手工创建一个 PyThreadState 对象.要创建 PyThreadState 对象，你需要用到既有的 PyInterpreterState 对象。PyInterpreterState 对象带有为所有参与的线程所共享的信息。当你初始化 Python 时，它就会创建一个 PyInterpreterState 对象，并将它附加在主线程的 PyThreadState 对象上。你可以使用该解释器对象为你自己的 C 现成创建新的 PyThreadState。请参考下面代码

```python
// get the global lock
PyEval_AcquireLock();
// get a reference to the PyInterpreterState
PyInterpreterState * mainInterpreterState = mainThreadState->interp;
// create a thread state object for this thread
PyThreadState * myThreadState = PyThreadState_New(mainInterpreterState);
// free the lock
PyEval_ReleaseLock();
```

## 执行 Python 代码
现在我们已创建 PyThreadState 对象，你的 C 线程就可以开始使用 Python API 执行 Python 脚本。从 C 线程执行 Python 代码时，你必须遵守一些简单的规则。首先，您在进行任何会改变当前线程状态的操作前必须持有全局解释器锁。第二，必须在执行任何 Python 代码之前，必须将该线程特定的 PyThreadState 对象加载到解释器。一旦您已经满足这些条件，您可以通过诸如 PyEval_SimpleString 函数来执行任意的 Python 代码，并记得在执行结束时切出 PyThreadState 对象并释放全局解释器锁。请参考下面代码，注意代码中“锁定、 切换、 执行、 切换，解锁”的对称关系：

```python
// grab the global interpreter lock
PyEval_AcquireLock();
// swap in my thread state
PyThreadState_Swap(myThreadState);
// execute some python code
PyEval_SimpleString("import sys\n");
PyEval_SimpleString("sys.stdout.write(‘Hello from a C thread!\n‘)\n");
// clear the thread state
PyThreadState_Swap(NULL);
// release our hold on the global interpreter
PyEval_ReleaseLock();
```

## 清除线程
一旦你的 C 线程不再需要 Python 解释器，你必须释放相关资源。为此，需要删除该线程的 PyThreadState 对象，相关代码如下：

```python
// grab the lock
PyEval_AcquireLock();
// swap my thread state out of the interpreter
PyThreadState_Swap(NULL);
// clear out any cruft from thread state object
PyThreadState_Clear(myThreadState);
// delete my thread state object
PyThreadState_Delete(myThreadState);
// release the lock
PyEval_ReleaseLock();
```

通过使用 Python API ，这个线程很有效率地完成了上述工作。现在你可以安全地调用 pthread_ext 来结束该线程的运行。
## 关闭解释器
一旦应用不在需要 Python 解释器，你可以用下面的代码将 Python 关闭掉：

```python
// shut down the interpreter
PyEval_AcquireLock();
Py_Finalize();
```

注意：因为 Python 已经被关系，这里就不需要释放锁。请确保在调用 Py_Finalize 之前用 PyThreadState_Clear 和 PyThreadState_Delete 删除掉所有线程状态对象。

## 小结：
作为嵌入式语言，Python 是一个不错的选择。Python 解释器同时支持嵌入和扩展，它允许 C 应用程序代码和嵌入的 Python 脚本之间的双向通信。此外，多线程支持促进了与多线程应用程序的集成，而且不影响性能。

你可以从本文的后面下载有关案例Python embedded HTTP Server (29)，该案例实现了一个内嵌 Python 解释器的多线程 HTTP 服务器。此外我推荐您去 http://www.python.org/docs/api/ 阅读有关的 Python C API 文档。另外 Python 解释器本身的代码也是一个很有价值的参考。


