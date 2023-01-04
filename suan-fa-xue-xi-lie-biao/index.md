# 算法学习列表


**参考列表**
[Wialliam Fiset's Algorithms](https://github.com/williamfiset/Algorithms)
[Keon's Algorithms](https://github.com/keon/algorithms)
[Xtaci's Algorithms](https://github.com/xtaci/algorithms)
[Labuladong's Fucking-Algorithms](https://github.com/labuladong/fucking-algorithm)
[Geekxh's Hello-Algorithms](https://github.com/geekxh/hello-algorithm)
[Algorithm-Visualizer](https://github.com/algorithm-visualizer/algorithm-visualizer)
[Justjavac's Free-Programming-Books](https://github.com/justjavac/free-programming-books-zh_CN)
[Imarvinle's Awesome-CS-Books](https://github.com/imarvinle/awesome-cs-books)
[Jeffe's Algorithms Course](http://jeffe.cs.illinois.edu/teaching/algorithms/)

主要关注其中 **操作系统**、**算法**、**图形学**、**设计模式**、**游戏引擎**、**C#**、 **C/C++** 、**编译器中间语言**、**元编程**

[Jeffe's Crouse](http://jeffe.cs.illinois.edu/teaching/algorithms/) 用于**学习**，[Wialliam Fiset's Algorithms](https://github.com/williamfiset/Algorithms), [Keon's Algorithms](https://github.com/keon/algorithms), [Xtaci's Algorithms](https://github.com/xtaci/algorithms) 用于**拓展补充**，[labuladong 的算法小抄](https://github.com/labuladong/fucking-algorithm),  [小浩算法](https://www.geekxh.com/) 用于**练习**

这是一个利用**空闲时间学习**的**长期计划**，或许需要几年时间，学不完没关系，学一点赚一点，加油！
## 数据结构
 - [ ] Balance Trees
    - [ ] AVL Tree (recursive)
    - [ ] Red Black Tree (recursive)
- [ ] Binary Search Tree
- [ ] Splay Tree
- [ ] Dynamic Array
    - [ ] Dynamic Array (integer only, fast)
- [ ] Fenwick Tree
    - [ ] Fenwick Tree (range query, point update)
    - [ ] (range update, point query)
- [ ] Fibonacci Heap
- [ ] Hashtable
    - [ ] Hashtable (double hashing)
    - [ ] Hashtable (linear probing)
    - [ ] Hashtable (quadratic probing)
    - [ ] Hashtable (sparate chaning)
- [ ] Linked List
- [ ] Priority Queue
    - [ ] Min Binary Heap
    - [ ] Min Indexed Binary Heap (sorted key-value pairs, similar to hash-table)
    - [ ] Min D-Heap
    - [ ] Min Indexed D-Heap (sorted key-value pairs, similar to hash table)
- [ ] Queue
    - [ ] Queue (integer only, fixed size, fast)
    - [ ] Queue (linked list, generic)
- [ ] Segment Tree
    - [ ] Segement Tree (array based, compact)
    - [ ] Segment Tree (linked list generic)
- [ ] Sparse Table
- [ ] Stack
    - [ ] Stack (integer only, fixed size, fast)
    - [ ] Stack (linked list, generic)
    - [ ] Stack (array, generic)
- [ ] Suffix Array
    - [ ] Suffix Array (O(n^2logn))
    - [ ] Suffix Array (O(nlog^2(n))
    - [ ] Suffix Array (O(nlogn))
- [ ] Trie
- [ ] Union Find
## 数组
[Array shuffle](https://github.com/xtaci/algorithms/blob/master/include/shuffle.h)
[2D Array](https://github.com/xtaci/algorithms/blob/master/include/2darray.h
)

## 动态编程
- [ ] Coin change problem
- [ ] Edit distance (iterative)
- [ ] Edit distance (recursive)
- [ ] Knapsack 0/1
- [ ] Knapsack unbounded (0/inf)
- [ ] Maximum contiguous subarray
- [ ] Longest Common Subsequence (LCS) 
- [ ] Longest Increasing Subsequence (LIS) 
- [ ] Longest Palindrome Subsequence (LPS)
- [ ]  Traveling Salesman Problem (dynamic programming, iterative)
- [ ]  Traveling Salesman Problem (dynamic programming, recursive)
- [ ]  Minimum Weight Perfect Matching (iterative, complete graph) 

Examples:
### Adhoc
- [ ]  Magic Cows
- [ ]  Narrow Art Gallery
### Tiling problems
- [ ] Tiling Dominoes
- [ ] Tiling Dominoes and Trominoes
- [ ] Mountain Scenes
## 计算几何
- [ ] Angle between 2D vectors - O(1)
- [ ] Angle between 3D vectors - O(1)
- [ ] Circle-circle intersection point(s) - O(1)
- [ ] Circle-line intersection point(s) - O(1)
- [ ] Circle-line segment intersection point(s) - O(1)
- [ ] Circle-point tangent line(s) - O(1)
- [ ] Closest pair of points (line sweeping algorithm) - O(nlog(n))
- [ ] Collinear points test (are three 2D points on the same line) - O(1)
- [ ] Convex hull (Graham Scan algorithm) - O(nlog(n))
- [ ] Convex hull (Monotone chain algorithm) - O(nlog(n))
- [ ] Convex polygon area - O(n)
- [ ] Convex polygon cut - O(n)
- [ ] Convex polygon contains points - O(log(n))
- [ ] Coplanar points test (are four 3D points on the same plane) - O(1)
- [ ] Line class (handy infinite line class) - O(1)
- [ ] Line-circle intersection point(s) - O(1)
- [ ] Line segment-circle intersection point(s) - O(1)
- [ ] Line segment to general form (ax + by = c) - O(1)
- [ ] Line segment-line segment intersection - O(1)
- [ ] Longitude-Latitude geographic distance - O(1)
- [ ] Point is inside triangle check - O(1)
- [ ] Point rotation about point - O(1)
- [ ] Triangle area algorithms - O(1)
- [ ] [UNTESTED] Circle-circle intersection area - O(1)
- [ ] [UNTESTED] Circular segment area - O(1)
## 图论
**Tree algorithms**
- [ ] 🎥 Rooting an undirected tree - O(V+E)
- [ ] 🎥 Identifying isomorphic trees - O(?)
- [ ] 🎥 Tree center(s) - O(V+E)
- [ ] Tree diameter - O(V+E)
- [ ] 🎥 Lowest Common Ancestor (LCA, Euler tour) - O(1) queries, O(nlogn) preprocessing
**Network flow**
- [ ] Bipartite graph verification (adjacency list) - O(V+E)
- [ ] 🎥 Max flow & Min cut (Ford-Fulkerson with DFS, adjacency list) - O(fE)
- [ ] Max flow & Min cut (Ford-Fulkerson with DFS, adjacency matrix) - O(fV2)
- [ ] 🎥 Max flow & Min cut (Edmonds-Karp, adjacency list) - O(VE2)
- [ ] 🎥 Max flow & Min cut (Capacity scaling, adjacency list) - O(E2log2(U))
- [ ] 🎥 Max flow & Min cut (Dinic's, adjacency list) - O(EV2) or O(E√V) for
- [ ] bipartite graphs
- [ ] Maximum Cardinality Bipartite Matching (augmenting path algorithm, adjacency list) - O(VE)
- [ ] Min Cost Max Flow (Bellman-Ford, adjacency list) - O(E2V2)
- [ ] Min Cost Max Flow (Johnson's algorithm, adjacency list) - O(E2Vlog(V))
**Main graph theory algorithms**
- [ ] Articulation points/cut vertices (adjacency list) - O(V+E)
- [ ] Bellman-Ford (edge list, negative cycles, fast & optimized) - O(VE)
- [ ] 🎥 Bellman-Ford (adjacency list, negative cycles) - O(VE)
- [ ] Bellman-Ford (adjacency matrix, negative cycles) - O(V3)
- [ ] 🎥 Breadth first search (adjacency list) - O(V+E)
- [ ] Breadth first search (adjacency list, fast queue) - O(V+E)
- [ ] Bridges/cut edges (adjacency list) - O(V+E)
- [ ] Find connected components (adjacency list, union find) - O(Elog(E))
- [ ] Find connected components (adjacency list, DFS) - O(V+E)
- [ ] Depth first search (adjacency list, iterative) - O(V+E)
- [ ] Depth first search (adjacency list, iterative, fast stack) - O(V+E)
- [ ] 🎥 Depth first search (adjacency list, recursive) - O(V+E)
- [ ] 🎥 Dijkstra's shortest path (adjacency list, lazy implementation) - O(Elog(V))
- [ ] 🎥 Dijkstra's shortest path (adjacency list, eager implementation + D-ary heap) - O(ElogE/V(V))
- [ ] 🎥 Eulerian Path (directed edges) - O(E+V)
- [ ] 🎥 Floyd Warshall algorithm (adjacency matrix, negative cycle check) - O(V3)
- [ ] Graph diameter (adjacency list) - O(VE)
- [ ] 🎥 Kahn's algorithm (topological sort, adjacency list) - O(E+V)
- [ ] Kruskal's min spanning tree algorithm (edge list, union find) - O(Elog(E))
- [ ] 🎥 Kruskal's min spanning tree algorithm (edge list, union find, lazy sorting) - O(Elog(E))
- [ ] Kosaraju's strongly connected components algorithm (adjacency list) - O(V+E)
- [ ] 🎥 Prim's min spanning tree algorithm (lazy version, adjacency list) - O(Elog(E))
- [ ] Prim's min spanning tree algorithm (lazy version, adjacency matrix) - O(V2)
- [ ] 🎥 Prim's min spanning tree algorithm (eager version, adjacency list) - O(Elog(V))
- [ ] Steiner tree (minimum spanning tree generalization) - O(V3 + V2 _ 2T + V _ 3T)
- [ ] 🎥 Tarjan's strongly connected components algorithm (adjacency list) - O(V+E)
- [ ] 🎥 Topological sort (acyclic graph, adjacency list) - O(V+E)
- [ ] Topological sort (acyclic graph, adjacency matrix) - O(V2)
- [ ] Traveling Salesman Problem (brute force) - O(n!)
- [ ] 🎥 Traveling Salesman Problem (dynamic programming, iterative) - O(n22n)
- [ ] Traveling Salesman Problem (dynamic programming, recursive) - O(n22n)
## 线性代数
Freivald's algorithm (matrix multiplication verification) - O(kn2)
Gaussian elimination (solve system of linear equations) - O(cr2)
Gaussian elimination (modular version, prime finite field) - O(cr2)
Linear recurrence solver (finds nth term in a recurrence relation) - O(m3log(n))
Matrix determinant (Laplace/cofactor expansion) - O((n+2)!)
Matrix inverse - O(n3)
Matrix multiplication - O(n3)
Matrix power - O(n3log(p))
Square matrix rotation - O(n2)
## 数学
[Arbitrary Integer](https://github.com/xtaci/algorithms/blob/master/include/integer.h)
[UNTESTED] Chinese remainder theorem
Prime number sieve (sieve of Eratosthenes) - O(nlog(log(n)))
Prime number sieve (sieve of Eratosthenes, compressed) - O(nlog(log(n)))
[Prime test(trial division)](https://github.com/xtaci/algorithms/blob/master/include/prime.h)
[Prime test(Miller-Rabin's method)](https://github.com/xtaci/algorithms/blob/master/include/prime.h)
Totient function (phi function, relatively prime number count) - O(n1/4)
Totient function using sieve (phi function, relatively prime number count) - O(nlog(log(n)))
Extended euclidean algorithm - ~O(log(a + b))
Greatest Common Divisor (GCD) - ~O(log(a + b))
Fast Fourier transform (quick polynomial multiplication) - O(nlog(n))
Fast Fourier transform (quick polynomial multiplication, complex numbers) - O(nlog(n))
Primality check - O(√n)
Primality check (Rabin-Miller) - O(k)
Least Common Multiple (LCM) - ~O(log(a + b))
Modular inverse - ~O(log(a + b))
Prime factorization (pollard rho) - O(n1/4)
Relatively prime check (coprimality check) - ~O(log(a + b))
## 搜索算法
Binary search (real numbers) - O(log(n))
Interpolation search (discrete discrete) - O(n) or O(log(log(n))) with uniform input
Ternary search (real numbers) - O(log(n))
Ternary search (discrete numbers) - O(log(n))
## 排序算法
Bubble sort - O(n2)
Bucket sort - Θ(n + k)
Counting sort - O(n + k)
Heapsort - O(nlog(n))
Insertion sort - O(n2)
Mergesort - O(nlog(n))
Quicksort (in-place, Hoare partitioning) - Θ(nlog(n))
Quicksort3 (Dutch National Flag algorithm) - Θ(nlog(n))
Selection sort - O(n2)
Radix sort - O(n*w)
## 字符串算法
Booth's algorithm (finds lexicographically smallest string rotation) - O(n)
Knuth-Morris-Pratt algorithm (finds pattern matches in text) - O(n+m)
Longest Common Prefix (LCP) array - O(nlog(n)) bounded by SA construction, otherwise O(n)
🎥 Longest Common Substring (LCS) - O(nlog(n)) bounded by SA construction, otherwise O(n)
🎥 Longest Repeated Substring (LRS) - O(nlog(n))
Manacher's algorithm (finds all palindromes in text) - O(n)
Rabin-Karp algorithm (finds pattern match positions in text) - O(n+m)
Substring verification with suffix array - O(nlog(n)) SA construction and O(mlog(n)) per query
## 其他
Bit manipulations - O(1)
List permutations - O(n!)
🎥 Power set (set of all subsets) - O(2n)
Set combinations - O(n choose r)
Set combinations with repetition - O((n+r-1) choose r)
Sliding Window Minimum/Maximum - O(1)
Square Root Decomposition - O(1) point updates, O(√n) range queries
Unique set combinations - O(n choose r)
Lazy Range Adder - O(1) range updates, O(n) to finalize all updates
[Linear congruential generator](https://github.com/xtaci/algorithms/blob/master/include/random.h)
[Maximum subarray problem](https://github.com/xtaci/algorithms/blob/master/include/max_subarray.h)
[Bit-Set](https://github.com/xtaci/algorithms/blob/master/include/bitset.h)
[Double linked list](https://github.com/xtaci/algorithms/blob/master/include/double_linked_list.h)
[Skip list](https://github.com/xtaci/algorithms/blob/master/include/skiplist.h)
[Dynamic order statistics](https://github.com/xtaci/algorithms/blob/master/include/dos_tree.h)
[Interval tree](https://github.com/xtaci/algorithms/blob/master/include/interval_tree.h)
[Prefix Tree(Trie)](https://github.com/xtaci/algorithms/blob/master/include/trie.h)
[Suffix Tree](https://github.com/xtaci/algorithms/blob/master/include/suffix_tree.h)
[B-Tree](https://github.com/xtaci/algorithms/blob/master/include/btree.h)
#### hash
[Hash by multiplication](https://github.com/xtaci/algorithms/blob/master/include/hash_multi.h)
[Hash table](https://github.com/xtaci/algorithms/blob/master/include/hash_table.h)
[Universal hash function](https://github.com/xtaci/algorithms/blob/master/include/universal_hash.h)
[Perfect hash](https://github.com/xtaci/algorithms/blob/master/include/perfect_hash.h)
[Java's string hash](https://github.com/xtaci/algorithms/blob/master/include/hash_string.h)
[FNV-1a string hash](https://github.com/xtaci/algorithms/blob/master/include/hash_string.h)
[SimHash](https://github.com/xtaci/algorithms/blob/master/include/simhash.h)

[Bloom Filter](https://github.com/xtaci/algorithms/blob/master/include/bloom_filter.h)
#### 密码学
[SHA-1 Message Digest Algorithm](https://github.com/xtaci/algorithms/blob/master/include/sha1.h)
[MD5](https://github.com/xtaci/algorithms/blob/master/include/md5.h)
[Base64](https://github.com/xtaci/algorithms/blob/master/include/base64.h)

[Push–Relabel algorithm](https://github.com/xtaci/algorithms/blob/master/include/relabel_to_front.h)
[Huffman Coding](https://github.com/xtaci/algorithms/blob/master/include/huffman.h)
[Word segementation](https://github.com/xtaci/algorithms/blob/master/include/word_seg.h)
[A* algorithm](https://github.com/xtaci/algorithms/blob/master/include/astar.h)
[K-Means](https://github.com/xtaci/algorithms/blob/master/include/k-means.h)
[Knuth–Morris–Pratt algorithm](https://github.com/xtaci/algorithms/blob/master/include/kmp.h)
[Disjoint-Set](https://github.com/xtaci/algorithms/blob/master/include/disjoint-set.h)
[8-Queen Problem](https://github.com/xtaci/algorithms/blob/master/include/8queen.h)



