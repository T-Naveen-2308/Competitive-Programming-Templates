// Author : Naveen

// Program Start

// Constants Start
const mod1 = 1000000007;
const mod2 = 998244353;
const maxIf = 1000000;
const inf = Number.MAX_SAFE_INTEGER;
const negInf = Number.MIN_SAFE_INTEGER;
const genInf = Number.POSITIVE_INFINITY;
const genNegInf = Number.NEGATIVE_INFINITY;
// Constants End

// -----------------------------------------------------------------

// Input and Output Start
// Normal Input Start
let inpInx = 0;
const input = require("fs").readFileSync(0, "utf-8").trim().split("\n");
const inpInt = () => parseInt(input[inpInx++]);
const inpFloat = () => parseFloat(input[inpInx++]);
// Normal Input End

// Map Input Start
const mapInt = () =>
    input[inpInx++]
        .trim()
        .split(" ")
        .map((s) => parseInt(s));
const mapFloat = () =>
    input[inpInx++]
        .trim()
        .split(" ")
        .map((s) => parseFloat(s));
// Map Input End

// List Input Start
const listChar = () => input[inpInx++].trim().split();
const listStr = () => input[inpInx++].trim().split(" ");
const listInt = () =>
    input[inpInx++]
        .trim()
        .split(" ")
        .map((s) => parseInt(s));
const listFloat = () =>
    input[inpInx++]
        .trim()
        .split(" ")
        .map((s) => parseFloat(s));
// List Input End

// Output Start
const print = console.log;
// Output End
// Input and Output End

// -----------------------------------------------------------------

// Basic Functions Start
// Destructing Math Start
const { max, min, sqrt, pow, floor } = Math;
// Destructing Math End

// Min Function Start
const minArr = (arr) => arr.reduce((prev, curr) => min(prev, curr), inf);
// Min Function End

// Max Function Start
const maxArr = (arr) => arr.reduce((prev, curr) => max(prev, curr), negInf);
// Max Function End

// Sum Function Start
const sum = (...numbers) => numbers.reduce((total, num) => total + num, 0);
const sumArr = (arr) => arr.reduce((total, num) => total + num, 0);
// Sum Function End

// Square Root Start
const sqrtInt = (n) => floor(sqrt(n));

const sqrtIntU = (n) => {
    const k = floor(sqrt(n));
    return k * k === n ? k : k + 1;
};
// Square Root End

// Dictionary Computation Start
const dictCom = (l) => {
    const d = {};
    for (let i of l) {
        if (!(i in d)) {
            d[i] = 0;
        }
        d[i] += 1;
    }
    return d;
};
// Dictionary Computation End

// Prime Number Start
const isPrime = (n, p = false) => {
    if (n < 0) {
        throw new Error("The number can't be negative.");
    }
    if (n <= 1) {
        return false;
    }
    if (n <= 3) {
        return true;
    }
    if (n % 2 === 0 || n % 3 === 0) {
        return false;
    }
    if (p) {
        for (let i = 5; i <= floor(sqrt(n)) + 1; i += 6) {
            if (n % i === 0 || n % (i + 2) === 0) {
                return false;
            }
        }
        return true;
    } else {
        let d = n - 1;
        while (d % 2 === 0) {
            d = floor(d / 2);
        }
        const miller = (a) => {
            if (a % n === 0) {
                return true;
            }
            let c = d;
            let x = pow(a, c) % n;
            if (x === 1 || x === n - 1) {
                return true;
            }
            while (c < n - 1) {
                x = (x * x) % n;
                if (x === n - 1) {
                    return true;
                }
                c <<= 1;
            }
            return false;
        };
        const bases64 = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];
        const bases32 = [2, 7, 61];
        const bases = n <= 4294967296 ? bases32 : bases64;
        for (let base of bases) {
            if (!miller(base)) {
                return false;
            }
        }
        return true;
    }
};

const primes = (prime, size = maxIf) => {
    const primeIs = Array(size + 1).fill(true);
    primeIs[0] = false;
    primeIs[1] = false;
    for (let i = 2; i <= sqrtIntU(size); ++i) {
        if (primeIs[i]) {
            for (let j = i * i; j <= size; j += i) {
                primeIs[j] = false;
            }
        }
    }
    for (let i = 2; i <= size; ++i) {
        if (primeIs[i]) {
            prime.push(i);
        }
    }
};
// Prime Number End

// KMP Algorithm Start
const substrIsIn = (text, pattern) => {
    const m = pattern.length;
    const n = text.length;
    const lps = Array(m).fill(0);
    let i = 1;
    let len = 0;
    while (i < m) {
        if (pattern[i] === pattern[len]) {
            lps[i++] = ++len;
        } else {
            if (len) {
                len = lps[len - 1];
            } else {
                lps[i++] = 0;
            }
        }
    }
    i = 0;
    let j = 0;
    while (i < n) {
        if (pattern[j] === text[i]) {
            ++i;
            ++j;
        }
        if (j === m) {
            return i - j;
        } else if (i < n && pattern[j] !== text[i]) {
            if (j) {
                j = lps[j - 1];
            } else {
                ++i;
            }
        }
    }
    return -1;
};
// KMP Algorithm End
// Basic Functions End

// -----------------------------------------------------------------

// Modular Arithmetic Start
// Modular Inverse Function Start
const modInv = (n, mod = mod1, p = false) => {
    if (n < 0) {
        throw new Error("The number can't be negative.");
    }
    if (p) {
        let t1 = 0;
        let t2 = 1;
        let r1 = mod;
        let r2 = n;
        while (r2) {
            const q = floor(r1 / r2);
            [t1, t2] = [t2, t1 - q * t2];
            [r1, r2] = [r2, r1 - q * r2];
        }
        if (r1 > 1) {
            throw new Error("Modular inverse does not exist.");
        }
        return t1 % mod;
    } else {
        return pow(n, mod - 2, mod);
    }
};
// Modular Inverse Function End

// Permutations and Combinations Start
const fact = (n, mod = Number.MAX_SAFE_INTEGER) => {
    if (n < 0) {
        throw new Error("The number can't be negative.");
    }
    let ans = 1;
    for (let i = 1; i <= n; ++i) {
        ans = (ans * i) % mod;
    }
    return ans;
};

const factArr = (lis, mod = Number.MAX_SAFE_INTEGER) => {
    lis[0] = 1;
    for (let i = 1; i < lis.length; ++i) {
        lis[i] = (i * lis[i - 1]) % mod;
    }
};

const perm = (n, r, mod = Number.MAX_SAFE_INTEGER) => {
    if (n < 0 || r < 0) {
        throw new Error("The numbers can't be negative.");
    }
    let ans = 1;
    for (let i = n - r + 1; i <= n; ++i) {
        ans = (ans * i) % mod;
    }
    return ans;
};

const comb = (n, r, mod = Number.MAX_SAFE_INTEGER, p = false) => {
    if (n < 0 || r < 0) {
        throw new Error("The numbers can't be negative.");
    }
    let num = 1;
    let den = 1;
    if (r > floor(n / 2)) {
        r = n - r;
    }
    n += 1;
    for (let i = 1; i <= r; ++i) {
        num = (num * (n - i)) % mod;
        den = (den * i) % mod;
    }
    ans = (num * modInv(den, mod, p)) % mod;
    return ans;
};

const fastFib = (n, mod = Number.MAX_SAFE_INTEGER) => {
    if (n < 0) {
        throw new Error("The number can't be negative.");
    }
    let a0 = 0;
    let a1 = 1;
    const str = n.toString(2);
    for (let i of str) {
        const f2 = (a0 * (2 * a1 - a0)) % mod;
        const f21 = (a0 ** 2 + a1 ** 2) % mod;
        if (i === "1") {
            [a0, a1] = [f21, (f2 + f21) % mod];
        } else {
            [a0, a1] = [f2, f21];
        }
    }
    return a0 % mod;
};
// Permutations and Combinations End
// Modular Inverse End

// Data Structures Start
// Queue Start
class Node {
    constructor(data) {
        this.data = data;
        this.next = null;
    }
}

class Queue {
    constructor() {
        this.front = null;
        this.rear = null;
        this.size = 0;
    }

    enqueue(data) {
        const newNode = new Node(data);
        if (!this.front) {
            this.front = newNode;
        } else {
            this.rear.next = newNode;
        }
        this.rear = newNode;
        this.size++;
    }

    dequeue() {
        if (!this.front) {
            throw new Error("Queue is empty");
        }
        const rn = this.front;
        this.front = this.front.next;
        if (!this.front) {
            this.rear = null;
        }
        this.size--;
        return rn.data;
    }

    peek() {
        if (!this.front) {
            throw new Error("Queue is empty");
        }
        return this.front.data;
    }

    isEmpty() {
        return this.size === 0;
    }

    getSize() {
        return this.size;
    }

    clear() {
        this.front = null;
        this.rear = null;
        this.size = 0;
    }
}
// Queue End

// Priority Queue Start
class PriorityQueue {
    constructor(comparator = (a, b) => a - b, heap = []) {
        this.heap = [];
        this.heapify(heap);
        this.comparator = comparator;
    }

    push(value) {
        this.heap.push(value);
        this.heapifyUp();
    }

    pop() {
        if (this.isEmpty()) {
            throw new Error("Priority queue is empty");
        }
        const minn = this.heap[0];
        const lastVal = this.heap.pop();
        if (!this.isEmpty()) {
            this.heap[0] = lastVal;
            this.heapifyDown();
        }
        return minn;
    }

    top() {
        if (this.isEmpty()) {
            throw new Error("Priority queue is empty");
        }
        return this.heap[0];
    }

    isEmpty() {
        return this.heap.length === 0;
    }

    heapifyUp() {
        let currInx = this.heap.length - 1;
        while (currInx > 0) {
            const prevInx = floor((currInx - 1) / 2);
            if (this.comparator(this.heap[currInx], this.heap[prevInx]) < 0) {
                [this.heap[currInx], this.heap[prevInx]] = [
                    this.heap[prevInx],
                    this.heap[currInx]
                ];
                currInx = prevInx;
            } else {
                break;
            }
        }
    }

    heapifyDown() {
        let currInx = 0;
        const len = this.heap.length;
        while (true) {
            let lci = 2 * currInx + 1;
            let rci = 2 * currInx + 2;
            let sci = currInx;
            if (
                lci < len &&
                this.comparator(this.heap[lci], this.heap[sci]) < 0
            ) {
                sci = lci;
            }
            if (
                rci < len &&
                this.comparator(this.heap[rci], this.heap[sci]) < 0
            ) {
                sci = rci;
            }
            if (sci === currInx) break;
            [this.heap[currInx], this.heap[sci]] = [
                this.heap[sci],
                this.heap[currInx]
            ];
            currInx = sci;
        }
    }

    heapify(arr) {
        this.heap = arr;
        for (let i = floor(this.heap.length / 2) - 1; i >= 0; --i) {
            this.heapifyDownI(i);
        }
    }

    heapifyDownI(index) {
        let currInx = index;
        const len = this.heap.length;
        while (true) {
            let lci = 2 * currInx + 1;
            let rci = 2 * currInx + 2;
            let sci = currInx;
            if (
                lci < len &&
                this.comparator(this.heap[lci], this.heap[sci]) < 0
            ) {
                sci = lci;
            }
            if (
                rci < len &&
                this.comparator(this.heap[rci], this.heap[sci]) < 0
            ) {
                sci = rci;
            }
            if (sci === currInx) break;
            [this.heap[currInx], this.heap[sci]] = [
                this.heap[sci],
                this.heap[currInx]
            ];
            currInx = sci;
        }
    }
}
// Priority Queue End

// Disjoint Set Union Start
class DisjointSet {
    constructor(n, start = 1, ds = null) {
        if (ds) {
            this.numSets = ds.numSets;
            this.maxSize = ds.maxSize;
            this.parent = [...ds.parent];
            this.minSet = [...ds.minSet];
            this.maxSet = [...ds.maxSet];
            this.depth = [...ds.depth];
            this.setSize = [...ds.setSize];
        } else {
            [this.numSets, this.maxSize] = [n, 1];
            n += start;
            this.parent = Array(n).fill(0);
            this.minSet = Array(n).fill(0);
            this.maxSet = Array(n).fill(0);
            for (let i = start; i < n; ++i) {
                this.parent[i] = this.minSet[i] = this.maxSet[i] = i;
            }
            [this.depth, this.setSize] = [Array(n).fill(0), Array(n).fill(1)];
        }
    }

    findSet(n, recur = true) {
        if (recur) {
            if (this.parent[n] === n) {
                return n;
            }
            this.parent[n] = this.findSet(this.parent[n]);
            return this.parent[n];
        } else {
            const st = [];
            while (n !== this.parent[n]) {
                st.push(n);
                n = this.parent[n];
            }
            while (st.length > 0) {
                const v = st.pop();
                this.parent[v] = n;
            }
            return n;
        }
    }

    isSameSet(a, b) {
        return this.findSet(a) === this.findSet(b);
    }

    unionSet(a, b) {
        if (this.isSameSet(a, b)) {
            return;
        }
        let [x, y] = [this.findSet(a), this.findSet(b)];
        if (this.depth[x] > this.depth[y]) {
            [x, y] = [y, x];
        }
        if (this.depth[x] === this.depth[y]) {
            this.depth[y] += 1;
        }
        this.parent[x] = y;
        this.setSize[y] += this.setSize[x];
        this.minSet[y] = min(this.minSet[y], this.minSet[x]);
        this.maxSet[y] = max(this.maxSet[y], this.maxSet[x]);
        this.numSets -= 1;
    }

    numOfSets() {
        return this.numSets;
    }

    sizeOfSet(n) {
        return this.setSize[this.findSet(n)];
    }

    minOfSet(n) {
        return this.minSet[this.findSet(n)];
    }

    maxOfSet(n) {
        return this.maxSet[this.findSet(n)];
    }

    maxSizeOfSets() {
        return this.maxSize;
    }
}
// Disjoint Set Union End

// Segment Tree Start
class SegmentTree {
    buildTree(arr, treeIndex, left, right) {
        if (left === right) {
            this.tree[treeIndex] = arr[left];
            return;
        }
        const mid = floor((left + right) / 2);
        this.buildTree(arr, 2 * treeIndex + 1, left, mid);
        this.buildTree(arr, 2 * treeIndex + 2, mid + 1, right);
        this.tree[treeIndex] = this.func(
            this.tree[2 * treeIndex + 1],
            this.tree[2 * treeIndex + 2]
        );
    }

    constructor(arr, fun) {
        this.size = arr.length;
        this.func = fun;
        this.tree = Array(4 * this.size).fill(0);
        this.buildTree(arr, 0, 0, this.size - 1);
    }

    query(treeIndex, left, right) {
        if (this.queryLeft <= left && right <= this.queryRight) {
            return this.tree[treeIndex];
        }
        const mid = floor((left + right) / 2);
        if (mid < this.queryLeft || left > this.queryRight) {
            return this.query(2 * treeIndex + 2, mid + 1, right);
        }
        if (right < this.queryLeft || mid + 1 > this.queryRight) {
            return this.query(2 * treeIndex + 1, left, mid);
        }
        return this.func(
            this.query(2 * treeIndex + 1, left, mid),
            this.query(2 * treeIndex + 2, mid + 1, right)
        );
    }

    update(treeIndex, left, right) {
        if (left === right) {
            this.tree[treeIndex] = this.updateNewValue;
            return;
        }
        const mid = floor((left + right) / 2);
        if (this.updateIndex <= mid) {
            this.update(2 * treeIndex + 1, left, mid);
        } else {
            this.update(2 * treeIndex + 2, mid + 1, right);
        }
        this.tree[treeIndex] = this.func(
            this.tree[2 * treeIndex + 1],
            this.tree[2 * treeIndex + 2]
        );
    }

    query(left, right) {
        if (left > right || left < 0 || right >= this.size) {
            throw new Error("Given query range is invalid or out of range.");
        }
        this.queryLeft = left;
        this.queryRight = right;
        return this.query(0, 0, this.size - 1);
    }

    update(index, newValue) {
        if (index < 0 || index >= this.size) {
            throw new Error("Given update index is out of range.");
        }
        this.updateIndex = index;
        this.updateNewValue = newValue;
        this.update(0, 0, this.size - 1);
    }
}
// Segment Tree End

// Graphs Start
// Unweighted Graphs Start
class UnweightedGraph {
    constructor(aList, start = 1) {
        this.aList = aList;
        this.start = start;
        this.end = aList.length + start;
    }

    dfs(src, iter = true) {
        this.visited = Array(this.end).fill(false);
        this.parent = Array(this.end).fill(-1);
        this.pre = Array(this.end).fill(-1);
        this.post = Array(this.end).fill(-1);
        let count = 0;
        if (iter) {
            const stack = [src];
            while (stack.length) {
                const p = stack[stack.length - 1];
                if (!this.visited[p]) {
                    this.visited[p] = true;
                    this.pre[p] = count;
                    count += 1;
                }
                let flag = true;
                for (let i of this.aList[p]) {
                    if (!this.visited[i]) {
                        this.parent[i] = p;
                        stack.push(i);
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    this.post[p] = count;
                    count += 1;
                    stack.pop();
                }
            }
        } else {
            const dfsCom = (v) => {
                this.visited[v] = true;
                this.pre[v] = count;
                count += 1;
                for (let i of this.aList[v]) {
                    if (!this.visited[i]) {
                        this.parent[i] = v;
                        dfsCom(i);
                    }
                }
                this.post[v] = count;
                count += 1;
            };
            dfsCom(src);
        }
    }

    bfs(src) {
        this.level = Array(this.end).fill(-1);
        this.parent = Array(this.end).fill(-1);
        this.level[src] = 0;
        const queue = new Queue();
        queue.enqueue(src);
        while (queue.length) {
            const v = queue.dequeue();
            for (let i of this.aList[v]) {
                if (this.level[i] === -1) {
                    this.level[i] = this.level[v] + 1;
                    this.parent[i] = v;
                    queue.enqueue(i);
                }
            }
        }
    }

    components() {
        this.component = Array(this.end).fill(-1);
        let seen = this.start;
        this.numOfComp = 0;
        while (seen < this.end) {
            let src = -1;
            for (let i = this.start; i < this.end; ++i) {
                if (this.component[i] === -1) {
                    src = i;
                    break;
                }
            }
            this.bfs(src);
            for (let i = this.start; i < this.end; ++i) {
                if (this.level[i] !== -1) {
                    this.component[i] = this.numOfComp;
                    seen += 1;
                }
            }
            this.numOfComp += 1;
        }
    }

    topologicalOrder() {
        const inDegree = Array(this.end).fill(0);
        this.topoOrder = [];
        this.pathLen = Array(this.end).fill(0);
        for (let i of this.aList) {
            for (let j of this.aList[i]) {
                inDegree[j] += 1;
            }
        }
        const queue = new Queue();
        for (let i of inDegree) {
            if (inDegree[i] === 0) {
                queue.enqueue(i);
            }
        }
        while (queue.length) {
            const i = queue.dequeue();
            this.topoOrder.enqueue(i);
            inDegree[i] = -1;
            for (let j of this.aList[i]) {
                inDegree[j] -= 1;
                this.pathLen[j] = max(this.pathLen[j], this.pathLen[i] + 1);
                if (inDegree[j] === 0) {
                    queue.enqueue(j);
                }
            }
        }
    }
}
// Unweighted Graphs End

// Weighted Graphs Start
class WeightedGraph {
    constructor(wList, start = 1) {
        this.wList = wList;
        this.start = start;
        this.end = wList.length + start;
    }

    dijkstra(src) {
        const visited = Array(this.end).fill(false);
        this.distance = Array(this.end).fill(inf);
        this.distance[src] = 0;
        const heap = new PriorityQueue();
        heap.push([0, src]);
        while (heap.length) {
            const [_, nextv] = heap.pop();
            if (visited[nextv]) {
                continue;
            }
            visited[nextv] = true;
            for (let [v, d] of this.wList[nextv]) {
                if (
                    !visited[v] &&
                    this.distance[nextv] + d < this.distance[v]
                ) {
                    this.distance[v] = this.distance[nextv] + d;
                    heap.push([this.distance[v], v]);
                }
            }
        }
    }

    bellmanFord(src) {
        this.distance = Array(this.end).fill(inf);
        this.distance[src] = 0;
        for (let _ = 0; _ < this.end - this.start - 1; ++_) {
            for (let u = this.start; u < this.end; ++u) {
                for (let [v, d] of this.wList[u]) {
                    this.distance[v] = min(
                        this.distance[v],
                        this.distance[u] + d
                    );
                }
            }
        }
        for (let u = this.start; u < this.end; ++u) {
            for (let [v, d] of this.wList[u]) {
                if (this.distance[u] + d < this.distance[v]) {
                    throw new Error("The graph has negative cycles.");
                }
            }
        }
    }

    floydWarshall() {
        this.distanceFw = Array(this.end).fill(Array(this.end).fill(inf));
        for (let u = this.start; u < this.end; ++u) {
            for (let [v, d] of this.wList[u]) {
                this.distanceFw[u][v] = d;
            }
        }
        for (let i = this.start; i < this.end; ++i) {
            for (let j = this.start; j < this.end; ++j) {
                for (let k = this.start; k < this.end; ++k) {
                    this.distanceFw[j][k] = min(
                        this.distanceFw[j][k],
                        this.distanceFw[j][i] + this.distanceFw[i][k]
                    );
                }
            }
        }
    }

    prim() {
        const visited = Array(this.end).fill(false);
        this.distance = Array(this.end).fill(inf);
        this.nbr = Array(this.end).fill(-1);
        this.distance[this.start] = 0;
        const heap = new PriorityQueue();
        heap.push([0, this.start]);
        while (heap.length) {
            const [_, nextv] = heap.pop();
            if (visited[nextv]) {
                continue;
            }
            visited[nextv] = true;
            for (let [v, d] of this.wList[nextv]) {
                if (!visited[v] && d < this.distance[v]) {
                    [this.distance[v], this.nbr[v]] = [d, nextv];
                    heap.push([this.distance[v], v]);
                }
            }
        }
    }

    kruskal() {
        const edges = new Set();
        this.component = new DisjointSet(this.end - this.start, this.start);
        this.edge = [];
        for (let u = this.start; u < this.wList.length + this.start; ++u) {
            for (let [v, d] of this.wList[u]) {
                edges.add([d, u, v]);
            }
        }
        const sortedEdges = Array.from(edges).sort();
        for (let [d, u, v] of sortedEdges) {
            if (this.component.isSameSet(u, v)) {
                continue;
            }
            this.edge.push([u, v]);
            this.component.unionSet(u, v);
        }
    }
}
// Weighted Graphs End
// Graphs End
// Data Structures End

// -----------------------------------------------------------------

// Solution Class Start
class Solution {
    main(index) {
        // -----------------------------------------------------------------

        const n = inpInt();

        // print(`Case #index: ans`);
    }

    // -----------------------------------------------------------------

    static testCases = true;
}

// Solution Class End

// Main Function Start
if (require.main === module) {
    const sol = new Solution();
    let test = 1;
    if (Solution.testCases) {
        test = inpInt();
    }
    for (let i = 1; i <= test; ++i) {
        sol.main(i);
    }
}
// Main Function End
// Program End
// -----------------------------------------------------------------
