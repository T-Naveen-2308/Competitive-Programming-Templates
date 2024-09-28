# Author : Naveen

# Program Start
# Libraries Start
from math import *
from collections import deque
from copy import deepcopy
import heapq
import sys
# Libraries End

# -----------------------------------------------------------------

# Constants Start
mod1 = 1000000007
mod2 = 998244353
max_if = 1000000
inf = float("inf")
neg_inf = float("-inf")
# Constants End

# -----------------------------------------------------------------

# Input Start
# Normal Input Start
inp_int = lambda: int(input())
inp_float = lambda: float(input())
# Normal Input End

# Map Input Start
map_int = lambda: map(int, input().split())
map_float = lambda: map(float, input().split())
# Map Input End

# List Input Start
list_char = lambda: list(input())
list_str = lambda: input().split()
list_int = lambda: list(map(int, input().split()))
list_float = lambda: list(map(float, input().split()))
# List Input End
# Input End

# -----------------------------------------------------------------

# Basic Functions Start
# Square Root Start
sqrt_int = lambda n: int(sqrt(n))

def sqrt_int_u(n):
    k = int(sqrt(n))
    return k if k * k == n else k + 1
# Square Root End

# Dictionary Computation Start
def dict_com(l):
    d = {}
    for i in l:
        if i not in d:
            d[i] = 0
        d[i] += 1
    return d
# Dictionary Computation End

# Prime Number Start
def is_prime(n, p=0):
    if n<=0:
        raise ValueError("The given number can't be negative.")
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    if p:
        for i in range(5, int(n**0.5) + 1, 6):
            if n % i == 0 or n % (i + 2) == 0:
                return False
        return True
    else:
        d = n - 1
        while d % 2 == 0:
            d //= 2
        def miller(a):
            if a % n == 0:
                return True
            c = d
            x = pow(a, c, n)
            if x == 1 or x == n - 1:
                return True
            while c < n - 1:
                x = x * x % n
                if x == n - 1:
                    return True
                c <<= 1
            return False
        bases64 = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
        bases32 = [2, 7, 61]
        bases = bases32 if n <= 4294967296 else bases64
        for base in bases:
            if not miller(base):
                return False
        return True

def primes(prime, size=max_if):
    prime_is = [True] * (size + 1)
    prime_is[0] = False
    prime_is[1] = False
    for i in range(2, sqrt_int_u(size) + 1):
        if prime_is[i]:
            for j in range(i * i, size + 1, i):
                prime_is[j] = False
    for i in range(2, size + 1):
        if prime_is[i]:
            prime.append(i)
# Prime Number End

# KMP Algorithm Start
def substr_is_in(text, pattern):
    m = len(pattern)
    n = len(text)
    lps = [0] * m
    while i < m:
        if pattern[i] == pattern[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            if len:
                len = lps[len - 1]
            else:
                lps[i] = 0
                i += 1
    i = 0
    j = 0
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            return i - j
        elif i < n and pattern[j] != text[i]:
            if j:
                j = lps[j - 1]
            else:
                i += 1
    return -1
# KMP Algorithm End
# Basic Functions End

# -----------------------------------------------------------------

# Modular Arithmetic Start
# Modular Inverse Function Start
def mod_inv(n, mod=mod1, p=0):
    if n<=0:
        raise ValueError("The number can't be zero or negative.")
    if p:
        t1, t2, r1, r2 = 0, 1, mod, n
        while r2:
            q = r1 // r2
            t1, t2 = t2, t1 - q * t2
            r1, r2 = r2, r1 - q * r2
        if r1 > 1:
            raise ValueError("Modular inverse does not exist.")
        return t1 % mod
    else:
        return pow(n, mod - 2, mod)
# Modular Inverse Function End

# Permutations and Combinations Start
def fact(n, mod=0):
    if n<0:
        ValueError("The number can't be negative.")
    ans = 1
    if mod == 0:
        for i in range(1, n + 1):
            ans = ans * i
    else:
        for i in range(1, n + 1):
            ans = ans * i % mod
    return ans

def fact_lis(lis, mod=0):
    lis[0] = 1
    if mod == 0:
        for i in range(1, len(lis)):
            lis[i] = i * lis[i - 1]
    else:
        for i in range(1, len(lis)):
            lis[i] = i * lis[i - 1] % mod


def perm(n, r, mod=0):
    if n < 0 or r < 0:
        ValueError("The number can't be negative.")
    ans = 1
    if mod == 0:
        for i in range(n - r + 1, n + 1):
            ans *= i
    else:
        for i in range(n - r + 1, n + 1):
            ans = ans * i % mod
    return ans


def comb(n, r, mod=0, p=0):
    if n < 0 or r < 0:
        ValueError("The number can't be negative.")
    num = 1
    den = 1
    if r > n // 2:
        r = n - r
    n += 1
    if mod == 0:
        for i in range(1, r + 1):
            num *= n - i
            den *= i
        ans = num // den
    else:
        for i in range(1, r + 1):
            num = num * (n - i) % mod
            den = den * i % mod
        ans = num * mod_inv(den, mod, p) % mod
    return ans


def fast_fib(n, mod=0):
    if n < 0:
        ValueError("The number can't be negative.")
    if mod==0:
        a0, a1 = 0, 1
        str = bin(n)[2:]
        for i in str:
            f2 = (a0 * (2 * a1 - a0))
            f21 = (a0**2 + a1**2)
            if i == "1":
                a0, a1 = f21, f2 + f21
            else:
                a0, a1 = f2, f21
        return a0
    else:
        a0, a1 = 0, 1
        str = bin(n)[2:]
        for i in str:
            f2 = (a0 * (2 * a1 - a0)) % mod
            f21 = (a0**2 + a1**2) % mod
            if i == "1":
                a0, a1 = f21, (f2 + f21) % mod
            else:
                a0, a1 = f2, f21
        return a0 % mod
# Permutations and Combinations End
# Modular Inverse End

# Data Structures Start
# Disjoint Set Union Start
class DisjointSet:
    def __init__(self, n, start=1, ds=None):
        if ds is not None:
            self.num_sets = ds.num_sets
            self.max_size = ds.max_size
            self.parent = list(ds.parent)
            self.min_set = list(ds.min_set)
            self.max_set = list(ds.max_set)
            self.depth = list(ds.depth)
            self.set_size = list(ds.set_size)
        else:
            self.num_sets, self.max_size = n, 1
            n += start
            self.parent, self.min_set, self.max_set = [0] * n, [0] * n, [0] * n
            for i in range(start, n):
                self.parent[i] = self.min_set[i] = self.max_set[i] = i
            self.depth, self.set_size = [0] * n, [1] * n

    def find_set(self, n, recur=True):
        if recur:
            if self.parent[n] == n:
                return n
            self.parent[n] = self.find_set(self.parent[n])
            return self.parent[n]
        else:
            st = []
            while n != self.parent[n]:
                st.append(n)
                n = self.parent[n]
            while len(st) > 0:
                v = st.pop()
                self.parent[v] = n
            return n

    def is_same_set(self, a, b):
        return self.find_set(a) == self.find_set(b)

    def union_set(self, a, b):
        if self.is_same_set(a, b):
            return
        x, y = self.find_set(a), self.find_set(b)
        if self.depth[x] > self.depth[y]:
            x, y = y, x
        if self.depth[x] == self.depth[y]:
            self.depth[y] += 1
        self.parent[x] = y
        self.set_size[y] += self.set_size[x]
        self.max_size = max(self.max_size, self.set_size[y])
        self.min_set[y] = min(self.min_set[y], self.min_set[x])
        self.max_set[y] = min(self.max_set[y], self.max_set[x])
        self.num_sets -= 1

    def num_of_sets(self):
        return self.num_sets

    def size_of_set(self, n):
        return self.set_size[self.find_set(n)]

    def min_of_set(self, n):
        return self.min_set[self.find_set(n)]

    def max_of_set(self, n):
        return self.max_set[self.find_set(n)]

    def max_size_of_sets(self):
        return self.max_size
# Disjoint Set Union End

# Segment Tree Start
class SegmentTree:
    def build_tree(self, arr, tree_index, left, right):
        if left == right:
            self.tree[tree_index] = arr[left]
            return
        mid = (left + right) // 2
        self.build_tree(arr, 2 * tree_index + 1, left, mid)
        self.build_tree(arr, 2 * tree_index + 2, mid + 1, right)
        self.tree[tree_index] = self.func(
            self.tree[2 * tree_index + 1], self.tree[2 * tree_index + 2]
        )

    def __init__(self, arr, fun):
        self.size = len(arr)
        self.func = fun
        self.tree = [0] * (4 * self.size)
        self.build_tree(arr, 0, 0, self.size - 1)

    def query(self, tree_index, left, right):
        if self.query_left <= left and right <= self.query_right:
            return self.tree[tree_index]
        mid = (left + right) // 2
        if mid < self.query_left or left > self.query_right:
            return self.query(2 * tree_index + 2, mid + 1, right)
        if right < self.query_left or mid + 1 > self.query_right:
            return self.query(2 * tree_index + 1, left, mid)
        return self.func(
            self.query(2 * tree_index + 1, left, mid),
            self.query(2 * tree_index + 2, mid + 1, right),
        )

    def update(self, tree_index, left, right):
        if left == right:
            self.tree[tree_index] = self.update_new_value
            return
        mid = (left + right) // 2
        if self.update_index <= mid:
            self.update(2 * tree_index + 1, left, mid)
        else:
            self.update(2 * tree_index + 2, mid + 1, right)
        self.tree[tree_index] = self.func(self.tree[2 * tree_index + 1], self.tree[2 * tree_index + 2])

    def query(self, left, right):
        if left > right or left < 0 or right >= self.size:
            raise ValueError("Given query range is invalid or out of range.")
        self.query_left = left
        self.query_right = right
        return self.query(0, 0, self.size - 1)

    def update(self, index, new_value):
        if index < 0 or index >= self.size:
            raise ValueError("Given update index is out of range.")
        self.update_index = index
        self.update_new_value = new_value
        self.update(0, 0, self.size - 1)
# Segment Tree End

# Graphs Start
# Unweighted Graphs Start
class UnweightedGraph:
    def __init__(self, a_list, start=1):
        self.a_list = a_list
        self.start = start
        self.end = len(a_list) + start

    def dfs(self, src, iter=True):
        self.visited = [False] * self.end
        self.parent = [-1] * self.end
        self.pre = [-1] * self.end
        self.post = [-1] * self.end
        count = 0
        if iter:
            stack = [src]
            while len(stack):
                p = stack[-1]
                if not self.visited[p]:
                    self.visited[p] = True
                    self.pre[p] = count
                    count += 1
                flag = True
                for i in self.a_list[p]:
                    if not self.visited[i]:
                        self.parent[i] = p
                        stack.append(i)
                        flag = False
                        break
                if flag:
                    self.post[p] = count
                    count += 1
                    stack.pop()
        else:
            def dfs_com(v):
                nonlocal count
                self.visited[v] = True
                self.pre[v] = count
                count += 1
                for i in self.a_list[v]:
                    if not self.visited[i]:
                        self.parent[i] = v
                        dfs_com(i)
                self.post[v] = count
                count += 1
            dfs_com(src)

    def bfs(self, src):
        self.level = [-1] * self.end
        self.parent = [-1] * self.end
        self.level[src] = 0
        queue = deque([src])
        while len(queue):
            v = queue.popleft()
            for i in self.a_list[v]:
                if self.level[i] == -1:
                    self.level[i] = self.level[v] + 1
                    self.parent[i] = v
                    queue.append(i)

    def components(self):
        self.component = [-1] * self.end
        seen = self.start
        self.num_of_comp = 0
        while seen < self.end:
            src = -1
            for i in range(self.start, self.end):
                if self.component[i] == -1:
                    src = i
                    break
            self.bfs(src)
            for i in range(self.start, self.end):
                if self.level[i] != -1:
                    self.component[i] = self.num_of_comp
                    seen += 1
            self.num_of_comp += 1

    def topological_order(self):
        in_degree = [0] * self.end
        self.topo_order = []
        self.path_len = [0] * self.end
        for i in self.a_list:
            for j in self.a_list[i]:
                in_degree[j] += 1
        queue = deque()
        for i in in_degree:
            if in_degree[i] == 0:
                queue.append(i)
        while len(queue):
            i = queue.popleft()
            self.topo_order.append(i)
            in_degree[i] = -1
            for j in self.a_list[i]:
                in_degree[j] -= 1
                self.path_len[j] = max(self.path_len[j], self.path_len[i] + 1)
                if in_degree[j] == 0:
                    queue.append(j)
# Unweighted Graphs End

# Weighted Graphs Start
class WeightedGraph:
    def __init__(self, w_list, start=1):
        self.w_list = w_list
        self.start = start
        self.end = len(w_list) + start

    def dijkstra(self, src):
        visited = [False] * self.end
        self.distance = [inf] * self.end
        self.distance[src] = 0
        heap = [(0, src)]
        while len(heap) > 0:
            nextv = heapq.heappop(heap)[1]
            if visited[nextv]:
                continue
            visited[nextv] = True
            for v, d in self.w_list[nextv]:
                if not visited[v] and self.distance[nextv] + d < self.distance[v]:
                    self.distance[v] = self.distance[nextv] + d
                    heapq.heappush(heap, (self.distance[v], v))

    def bellman_ford(self, src):
        self.distance = [inf] * self.end
        self.distance[src] = 0
        for _ in range(self.end - self.start - 1):
            for u in range(self.start, self.end):
                for v, d in self.w_list[u]:
                    self.distance[v] = min(self.distance[v], self.distance[u] + d)
        for u in range(self.start, self.end):
            for v, d in self.w_list[u]:
                if self.distance[u] + d < self.distance[v]:
                    raise Exception("The graph has negative cycles.")

    def floyd_warshall(self):
        self.distance_fw = [[inf] * self.end for _ in range(self.end)]
        for u in range(self.start, self.end):
            for v, d in self.w_list[u]:
                self.distance_fw[u][v] = d
        for i in range(self.start, self.send):
            for j in range(self.start, self.end):
                for k in range(self.start, self.end):
                    self.distance_fw[j][k] = min(self.distance_fw[j][k], self.distance_fw[j][i] + self.distance_fw[i][k])

    def prim(self):
        visited = [False] * self.end
        self.distance = [inf] * self.end
        self.nbr = [-1] * self.end
        self.distance[self.start] = 0
        heap = []
        heapq.heappush(heap, (0, self.start))
        while len(heap) > 0:
            nextv = heapq.heappop(heap)[1]
            if visited[nextv]:
                continue
            visited[nextv] = True
            for v, d in self.w_list[nextv]:
                if not visited[v] and d < self.distance[v]:
                    self.distance[v], self.nbr[v] = d, nextv
                    heapq.heappush(heap, (self.distance[v], v))

    def kruskal(self):
        edges = {}
        self.component = DisjointSet(self.end - self.start, self.start)
        self.edge = []
        for u in range(self.start, len(self.w_list) + self.start):
            for v, d in self.w_list[u]:
                edges.add((d, u, v))
        edges = list(edges)
        edges.sort()
        for d, u, v in edges:
            if self.component.is_same_set(u, v):
                continue
            self.edge.append((u, v))
            self.component.union_set(u, v)
# Weighted Graphs End
# Graphs End
# Data Structures End

# -----------------------------------------------------------------

# Solution Class Start
class Solution:

    def main(self, index):
        # -----------------------------------------------------------------

        n = inp_int()
        
        # printf(f'Case #{index}: {ans}')

    # -----------------------------------------------------------------

    test_cases = True

# Solution Class End

# Main Function Start
if __name__ == "__main__":
    sol = Solution()
    input = lambda: sys.stdin.buffer.readline().decode().strip()
    printf = lambda s: sys.stdout.write(s)
    test_case = 1
    if Solution.test_cases:
        test_case = int(input())
    for i in range(1, test_case + 1):
        sol.main(i)
# Main Function End
# Program End
# -----------------------------------------------------------------
