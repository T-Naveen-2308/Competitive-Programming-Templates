// Author : Naveen

// Program Start
// Libraries Start
import java.util.*;
import java.lang.*;
import java.io.*;
// Libraries End

//----------------------------------------------------------------

// Main Class Start
public class Main {
    // Input Start
    static class Reader {
        final private int BUFFER_SIZE = 1 << 16;
        private DataInputStream din;
        private byte[] buffer;
        private int bufferPointer, bytesRead;

        public Reader() {
            din = new DataInputStream(System.in);
            buffer = new byte[BUFFER_SIZE];
            bufferPointer = bytesRead = 0;
        }

        public Reader(String file_name) throws IOException {
            din = new DataInputStream(new FileInputStream(file_name));
            buffer = new byte[BUFFER_SIZE];
            bufferPointer = bytesRead = 0;
        }

        public String readLine() throws IOException {
            byte[] buf = new byte[64];
            int cnt = 0, c;
            while ((c = read()) != -1) {
                if (c == '\n') {
                    if (cnt != 0) {
                        break;
                    } else {
                        continue;
                    }
                }
                buf[cnt++] = (byte) c;
            }
            return new String(buf, 0, cnt);
        }

        public int nextInt() throws IOException {
            int ret = 0;
            byte c = read();
            while (c <= ' ') {
                c = read();
            }
            boolean neg = (c == '-');
            if (neg)
                c = read();
            do {
                ret = ret * 10 + c - '0';
            } while ((c = read()) >= '0' && c <= '9');

            if (neg) {
                return -ret;
            }
            return ret;
        }

        public long nextLong() throws IOException {
            long ret = 0;
            byte c = read();
            while (c <= ' ')
                c = read();
            boolean neg = (c == '-');
            if (neg)
                c = read();
            do {
                ret = ret * 10 + c - '0';
            } while ((c = read()) >= '0' && c <= '9');
            if (neg) {
                return -ret;
            }
            return ret;
        }

        public double nextDouble() throws IOException {
            double ret = 0, div = 1;
            byte c = read();
            while (c <= ' ')
                c = read();
            boolean neg = (c == '-');
            if (neg) {
                c = read();
            }

            do {
                ret = ret * 10 + c - '0';
            } while ((c = read()) >= '0' && c <= '9');

            if (c == '.') {
                while ((c = read()) >= '0' && c <= '9') {
                    ret += (c - '0') / (div *= 10);
                }
            }

            if (neg)
                return -ret;
            return ret;
        }

        private void fillBuffer() throws IOException {
            bytesRead = din.read(buffer, bufferPointer = 0, BUFFER_SIZE);
            if (bytesRead == -1) {
                buffer[0] = -1;
            }
        }

        private byte read() throws IOException {
            if (bufferPointer == bytesRead) {
                fillBuffer();
            }
            return buffer[bufferPointer++];
        }

        public void close() throws IOException {
            if (din == null) {
                return;
            }
            din.close();
        }
    }
    // Input End

    // ----------------------------------------------------------------

    static class Functions {
        // Constants Start
        static final long mod1 = 1000000007;
        static final long mod2 = 998244353;
        static final long maxIf = 1000000L;
        static final long inf = Long.MAX_VALUE;
        static final long negInf = Long.MIN_VALUE;
        static final char spc = ' ';
        static final char newl = '\n';
        // Constants End

        // Min Function Start
        @SafeVarargs
        static <T extends Comparable<T>> T min(T val, T... values) {
            if (values.length == 0) {
                throw new IllegalArgumentException("No values provided");
            }
            T mini = val;
            for (T value : values) {
                if (value.compareTo(mini) < 0) {
                    mini = value;
                }
            }
            return mini;
        }

        static <T extends Comparable<T>> T min(T[] array) {
            if (array.length == 0) {
                throw new IllegalArgumentException("Empty array");
            }
            T mini = array[0];
            for (T value : array) {
                if (value.compareTo(mini) < 0) {
                    mini = value;
                }
            }
            return mini;
        }
        // Min Function End

        // Max Function Start
        @SafeVarargs
        static <T extends Comparable<T>> T max(T val, T... values) {
            if (values.length == 0) {
                throw new IllegalArgumentException("No values provided");
            }
            T maxi = val;
            for (T value : values) {
                if (value.compareTo(maxi) > 0) {
                    maxi = value;
                }
            }
            return maxi;
        }

        static <T extends Comparable<T>> T max(T[] array) {
            if (array.length == 0) {
                throw new IllegalArgumentException("Empty array");
            }
            T maxi = array[0];
            for (T value : array) {
                if (value.compareTo(maxi) > 0) {
                    maxi = value;
                }
            }
            return maxi;
        }
        // Max Function End

        // Sum Function Start
        @SafeVarargs
        static <T extends Number> long sum(T val, T... values) {
            if (values.length == 0) {
                throw new IllegalArgumentException("No values provided");
            }
            long sumi = (long) val;
            for (T value : values) {
                sumi += (long) value;
            }
            return sumi;
        }

        static <T extends Number> long sum(T[] array) {
            if (array.length == 0) {
                throw new IllegalArgumentException("Empty array");
            }
            long sumi = 0L;
            for (T value : array) {
                sumi += (long) value;
            }
            return sumi;
        }
        // Sum Function End

        // Binary Start
        static <T extends Number> String bin(T number) {
            if (!(number instanceof Integer) && !(number instanceof Long)) {
                throw new IllegalArgumentException("Unsupported number type");
            }
            long value = number.longValue();
            String binStr = Long.toBinaryString(value);
            int firstOne = binStr.indexOf('1');
            if (firstOne != -1) {
                binStr = binStr.substring(firstOne);
            } else {
                binStr = "0";
            }
            return binStr;
        }

        @SuppressWarnings("unchecked")
        static <T extends Number> long int2(String binStr) {
            BitSet bits = new BitSet(64);
            for (int i = 0; i < binStr.length(); ++i) {
                if (binStr.charAt(i) == '1') {
                    bits.set(i + (64 - binStr.length()));
                }
            }
            long dn = 0;
            for (int i = 0; i < 64; ++i) {
                if (bits.get(i)) {
                    dn |= (1L << (63 - i));
                }
            }
            return dn;
        }
        // Binary End

        // Square Root Start
        static long sqrtInt(long n) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            return (long) Math.sqrt(n);
        }

        static long sqrtIntU(long n) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            long k = (long) Math.sqrt(n);
            return (k * k == n) ? k : k + 1;
        }
        // Square Root End

        // Map Computation Start
        static void dictCom(HashMap<Long, Long> dict, long[] arr) {
            for (long i : arr) {
                if (!dict.containsKey(i)) {
                    dict.put(i, 0L);
                }
                dict.put(i, dict.get(i) + 1);
            }
        }
        // Map Computation End

        // Binary Exponentiation Start
        static long power(long a, long b, long mod) throws IllegalArgumentException {
            if (a < 0 || b < 0 || (a == 0 && b == 0)) {
                throw new IllegalArgumentException("The numbers can't be negative.");
            }
            long res = 1;
            a %= mod;
            while (b != 0) {
                if ((b & 1) != 0) {
                    res = res * a % mod;
                }
                a = a * a % mod;
                b >>= 1;
            }
            return res;
        }
        static long power(long a, long b) throws IllegalArgumentException {
            return power(a, b, inf);
        }
        // Binary Exponentiation End

        // Prime Number Start
        static boolean isPrime(long n, boolean p) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            if (n <= 1) {
                return false;
            }
            if (n <= 3) {
                return true;
            }
            if (n % 2 == 0 || n % 3 == 0) {
                return false;
            }
            for (long i = 5; i <= Math.sqrt(n) + 1; i += 6) {
                if (n % i == 0 || n % (i + 2) == 0) {
                    return false;
                }
            }
            return true;
        }
        private static boolean miller(long a, long n, long d) {
            if (a % n == 0) {
                return true;
            }
            long c = d;
            long x = power(a, c, n);
            if (x == 1 || x == n - 1) {
                return true;
            }
            while (c < n - 1) {
                x = x * x % n;
                if (x == n - 1) {
                    return true;
                }
                c <<= 1;
            }
            return false;
        }
        static boolean isPrime(long n) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            if (n <= 1) {
                return false;
            }
            if (n <= 3) {
                return true;
            }
            if (n % 2 == 0 || n % 3 == 0) {
                return false;
            }
            long d = n - 1;
            while (d % 2 == 0) {
                d /= 2;
            }
            long[] bases64 = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 };
            long[] bases32 = { 2, 7, 61 };
            long[] bases = (n <= 4294967296L) ? bases32 : bases64;
            for (long base : bases) {
                if (!miller(base, n, d)) {
                    return false;
                }
            }
            return true;
        }
        // Prime Number End

        // KMP Algorithm Start
        static int substringIsIn(String text, String pattern) {
            int m = pattern.length();
            int n = text.length();
            int[] lps = new int[m];
            int i = 1, len = 0;
            while (i < m) {
                if (pattern.charAt(i) == pattern.charAt(len)) {
                    lps[i++] = ++len;
                } else {
                    if (len != 0) {
                        len = lps[len - 1];
                    } else {
                        lps[i++] = 0;
                    }
                }
            }
            i = 0;
            int j = 0;
            while (i < n) {
                if (pattern.charAt(j) == text.charAt(i)) {
                    ++i;
                    ++j;
                }
                if (j == m) {
                    return i - j;
                } else if (i < n && pattern.charAt(j) != text.charAt(i)) {
                    if (j != 0) {
                        j = lps[j - 1];
                    } else {
                        ++i;
                    }
                }
            }
            return -1;
        }
        // KMP Algorithm End

        // ----------------------------------------------------------------

        // Modular Arithmetic Start
        // Modular Inverse Function Start
        static long modInv(long n, long mod) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            return power(n, mod - 2, mod);
        }

        static long modInv(long n, int mod, boolean p) throws IllegalArgumentException, ArithmeticException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            long t1 = 0, t2 = 1, r1 = mod, r2 = n, q, temp;
            while (r2 != 0) {
                q = r1 / r2;
                temp = t1;
                t1 = t2;
                t2 = temp - q * t2;
                temp = r1;
                r1 = r2;
                r2 = temp - q * r2;
            }
            if (r1 > 1) {
                throw new ArithmeticException("Modular inverse does not exist.");
            }
            return (t1 % mod + mod) % mod;
        }
        // Modular Inverse Function End

        // Permutations and Combinations Start
        static long fact(long n, long mod) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            long ans = 1;
            for (long i = 1; i <= n; ++i) {
                ans = ans * i % mod;
            }
            return ans;
        }
        static long fact(long n) throws IllegalArgumentException {
            return fact(n, inf);
        }

        static void factArr(long[] arr, long mod) {
            arr[0] = 1;
            for (int i = 1; i < arr.length; ++i) {
                arr[i] = (i * arr[i - 1]) % mod;
            }
        }
        static void factArr(long[] arr) {
            factArr(arr, inf);
        }

        static long perm(long n, long r, long mod) throws IllegalArgumentException {
            if (n < 0 || r < 0) {
                throw new IllegalArgumentException("The numbers can't be negative.");
            }
            long ans = 1;
            for (long i = n - r + 1; i <= n; ++i) {
                ans = ans * i % mod;
            }
            return ans;
        }
        static long perm(long n, long r) throws IllegalArgumentException {
            return perm(n, r, inf);
        }

        static long comb(long n, long r) throws IllegalArgumentException {
            if (n < 0 || r < 0) {
                throw new IllegalArgumentException("The numbers can't be negative.");
            }
            long num = 1;
            long den = 1;
            if (r > n / 2) {
                r = n - r;
            }
            ++n;
            for (long i = 1; i <= r; ++i) {
                num *= n - i;
                den *= i;
            }
            return num / den;
        }
        static long comb(long n, long r, long mod) throws IllegalArgumentException {
            if (n < 0 || r < 0) {
                throw new IllegalArgumentException("The numbers can't be negative.");
            }
            long num = 1;
            long den = 1;
            if (r > n / 2) {
                r = n - r;
            }
            ++n;
            for (long i = 1; i <= r; ++i) {
                num = num * (n - i) % mod;
                den = den * i % mod;
            }
            return num * modInv(den, mod) % mod;
        }

        static long fastFib(long n, long mod) throws IllegalArgumentException {
            if (n < 0) {
                throw new IllegalArgumentException("The number can't be negative.");
            }
            long a0 = 0, a1 = 1, f2, f21, t;
            for (int i = 61; i >= 0; --i) {
                f2 = (a0 * (2 * a1 - a0)) % mod;
                f21 = (a0 * a0 + a1 * a1) % mod;
                if ((n & (1L << i)) > 0) {
                    a0 = f21;
                    a1 = (f2 + f21) % mod;
                } else {
                    a0 = f2;
                    a1 = f21;
                }
            }
            return a0 % mod;
        }
        static long fastFib(long n) throws IllegalArgumentException {
            return fastFib(n, inf);
        }
        // Permutations and Combinations End
        // Modular Arithmetic End
    }

    // ----------------------------------------------------------------

    // Data Structures Start
    // Pair Start
    static class Pair<T1, T2> {
        public final T1 F;
        public final T2 S;

        public Pair(T1 F, T2 S) {
            this.F = F;
            this.S = S;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            Pair<?, ?> pair = (Pair<?, ?>) o;
            if (!F.equals(pair.F))
                return false;
            return S.equals(pair.S);
        }

        @Override
        public int hashCode() {
            int result = F.hashCode();
            result = 31 * result + S.hashCode();
            return result;
        }

        @Override
        public String toString() {
            return "(" + F + ", " + S + ")";
        }
    }
    // Pair End

    // Tuple Start
    static class Tuple<T1, T2, T3> {
        public final T1 F;
        public final T2 S;
        public final T3 T;

        public Tuple(T1 F, T2 S, T3 T) {
            this.F = F;
            this.S = S;
            this.T = T;
        }

        @Override
        public String toString() {
            return "(" + F + ", " + S + ", " + T + ")";
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            Tuple<?, ?, ?> tuple = (Tuple<?, ?, ?>) o;
            if (F != null ? !F.equals(tuple.F) : tuple.F != null) {
                return false;
            }
            if (S != null ? !S.equals(tuple.S) : tuple.S != null) {
                return false;
            }
            return T != null ? T.equals(tuple.T) : tuple.T == null;
        }

        @Override
        public int hashCode() {
            int result = F != null ? F.hashCode() : 0;
            result = 31 * result + (S != null ? S.hashCode() : 0);
            result = 31 * result + (T != null ? T.hashCode() : 0);
            return result;
        }
    }
    // Tuple End

    // Disjoint Set Union Start
    static class DisjointSet {
        private List<Integer> parent, depth, setSize, maxSet, minSet;
        private int numSets, maxSize;

        public DisjointSet() {
        }

        public DisjointSet(int n, int start) {
            init(n, start);
        }

        public DisjointSet(DisjointSet obj) {
            this.maxSize = obj.maxSize;
            this.numSets = obj.numSets;
            this.parent = new ArrayList<>(obj.parent);
            this.depth = new ArrayList<>(obj.depth);
            this.setSize = new ArrayList<>(obj.setSize);
            this.minSet = new ArrayList<>(obj.minSet);
            this.maxSet = new ArrayList<>(obj.maxSet);
        }

        public void init(int n, int start) {
            numSets = n;
            maxSize = 1;
            n += start;
            parent = new ArrayList<>(Collections.nCopies(n, 0));
            maxSet = new ArrayList<>(Collections.nCopies(n, 0));
            minSet = new ArrayList<>(Collections.nCopies(n, 0));
            for (int i = start; i < n; ++i) {
                parent.set(i, maxSet.set(i, minSet.set(i, i)));
            }
            depth = new ArrayList<>(Collections.nCopies(n, 0));
            setSize = new ArrayList<>(Collections.nCopies(n, 1));
        }

        public int findSet(int n) {
            return parent.set(n, (parent.get(n) == n ? n : findSet(parent.get((n)))));
        }

        public int findSet(int n, boolean p) {
            Stack<Integer> st = new Stack<>();
            int v;
            while (n != parent.get(n)) {
                st.push(n);
                n = parent.get(n);
            }
            while (!st.empty()) {
                v = st.pop();
                parent.set(v, n);
            }
            return n;
        }

        public boolean isSameSet(int a, int b) {
            return findSet(a) == findSet(b);
        }

        public void unionSet(int a, int b) {
            int x = findSet(a), y = findSet(b);
            if (x == y) {
                return;
            }
            if (depth.get(x) > depth.get(y)) {
                int t = x;
                x = y;
                y = t;
            }
            if (depth.get(x) == depth.get(y)) {
                depth.set(y, depth.get(y) + 1);
            }
            parent.set(x, y);
            setSize.set(y, setSize.get(y) + setSize.get(x));
            minSet.set(y, Math.min(minSet.get(y), minSet.get(x)));
            maxSet.set(y, Math.max(maxSet.get(y), maxSet.get(x)));
            numSets--;
        }

        public int numOfSets() {
            return numSets;
        }

        public int sizeOfSet(int n) {
            return setSize.get(findSet(n));
        }

        public int maxOfSet(int n) {
            return maxSet.get(findSet(n));
        }

        public int minOfSet(int n) {
            return minSet.get(findSet(n));
        }

        public int maxSizeOfSets() {
            return maxSize;
        }
    }

    // Disjoin Set Union End

    // Segment Tree Start
    static class SegmentTree {
        private long[] tree;
        private int size;
        private java.util.function.BiFunction<Long, Long, Long> func;

        private void buildTree(long[] arr, int treeIndex, int left, int right) {
            if (left == right) {
                tree[treeIndex] = arr[left];
                return;
            }
            int mid = (left + right) / 2;
            buildTree(arr, 2 * treeIndex + 1, left, mid);
            buildTree(arr, 2 * treeIndex + 2, mid + 1, right);
            tree[treeIndex] = func.apply(tree[2 * treeIndex + 1], tree[2 * treeIndex + 2]);
        }

        public SegmentTree(long[] arr, java.util.function.BiFunction<Long, Long, Long> fun) {
            size = arr.length;
            func = fun;
            tree = new long[2 * size - 1];
            buildTree(arr, 0, 0, size - 1);
        }

        private long query(int treeIndex, int left, int right, int queryLeft, int queryRight) {
            if (queryLeft <= left && right <= queryRight) {
                return tree[treeIndex];
            }
            int mid = (left + right) / 2;
            if (mid < queryLeft || left > queryRight) {
                return query(2 * treeIndex + 2, mid + 1, right, queryLeft, queryRight);
            }
            if (right < queryLeft || mid + 1 > queryRight) {
                return query(2 * treeIndex + 1, left, mid, queryLeft, queryRight);
            }
            return func.apply(query(2 * treeIndex + 1, left, mid, queryLeft, queryRight),
                    query(2 * treeIndex + 2, mid + 1, right, queryLeft, queryRight));
        }

        public long query(int left, int right) throws java.lang.Exception {
            if (left > right || left < 0 || right >= size) {
                throw new IllegalArgumentException("Given query range is invalid or out of range.");
            }
            return query(0, 0, size - 1, left, right);
        }

        private void update(int treeIndex, int left, int right, int updateIndex, int updateNewValue) {
            if (left == right) {
                tree[treeIndex] = updateNewValue;
                return;
            }
            int mid = (left + right) / 2;
            if (updateIndex <= mid) {
                update(2 * treeIndex + 1, left, mid, updateIndex, updateNewValue);
            } else {
                update(2 * treeIndex + 2, mid + 1, right, updateIndex, updateNewValue);
            }
            tree[treeIndex] = func.apply(tree[2 * treeIndex + 1], tree[2 * treeIndex + 2]);
        }

        public void update(int index, int newValue) throws java.lang.Exception {
            if (index < 0 || index >= size) {
                throw new IllegalArgumentException("Given update index is out of range.");
            }
            update(0, 0, size - 1, index, newValue);
        }
    }
    // Segment Tree End

    // Graphs Start
    // Unweighted Graphs Start
    static class UnweightedGraph {
        private int count;

        private void dfsPr(int src) {
            visited.set(src, true);
            pre.set(src, count);
            ++count;
            for (int it : aList.get(src)) {
                if (!visited.get(it)) {
                    parent.set(it, src);
                    dfsPr(it);
                }
            }
            post.set(src, count);
            ++count;
        }

        public List<List<Integer>> aList;
        public List<Boolean> visited;
        public List<Integer> level, component, topoOrder, pathLen, parent, pre, post;
        public int start, end, numOfComp;

        public UnweightedGraph(final List<List<Integer>> aLis, int star) {
            aList = aLis;
            start = star;
            end = aLis.size() + star;
        }

        public void dfs(int src, boolean iter) {
            visited = new ArrayList<>(Collections.nCopies(end, false));
            parent = new ArrayList<>(Collections.nCopies(end, -1));
            pre = new ArrayList<>(Collections.nCopies(end, -1));
            post = new ArrayList<>(Collections.nCopies(end, -1));
            if (iter) {
                int cou = 0;
                Stack<Integer> st = new Stack<>();
                int u;
                boolean flag;
                st.push(src);
                while (!st.empty()) {
                    u = st.peek();
                    if (!visited.get(u)) {
                        visited.set(u, true);
                        pre.set(u, cou);
                        ++cou;
                    }
                    flag = true;
                    for (int v : aList.get(u)) {
                        if (!visited.get(v)) {
                            parent.set(v, u);
                            st.push(v);
                            flag = false;
                            break;
                        }
                    }
                    if (flag) {
                        post.set(u, cou);
                        ++cou;
                        st.pop();
                    }
                }
            } else {
                count = 0;
                dfsPr(src);
            }
        }

        public void bfs(int src) {
            level = new ArrayList<>(Collections.nCopies(end, -1));
            parent = new ArrayList<>(Collections.nCopies(end, -1));
            Queue<Integer> que = new LinkedList<>();
            que.add(src);
            level.set(src, 0);
            int v;
            while (!que.isEmpty()) {
                v = que.poll();
                for (int it : aList.get(v)) {
                    if (level.get(it) == -1) {
                        level.set(it, level.get(v) + 1);
                        parent.set(it, v);
                        que.add(it);
                    }
                }
            }
        }

        public void components() {
            component = new ArrayList<>(Collections.nCopies(end, -1));
            int seen = start, src, i;
            numOfComp = 0;
            while (seen < end) {
                src = -1;
                for (i = start; i < end; ++i) {
                    if (component.get(i) == -1) {
                        src = i;
                        break;
                    }
                }
                bfs(src);
                for (int it : level) {
                    if (it != -1) {
                        component.set(it, numOfComp);
                        ++seen;
                    }
                }
                ++numOfComp;
            }
        }

        public void topologicalOrder() {
            List<Integer> inDegree = new ArrayList<>(Collections.nCopies(end, 0));
            pathLen = new ArrayList<>(Collections.nCopies(end, 0));
            int i;
            for (i = start; i < end; ++i) {
                for (int it : aList.get(i)) {
                    inDegree.set(it, inDegree.get(it) + 1);
                }
            }
            Queue<Integer> que = new LinkedList<>();
            for (i = start; i < end; ++i) {
                if (inDegree.get(i) == 0) {
                    que.add(i);
                }
            }
            int v;
            while (!que.isEmpty()) {
                v = que.poll();
                topoOrder.add(v);
                inDegree.set(v, -1);
                for (int it : aList.get(v)) {
                    inDegree.set(it, inDegree.get(it) - 1);
                    pathLen.set(it, Math.max(pathLen.get(it), pathLen.get(v) + 1));
                    if (inDegree.get(it) == 0) {
                        que.add(it);
                    }
                }
            }
        }
    }
    // Unweighted Graphs End

    // Weighted Graphs Start
    static class WeightedGraph {
        public long inf = 9223372036854775807L;
        public List<List<Pair<Integer, Long>>> wList;
        public List<List<Long>> distanceFw;
        public DisjointSet component;
        public List<Pair<Integer, Integer>> edge;
        public List<Long> distance;
        public int start, end;

        public WeightedGraph(final List<List<Pair<Integer, Long>>> wLis, int star) {
            wList = wLis;
            start = star;
            end = wLis.size() + star;
        }

        public void dijkstra(int src) {
            List<Boolean> visited = new ArrayList<>(Collections.nCopies(end, false));
            distance = new ArrayList<>(Collections.nCopies(end, inf));
            distance.set(src, 0L);
            PriorityQueue<Pair<Long, Integer>> heap = new PriorityQueue<>(
                    new Comparator<Pair<Long, Integer>>() {
                        public int compare(Pair<Long, Integer> p1, Pair<Long, Integer> p2) {
                            return p1.F.compareTo(p2.F);
                        }
                    });
            heap.add(new Pair<Long, Integer>(0L, src));
            int nextv;
            while (!heap.isEmpty()) {
                nextv = heap.peek().S;
                heap.poll();
                if (visited.get(nextv)) {
                    continue;
                }
                visited.set(nextv, true);
                for (Pair<Integer, Long> it : wList.get((int) nextv)) {
                    if (!visited.get(it.F) && distance.get(nextv) + it.S < distance.get(it.F)) {
                        distance.set(it.F, distance.get(nextv) + it.S);
                        heap.add(new Pair<Long, Integer>(distance.get(it.F), it.F));
                    }
                }
            }
        }

        public void bellmanFord(int src) {
            distance = new ArrayList<>(Collections.nCopies(end, inf));
            distance.set(src, 0L);
            int i, u;
            for (i = start; i < end - 1; ++i) {
                for (u = start; u < end; ++u) {
                    for (Pair<Integer, Long> it : wList.get(u)) {
                        distance.set(it.F, Math.min(distance.get(it.F), distance.get(u) + it.S));
                    }
                }
            }
            boolean flag = false;
            for (u = start; u < end; ++u) {
                for (Pair<Integer, Long> it : wList.get(u)) {
                    if (distance.get(u) + it.S < distance.get(it.F)) {
                        throw new RuntimeException("The graph has negative cycles.");
                    }
                }
            }
        }

        public void floydWarshall() {
            distanceFw = new ArrayList<>(
                    Collections.nCopies(end, new ArrayList<Long>(Collections.nCopies(end, inf))));
            int i, j, k;
            for (i = start; i < end; ++i) {
                for (Pair<Integer, Long> it : wList.get(i)) {
                    distanceFw.get(i).set(it.F, it.S);
                }
            }
            for (i = start; i < end; ++i) {
                for (j = start; j < end; ++j) {
                    for (k = start; k < end; ++k) {
                        distanceFw.get(j).set(k, Math.min(distanceFw.get(j).get(k),
                                distanceFw.get(j).get(i) + distanceFw.get(i).get(k)));
                    }
                }
            }
        }

        public void prim() {
            List<Boolean> visited = new ArrayList<>(Collections.nCopies(end, false));
            distance = new ArrayList<>(Collections.nCopies(end, inf));
            distance.set(start, 0L);
            PriorityQueue<Pair<Long, Integer>> heap = new PriorityQueue<>(
                    new Comparator<Pair<Long, Integer>>() {
                        public int compare(Pair<Long, Integer> p1, Pair<Long, Integer> p2) {
                            return p1.F.compareTo(p2.F);
                        }
                    });
            heap.add(new Pair<Long, Integer>(0L, start));
            int nextv;
            while (!heap.isEmpty()) {
                nextv = heap.peek().S;
                if (visited.get(nextv)) {
                    continue;
                }
                visited.set(nextv, true);
                for (Pair<Integer, Long> it : wList.get(nextv)) {
                    if (!visited.get(it.F) && it.S < distance.get(it.F)) {
                        distance.set(it.F, it.S);
                        heap.add(new Pair<Long, Integer>(it.S, it.F));
                    }
                }
            }
        }

        public void kruskal() {
            component.init(end - start, start);
            Set<Tuple<Long, Integer, Integer>> edges = new HashSet<>();
            int u, v;
            for (u = start; u < end; ++u) {
                for (Pair<Integer, Long> it : wList.get(u)) {
                    edges.add(new Tuple<Long, Integer, Integer>(it.S, u, it.F));
                }
            }
            for (Tuple<Long, Integer, Integer> it : edges) {
                u = it.S;
                v = it.T;
                if (component.isSameSet(u, v)) {
                    continue;
                }
                edge.add(new Pair<Integer, Integer>(u, v));
                component.unionSet(u, v);
            }
        }
    }
    // Weighted Graphs End
    // Graphs End
    // Data Structures End

    // ----------------------------------------------------------------

    // Solution Class Start
    static class Solution extends Functions {
        public static Reader in = new Reader();
        public static PrintWriter out = new PrintWriter(System.out, true);

        public void solve(long index) throws java.lang.Exception {
            // ----------------------------------------------------------------

            int n, i, j;
            n = in.nextInt();

            // out.println("Case #" + index + ": " + ans + newl);

            // ----------------------------------------------------------------
        }

        public boolean testCases = true;
    }
    // Solution Class End

    // Main Function Start
    public static void main(String[] args) throws java.lang.Exception {
        int testCases = 1;
        Solution sol = new Solution();
        if (sol.testCases) {
            testCases = Solution.in.nextInt();
        }
        for (int testCase = 1; testCase <= testCases; ++testCase) {
            sol.solve(testCase);
        }
    }
}
// Main Class End
// Program End
// ----------------------------------------------------------------