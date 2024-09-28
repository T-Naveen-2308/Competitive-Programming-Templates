// Author : Naveen

// Program Start
// Libraries and Namespace Start
#include <bits/stdc++.h>
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/tree_policy.hpp>
using namespace std;
using namespace __gnu_pbds;
// Libraries and Namespace End

//----------------------------------------------------------------

// Important Shortcuts Start
// Macros Start
#define F first
#define S second
#define pb push_back
#define yes cout << "yes\n"
#define no cout << "no\n"
#define Yes cout << "Yes\n"
#define No cout << "No\n"
#define YES cout << "YES\n"
#define NO cout << "NO\n"
#define all(v) v.begin(), v.end()
// Macros End

// Custom Hash Function Start
struct custom_hash {
    static uint64_t splitMix64(uint64_t x) {
        x += 0x9e3779b97f4a7c15;
        x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
        x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
        return x ^ (x >> 31);
    }

    size_t operator()(uint64_t x) const {
        static const uint64_t FIXED_RANDOM = chrono::steady_clock::now().time_since_epoch().count();
        return splitMix64(x + FIXED_RANDOM);
    }
};
// Custom Hash Function End

// Declarations Start
typedef long long int ll;
typedef unsigned long long int ull;
typedef long double ld;
typedef string str;
typedef pair<ll, ll> pll;
typedef pair<ull, ull> pull;
typedef tuple<ll, ll, ll> tll;
typedef tuple<ull, ull, ull> tull;
typedef stack<ll> stll;
typedef queue<ll> qll;
typedef vector<bool> vb;
typedef vector<ll> vll;
typedef vector<ull> vull;
typedef vector<pll> vpll;
typedef vector<pull> vpull;
typedef vector<tll> vtll;
typedef vector<tull> vtull;
typedef vector<ld> vld;
typedef set<ll> sll;
typedef unordered_set<ll, custom_hash> usll;
typedef multiset<ll> msll;
typedef unordered_multiset<ll, custom_hash> umsll;
typedef map<ll, ll> mll;
typedef unordered_map<ll, ll, custom_hash> umll;
typedef multimap<ll, ll> mmll;
typedef unordered_multimap<ll, ll, custom_hash> ummll;
typedef priority_queue<ll> maxheap;
typedef priority_queue<pll, vector<pll>, less<pll>> maxheappll;
typedef priority_queue<ll, vector<ll>, greater<ll>> minheap;
typedef priority_queue<pll, vector<pll>, greater<pll>> minheappll;
typedef tree<ll, null_type, less<ll>, rb_tree_tag, tree_order_statistics_node_update> pbdssll;
typedef tree<ll, null_type, less_equal<ll>, rb_tree_tag, tree_order_statistics_node_update> pbdsmsll;
typedef tree<pll, null_type, less<pll>, rb_tree_tag, tree_order_statistics_node_update> pbdsmll;
typedef tree<pll, null_type, less_equal<pll>, rb_tree_tag, tree_order_statistics_node_update> pbdsmmll;
// Declarations End

// Constants Start
const ull mod1 = 1000000007ull;
const ull mod2 = 998244353ull;
const ull max_if = 1000000ull;
const ull inf = 9223372036854775807ull;
const ll neg_inf = -9223372036854775807ll - 1ll;
const char spc = ' ';
const char newl = '\n';
// Constants End

// Input and Output Functions Start
template <class T>
istream& operator>>(istream& is, vector<T>& a) {
    for (T& x : a) {
        is >> x;
    }
    return is;
}

inline void yn(const bool con) {
    cout << (con ? "yes" : "no") << '\n';
}
inline void Yn(const bool con) {
    cout << (con ? "Yes" : "No") << '\n';
}
inline void YN(const bool con) {
    cout << (con ? "YES" : "NO") << '\n';
}

template <class T>
inline void print(const T& val) {
    cout << val << ' ';
}
template <class T>
void print(const vector<T>& vec) {
    for (T i : vec) {
        cout << i << ' ';
    }
}

template <class T>
inline void println(const T& val) {
    cout << val << '\n';
}
template <class T>
void println(const vector<T>& vec) {
    for (T i : vec) {
        cout << i << ' ';
    }
    cout << '\n';
}

template <class T, class V>
ostream& operator<<(ostream& os, const pair<T, V>& p) {
    os << "(" << p.F << ", " << p.S << ")";
    return os;
}
template <size_t Index, typename... Types>
typename enable_if<Index == sizeof...(Types)>::type
printTupleHelper(ostream& os, const tuple<Types...>&) {}
template <size_t Index, typename... Types>
typename enable_if < Index<sizeof...(Types)>::type
    printTupleHelper(ostream& os, const tuple<Types...>& t) {
    os << get<Index>(t);
    if (Index + 1 < sizeof...(Types))
        os << ", ";
    printTupleHelper<Index + 1>(os, t);
}
template <typename... Types>
ostream& operator<<(ostream& os, const tuple<Types...>& t) {
    os << "(";
    printTupleHelper<0>(os, t);
    os << ")";
    return os;
}
template <class T>
ostream& operator<<(ostream& os, const vector<T>& vec) {
    size_t c = 0;
    const size_t n = vec.size();
    os << "[";
    for (T i : vec) {
        os << i << (++c < n ? ", " : "");
    }
    os << "]";
    return os;
}
template <class T>
ostream& operator<<(ostream& os, const set<T>& set_v) {
    size_t c = 0;
    const size_t n = set_v.size();
    os << "{";
    for (T i : set_v) {
        os << i << (++c < n ? ", " : "");
    }
    os << "}";
    return os;
}
template <class T>
ostream& operator<<(ostream& os, const multiset<T>& set_v) {
    size_t c = 0;
    const size_t n = set_v.size();
    os << "{";
    for (T i : set_v) {
        os << i << (++c < n ? ", " : "");
    }
    os << "}";
    return os;
}
template <class T, class V>
ostream& operator<<(ostream& os, const map<T, V> dict) {
    size_t c = 0;
    const size_t n = dict.size();
    os << "{";
    for (auto it : dict) {
        os << it.first << ": " << it.second << (++c < n ? ", " : "");
    }
    os << "}";
    return os;
}
template <class T, class V>
ostream& operator<<(ostream& os, const unordered_map<T, V> dict) {
    size_t c = 0;
    const size_t n = dict.size();
    os << "{";
    for (auto it : dict) {
        os << it.first << ": " << it.second << (++c < n ? ", " : "");
    }
    os << "}";
    return os;
}
// Input and Output Functions End
// Important Shortcuts End

//----------------------------------------------------------------

// Basic Functions Start
// String Operator Overload Start
template <class T>
string operator*(const string& str, const T& n) {
    string res;
    res.reserve(str.size() * n);
    for (T i = 0; i < n; ++i)
        res += str;
    return res;
}
// String Operator Overload End

// Min Function Start
template <class T>
inline T min(const vector<T>& vec) {
    return *min_element(vec.begin(), vec.end());
}
// Min Function End

// Max Function Start
template <class T>
inline T max(const vector<T>& vec) {
    return *max_element(vec.begin(), vec.end());
}
// Max Function End

// Sum Function Start
template <class T>
inline T sum(const T& val) {
    return val;
}
template <class T, class... Args>
T sum(const T& val, Args &&...args) {
    return val + sum(forward<Args>(args)...);
}
template <class T>
inline T sum(const vector<T>& vec) {
    return accumulate(vec.begin(), vec.end(), T(0));
}
// Sum Function End

// Binary Start
template <class T>
string bin(const T& dn) {
    string bin_str = bitset<64>(dn).to_string();
    ull firstOne = bin_str.find('1');
    if (firstOne != string::npos) {
        bin_str = bin_str.substr(firstOne);
        return bin_str;
    }
    return "0";
}

ull int2(const string& bin_str) {
    bitset<64> bits(bin_str);
    ull dn = static_cast<ull>(bits.to_ulong());
    return dn;
}
// Binary End

// Square Root Start
inline ull sqrt_int(ll n) {
    if (n < 0) {
        throw invalid_argument("The number can't be negative.");
    }
    return static_cast<ull>(sqrt(n));
}

ull sqrt_int_u(ll n) {
    if (n < 0) {
        throw invalid_argument("The number can't be negative.");
    }
    ull k = sqrt(n);
    return k * k == n ? k : k + 1;
}
// Square Root End

// Map Computation Start
template <class T>
void map_com(map<T, ll>& ma, const vector<T>& vec) {
    for (T i : vec) {
        ma[i]++;
    }
}
// Map Computation End

// Binary Exponentiation Start
ull power(ll b, ll p, const ll mod = inf) {
    if (b <= 0 || p < 0) {
        throw invalid_argument("The numbers can't be negative.");
    }
    ull res = 1;
    b %= mod;
    while (p > 0) {
        if (p & 1) {
            res = res * b % mod;
        }
        b = b * b % mod;
        p >>= 1;
    }
    return res;
}
// Binary Exponentiation End

// Prime Number Start
bool is_prime(ll n, bool old = true) {
    if (n <= 0) {
        throw invalid_argument("The given number can't be negative.");
    }
    if (n == 1) {
        return false;
    }
    if (n <= 3) {
        return true;
    }
    if (n % 2 == 0 || n % 3 == 0) {
        return false;
    }
    if (old) {
        for (ll i = 5; i < (int)sqrt(n) + 1; i += 6) {
            if (n % i == 0 || n % (i + 2) == 0) {
                return false;
            }
        }
        return true;
    }
    ull d = n - 1;
    while (d % 2 == 0) {
        d /= 2;
    }
    auto miller = [&](ull a) {
        if (a % n == 0) {
            return true;
        }
        ull x = power(a, d, n);
        if (x == 1 || x == n - 1) {
            return true;
        }
        ull c = d;
        while (c < n - 1) {
            x = x * x % n;
            if (x == n - 1) {
                return true;
            }
            c <<= 1;
        }
        return false;
        };
    vector<ull> bases64 = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 };
    vector<ull> bases32 = { 2, 7, 61 };
    vector<ull>& bases = n <= 4294967296ull ? bases32 : bases64;
    for (ull base : bases) {
        if (!miller(base)) {
            return false;
        }
    }
    return true;
}

void primes(vector<ll>& prime, const ull size = max_if) {
    ull i, j;
    vector<bool> prime_is(size + 1, true);
    prime_is[0] = false;
    prime_is[1] = false;
    for (i = 2; i <= sqrt_int_u(size); ++i) {
        if (prime_is[i]) {
            for (j = i * i; j <= size; j += i) {
                prime_is[j] = false;
            }
        }
    }
    for (i = 2; i <= size; ++i) {
        if (prime_is[i]) {
            prime.pb(i);
        }
    }
}
// Prime Number End

// KMP Algorithm Start
ll substr_is_in(const string& text, const string& pattern) {
    const size_t m = pattern.size();
    const size_t n = text.size();
    vull lps(m, 0);
    ull len = 0, i = 1, j;
    while (i < m) {
        if (pattern[i] == pattern[len]) {
            lps[i++] = ++len;
        }
        else {
            if (len != 0) {
                len = lps[len - 1];
            }
            else {
                lps[i++] = 0;
            }
        }
    }
    i = 0, j = 0;
    while (i < n) {
        if (pattern[j] == text[i]) {
            ++i;
            ++j;
        }
        if (j == m) {
            return i - j;
        }
        else if (i < n && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            }
            else {
                ++i;
            }
        }
    }
    return -1;
}
// KMP Algorithm End
// Basic Functions End

//-----------------------------------------------------------------

// Modular Arithmetic Start
// Modular Inverse Function Start
ll mod_inv(ll n, const ll mod, const bool p) {
    if (n <= 0) {
        throw invalid_argument("The number can't be zero or negative.");
    }
    ll t1 = 0, t2 = 1, r1 = mod, r2 = n, q, temp;
    while (r2) {
        q = r1 / r2;
        temp = t1;
        t1 = t2;
        t2 = temp - q * t1;
        temp = r1;
        r1 = r2;
        r2 = temp - q * r1;
    }
    if (r1 > 1) {
        throw invalid_argument("Modular inverse does not exist.");
    }
    return (t1 % mod + mod) % mod;
}
inline ll mod_inv(ll n, const ll mod = mod1) {
    if (n < 0) {
        throw invalid_argument("The number can't be negative.");
    }
    return power(n, mod - 2, mod);
}
// Modular Inverse Function End

// Permutations and Combinations Start
inline ull fact(ll n, const ll mod = inf) {
    if (n < 0) {
        throw invalid_argument("The number can't be negative.");
    }
    ull ans = 1;
    for (ull i = 1; i <= n; ++i) {
        ans = (ans * i) % mod;
    }
    return ans;
}

inline void fact_arr(vector<ll>& v, const ll mod = inf) {
    v[0] = 1;
    const size_t n = v.size();
    for (size_t i = 1; i < n; ++i) {
        v[i] = (i * v[i - 1]) % mod;
    }
}

inline ull perm(ll n, ll r, const ll mod = inf) {
    if (n < 0 || r < 0) {
        throw invalid_argument("The numbers can't be negative.");
    }
    ull ans = 1;
    for (ull i = n - r + 1; i <= n; ++i) {
        ans = (ans * i) % mod;
    }
    return ans;
}

ull comb(ll n, ll r) {
    if (n < 0 || r < 0) {
        throw invalid_argument("The numbers can't be negative.");
    }
    ull num = 1, den = 1;
    if (r > n / 2) {
        r = n - r;
    }
    ++n;
    for (ull i = 1; i <= r; ++i) {
        num = num * (n - i);
        den = den * i;
    }
    return num / den;
}
ull comb(ll n, ll r, const ll mod, const bool p = true) {
    if (n < 0 || r < 0) {
        throw invalid_argument("The numbers can't be negative.");
    }
    ull num = 1, den = 1;
    if (r > n / 2) {
        r = n - r;
    }
    ++n;
    for (ull i = 1; i <= r; ++i) {
        num = (num * (n - i)) % mod;
        den = (den * i) % mod;
    }
    return num * (p ? mod_inv(den, mod) : mod_inv(den, mod, true)) % mod;
}

ull fast_fib(ll n, const ll mod = inf) {
    if (n < 0) {
        throw invalid_argument("The numbers can't be negative.");
    }
    ull a0 = 0, a1 = 1, f2, f21, t;
    for (ll i = 61; i >= 0; --i) {
        f2 = (a0 * (2 * a1 - a0)) % mod;
        f21 = (a0 * a0 + a1 * a1) % mod;
        if (n & (1LL << i)) {
            a0 = f21;
            a1 = (f2 + f21) % mod;
        }
        else {
            a0 = f2;
            a1 = f21;
        }
    }
    return a0 % mod;
}
// Permutations and Combinations End
// Modular Arithmetic End

//-----------------------------------------------------------------

// Data Structures Start
// Disjoint Set Union Start
class DisjointSet {
private:
    vll parent, depth, set_size, max_set, min_set;
    ull num_sets, max_size;

public:
    DisjointSet() {}

    DisjointSet(ll n, ll start = 1) {
        init(n, start);
    }

    DisjointSet(const DisjointSet& obj) : parent(all(obj.parent)), depth(all(obj.depth)), set_size(all(obj.set_size)), min_set(all(obj.min_set)), max_set(all(obj.max_set)), max_size(obj.max_size), num_sets(obj.num_sets) {}

    void init(ll n, ll start = 1) {
        num_sets = n;
        max_size = 1;
        n += start;
        parent.assign(n, 0);
        max_set.assign(n, 0);
        min_set.assign(n, 0);
        for (ll i = start; i < n; ++i) {
            parent[i] = max_set[i] = min_set[i] = i;
        }
        depth.assign(n, 0);
        set_size.assign(n, 1);
    }

    ll find_set(ll n) {
        return parent[n] = (parent[n] == n ? n : find_set(parent[n]));
    }
    ll find_set(ll n, bool p) {
        stack<ll> st;
        ll v;
        while (n != parent[n]) {
            st.push(n);
            n = parent[n];
        }
        while (!st.empty()) {
            v = st.top();
            st.pop();
            parent[v] = n;
        }
        return n;
    }

    bool is_same_set(ll a, ll b) {
        return find_set(a) == find_set(b);
    }

    void union_set(ll a, ll b) {
        ll x = find_set(a), y = find_set(b);
        if (x == y) {
            return;
        }
        if (depth[x] > depth[y]) {
            swap(x, y);
        }
        if (depth[x] == depth[y]) {
            depth[y]++;
        }
        parent[x] = y;
        set_size[y] += set_size[x];
        max_size = max((ll)max_size, set_size[y]);
        min_set[y] = min(min_set[y], min_set[x]);
        max_set[y] = max(max_set[y], max_set[x]);
        num_sets--;
    }

    ll num_of_sets() {
        return num_sets;
    }

    ll size_of_set(ll n) {
        return set_size[find_set(n)];
    }

    ll max_of_set(ll n) {
        return max_set[find_set(n)];
    }

    ll min_of_set(ll n) {
        return min_set[find_set(n)];
    }

    ll max_size_of_sets() {
        return max_size;
    }
};
// Disjoint Set Union End

// Segment Tree Start
template <class T>
class SegmentTree {
private:
    const function<T(T, T)>& func;
    vector<T> tree;
    const size_t size;
    size_t query_left, query_right, update_index, update_new_value;

    void build_tree(size_t treeIndex, size_t left, size_t right) {
        if (left == right) {
            tree[treeIndex] = data[left];
            return;
        }
        size_t mid = left + (right - left) / 2;
        build_tree(2 * treeIndex + 1, left, mid);
        build_tree(2 * treeIndex + 2, mid + 1, right);
        tree[treeIndex] = func(tree[2 * treeIndex + 1], tree[2 * treeIndex + 2]);
    }

    T query(size_t treeIndex, size_t left, size_t right) {
        if (query_left <= left && right <= query_right) {
            return tree[treeIndex];
        }
        size_t mid = left + (right - left) / 2;
        if (mid < query_left || left > query_right) {
            return query(2 * treeIndex + 2, mid + 1, right);
        }
        if (right < query_left || mid + 1 > query_right) {
            return query(2 * treeIndex + 1, left, mid);
        }
        return func(query(2 * treeIndex + 1, left, mid), query(2 * treeIndex + 2, mid + 1, right));
    }

    void update(size_t treeIndex, size_t left, size_t right) {
        if (left == right) {
            tree[treeIndex] = update_new_value;
            return;
        }
        size_t mid = left + (right - left) / 2;
        if (update_index <= mid) {
            update(2 * treeIndex + 1, left, mid);
        }
        else {
            update(2 * treeIndex + 2, mid + 1, right);
        }
        tree[treeIndex] = func(tree[2 * treeIndex + 1], tree[2 * treeIndex + 2]);
    }

public:
    vector<T> data;

    SegmentTree(const vector<T>& vec, const function<T(T, T)>& fun) : size(vec.size()), func(fun), data(vec.begin(), vec.end()) {
        tree.resize(4 * size);
        build_tree(0, 0, size - 1);
    }

    T query(ll left, ll right) {
        if (left > right || left < 0 || right >= size) {
            throw invalid_argument("Given query range is invalid or out of range.");
        }
        query_left = left;
        query_right = right;
        return query(0, 0, size - 1);
    }

    void update(ll index, T newValue) {
        if (index < 0 || index >= size) {
            throw invalid_argument("Given update index is out of range.");
        }
        data[index] = newValue;
        update_index = index;
        update_new_value = newValue;
        update(0, 0, size - 1);
    }
};
// Segment Tree End

// Graphs Start
// Unweighted Graphs Start
class UnweightedGraph {
private:
    ll count;

    void dfsPr(ll src) {
        visited[src] = true;
        pre[src] = count;
        count++;
        for (auto it : a_list[src]) {
            if (!visited[it]) {
                parent[it] = src;
                dfsPr(it);
            }
        }
        post[src] = count;
        count++;
    }

public:
    vector<vll> a_list;
    vb visited;
    vll level, component, topo_order, path_len, parent, pre, post;
    ll start, end, num_of_comp;

    UnweightedGraph(const vector<vll>& a_lis, ll star = 1) {
        a_list = a_lis;
        start = star;
        end = a_lis.size() + start;
    }

    void dfs(ll src, bool iter = true) {
        visited.assign(end, false);
        parent.assign(end, -1);
        pre.assign(end, -1);
        post.assign(end, -1);
        if (iter) {
            ll cou = 0;
            stack<ll> st;
            ll u;
            bool flag;
            st.push(src);
            while (!st.empty()) {
                u = st.top();
                if (!visited[u]) {
                    visited[u] = true;
                    pre[u] = cou;
                    ++cou;
                }
                flag = true;
                for (ll v : a_list[u]) {
                    if (!visited[v]) {
                        parent[v] = u;
                        st.push(v);
                        flag = false;
                        break;
                    }
                }
                if (flag) {
                    post[u] = cou;
                    ++cou;
                    st.pop();
                }
            }
        }
        else {
            count = 0;
            dfsPr(src);
        }
    }

    void bfs(ll src) {
        level.assign(end, -1);
        parent.assign(end, -1);
        queue<ll> que;
        que.push(src);
        level[src] = 0;
        ll v;
        while (!que.empty()) {
            v = que.front();
            que.pop();
            for (auto it : a_list[v]) {
                if (level[it] == -1) {
                    level[it] = level[v] + 1;
                    parent[it] = v;
                    que.push(it);
                }
            }
        }
    }

    void components() {
        component.assign(end, -1);
        ll seen = start, src, i;
        num_of_comp = 0;
        while (seen < end) {
            src = -1;
            for (i = start; i < end; ++i) {
                if (component[i] == -1) {
                    src = i;
                    break;
                }
            }
            bfs(src);
            for (auto it : level) {
                if (level[it] != -1) {
                    component[it] = num_of_comp;
                    ++seen;
                }
            }
            ++num_of_comp;
        }
    }

    void topological_order() {
        vll in_degree(end, 0);
        path_len.assign(end, 0);
        ll i;
        for (i = start; i < end; ++i) {
            for (auto it : a_list[i]) {
                ++in_degree[it];
            }
        }
        queue<ll> que;
        for (i = start; i < end; ++i) {
            if (in_degree[i] == 0) {
                que.push(i);
            }
        }
        ll v;
        while (!que.empty()) {
            v = que.front();
            que.pop();
            topo_order.pb(v);
            in_degree[v] = -1;
            for (auto it : a_list[v]) {
                in_degree[it]--;
                path_len[it] = max(path_len[it], path_len[v] + 1);
                if (in_degree[it] == 0) {
                    que.push(it);
                }
            }
        }
    }
};
// Unweighted Graphs End

// Weighted Graphs Start
class WeightedGraph {
public:
    vector<vpll> w_list;
    vector<vll> distance_fw;
    DisjointSet component;
    vpll edge;
    vll distance;
    ll start, end;

    WeightedGraph(const vector<vpll>& w_lis, ll star = 1) {
        w_list = w_lis;
        start = star;
        end = w_lis.size() + star;
    }

    void dijkstra(ll src) {
        vb visited(end, false);
        distance.assign(end, inf);
        distance[src] = 0;
        minheappll heap;
        heap.push({ 0, src });
        ll nextv;
        while (!heap.empty()) {
            nextv = heap.top().S;
            heap.pop();
            if (visited[nextv]) {
                continue;
            }
            visited[nextv] = true;
            for (auto it : w_list[nextv]) {
                if (!visited[it.F] && distance[nextv] + it.S < distance[it.F]) {
                    distance[it.F] = distance[nextv] + it.S;
                    heap.push({ distance[it.F], it.F });
                }
            }
        }
    }

    void bellman_ford(ll src) {
        distance.assign(end, inf);
        distance[src] = 0;
        ll i, u;
        for (i = start; i < end - 1; ++i) {
            for (u = start; u < end; ++u) {
                for (auto it : w_list[u]) {
                    distance[it.F] = min(distance[it.F], distance[u] + it.S);
                }
            }
        }
        bool flag = false;
        for (u = start; u < end; ++u) {
            for (auto it : w_list[u]) {
                if (distance[u] + it.S < distance[it.F]) {
                    throw invalid_argument("The graph has negative cycles.");
                }
            }
        }
    }

    void floyd_warshall() {
        distance_fw.assign(end, vll(end, inf));
        ll i, j, k;
        for (i = start; i < end; ++i) {
            for (auto it : w_list[i]) {
                distance_fw[i][it.F] = it.S;
            }
        }
        for (i = start; i < end; ++i) {
            for (j = start; j < end; ++j) {
                for (k = start; k < end; ++k) {
                    distance_fw[j][k] = min(distance_fw[j][k], distance_fw[j][i] + distance_fw[i][k]);
                }
            }
        }
    }

    void prim() {
        vb visited(end, false);
        distance.assign(end, inf);
        distance[start] = 0;
        minheappll heap;
        heap.push({ 0, start });
        ll nextv;
        while (!heap.empty()) {
            nextv = heap.top().S;
            if (visited[nextv]) {
                continue;
            }
            visited[nextv] = true;
            for (auto it : w_list[nextv]) {
                if (!visited[it.F] && it.S < distance[it.F]) {
                    distance[it.F] = it.S;
                    heap.push({ it.S, it.F });
                }
            }
        }
    }

    void kruskal() {
        component.init(end - start, start);
        set<tll> edges;
        ll u, v;
        for (u = start; u < end; ++u) {
            for (auto it : w_list[u]) {
                edges.insert({ it.S, u, it.F });
            }
        }
        for (auto it : edges) {
            u = get<1>(it);
            v = get<2>(it);
            if (component.is_same_set(u, v)) {
                continue;
            }
            edge.pb({ u, v });
            component.union_set(u, v);
        }
    }
};
// Weighted Graphs End
// Graphs End
// Data Structures End

//----------------------------------------------------------------

// Solution Class Start
class Solution {
public:
    void solve(ull index) {
        //----------------------------------------------------------------

        ll n, i, j, k;
        cin >> n;

        // cout << "Case #" << index << ": " << ans << newl;

        //----------------------------------------------------------------
    }

    bool test_cases = true;
};
// Solution Class End

// Main Function Start
int main() {
    Solution sol;
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);
    cout.tie(nullptr);
    ull test_cases = 1;
    if (sol.test_cases) {
        cin >> test_cases;
    }
    for (ull test_case = 1; test_case <= test_cases; ++test_case) {
        sol.solve(test_case);
    }
    return 0;
}
// Main Function End
// Program End
//----------------------------------------------------------------