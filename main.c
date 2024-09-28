// Author : Naveen

// Program Start
// Libraries Start
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <limits.h>
#include <conio.h>
#include <errno.h>
#include <assert.h>
#include <time.h>
// Libraries End

//----------------------------------------------------------------

// Important Shortcuts Start
// Macros Start
#define yes printf("yes\n")
#define no printf("no\n")
#define Yes printf("Yes\n")
#define No printf("No\n")
#define YES printf("Yes\n")
#define NO printf("No\n")
// Macros End

// Declarations Start
typedef long long int ll;
typedef unsigned long long int ull;
typedef long double ld;
// Declarations End

// Constants Start
const ull mod1 = 1000000007LL;
const ull mod2 = 998244353LL;
const ull max_if = 1000000LL;
const ll inf = __LONG_LONG_MAX__;
const ll neg_inf = -__LONG_LONG_MAX__ - 1ll;
const char spc = ' ';
const char newl = '\n';
// Constants End


// Input and Output Functions Start
void input(const ll arr[], const ull size) {
    for (ull i = 0;i < size;++i) {
        scanf("%lld", arr + i);
    }
}

void Yn(const bool con) {
    printf("%s\n", con ? "Yes" : "No");
}
void YN(const bool con) {
    printf("%s\n", con ? "YES" : "NO");
}

void print(const ll val) {
    printf("%s ", val);
}

void print_arr(const ll arr[], const ull size) {
    for (ull i = 0;i < size;++i) {
        printf("%lld ", arr[i]);
    }
}

void println(const ll val) {
    printf("%s\n", val);
}

void println_arr(const ll arr[], const ull size) {
    for (ull i = 0;i < size;++i) {
        printf("%lld ", arr[i]);
    }
    printf("\n");
}
// Input and Output Functions End
// Important Shortcuts End

//----------------------------------------------------------------

// Basic Functions Start
// Swap Function Start
void swap(ll* a, ll* b) {
    ll temp = *a;
    *a = *b;
    *b = temp;
}
// Swap Function End

// Min Function Start
ll min(const ll a, const ll b) {
    return a < b ? a : b;
}
ll min_arr(const ll arr[], const ull size) {
    ll mini = arr[0];
    for (ull i = 1; i < size; ++i) {
        mini = min(mini, arr[i]);
    }
    return mini;
}
// Min Function End

// Max Function Start
ll max(const ll a, const ll b) {
    return a > b ? a : b;
}
ll max_arr(const ll arr[], const ull size) {
    ll maxi = arr[0];
    for (ull i = 1;i < size;++i) {
        maxi = max(maxi, arr[i]);
    }
    return maxi;
}
// Max Function End

// Sum Function Start
ll sum(const ll arr[], const ull size) {
    ll sumi = 0;
    for (ull i = 0;i < size;++i) {
        sumi += arr[i];
    }
    return sumi;
}
// Sum Function End

// Compare Function Start
int compare(const void* a, const void* b) {
    return (*(ll*)a - *(ll*)b);
}
// Compare Function End

// Sort Start
ll sort(ll arr[], const ull size) {
    qsort(arr, size, sizeof(ll), compare);
}
// Sort End

// Binary Search Start
ll bin_search(ll arr[], const ull size, const ll key) {
    ll* res = (ll*)bsearch(&key, arr, size, sizeof(ll), compare);
    if (res != NULL) {
        return res - arr;
    }
    else {
        return -1;
    }
}
// Binary Search End

// Binary Start
char* bin(ull dn) {
    static char binStr[65];
    int i = 63;
    while (!(dn & (1ULL << i)) && i >= 0) {
        --i;
    }
    if (i >= 0) {
        int j;
        for (j = 0; i >= 0; --i, j++) {
            binStr[j] = ((dn & (1ULL << i)) ? '1' : '0');
        }
        binStr[j] = '\0';
        return binStr;
    }
    return "0";
}

ull int2(const char* bin_str) {
    ull dn = 0;
    int len = strlen(bin_str);
    int i;
    for (i = 0; i < len; ++i) {
        dn <<= 1;
        if (bin_str[i] == '1') {
            dn |= 1;
        }
    }
    return dn;
}
// Binary End

// Square Root Start
ull sqrt_int(ll n) {
    if (n < 0) {
        return 11;
    }
    return (ull)sqrt(n);
}

ull sqrt_int_u(ll n) {
    if (n < 0) {
        return -1;
    }
    ull k = sqrt(n);
    return k * k == n ? k : k + 1;
}
// Square Root End

// Binary Exponentiation Start
ull power_mod(ull b, ull p, const ull mod) {
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

ull power(ull b, ull p) {
    return power_mod(b, p, inf);
}
// Binary Exponentiation End

// Prime Number Start
bool is_prime_old(ll n) {
    if (n <= 0) {
        return -1;
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
    for (ll i = 5; i < sqrt(n) + 1; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

bool miller(ull d, ull n, ull a) {
    if (a % n == 0) {
        return true;
    }
    ull x = power_mod(a, d, n);
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
}
bool is_prime(ll n) {
    if (n <= 0) {
        return -1;
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
    ull d = n - 1;
    while (d % 2 == 0) {
        d /= 2;
    }
    ull bases64_size = 7;
    ull bases64[] = { 2, 325, 9375, 28178, 450775, 9780504, 1795265022 };
    ull bases32_size = 3;
    ull bases32[] = { 2, 7, 61 };
    ull bases_size = n <= 4294967296ll ? bases32_size : bases64_size;
    ull* bases = n <= 4294967296ll ? bases32 : bases64;
    for (ull i = 0; i < bases_size; ++i) {
        if (!miller(d, n, bases[i])) {
            return false;
        }
    }
    return true;
}

void primes(ll prime[], ll* cou, const ull size) {
    ull i, j;
    bool prime_is[size + 1];
    prime_is[0] = false;
    prime_is[1] = false;
    for (i = 2; i <= size; ++i) {
        prime_is[i] = true;
    }
    for (i = 2; i <= sqrt_int_u(size); ++i) {
        if (prime_is[i]) {
            for (j = i * i; j <= size; j += i) {
                prime_is[j] = false;
            }
        }
    }
    for (i = 2; i <= size; ++i) {
        if (prime_is[i]) {
            prime[*cou++] = i;
        }
    }
}
// Prime Number End

// KMP Algorithm Start
ll substr_is_in(const char* text, const char* pattern) {
    const size_t m = strlen(pattern);
    const size_t n = strlen(text);
    ull lps[m];
    for (ll i = 0; i < m; ++i) {
        lps[i] = 0;
    }
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
ll mod_inv_gcd(ll n, const ll mod) {
    if (n < 0) {
        return -1;
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
        return 1;
    }
    return (t1 % mod + mod) % mod;
}

ll mod_inv(ll n, const ll mod) {
    if (n < 0) {
        return -1;
    }
    return power_mod(n, mod - 2, mod);
}
// Modular Inverse Function End

// Permutations and Combinations Start
ull fact_mod(ll n, const ull mod) {
    if (n < 0) {
        return -1;
    }
    ull ans = 1;
    for (ull i = 1; i <= n; ++i) {
        ans = ans * i % mod;
    }
    return ans;
}
ull fact(ll n) {
    return fact_mod(n, inf);
}

void fact_arr_mod(ll arr[], const ull size, const ll mod) {
    arr[0] = 1;
    for (size_t i = 1; i < size; ++i) {
        arr[i] = i * arr[i - 1] % mod;
    }
}
void fact_arr(ll arr[], const ull size) {
    fact_arr_mod(arr, size, inf);
}

ull perm_mod(ll n, ll r, const ull mod) {
    if (n < 0 || r < 0) {
        return -1;
    }
    ull ans = 1;
    for (ull i = n - r + 1; i <= n; ++i) {
        ans = ans * i % mod;
    }
    return ans;
}
ull perm(ll n, ll r) {
    return perm_mod(n, r, inf);
}

ull comb(ll n, ll r) {
    if (n < 0 || r < 0) {
        return -1;
    }
    ull num = 1, den = 1;
    if (r > n / 2) {
        r = n - r;
    }
    n++;
    for (ull i = 1; i <= r; ++i) {
        num *= n - i;
        den *= i;
    }
    return num / den;
}
ull comb_mod(ll n, ll r, const ull mod) {
    if (n < 0 || r < 0) {
        return -1;
    }
    ull num = 1, den = 1;
    if (r > n / 2) {
        r = n - r;
    }
    n++;
    for (ull i = 1; i <= r; ++i) {
        num = num * (n - i) % mod;
        den = den * i % mod;
    }
    return (num * mod_inv(den, mod)) % mod;
}

ull fast_fib_mod(ll n, const ull mod) {
    if (n < 0) {
        return -1;
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
ull fast_fib(ll n) {
    return fast_fib_mod(n, inf);
}
// Permutations and Combinations End
// Modular Arithmetic End

//-----------------------------------------------------------------

// Solution Function Start
void solve(ull index) {
    //----------------------------------------------------------------

    ll n, i, j, k;
    scanf("%lld", &n);
    getchar();

    // printf("Case #%llu: %lld\n", index, ans);

    //----------------------------------------------------------------
}
// Solution Function End

// Main Function Start
int main() {
    bool test_cases = true;
    ull tests = 1;
    if (test_cases) {
        scanf("%llu", &tests);
        getchar();
    }
    for (ull test = 1; test <= tests; ++test) {
        solve(test);
    }
    return 0;
}
// Main Function End
// Program End
//----------------------------------------------------------------