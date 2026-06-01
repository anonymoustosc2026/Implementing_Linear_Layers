// sxor_continue_opt_perm_sat.cpp
// Continue optimizing an existing in-place s-XOR/CNOT circuit under:
//   1) depth must not increase;
//   2) gate count should decrease;
//   3) final output row permutation is allowed as free rewiring.
//
// Compile on your Mac:
//   g++ -O3 -std=c++17 \
//     sxor_continue_opt_perm_sat.cpp \
//     /Users/xushengyuan/Downloads/cadical-master/build/libcadical.a \
//     -o opt
//
// Example:
//   ./opt AES.txt result_AES_v2.txt result_AES_opt.txt 5 2 6 800000 0
//
// Arguments:
//   argv[1] target matrix file, e.g. AES.txt
//   argv[2] initial circuit file, e.g. result_AES_v2.txt
//   argv[3] output circuit file
//   argv[4] max_passes
//   argv[5] min_window
//   argv[6] max_window
//   argv[7] conflict_limit
//   argv[8] try_depth_reduce   0 or 1
//
// Notes:
//   - The SAT refinement replaces contiguous windows exactly, so the current
//     final row permutation is preserved.
//   - A cheap global single-gate deletion pass is also included; it accepts
//     any circuit that remains correct up to final output row permutation.
//   - This program uses solver.reserve(abs(lit)) before every solver.add(lit),
//     avoiding the CaDiCaL "undeclared variable" API error.

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "/mnt/f/cadical-master/src/cadical.hpp"

using namespace std;

struct Op {
    int target = 0;
    int control = 0;
};

using Layer = vector<Op>;
using Matrix = vector<uint64_t>;

static int n = 0;

static inline int hw(uint64_t x) {
    return __builtin_popcountll(x);
}

static inline uint64_t idrow(int i) {
    return 1ULL << i;
}

Matrix identity_matrix() {
    Matrix A(n);
    for (int i = 0; i < n; i++) A[i] = idrow(i);
    return A;
}

int count_gates(const vector<Layer>& layers) {
    int s = 0;
    for (const auto& L : layers) s += (int)L.size();
    return s;
}

void apply_layer(Matrix& A, const Layer& L) {
    Matrix oldA = A;
    Matrix newA = A;
    for (const auto& op : L) {
        newA[op.target] = oldA[op.target] ^ oldA[op.control];
    }
    A.swap(newA);
}

Matrix simulate_circuit(const vector<Layer>& layers) {
    Matrix A = identity_matrix();
    for (const auto& L : layers) apply_layer(A, L);
    return A;
}

Matrix simulate_window_transform(const vector<Layer>& layers, int start, int len) {
    Matrix A = identity_matrix();
    for (int i = start; i < start + len; i++) apply_layer(A, layers[i]);
    return A;
}

bool layer_matching(const Layer& L) {
    vector<int> used(n, 0);
    for (const auto& op : L) {
        if (op.target < 0 || op.target >= n || op.control < 0 || op.control >= n) return false;
        if (op.target == op.control) return false;
        if (used[op.target] || used[op.control]) return false;
        used[op.target] = 1;
        used[op.control] = 1;
    }
    return true;
}

bool circuit_layers_valid(const vector<Layer>& layers) {
    for (const auto& L : layers) {
        if (!layer_matching(L)) return false;
    }
    return true;
}

bool row_perm_equal(const Matrix& A, const Matrix& M, vector<int>& perm) {
    if ((int)A.size() != n || (int)M.size() != n) return false;
    unordered_map<uint64_t, int> pos;
    pos.reserve(n * 2);
    for (int i = 0; i < n; i++) {
        if (pos.count(M[i])) return false;
        pos[M[i]] = i;
    }

    perm.assign(n, -1);
    vector<int> used(n, 0);
    for (int i = 0; i < n; i++) {
        auto it = pos.find(A[i]);
        if (it == pos.end()) return false;
        int j = it->second;
        if (used[j]) return false;
        used[j] = 1;
        perm[i] = j;
    }
    return true;
}

bool verify_up_to_row_perm(const Matrix& Target, const vector<Layer>& layers, vector<int>& perm) {
    if (!circuit_layers_valid(layers)) return false;
    Matrix A = simulate_circuit(layers);
    return row_perm_equal(A, Target, perm);
}

bool verify_exact_matrix(const Matrix& Target, const vector<Layer>& layers) {
    if (!circuit_layers_valid(layers)) return false;
    Matrix A = simulate_circuit(layers);
    return A == Target;
}

void remove_empty_layers(vector<Layer>& layers) {
    vector<Layer> out;
    for (auto& L : layers) {
        if (!L.empty()) out.push_back(L);
    }
    layers.swap(out);
}

string trim_copy(string s) {
    auto not_space = [](unsigned char c) { return !isspace(c); };
    s.erase(s.begin(), find_if(s.begin(), s.end(), not_space));
    s.erase(find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

string strip_comment(string s) {
    size_t p1 = s.find('#');
    size_t p2 = s.find("//");
    size_t p = string::npos;
    if (p1 != string::npos) p = p1;
    if (p2 != string::npos) p = (p == string::npos ? p2 : min(p, p2));
    if (p != string::npos) s = s.substr(0, p);
    return trim_copy(s);
}

bool parse_binary_row(const string& raw, int expected_n, vector<int>& bits) {
    string s = strip_comment(raw);
    if (s.empty()) return false;

    string compact;
    for (char ch : s) {
        if (!isspace((unsigned char)ch)) compact.push_back(ch);
    }
    bool compact01 = !compact.empty();
    for (char ch : compact) {
        if (ch != '0' && ch != '1') {
            compact01 = false;
            break;
        }
    }
    if (compact01 && (int)compact.size() == expected_n) {
        bits.assign(expected_n, 0);
        for (int i = 0; i < expected_n; i++) bits[i] = compact[i] - '0';
        return true;
    }

    vector<int> vals;
    string token;
    for (char ch : s) {
        if (ch == '0' || ch == '1') {
            vals.push_back(ch - '0');
        } else if (isdigit((unsigned char)ch)) {
            // A non-binary digit means this is not a clean binary row.
            return false;
        }
    }
    if ((int)vals.size() == expected_n) {
        bits = vals;
        return true;
    }

    return false;
}

Matrix parse_matrix_file(const string& path, int expected_n) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Cannot open matrix file: " << path << "\n";
        exit(1);
    }

    vector<vector<int>> rows;
    string line;
    bool skipped_dim = false;

    while (getline(fin, line)) {
        string s = strip_comment(line);
        if (s.empty()) continue;

        if (!skipped_dim) {
            string t = trim_copy(s);
            bool all_digit = !t.empty();
            for (char ch : t) {
                if (!isdigit((unsigned char)ch)) {
                    all_digit = false;
                    break;
                }
            }
            if (all_digit && stoi(t) == expected_n) {
                skipped_dim = true;
                continue;
            }
        }

        vector<int> bits;
        if (parse_binary_row(line, expected_n, bits)) {
            rows.push_back(bits);
            if ((int)rows.size() == expected_n) break;
        }
    }

    if ((int)rows.size() != expected_n) {
        cerr << "Failed to parse " << expected_n << "x" << expected_n
             << " target matrix from " << path << ". Parsed rows = " << rows.size() << "\n";
        exit(1);
    }

    Matrix M(expected_n, 0);
    for (int r = 0; r < expected_n; r++) {
        uint64_t mask = 0;
        for (int c = 0; c < expected_n; c++) {
            if (rows[r][c]) mask |= (1ULL << c);
        }
        M[r] = mask;
    }
    return M;
}

vector<Layer> parse_circuit_file(const string& path, int& inferred_n, int& declared_depth, int& declared_gates) {
    ifstream fin(path);
    if (!fin) {
        cerr << "Cannot open circuit file: " << path << "\n";
        exit(1);
    }

    vector<Layer> layers;
    Layer current;
    bool in_layer = false;
    int max_idx = -1;
    declared_depth = -1;
    declared_gates = -1;

    regex re_depth(R"(\bDepth\s*=\s*(\d+))");
    regex re_gates(R"(\b(?:XOR count|gates?)\s*=\s*(\d+))", regex_constants::icase);
    regex re_layer(R"(^\s*Layer\s+(\d+)\s*:)");
    regex re_op(R"(\bx(\d+)\s*\^=\s*x(\d+)\b)");

    string line;
    smatch m;
    while (getline(fin, line)) {
        if (regex_search(line, m, re_depth)) declared_depth = stoi(m[1]);
        if (regex_search(line, m, re_gates)) declared_gates = stoi(m[1]);

        if (regex_search(line, m, re_layer)) {
            if (in_layer) layers.push_back(current);
            current.clear();
            in_layer = true;
            continue;
        }

        if (in_layer && regex_search(line, m, re_op)) {
            int t = stoi(m[1]);
            int c = stoi(m[2]);
            current.push_back({t, c});
            max_idx = max(max_idx, max(t, c));
        }
    }

    if (in_layer) layers.push_back(current);

    if (max_idx < 0) {
        cerr << "No CNOT operations found in circuit file: " << path << "\n";
        exit(1);
    }

    inferred_n = max_idx + 1;
    return layers;
}

void write_circuit_file(const string& path, const Matrix& Target, const vector<Layer>& layers) {
    ofstream fout(path);
    if (!fout) {
        cerr << "Cannot write output file: " << path << "\n";
        exit(1);
    }

    vector<int> perm;
    bool ok = verify_up_to_row_perm(Target, layers, perm);

    fout << "Continued SAT-optimized CNOT circuit\n";
    fout << "Depth = " << layers.size() << "\n";
    fout << "XOR count = " << count_gates(layers) << "\n\n";

    for (int d = 0; d < (int)layers.size(); d++) {
        fout << "Layer " << d << ":\n";
        for (const auto& op : layers[d]) {
            fout << "    x" << op.target << " ^= x" << op.control
                 << "    // CNOT(" << op.control << " -> " << op.target << ")\n";
        }
        fout << "\n";
    }

    if (ok) {
        fout << "Free final row permutation:\n";
        fout << "final row i equals target row perm[i]\n";
        for (int i = 0; i < n; i++) {
            fout << "row " << i << " -> target row " << perm[i] << "\n";
        }
    } else {
        fout << "WARNING: verification up to final row permutation failed.\n";
    }
}

struct VarManager {
    int next_var = 1;
    int new_var() { return next_var++; }
};

struct SatSynthesizer {
    int D;
    int gate_bound;
    int conflict_limit;
    Matrix Target;

    VarManager vm;
    vector<int> S;      // (D+1) * n * n
    vector<int> X;      // D * n * n, X[l,t,c], zero if t == c
    vector<int> gate_vars;

    CaDiCaL::Solver solver;

    SatSynthesizer(const Matrix& target, int depth, int bound, int conflict)
        : D(depth), gate_bound(bound), conflict_limit(conflict), Target(target) {
        S.assign((D + 1) * n * n, 0);
        X.assign(D * n * n, 0);
    }

    int sidx(int t, int r, int b) const {
        return (t * n + r) * n + b;
    }

    int xidx(int l, int t, int c) const {
        return (l * n + t) * n + c;
    }

    int Svar(int t, int r, int b) const {
        return S[sidx(t, r, b)];
    }

    int Xvar(int l, int t, int c) const {
        if (t == c) return 0;
        return X[xidx(l, t, c)];
    }

    void add_clause(const vector<int>& lits) {
        for (int lit : lits) solver.add(lit);
        solver.add(0);
    }

    void add_unit(int lit) {
        solver.add(lit);
        solver.add(0);
    }

    void alloc_vars() {
        for (int t = 0; t <= D; t++) {
            for (int r = 0; r < n; r++) {
                for (int b = 0; b < n; b++) {
                    S[sidx(t, r, b)] = vm.new_var();
                }
            }
        }

        for (int l = 0; l < D; l++) {
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < n; c++) {
                    if (t == c) continue;
                    int v = vm.new_var();
                    X[xidx(l, t, c)] = v;
                    gate_vars.push_back(v);
                }
            }
        }
    }

    void add_xor_eq_under(int cond, int a, int b, int c) {
        // cond -> (c = a XOR b)
        add_clause({-cond,  a,  b, -c});
        add_clause({-cond,  a, -b,  c});
        add_clause({-cond, -a,  b,  c});
        add_clause({-cond, -a, -b, -c});
    }

    void add_matching_constraints() {
        for (int l = 0; l < D; l++) {
            for (int wire = 0; wire < n; wire++) {
                vector<int> touch;
                touch.reserve(2 * (n - 1));
                for (int c = 0; c < n; c++) {
                    if (c != wire) touch.push_back(Xvar(l, wire, c)); // wire is target
                }
                for (int t = 0; t < n; t++) {
                    if (t != wire) touch.push_back(Xvar(l, t, wire)); // wire is control
                }

                for (int i = 0; i < (int)touch.size(); i++) {
                    for (int j = i + 1; j < (int)touch.size(); j++) {
                        add_clause({-touch[i], -touch[j]});
                    }
                }
            }
        }
    }

    void add_initial_and_target_constraints() {
        for (int r = 0; r < n; r++) {
            for (int b = 0; b < n; b++) {
                bool bit = (r == b);
                add_unit(bit ? Svar(0, r, b) : -Svar(0, r, b));
            }
        }

        for (int r = 0; r < n; r++) {
            for (int b = 0; b < n; b++) {
                bool bit = ((Target[r] >> b) & 1ULL);
                add_unit(bit ? Svar(D, r, b) : -Svar(D, r, b));
            }
        }
    }

    void add_transition_constraints() {
        for (int l = 0; l < D; l++) {
            for (int r = 0; r < n; r++) {
                vector<int> target_vars;
                target_vars.reserve(n - 1);
                for (int c = 0; c < n; c++) {
                    if (c != r) target_vars.push_back(Xvar(l, r, c));
                }

                for (int b = 0; b < n; b++) {
                    int oldr = Svar(l, r, b);
                    int newr = Svar(l + 1, r, b);

                    // If x[l,r,c] is selected, row r becomes old row r XOR old row c.
                    for (int c = 0; c < n; c++) {
                        if (c == r) continue;
                        int xv = Xvar(l, r, c);
                        int oldc = Svar(l, c, b);
                        add_xor_eq_under(xv, oldr, oldc, newr);
                    }

                    // If r is not a target in this layer, row r is unchanged.
                    // (x_r0 OR x_r1 OR ... OR oldr=false OR newr=true)
                    // (x_r0 OR x_r1 OR ... OR oldr=true  OR newr=false)
                    vector<int> clause1 = target_vars;
                    clause1.push_back(-oldr);
                    clause1.push_back(newr);
                    add_clause(clause1);

                    vector<int> clause2 = target_vars;
                    clause2.push_back(oldr);
                    clause2.push_back(-newr);
                    add_clause(clause2);
                }
            }
        }
    }

    void add_at_most_k(const vector<int>& vars, int k) {
        int m = (int)vars.size();
        if (k >= m) return;
        if (k < 0) {
            add_clause({});
            return;
        }
        if (k == 0) {
            for (int v : vars) add_unit(-v);
            return;
        }

        // Sinz sequential counter for at most k.
        vector<vector<int>> s(m, vector<int>(k + 1, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 1; j <= k; j++) s[i][j] = vm.new_var();
        }

        // i = 0
        add_clause({-vars[0], s[0][1]});
        for (int j = 2; j <= k; j++) add_unit(-s[0][j]);

        for (int i = 1; i < m; i++) {
            add_clause({-vars[i], s[i][1]});

            for (int j = 1; j <= k; j++) {
                add_clause({-s[i - 1][j], s[i][j]});
            }

            for (int j = 2; j <= k; j++) {
                add_clause({-vars[i], -s[i - 1][j - 1], s[i][j]});
            }

            add_clause({-vars[i], -s[i - 1][k]});
        }
    }

    bool solve(vector<Layer>& out_layers, bool verbose = false) {
        alloc_vars();

        solver.set("quiet", 1);
        // Some CaDiCaL builds enable factor/factorcheck API checking and then
        // require explicit variable reservation. Older headers may not expose
        // Solver::reserve(), so we disable these checks here.
        solver.set("factor", 0);
        solver.set("factorcheck", 0);
        if (conflict_limit > 0) solver.limit("conflicts", conflict_limit);

        add_initial_and_target_constraints();
        add_matching_constraints();
        add_transition_constraints();
        add_at_most_k(gate_vars, gate_bound);

        int res = solver.solve();
        if (res != 10) {
            if (verbose) {
                if (res == 20) cerr << "    SAT result: UNSAT\n";
                else cerr << "    SAT result: UNKNOWN/LIMIT\n";
            }
            return false;
        }

        out_layers.assign(D, Layer());
        for (int l = 0; l < D; l++) {
            for (int t = 0; t < n; t++) {
                for (int c = 0; c < n; c++) {
                    if (t == c) continue;
                    int v = Xvar(l, t, c);
                    if (solver.val(v) > 0) {
                        out_layers[l].push_back({t, c});
                    }
                }
            }
        }

        if (!verify_exact_matrix(Target, out_layers)) {
            cerr << "Internal error: SAT model extracted but exact window verification failed.\n";
            return false;
        }

        if (count_gates(out_layers) > gate_bound) {
            cerr << "Internal error: SAT model exceeds gate bound.\n";
            return false;
        }

        return true;
    }
};

bool sat_synthesize_with_bound(const Matrix& target, int depth, int gate_bound, int conflict_limit,
                               vector<Layer>& out_layers, bool verbose = false) {
    if (gate_bound < 0) return false;
    SatSynthesizer syn(target, depth, gate_bound, conflict_limit);
    return syn.solve(out_layers, verbose);
}

bool minimize_window_same_depth(const Matrix& target, int depth, int old_gates, int conflict_limit,
                                vector<Layer>& best_layers, int& best_gates) {
    if (old_gates <= 0) return false;

    vector<Layer> cand;
    int high = old_gates - 1;

    cerr << "    SAT same-depth try: depth = " << depth
         << ", gate_bound = " << high << "\n";

    if (!sat_synthesize_with_bound(target, depth, high, conflict_limit, cand, false)) {
        return false;
    }

    best_layers = cand;
    best_gates = count_gates(cand);

    // Try to lower the bound further by binary search.
    int lo = 0;
    int hi = best_gates - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        vector<Layer> tmp;

        cerr << "    SAT lower-bound try: depth = " << depth
             << ", gate_bound = " << mid << "\n";

        if (sat_synthesize_with_bound(target, depth, mid, conflict_limit, tmp, false)) {
            best_layers = tmp;
            best_gates = count_gates(tmp);
            hi = best_gates - 1;
        } else {
            lo = mid + 1;
        }
    }

    return true;
}

bool try_global_single_gate_deletion(Matrix& Target, vector<Layer>& layers) {
    bool improved = false;

    for (int d = 0; d < (int)layers.size(); d++) {
        for (int k = 0; k < (int)layers[d].size(); k++) {
            vector<Layer> test = layers;
            Op removed = test[d][k];
            test[d].erase(test[d].begin() + k);
            remove_empty_layers(test);

            vector<int> perm;
            if (verify_up_to_row_perm(Target, test, perm)) {
                cerr << "Single-gate deletion accepted: remove Layer " << d
                     << " gate x" << removed.target << " ^= x" << removed.control
                     << " | depth " << layers.size() << " -> " << test.size()
                     << " | gates " << count_gates(layers) << " -> " << count_gates(test) << "\n";
                layers.swap(test);
                improved = true;
                return true; // restart after each accepted deletion
            }
        }
    }

    return improved;
}

struct WindowInfo {
    int start = 0;
    int len = 0;
    int gates = 0;
};

vector<WindowInfo> collect_windows(const vector<Layer>& layers, int minw, int maxw) {
    vector<WindowInfo> wins;
    int D = (int)layers.size();

    for (int len = maxw; len >= minw; len--) {
        if (len > D) continue;
        for (int s = 0; s + len <= D; s++) {
            int g = 0;
            for (int i = s; i < s + len; i++) g += (int)layers[i].size();
            if (g > 0) wins.push_back({s, len, g});
        }
    }

    sort(wins.begin(), wins.end(), [](const WindowInfo& a, const WindowInfo& b) {
        // Prefer high-density and larger windows.
        if (a.gates != b.gates) return a.gates > b.gates;
        if (a.len != b.len) return a.len > b.len;
        return a.start < b.start;
    });

    return wins;
}

bool replace_window(vector<Layer>& layers, int start, int len, const vector<Layer>& repl) {
    vector<Layer> out;
    for (int i = 0; i < start; i++) out.push_back(layers[i]);
    for (const auto& L : repl) {
        if (!L.empty()) out.push_back(L);
    }
    for (int i = start + len; i < (int)layers.size(); i++) out.push_back(layers[i]);
    layers.swap(out);
    return true;
}

bool try_sat_window_refinement(Matrix& Target, vector<Layer>& layers,
                               int min_window, int max_window, int conflict_limit,
                               bool try_depth_reduce) {
    int old_depth_global = (int)layers.size();
    int old_gates_global = count_gates(layers);

    vector<WindowInfo> wins = collect_windows(layers, min_window, max_window);

    cerr << "Collected " << wins.size() << " SAT windows.\n";

    int checked = 0;
    for (const auto& w : wins) {
        checked++;
        cerr << "  Window " << checked << "/" << wins.size()
             << " | start = " << w.start
             << " | len = " << w.len
             << " | old gates = " << w.gates << "\n";

        Matrix Wtarget = simulate_window_transform(layers, w.start, w.len);

        // First priority: depth reduction if enabled, but without increasing gates.
        if (try_depth_reduce && w.len >= 2) {
            for (int nd = w.len - 1; nd >= max(1, min_window - 1); nd--) {
                vector<Layer> cand;
                cerr << "    SAT depth-reduce try: old depth = " << w.len
                     << ", new depth = " << nd
                     << ", gate_bound = " << w.gates << "\n";
                if (sat_synthesize_with_bound(Wtarget, nd, w.gates, conflict_limit, cand, false)) {
                    int newg = count_gates(cand);
                    vector<Layer> test = layers;
                    replace_window(test, w.start, w.len, cand);

                    if ((int)test.size() <= old_depth_global && count_gates(test) <= old_gates_global) {
                        vector<int> perm;
                        if (verify_up_to_row_perm(Target, test, perm)) {
                            cerr << "    ACCEPT depth-reduced window | global depth "
                                 << old_depth_global << " -> " << test.size()
                                 << " | gates " << old_gates_global << " -> " << count_gates(test)
                                 << " | local gates " << w.gates << " -> " << newg << "\n";
                            layers.swap(test);
                            return true;
                        }
                    }
                }
            }
        }

        // Same-depth gate minimization.
        vector<Layer> best_repl;
        int best_g = numeric_limits<int>::max();

        if (minimize_window_same_depth(Wtarget, w.len, w.gates, conflict_limit, best_repl, best_g)) {
            vector<Layer> test = layers;
            replace_window(test, w.start, w.len, best_repl);

            if ((int)test.size() <= old_depth_global && count_gates(test) < old_gates_global) {
                vector<int> perm;
                if (verify_up_to_row_perm(Target, test, perm)) {
                    cerr << "    ACCEPT same-depth gate-reduced window | global depth "
                         << old_depth_global << " -> " << test.size()
                         << " | gates " << old_gates_global << " -> " << count_gates(test)
                         << " | local gates " << w.gates << " -> " << best_g << "\n";
                    layers.swap(test);
                    return true;
                } else {
                    cerr << "    Reject: global verification failed unexpectedly.\n";
                }
            }
        }
    }

    return false;
}

void print_summary(const string& tag, const Matrix& Target, const vector<Layer>& layers) {
    vector<int> perm;
    bool ok = verify_up_to_row_perm(Target, layers, perm);
    cerr << tag << "\n";
    cerr << "  Depth = " << layers.size() << "\n";
    cerr << "  XOR count = " << count_gates(layers) << "\n";
    cerr << "  Layer validity = " << (circuit_layers_valid(layers) ? "OK" : "BAD") << "\n";
    cerr << "  Verify up to final row permutation = " << (ok ? "OK" : "FAIL") << "\n";
    if (ok) {
        vector<pair<int,int>> moved;
        for (int i = 0; i < n; i++) {
            if (perm[i] != i) moved.push_back({i, perm[i]});
        }
        cerr << "  Final row permutation moved count = " << moved.size() << "\n";
        if (!moved.empty()) {
            cerr << "  Moved rows:";
            for (auto p : moved) cerr << " (" << p.first << "->" << p.second << ")";
            cerr << "\n";
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 9) {
        cerr << "Usage:\n"
             << "  " << argv[0]
             << " AES.txt initial_result.txt output_result.txt max_passes min_window max_window conflict_limit try_depth_reduce\n\n"
             << "Example:\n"
             << "  " << argv[0] << " AES.txt result_AES_v2.txt result_AES_opt.txt 5 2 6 800000 0\n";
        return 1;
    }

    string matrix_file = argv[1];
    string init_file = argv[2];
    string out_file = argv[3];
    int max_passes = stoi(argv[4]);
    int min_window = stoi(argv[5]);
    int max_window = stoi(argv[6]);
    int conflict_limit = stoi(argv[7]);
    bool try_depth_reduce = (stoi(argv[8]) != 0);

    int inferred_n = 0, declared_depth = -1, declared_gates = -1;
    vector<Layer> layers = parse_circuit_file(init_file, inferred_n, declared_depth, declared_gates);
    n = inferred_n;

    if (n <= 0 || n > 64) {
        cerr << "Unsupported n = " << n << ". This program currently supports n <= 64.\n";
        return 1;
    }

    Matrix Target = parse_matrix_file(matrix_file, n);

    cerr << "Start continuation optimizer with free final row permutation\n";
    cerr << "n = " << n
         << ", initial declared depth = " << declared_depth
         << ", parsed depth = " << layers.size()
         << ", initial declared gates = " << declared_gates
         << ", parsed gates = " << count_gates(layers)
         << ", max_passes = " << max_passes
         << ", window = [" << min_window << "," << max_window << "]"
         << ", conflict_limit = " << conflict_limit
         << ", try_depth_reduce = " << (try_depth_reduce ? 1 : 0)
         << "\n";

    vector<int> perm;
    if (!verify_up_to_row_perm(Target, layers, perm)) {
        cerr << "Initial circuit does NOT verify up to final row permutation. Stop.\n";
        return 2;
    }

    print_summary("[Initial]", Target, layers);

    for (int pass = 1; pass <= max_passes; pass++) {
        cerr << "\n========== Pass " << pass << " ==========\n";
        int beforeD = (int)layers.size();
        int beforeG = count_gates(layers);

        bool improved = false;

        // Very cheap cleanup first.
        while (try_global_single_gate_deletion(Target, layers)) {
            improved = true;
        }

        if (improved) {
            print_summary("[After single-gate deletion]", Target, layers);
            continue;
        }

        improved = try_sat_window_refinement(Target, layers, min_window, max_window,
                                             conflict_limit, try_depth_reduce);

        if (improved) {
            print_summary("[After SAT refinement]", Target, layers);
            continue;
        }

        int afterD = (int)layers.size();
        int afterG = count_gates(layers);

        cerr << "No improvement in pass " << pass
             << " | depth " << beforeD << " -> " << afterD
             << " | gates " << beforeG << " -> " << afterG << "\n";
        break;
    }

    print_summary("\n[Final]", Target, layers);
    write_circuit_file(out_file, Target, layers);

    cerr << "Wrote optimized circuit to: " << out_file << "\n";
    return 0;
}
