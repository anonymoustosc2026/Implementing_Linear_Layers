// sxor_depth_priority_satfix.cpp
// Depth-priority

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "/mnt/f/cadical-master/src/cadical.hpp"
using namespace std;

struct Op {
    int target;
    int control;
};

using Layer = vector<Op>;

struct RawOp {
    int target;
    int control;
    int type; // 0 row, 1 column
};

struct Result {
    int depth = INT_MAX;
    int gates = INT_MAX;
    vector<Layer> layers;
    vector<int> final_perm;
};

struct VarManager {
    int next_var = 1;
    int new_var() { return next_var++; }
};

int n;
int CONFLICT_LIMIT = 200000;
int POOL_PER_DEPTH = 20;

mt19937_64 rng((uint64_t)chrono::steady_clock::now().time_since_epoch().count());
VarManager vm;

vector<vector<vector<int>>> S;
vector<vector<vector<int>>> X;

uint64_t idrow(int i) {
    return 1ULL << i;
}

int hw(uint64_t x) {
    return __builtin_popcountll(x);
}

vector<uint64_t> identity_matrix() {
    vector<uint64_t> A(n);
    for (int i = 0; i < n; i++) A[i] = idrow(i);
    return A;
}

uint64_t col_bits(const vector<uint64_t>& A, int c) {
    uint64_t x = 0;
    for (int r = 0; r < n; r++) {
        if ((A[r] >> c) & 1ULL) x |= (1ULL << r);
    }
    return x;
}

void apply_layer(vector<uint64_t>& A, const Layer& L) {
    vector<uint64_t> oldA = A;
    vector<uint64_t> newA = A;

    for (auto op : L) {
        newA[op.target] = oldA[op.target] ^ oldA[op.control];
    }

    A = newA;
}

void apply_row_op(vector<uint64_t>& A, int target, int control) {
    A[target] ^= A[control];
}

void apply_col_op(vector<uint64_t>& A, int target, int control) {
    uint64_t mt = 1ULL << target;
    uint64_t mc = 1ULL << control;

    for (int r = 0; r < n; r++) {
        if (A[r] & mc) A[r] ^= mt;
    }
}

bool invert_matrix(const vector<uint64_t>& A, vector<uint64_t>& Inv) {
    vector<uint64_t> L = A;
    Inv = identity_matrix();

    for (int col = 0; col < n; col++) {
        int piv = -1;

        for (int r = col; r < n; r++) {
            if ((L[r] >> col) & 1ULL) {
                piv = r;
                break;
            }
        }

        if (piv == -1) return false;

        if (piv != col) {
            swap(L[piv], L[col]);
            swap(Inv[piv], Inv[col]);
        }

        for (int r = 0; r < n; r++) {
            if (r == col) continue;

            if ((L[r] >> col) & 1ULL) {
                L[r] ^= L[col];
                Inv[r] ^= Inv[col];
            }
        }
    }

    return true;
}

double hsq_rows(const vector<uint64_t>& A) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        int w = hw(A[i]);
        s += 1.0 * w * w;
    }
    return s;
}

double hsq_cols(const vector<uint64_t>& A) {
    double s = 0;
    for (int c = 0; c < n; c++) {
        int w = hw(col_bits(A, c));
        s += 1.0 * w * w;
    }
    return s;
}

double hprod_rows(const vector<uint64_t>& A) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        int w = hw(A[i]);
        if (w == 0) return 1e100;
        s += log2((double)w);
    }
    return s;
}

double hprod_cols(const vector<uint64_t>& A) {
    double s = 0;
    for (int c = 0; c < n; c++) {
        int w = hw(col_bits(A, c));
        if (w == 0) return 1e100;
        s += log2((double)w);
    }
    return s;
}

double cost_row_eval(const vector<uint64_t>& A, int mode) {
    vector<uint64_t> Inv;
    if (!invert_matrix(A, Inv)) return 1e100;

    if (mode == 0) return hsq_rows(A) + hsq_cols(Inv);
    return hprod_rows(A) + hprod_cols(Inv);
}

double cost_col_eval(const vector<uint64_t>& A, int mode) {
    vector<uint64_t> Inv;
    if (!invert_matrix(A, Inv)) return 1e100;

    if (mode == 0) return hsq_cols(A) + hsq_rows(Inv);
    return hprod_cols(A) + hprod_rows(Inv);
}

double current_cost(const vector<uint64_t>& A, int mode) {
    return max(cost_row_eval(A, mode), cost_col_eval(A, mode));
}

bool is_permutation_matrix(const vector<uint64_t>& A, vector<int>& perm) {
    perm.assign(n, -1);
    vector<int> used(n, 0);

    for (int r = 0; r < n; r++) {
        if (hw(A[r]) != 1) return false;

        int c = __builtin_ctzll(A[r]);
        if (c < 0 || c >= n || used[c]) return false;

        used[c] = 1;
        perm[r] = c;
    }

    return true;
}

bool raw_conflict(const vector<RawOp>& L, int t, int c) {
    for (auto op : L) {
        if (t == op.target || t == op.control ||
            c == op.target || c == op.control) {
            return true;
        }
    }
    return false;
}

bool layer_matching(const Layer& L) {
    vector<int> used(n, 0);

    for (auto op : L) {
        if (op.target == op.control) return false;
        if (op.target < 0 || op.target >= n) return false;
        if (op.control < 0 || op.control >= n) return false;
        if (used[op.target] || used[op.control]) return false;

        used[op.target] = 1;
        used[op.control] = 1;
    }

    return true;
}

bool row_perm_equal(const vector<uint64_t>& A, const vector<uint64_t>& M, vector<int>& perm) {
    perm.assign(n, -1);
    vector<int> used(n, 0);

    for (int i = 0; i < n; i++) {
        bool ok = false;

        for (int k = 0; k < n; k++) {
            if (!used[k] && A[i] == M[k]) {
                perm[i] = k;
                used[k] = 1;
                ok = true;
                break;
            }
        }

        if (!ok) return false;
    }

    return true;
}

bool verify_circuit(const vector<uint64_t>& M, const vector<Layer>& layers, vector<int>& perm) {
    vector<uint64_t> A = identity_matrix();

    for (auto& L : layers) {
        if (!layer_matching(L)) return false;
        apply_layer(A, L);
    }

    if (A == M) {
        perm.resize(n);
        iota(perm.begin(), perm.end(), 0);
        return true;
    }

    return row_perm_equal(A, M, perm);
}

bool verify_exact_matrix(const vector<uint64_t>& Target, const vector<Layer>& layers) {
    vector<uint64_t> A = identity_matrix();

    for (auto& L : layers) {
        if (!layer_matching(L)) return false;
        apply_layer(A, L);
    }

    return A == Target;
}

int count_gates(const vector<Layer>& layers) {
    int s = 0;
    for (auto& L : layers) s += (int)L.size();
    return s;
}


int ceil_log2_int(int x) {
    if (x <= 1) return 0;
    int v = 1, k = 0;
    while (v < x) {
        v <<= 1;
        k++;
    }
    return k;
}

int max_row_weight(const vector<uint64_t>& A) {
    int mx = 0;
    for (auto r : A) mx = max(mx, hw(r));
    return mx;
}

int max_col_weight(const vector<uint64_t>& A) {
    int mx = 0;
    for (int c = 0; c < n; c++) mx = max(mx, hw(col_bits(A, c)));
    return mx;
}

double depth_pressure(const vector<uint64_t>& A) {
    vector<uint64_t> Inv;
    if (!invert_matrix(A, Inv)) return 1e100;

    int a = max_row_weight(A);
    int b = max_col_weight(A);
    int c = max_row_weight(Inv);
    int d = max_col_weight(Inv);

    return (double)max({
        ceil_log2_int(a),
        ceil_log2_int(b),
        ceil_log2_int(c),
        ceil_log2_int(d)
        });
}


double depth_first_score(const vector<uint64_t>& A, int mode) {
    return current_cost(A, mode);
}

double operation_oriented_score(const vector<uint64_t>& A, int type, int mode) {
    if (type == 0) return cost_row_eval(A, mode);
    return cost_col_eval(A, mode);
}

void apply_raw_layer_to_matrix(vector<uint64_t>& A, const vector<RawOp>& L) {
    if (L.empty()) return;

    for (auto op : L) {
        if (op.type == 0) apply_row_op(A, op.target, op.control);
        else apply_col_op(A, op.target, op.control);
    }
}

struct MatchingBuildResult {
    vector<RawOp> layer;
    vector<uint64_t> after;
    double score = 1e100;
    double gain = -1e100;
};

MatchingBuildResult build_best_matching_layer(
    const vector<uint64_t>& B,
    int type,
    int mode
) {
    const double EPS = 1e-9;
    double base_score = current_cost(B, mode);

    struct Cand {
        int t;
        int c;
        double gain;
        double single_score;
        uint64_t rnd;
    };

    vector<Cand> cand;

    for (int t = 0; t < n; t++) {
        for (int c = 0; c < n; c++) {
            if (t == c) continue;

            vector<uint64_t> T = B;
            if (type == 0) apply_row_op(T, t, c);
            else apply_col_op(T, t, c);

            double sc = operation_oriented_score(T, type, mode);
            double g = base_score - sc;

            if (g > EPS) {
                cand.push_back({ t, c, g, sc, rng() });
            }
        }
    }

    sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) {
        if (fabs(a.gain - b.gain) > 1e-9) return a.gain > b.gain;
        return a.rnd < b.rnd;
        });

    vector<int> used(n, 0);
    vector<RawOp> L;

    int choice_width = 1;
    uint64_t rr = rng() % 100;
    if (rr < 55) choice_width = 1;  
    else if (rr < 80) choice_width = 3;
    else if (rr < 94) choice_width = 6;
    else choice_width = 12;

    while (true) {
        vector<int> avail;
        for (int idx = 0; idx < (int)cand.size(); idx++) {
            auto& x = cand[idx];
            if (!used[x.t] && !used[x.c]) avail.push_back(idx);
        }
        if (avail.empty()) break;

        int k = min(choice_width, (int)avail.size());
        int pick_pos = 0;
        if (k > 1) {
            uint64_t z = rng() % 100;
            if (z < 60) pick_pos = 0;
            else pick_pos = (int)(rng() % k);
        }
        auto x = cand[avail[pick_pos]];
        used[x.t] = used[x.c] = 1;
        L.push_back({ x.t, x.c, type });
    }

    MatchingBuildResult ret;
    ret.after = B;

    if (L.empty()) {
        ret.score = base_score;
        ret.gain = 0.0;
        return ret;
    }

    while (!L.empty()) {
        vector<uint64_t> T = B;
        apply_raw_layer_to_matrix(T, L);
        double sc = operation_oriented_score(T, type, mode);

        if (sc + EPS < base_score) {
            ret.layer = L;
            ret.after = T;
            ret.score = sc;
            ret.gain = base_score - sc;
            return ret;
        }

        L.pop_back();
    }

    ret.score = base_score;
    ret.gain = 0.0;
    return ret;
}

vector<Layer> compact_layers_preserve_order(const vector<Layer>& layers) {
    vector<Layer> compacted;
    vector<int> last(n, -1);

    for (const auto& L : layers) {
        for (auto op : L) {
            int earliest = max(last[op.target], last[op.control]) + 1;

            while ((int)compacted.size() <= earliest) {
                compacted.push_back(Layer());
            }

            compacted[earliest].push_back(op);
            last[op.target] = earliest;
            last[op.control] = earliest;
        }
    }

    vector<Layer> cleaned;
    for (auto& L : compacted) {
        if (!L.empty()) cleaned.push_back(L);
    }
    return cleaned;
}

bool compact_result_if_valid(const vector<uint64_t>& M, Result& res) {
    vector<Layer> old = res.layers;
    vector<Layer> nl = compact_layers_preserve_order(res.layers);

    vector<int> perm;
    if (!verify_circuit(M, nl, perm)) {
        res.layers = old;
        return false;
    }

    int oldD = res.depth;
    int oldG = res.gates;

    res.layers = nl;
    res.depth = (int)nl.size();
    res.gates = count_gates(nl);
    res.final_perm = perm;

    return res.depth < oldD || res.gates < oldG;
}

bool better_depth_first(const Result& a, const Result& b) {
    if (a.depth != b.depth) return a.depth < b.depth;
    return a.gates < b.gates;
}

Layer convert_col_layer(const vector<RawOp>& LC, int variant) {
    Layer L;

    for (auto op : LC) {
        if (variant == 0) L.push_back({ op.control, op.target });
        else L.push_back({ op.target, op.control });
    }

    return L;
}

Layer convert_row_layer(const vector<RawOp>& LR, const vector<int>& perm, int variant) {
    vector<int> invperm(n, -1);

    for (int i = 0; i < n; i++) {
        if (perm[i] >= 0 && perm[i] < n) invperm[perm[i]] = i;
    }

    Layer L;

    for (auto op : LR) {
        int t = op.target;
        int c = op.control;

        if (variant == 0) L.push_back({ perm[t], perm[c] });
        else if (variant == 1) L.push_back({ perm[c], perm[t] });
        else if (variant == 2) L.push_back({ invperm[t], invperm[c] });
        else L.push_back({ invperm[c], invperm[t] });
    }

    return L;
}

bool build_and_verify(
    const vector<uint64_t>& M,
    const vector<vector<RawOp>>& row_layers,
    const vector<vector<RawOp>>& col_layers,
    const vector<int>& perm,
    Result& res
) {
    for (int cv = 0; cv < 2; cv++) {
        for (int rv = 0; rv < 4; rv++) {
            vector<Layer> layers;

            for (auto& rawL : col_layers) {
                Layer L = convert_col_layer(rawL, cv);
                if (!L.empty()) layers.push_back(L);
            }

            for (int i = (int)row_layers.size() - 1; i >= 0; i--) {
                Layer L = convert_row_layer(row_layers[i], perm, rv);
                if (!L.empty()) layers.push_back(L);
            }

            vector<int> final_perm;

            if (verify_circuit(M, layers, final_perm)) {
                res.layers = layers;
                res.depth = (int)layers.size();
                res.gates = count_gates(layers);
                res.final_perm = final_perm;
                return true;
            }
        }
    }

    return false;
}

vector<uint64_t> transpose_matrix_bits(const vector<uint64_t>& A);

bool finish_with_one_row_layer(const vector<uint64_t>& B, vector<RawOp>& finish) {
    finish.clear();

    vector<int> singleton_row_of_col(n, -1);
    vector<int> weight(n, 0);
    vector<pair<int,int>> pair_cols(n, {-1, -1});

    for (int r = 0; r < n; r++) {
        int w = hw(B[r]);
        weight[r] = w;
        if (w == 0 || w > 2) return false;
        if (w == 1) {
            int c = __builtin_ctzll(B[r]);
            if (singleton_row_of_col[c] == -1) singleton_row_of_col[c] = r;
        } else {
            int a = __builtin_ctzll(B[r]);
            int b = __builtin_ctzll(B[r] & ~(1ULL << a));
            pair_cols[r] = {a, b};
        }
    }

    vector<int> col_used_by_pair(n, 0);
    for (int r = 0; r < n; r++) {
        if (weight[r] != 2) continue;
        int a = pair_cols[r].first, b = pair_cols[r].second;
        if (col_used_by_pair[a] || col_used_by_pair[b]) return false;
        col_used_by_pair[a] = col_used_by_pair[b] = 1;
    }

    vector<int> row_used(n, 0);
    for (int r = 0; r < n; r++) {
        if (weight[r] != 2) continue;
        int a = pair_cols[r].first, b = pair_cols[r].second;
        int ctrl = -1;
        if (singleton_row_of_col[a] != -1 && !row_used[singleton_row_of_col[a]]) ctrl = singleton_row_of_col[a];
        else if (singleton_row_of_col[b] != -1 && !row_used[singleton_row_of_col[b]]) ctrl = singleton_row_of_col[b];
        else return false;
        if (ctrl == r || row_used[r]) return false;
        row_used[r] = row_used[ctrl] = 1;
        finish.push_back({r, ctrl, 0});
    }

    if (finish.empty()) return false;
    vector<uint64_t> T = B;
    apply_raw_layer_to_matrix(T, finish);
    vector<int> perm;
    return is_permutation_matrix(T, perm);
}

bool finish_with_one_col_layer(const vector<uint64_t>& B, vector<RawOp>& finish) {
    vector<uint64_t> Bt = transpose_matrix_bits(B);
    vector<RawOp> row_finish;
    if (!finish_with_one_row_layer(Bt, row_finish)) return false;
    finish.clear();
    for (auto op : row_finish) finish.push_back({op.target, op.control, 1});
    vector<uint64_t> T = B;
    apply_raw_layer_to_matrix(T, finish);
    vector<int> perm;
    return is_permutation_matrix(T, perm);
}

bool greedy_from_prefix(
    const vector<uint64_t>& M,
    vector<uint64_t> B,
    vector<vector<RawOp>> row_layers,
    vector<vector<RawOp>> col_layers,
    Result& res,
    int depth_limit
) {
    int mode = (int)(rng() & 1);
    int no_progress = 0;

    while (true) {
        vector<int> perm;

        if (is_permutation_matrix(B, perm)) {
            if (!build_and_verify(M, row_layers, col_layers, perm, res)) return false;

            compact_result_if_valid(M, res);
            return true;
        }
        {
            vector<RawOp> finishR, finishC;
            Result best_finish;
            bool ok = false;

            if (finish_with_one_row_layer(B, finishR)) {
                auto rr = row_layers;
                rr.push_back(finishR);
                vector<uint64_t> T = B;
                apply_raw_layer_to_matrix(T, finishR);
                vector<int> p2;
                if (is_permutation_matrix(T, p2)) {
                    Result cand;
                    if (build_and_verify(M, rr, col_layers, p2, cand)) {
                        compact_result_if_valid(M, cand);
                        best_finish = cand;
                        ok = true;
                    }
                }
            }

            if (finish_with_one_col_layer(B, finishC)) {
                auto cc = col_layers;
                cc.push_back(finishC);
                vector<uint64_t> T = B;
                apply_raw_layer_to_matrix(T, finishC);
                vector<int> p2;
                if (is_permutation_matrix(T, p2)) {
                    Result cand;
                    if (build_and_verify(M, row_layers, cc, p2, cand)) {
                        compact_result_if_valid(M, cand);
                        if (!ok || better_depth_first(cand, best_finish)) best_finish = cand;
                        ok = true;
                    }
                }
            }

            if (ok) {
                res = best_finish;
                return true;
            }
        }

        if ((int)row_layers.size() + (int)col_layers.size() > depth_limit) {
            return false;
        }

        MatchingBuildResult R = build_best_matching_layer(B, 0, mode);
        MatchingBuildResult C = build_best_matching_layer(B, 1, mode);

        bool take_row = false;
        bool take_col = false;

        if (!R.layer.empty() || !C.layer.empty()) {
            if (R.gain > C.gain + 1e-9) take_row = true;
            else if (C.gain > R.gain + 1e-9) take_col = true;
            else {
                if (R.layer.size() >= C.layer.size()) take_row = !R.layer.empty();
                else take_col = !C.layer.empty();
            }
        }

        if (!take_row && !take_col) {
            no_progress++;
            mode ^= 1;

            if (no_progress > 4) return false;
            continue;
        }

        no_progress = 0;

        if (take_row) {
            B = R.after;
            row_layers.push_back(R.layer);
        }
        else {
            B = C.after;
            col_layers.push_back(C.layer);
        }
    }
}

bool greedy_once(const vector<uint64_t>& M, Result& res, int depth_limit) {
    return greedy_from_prefix(M, M, vector<vector<RawOp>>(), vector<vector<RawOp>>(), res, depth_limit);
}

void add_clause(CaDiCaL::Solver& solver, initializer_list<int> lits) {
    for (int x : lits) solver.add(x);
    solver.add(0);
}

void add_clause_vec(CaDiCaL::Solver& solver, const vector<int>& lits) {
    for (int x : lits) solver.add(x);
    solver.add(0);
}

void add_unit(CaDiCaL::Solver& solver, int lit) {
    solver.add(lit);
    solver.add(0);
}

void add_at_most_one(CaDiCaL::Solver& solver, const vector<int>& lits) {
    for (int i = 0; i < (int)lits.size(); i++) {
        for (int j = i + 1; j < (int)lits.size(); j++) {
            add_clause(solver, { -lits[i], -lits[j] });
        }
    }
}

void add_at_most_k(CaDiCaL::Solver& solver, const vector<int>& lits, int K) {
    int m = (int)lits.size();

    if (K >= m) return;

    if (K < 0) {
        solver.add(0);
        return;
    }

    if (K == 0) {
        for (int x : lits) add_unit(solver, -x);
        return;
    }

    vector<vector<int>> s(m, vector<int>(K + 1, 0));

    for (int i = 0; i < m; i++) {
        for (int j = 1; j <= K; j++) {
            s[i][j] = vm.new_var();
        }
    }

    add_clause(solver, { -lits[0], s[0][1] });

    for (int j = 2; j <= K; j++) {
        add_unit(solver, -s[0][j]);
    }

    for (int i = 1; i < m; i++) {
        add_clause(solver, { -lits[i], s[i][1] });

        for (int j = 1; j <= K; j++) {
            add_clause(solver, { -s[i - 1][j], s[i][j] });
        }

        for (int j = 2; j <= K; j++) {
            add_clause(solver, { -lits[i], -s[i - 1][j - 1], s[i][j] });
        }

        add_clause(solver, { -lits[i], -s[i - 1][K] });
    }
}

void add_xor_gate(CaDiCaL::Solver& solver, int gate, int x, int y, int z) {
    add_clause(solver, { -gate, -x, -y, -z });
    add_clause(solver, { -gate, -x,  y,  z });
    add_clause(solver, { -gate,  x, -y,  z });
    add_clause(solver, { -gate,  x,  y, -z });
}

bool sat_synthesize_exact(
    const vector<uint64_t>& Target,
    int depth,
    int gate_bound,
    vector<Layer>& circuit
) {
    vm.next_var = 1;
    CaDiCaL::Solver solver;
    solver.set("factor", 0);
    solver.set("factorcheck", 0);
    solver.set("quiet", 1);

    S.assign(depth + 1, vector<vector<int>>(n, vector<int>(n)));
    X.assign(depth, vector<vector<int>>(n, vector<int>(n, 0)));

    for (int d = 0; d <= depth; d++) {
        for (int r = 0; r < n; r++) {
            for (int b = 0; b < n; b++) {
                S[d][r][b] = vm.new_var();
            }
        }
    }

    vector<int> all_gates;

    for (int d = 0; d < depth; d++) {
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < n; c++) {
                if (t == c) continue;

                X[d][t][c] = vm.new_var();
                all_gates.push_back(X[d][t][c]);
            }
        }
    }

    for (int r = 0; r < n; r++) {
        for (int b = 0; b < n; b++) {
            bool bit = (r == b);
            add_unit(solver, bit ? S[0][r][b] : -S[0][r][b]);
        }
    }

    for (int r = 0; r < n; r++) {
        for (int b = 0; b < n; b++) {
            bool bit = (Target[r] >> b) & 1ULL;
            add_unit(solver, bit ? S[depth][r][b] : -S[depth][r][b]);
        }
    }

    for (int d = 0; d < depth; d++) {
        for (int q = 0; q < n; q++) {
            vector<int> incident;

            for (int c = 0; c < n; c++) {
                if (c != q) incident.push_back(X[d][q][c]);
            }

            for (int t = 0; t < n; t++) {
                if (t != q) incident.push_back(X[d][t][q]);
            }

            add_at_most_one(solver, incident);
        }
    }

    add_at_most_k(solver, all_gates, gate_bound);

    for (int d = 0; d < depth; d++) {
        for (int r = 0; r < n; r++) {
            vector<int> incoming;

            for (int c = 0; c < n; c++) {
                if (c != r) incoming.push_back(X[d][r][c]);
            }

            for (int b = 0; b < n; b++) {
                int oldb = S[d][r][b];
                int newb = S[d + 1][r][b];

                vector<int> eq1 = incoming;
                eq1.push_back(-oldb);
                eq1.push_back(newb);
                add_clause_vec(solver, eq1);

                vector<int> eq2 = incoming;
                eq2.push_back(oldb);
                eq2.push_back(-newb);
                add_clause_vec(solver, eq2);

                for (int c = 0; c < n; c++) {
                    if (c == r) continue;

                    add_xor_gate(
                        solver,
                        X[d][r][c],
                        oldb,
                        S[d][c][b],
                        newb
                    );
                }
            }
        }
    }

    if (CONFLICT_LIMIT > 0) {
        solver.limit("conflicts", CONFLICT_LIMIT);
    }

    int ret = solver.solve();

    // CaDiCaL returns 10 for SAT, 20 for UNSAT, and 0 for UNKNOWN/limit.
    if (ret != 10) return false;

    circuit.assign(depth, Layer());

    for (int d = 0; d < depth; d++) {
        for (int t = 0; t < n; t++) {
            for (int c = 0; c < n; c++) {
                if (t == c) continue;

                if (solver.val(X[d][t][c]) > 0) {
                    circuit[d].push_back({ t, c });
                }
            }
        }
    }

    return verify_exact_matrix(Target, circuit);
}

vector<uint64_t> window_transform(const vector<Layer>& layers, int l, int r) {
    vector<uint64_t> A = identity_matrix();

    for (int i = l; i <= r; i++) {
        apply_layer(A, layers[i]);
    }

    return A;
}

bool try_sat_window_depth_first(
    vector<Layer>& layers,
    int start,
    int win_len,
    int extra_gate_allow
) {
    int end = start + win_len - 1;

    if (start < 0 || end >= (int)layers.size()) return false;

    vector<uint64_t> Target = window_transform(layers, start, end);

    vector<Layer> old_window;
    for (int i = start; i <= end; i++) {
        old_window.push_back(layers[i]);
    }

    int oldD = win_len;
    int oldG = count_gates(old_window);

    vector<Layer> best_local;
    bool improved = false;

    for (int d = oldD - 1; d >= max(1, oldD - 2); d--) {
        int gate_cap = oldG + extra_gate_allow;
        vector<Layer> cand;

        if (sat_synthesize_exact(Target, d, gate_cap, cand)) {
            best_local = cand;
            improved = true;
            break;
        }
    }

    // Depth-priority mode:
    // Gate count is not the objective here.  Do not spend SAT time on
    // same-depth gate minimization.  If no lower-depth replacement is found,
    // leave the window unchanged.

    if (!improved) return false;

    vector<Layer> nl;

    for (int i = 0; i < start; i++) nl.push_back(layers[i]);
    for (auto& L : best_local) nl.push_back(L);
    for (int i = end + 1; i < (int)layers.size(); i++) nl.push_back(layers[i]);

    layers.swap(nl);

    return true;
}

string circuit_signature(const Result& r) {
    string s;

    for (const auto& L : r.layers) {
        s += "[";
        vector<pair<int, int>> ops;

        for (auto op : L) {
            ops.push_back({ op.target, op.control });
        }

        sort(ops.begin(), ops.end());

        for (auto p : ops) {
            s += to_string(p.first) + "," + to_string(p.second) + ";";
        }

        s += "]";
    }

    return s;
}

void insert_candidate(
    vector<Result>& pool,
    unordered_set<string>& seen,
    const Result& cand,
    int pool_per_depth
) {
    string sig = circuit_signature(cand);

    if (seen.count(sig)) return;
    seen.insert(sig);

    pool.push_back(cand);

    map<int, vector<Result>> by_depth;

    for (auto& r : pool) {
        by_depth[r.depth].push_back(r);
    }

    pool.clear();

    for (auto& kv : by_depth) {
        auto& vec = kv.second;

        sort(vec.begin(), vec.end(), [](const Result& a, const Result& b) {
            if (a.gates != b.gates) return a.gates < b.gates;
            return circuit_signature(a) < circuit_signature(b);
            });

        if ((int)vec.size() > pool_per_depth) {
            int elite = max(1, pool_per_depth / 2);
            vector<Result> kept;
            for (int i = 0; i < elite && i < (int)vec.size(); i++) {
                kept.push_back(vec[i]);
            }

            vector<Result> rest;
            for (int i = elite; i < (int)vec.size(); i++) {
                rest.push_back(vec[i]);
            }
            shuffle(rest.begin(), rest.end(), rng);

            for (auto& r : rest) {
                if ((int)kept.size() >= pool_per_depth) break;
                kept.push_back(r);
            }
            vec.swap(kept);
        }

        for (auto& r : vec) {
            pool.push_back(r);
        }
    }

    sort(pool.begin(), pool.end(), [](const Result& a, const Result& b) {
        if (a.depth != b.depth) return a.depth < b.depth;
        return a.gates < b.gates;
        });
}

void prune_pool_to_current_best_depth(vector<Result>& pool, int best_depth) {
    vector<Result> np;
    for (auto& r : pool) {
        if (r.depth <= best_depth) np.push_back(r);
    }
    pool.swap(np);
}

void print_pool_summary(const vector<Result>& pool) {
    map<int, int> best_gate;

    for (auto& r : pool) {
        if (!best_gate.count(r.depth)) {
            best_gate[r.depth] = r.gates;
        }
        else {
            best_gate[r.depth] = min(best_gate[r.depth], r.gates);
        }
    }

    cout << "Candidate pool summary:\n";

    for (auto& kv : best_gate) {
        cout << "  depth " << kv.first
            << " | best XOR count = " << kv.second << "\n";
    }
}

bool sat_refine_candidate(
    const vector<uint64_t>& M,
    Result& cand,
    int sat_rounds,
    int min_window,
    int max_window,
    int extra_gate_allow
) {
    bool any_changed = false;

    for (int sr = 0; sr < sat_rounds; sr++) {
        bool changed = false;

        for (int w = max_window; w >= min_window; w--) {
            for (int s = 0; s + w <= (int)cand.layers.size(); s++) {
                int oldD = cand.depth;
                int oldG = cand.gates;

                if (try_sat_window_depth_first(cand.layers, s, w, extra_gate_allow)) {
                    vector<int> perm;

                    if (!verify_circuit(M, cand.layers, perm)) {
                        cerr << "Global verification failed after SAT replacement.\n";
                        exit(1);
                    }

                    cand.depth = (int)cand.layers.size();
                    cand.gates = count_gates(cand.layers);
                    cand.final_perm = perm;

                    compact_result_if_valid(M, cand);

                    cout << "SAT improve"
                        << " | sat_round = " << sr
                        << " | window_start = " << s
                        << " | window_len = " << w
                        << " | depth " << oldD << " -> " << cand.depth
                        << " | gates " << oldG << " -> " << cand.gates
                        << endl;

                    changed = true;
                    any_changed = true;

                    s = max(-1, s - w);
                }
            }
        }

        if (!changed) break;
    }

    return any_changed;
}

void save_result(const Result& res, const string& file) {
    ofstream out(file);

    out << "Candidate-pool Greedy + SAT-guided CNOT circuit\n";
    out << "Depth = " << res.depth << "\n";
    out << "XOR count = " << res.gates << "\n\n";

    for (int d = 0; d < (int)res.layers.size(); d++) {
        out << "Layer " << d << ":\n";

        for (auto op : res.layers[d]) {
            out << "    x" << op.target << " ^= x" << op.control
                << "    // CNOT(" << op.control << " -> " << op.target << ")\n";
        }

        out << "\n";
    }

    out << "Free final row permutation:\n";
    out << "final row i equals target row perm[i]\n";

    for (int i = 0; i < (int)res.final_perm.size(); i++) {
        out << "row " << i << " -> target row " << res.final_perm[i] << "\n";
    }
}

bool read_matrix(vector<uint64_t>& M) {
    int T;
    cin >> T;

    if (T != 1) return false;

    int rows, cols;
    cin >> rows >> cols;

    if (rows != cols || rows > 63) return false;

    n = rows;
    M.assign(n, 0);

    for (int i = 0; i < n; i++) {
        uint64_t row = 0;

        for (int j = 0; j < n; j++) {
            int bit;
            cin >> bit;

            if (bit) row |= (1ULL << j);
        }

        M[i] = row;
    }

    return true;
}

vector<int> identity_perm(int m) {
    vector<int> p(m);
    iota(p.begin(), p.end(), 0);
    return p;
}

vector<int> inverse_perm_vec(const vector<int>& p) {
    vector<int> inv((int)p.size(), -1);
    for (int i = 0; i < (int)p.size(); i++) inv[p[i]] = i;
    return inv;
}

bool is_perm_vec(const vector<int>& p) {
    if ((int)p.size() != n) return false;
    vector<int> seen(n, 0);
    for (int x : p) {
        if (x < 0 || x >= n || seen[x]) return false;
        seen[x] = 1;
    }
    return true;
}

vector<uint64_t> inverse_conjugate_by_perm(const vector<uint64_t>& M, const vector<int>& p) {
    vector<uint64_t> B(n, 0);
    for (int i = 0; i < n; i++) {
        uint64_t row = 0;
        uint64_t src = M[p[i]];
        for (int j = 0; j < n; j++) {
            if ((src >> p[j]) & 1ULL) row |= (1ULL << j);
        }
        B[i] = row;
    }
    return B;
}

Result map_result_by_perm(const Result& r, const vector<int>& p) {
    Result out;
    out.layers.clear();
    for (const auto& L : r.layers) {
        Layer NL;
        for (auto op : L) NL.push_back({ p[op.target], p[op.control] });
        out.layers.push_back(NL);
    }
    out.depth = (int)out.layers.size();
    out.gates = count_gates(out.layers);
    out.final_perm.clear();
    return out;
}

vector<uint64_t> transpose_matrix_bits(const vector<uint64_t>& A) {
    vector<uint64_t> T(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((A[i] >> j) & 1ULL) T[j] |= (1ULL << i);
        }
    }
    return T;
}

Result reverse_inverse_result(const Result& r, bool swap_target_control) {
    Result out;
    for (int i = (int)r.layers.size() - 1; i >= 0; i--) {
        Layer NL;
        for (auto op : r.layers[i]) {
            if (swap_target_control) NL.push_back({ op.control, op.target });
            else NL.push_back(op);
        }
        out.layers.push_back(NL);
    }
    out.depth = (int)out.layers.size();
    out.gates = count_gates(out.layers);
    return out;
}

vector<vector<int>> generate_structure_perms(int random_perm_count) {
    vector<vector<int>> perms;
    auto addp = [&](vector<int> p) {
        if (!is_perm_vec(p)) return;
        for (auto& q : perms) if (q == p) return;
        perms.push_back(p);
    };

    addp(identity_perm(n));

    for (int s = 1; s < n; s++) {
        vector<int> p(n);
        for (int i = 0; i < n; i++) p[i] = (i + s) % n;
        addp(p);
    }
    vector<int> block_sizes = {2, 4, 8, 16};
    for (int bs : block_sizes) {
        if (bs <= 1 || bs >= n || n % bs != 0) continue;
        int nb = n / bs;

        for (int sh = 1; sh < nb; sh++) {
            vector<int> p(n);
            for (int b = 0; b < nb; b++) {
                for (int k = 0; k < bs; k++) {
                    p[b * bs + k] = ((b + sh) % nb) * bs + k;
                }
            }
            addp(p);
        }

        for (int sh = 1; sh < bs; sh++) {
            vector<int> p(n);
            for (int b = 0; b < nb; b++) {
                for (int k = 0; k < bs; k++) {
                    p[b * bs + k] = b * bs + ((k + sh) % bs);
                }
            }
            addp(p);
        }

        if (nb * bs == n) {
            vector<int> p(n);
            for (int b = 0; b < nb; b++) {
                for (int k = 0; k < bs; k++) {
                    p[b * bs + k] = k * nb + b;
                }
            }
            addp(p);
        }
    }
    {
        vector<int> p(n);
        for (int i = 0; i < n; i++) p[i] = n - 1 - i;
        addp(p);
    }

    for (int r = 0; r < random_perm_count; r++) {
        vector<int> p = identity_perm(n);
        shuffle(p.begin(), p.end(), rng);
        addp(p);
    }

    return perms;
}

bool normalize_candidate_against_target(const vector<uint64_t>& M, Result& cand) {
    vector<int> perm;
    if (!verify_circuit(M, cand.layers, perm)) return false;
    cand.depth = (int)cand.layers.size();
    cand.gates = count_gates(cand.layers);
    cand.final_perm = perm;
    compact_result_if_valid(M, cand);
    return verify_circuit(M, cand.layers, cand.final_perm);
}

bool greedy_on_permuted_target(
    const vector<uint64_t>& M,
    const vector<int>& p,
    Result& out,
    int depth_limit
) {
    vector<uint64_t> B = inverse_conjugate_by_perm(M, p);
    Result tmp;
    if (!greedy_once(B, tmp, depth_limit)) return false;
    out = map_result_by_perm(tmp, p);
    return normalize_candidate_against_target(M, out);
}

bool greedy_on_inverse_target(const vector<uint64_t>& M, Result& out, int depth_limit) {
    vector<uint64_t> Inv;
    if (!invert_matrix(M, Inv)) return false;
    Result tmp;
    if (!greedy_once(Inv, tmp, depth_limit)) return false;
    out = reverse_inverse_result(tmp, false);
    return normalize_candidate_against_target(M, out);
}

bool greedy_on_transpose_target(const vector<uint64_t>& M, Result& out, int depth_limit) {
    vector<uint64_t> Mt = transpose_matrix_bits(M);
    Result tmp;
    if (!greedy_once(Mt, tmp, depth_limit)) return false;
    out = reverse_inverse_result(tmp, true);
    return normalize_candidate_against_target(M, out);
}

vector<RawOp> make_block_raw_layer(int block_size, bool is_row, const vector<pair<int,int>>& block_ops, const vector<int>& bp) {
    vector<RawOp> L;
    int nb = n / block_size;
    vector<int> used_block(nb, 0);
    for (auto pr : block_ops) {
        int tb = bp[pr.first];
        int cb = bp[pr.second];
        if (tb == cb || tb < 0 || tb >= nb || cb < 0 || cb >= nb) return {};
        if (used_block[tb] || used_block[cb]) return {};
        used_block[tb] = used_block[cb] = 1;
        for (int k = 0; k < block_size; k++) {
            L.push_back({tb * block_size + k, cb * block_size + k, is_row ? 0 : 1});
        }
    }
    return L;
}

struct PrefixSeed {
    string name;
    vector<vector<RawOp>> row_layers;
    vector<vector<RawOp>> col_layers;
    vector<uint64_t> B;
};

vector<PrefixSeed> generate_aes_block_cyclic_seeds(const vector<uint64_t>& M) {
    vector<PrefixSeed> seeds;
    if (n != 32) return seeds;

    const int bs = 8;
    const int nb = 4;

    // Zhang et al. observation: block-level phased row/column operations for AES MixColumn.
    vector<pair<bool, vector<pair<int,int>>>> base = {
        {true,  {{1,0}, {3,2}}},
        {false, {{0,2}, {1,3}}},
        {true,  {{2,0}, {3,1}}},
        {false, {{0,1}, {2,3}}}
    };

    vector<int> bp = {0,1,2,3};
    int id = 0;
    sort(bp.begin(), bp.end());
    do {
        PrefixSeed seed;
        seed.name = string("aes-blockcyclic-") + to_string(id++);
        seed.B = M;
        bool ok = true;
        for (auto& step : base) {
            vector<RawOp> L = make_block_raw_layer(bs, step.first, step.second, bp);
            if (L.empty()) { ok = false; break; }
            apply_raw_layer_to_matrix(seed.B, L);
            if (step.first) seed.row_layers.push_back(L);
            else seed.col_layers.push_back(L);
        }
        if (ok) seeds.push_back(seed);
    } while (next_permutation(bp.begin(), bp.end()));

    bp = {0,1,2,3};
    id = 0;
    reverse(base.begin(), base.end());
    do {
        PrefixSeed seed;
        seed.name = string("aes-blockcyclic-rev-") + to_string(id++);
        seed.B = M;
        bool ok = true;
        for (auto& step : base) {
            vector<RawOp> L = make_block_raw_layer(bs, step.first, step.second, bp);
            if (L.empty()) { ok = false; break; }
            apply_raw_layer_to_matrix(seed.B, L);
            if (step.first) seed.row_layers.push_back(L);
            else seed.col_layers.push_back(L);
        }
        if (ok) seeds.push_back(seed);
    } while (next_permutation(bp.begin(), bp.end()));

    return seeds;
}

bool greedy_on_prefix_seed(const vector<uint64_t>& M, const PrefixSeed& seed, Result& out, int depth_limit) {
    if (!greedy_from_prefix(M, seed.B, seed.row_layers, seed.col_layers, out, depth_limit)) return false;
    return normalize_candidate_against_target(M, out);
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int greedy_rounds = 50000;
    int sat_rounds = 3;
    int min_window = 2;
    int max_window = 5;
    int extra_gate_allow = 8;
    string output_file = "pool_hybrid_result.txt";

    int perm_rounds = 2000;
    int block_rounds = 50000;
    int random_perm_count = 64;
    int refine_depth_slack = 0;
    bool has_fixed_seed = false;
    uint64_t fixed_seed = 0;

    // Usage:
    // ./main greedy_rounds sat_rounds min_window max_window extra_gate_allow pool_per_depth conflict_limit output_file [perm_rounds] [block_rounds] [random_perm_count] [refine_depth_slack] [rng_seed]
    if (argc >= 2) greedy_rounds = atoi(argv[1]);
    if (argc >= 3) sat_rounds = atoi(argv[2]);
    if (argc >= 4) min_window = atoi(argv[3]);
    if (argc >= 5) max_window = atoi(argv[4]);
    if (argc >= 6) extra_gate_allow = atoi(argv[5]);
    if (argc >= 7) POOL_PER_DEPTH = atoi(argv[6]);
    if (argc >= 8) CONFLICT_LIMIT = atoi(argv[7]);
    if (argc >= 9) output_file = argv[8];
    if (argc >= 10) perm_rounds = atoi(argv[9]);
    if (argc >= 11) block_rounds = atoi(argv[10]);
    if (argc >= 12) random_perm_count = atoi(argv[11]);
    if (argc >= 13) refine_depth_slack = atoi(argv[12]);
    if (argc >= 14) {
        fixed_seed = strtoull(argv[13], nullptr, 10);
        has_fixed_seed = true;
        rng.seed(fixed_seed);
    }

    vector<uint64_t> M;

    if (!read_matrix(M)) {
        cerr << "Input matrix error.\n";
        return 1;
    }

    cout << "Start depth-priority Greedy + depth-only SAT + block-cyclic-prefix hybrid search\n";
    cout << "n = " << n
        << ", greedy_rounds = " << greedy_rounds
        << ", sat_rounds = " << sat_rounds
        << ", window = [" << min_window << "," << max_window << "]"
        << ", extra_gate_allow = " << extra_gate_allow
        << ", pool_per_depth = " << POOL_PER_DEPTH
        << ", conflict_limit = " << CONFLICT_LIMIT
        << ", perm_rounds = " << perm_rounds
        << ", block_rounds = " << block_rounds
        << ", random_perm_count = " << random_perm_count
        << ", refine_depth_slack = " << refine_depth_slack;
    if (has_fixed_seed) cout << ", rng_seed = " << fixed_seed;
    cout << "\n";

    vector<Result> pool;
    unordered_set<string> seen;
    Result global_best;

    auto accept_candidate = [&](Result cur, const string& tag, int round_id) {
        if (!normalize_candidate_against_target(M, cur)) return;

        insert_candidate(pool, seen, cur, POOL_PER_DEPTH);

        if (better_depth_first(cur, global_best)) {
            int old_best_depth = global_best.depth;
            global_best = cur;

            cout << tag << " improve"
                << " | round = " << round_id
                << " | Depth = " << global_best.depth
                << " | XOR count = " << global_best.gates << endl;

            save_result(global_best, output_file);

            // Once a better depth is found, deeper candidates are irrelevant
            // for strict depth-first search and SAT refinement.
            if (global_best.depth < old_best_depth) {
                prune_pool_to_current_best_depth(pool, global_best.depth);
                cout << "Depth-focused prune | keep depth <= " << global_best.depth
                     << " | pool size = " << pool.size() << endl;
            }
        }
    };

    // ===== Phase 1 =====
    cout << "\n[Phase 1] Greedy on original target\n";
    for (int r = 0; r < greedy_rounds; r++) {
        Result cur;
        if (!greedy_once(M, cur, 200)) continue;
        accept_candidate(cur, "Original greedy", r);

        if ((r + 1) % 5000 == 0) {
            cout << "Original greedy progress | round = " << (r + 1)
                << " | pool size = " << pool.size() << "\n";
            print_pool_summary(pool);
        }
    }

    // ===== Phase 2:  =====
    if (perm_rounds > 0) {
        cout << "\n[Phase 2] Structure-aware permuted targets\n";
        vector<vector<int>> perms = generate_structure_perms(random_perm_count);
        cout << "Generated structure/random permutations = " << perms.size() << "\n";

        for (int r = 0; r < perm_rounds; r++) {
            const vector<int>& p = perms[r % perms.size()];
            Result cur;
            if (!greedy_on_permuted_target(M, p, cur, 200)) continue;
            accept_candidate(cur, "Permuted greedy", r);

            if ((r + 1) % 1000 == 0) {
                cout << "Permuted greedy progress | round = " << (r + 1)
                    << " | pool size = " << pool.size() << "\n";
                print_pool_summary(pool);
            }
        }
    }

    // ===== Phase 2b: AES block-cyclic row/column prefix seeds =====
    if (block_rounds > 0) {
        vector<PrefixSeed> seeds = generate_aes_block_cyclic_seeds(M);
        if (!seeds.empty()) {
            cout << "\n[Phase 2b] AES block-cyclic prefix seeds | count = " << seeds.size() << "\n";
            int block_seed_rounds = max((int)seeds.size(), block_rounds);
            for (int r = 0; r < block_seed_rounds; r++) {
                const PrefixSeed& seed = seeds[r % seeds.size()];
                Result cur;
                if (!greedy_on_prefix_seed(M, seed, cur, 200)) continue;
                accept_candidate(cur, "Block-cyclic prefix greedy", r);

                if ((r + 1) % 500 == 0) {
                    cout << "Block-cyclic prefix progress | round = " << (r + 1)
                         << " | pool size = " << pool.size() << "\n";
                    print_pool_summary(pool);
                }
            }
        }
    }

    // ===== Phase 3: =====
    int derived_rounds = max(100, perm_rounds / 10);
    cout << "\n[Phase 3] Inverse / transpose derived targets\n";
    for (int r = 0; r < derived_rounds; r++) {
        Result inv_cand;
        if (greedy_on_inverse_target(M, inv_cand, 200)) {
            accept_candidate(inv_cand, "Inverse-derived greedy", r);
        }

        Result tr_cand;
        if (greedy_on_transpose_target(M, tr_cand, 200)) {
            accept_candidate(tr_cand, "Transpose-derived greedy", r);
        }

        if ((r + 1) % 100 == 0) {
            cout << "Derived greedy progress | round = " << (r + 1)
                << " | pool size = " << pool.size() << "\n";
            print_pool_summary(pool);
        }
    }

    if (pool.empty()) {
        cout << "No valid greedy candidate found.\n";
        return 0;
    }

    cout << "\nGreedy/multi-source phase finished.\n";
    cout << "Pool size = " << pool.size() << "\n";
    print_pool_summary(pool);

    sort(pool.begin(), pool.end(), [](const Result& a, const Result& b) {
        if (a.depth != b.depth) return a.depth < b.depth;
        return a.gates < b.gates;
        });

    int min_depth_in_pool = pool.front().depth;
    cout << "\nBest depth before SAT = " << min_depth_in_pool << "\n";
    cout << "Start depth-only SAT refinement over best-depth candidate pool\n";

    // ===== Phase 4: =====
    for (int idx = 0; idx < (int)pool.size(); idx++) {
        Result cand = pool[idx];
        if (cand.depth > min_depth_in_pool + refine_depth_slack) continue;

        cout << "Refine candidate " << idx
            << " | Depth = " << cand.depth
            << " | XOR count = " << cand.gates << endl;

        sat_refine_candidate(
            M,
            cand,
            sat_rounds,
            min_window,
            max_window,
            extra_gate_allow
        );

        if (better_depth_first(cand, global_best)) {
            global_best = cand;

            cout << "Global improve after SAT"
                << " | candidate = " << idx
                << " | Depth = " << global_best.depth
                << " | XOR count = " << global_best.gates << endl;

            save_result(global_best, output_file);
        }
    }

    cout << "\nFinal result\n";
    cout << "Depth = " << global_best.depth << "\n";
    cout << "XOR count = " << global_best.gates << "\n";
    cout << "Saved to " << output_file << "\n";

    return 0;
}

