// anvil_host.h — the run-time support that the GPU backends (CUDA, Metal) share: the counter-based
// random numbers and integer helpers (on the host and in kernels: ANVIL_HD), printing, data files,
// checkpoints, input. Plain C (C++ for the CUDA file); the includer defines ANVIL_HD (empty on the host).
#pragma once
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <zlib.h>
#ifndef ANVIL_HD
#define ANVIL_HD
#endif

ANVIL_HD static inline uint32_t anvil_lowbias32(uint32_t x) {
    x ^= x >> 16; x *= 0x7FEB352Du; x ^= x >> 15; x *= 0x846CA68Bu; x ^= x >> 16; return x;
}
// The counter-based generator shared by every backend: uniform in [0, 1), or (0, 1] if open_low.
ANVIL_HD static inline uint32_t anvil_rand_bits(uint32_t idx, uint32_t stream, uint32_t salt, uint32_t seed) {
    uint32_t k = stream * 0x85EBCA77u + salt * 0xC2B2AE3Du + seed + 0x632BE5ABu;
    return anvil_lowbias32(anvil_lowbias32(idx * 0x9E3779B1u + k));
}
ANVIL_HD static inline float anvil_rand(uint32_t idx, uint32_t stream, uint32_t salt, uint32_t seed, int open_low) {
    double hi = (double)(anvil_rand_bits(idx, stream, salt, seed) >> 8);
    if (open_low) hi += 1.0;
    return (float)(hi * (1.0 / 16777216.0));
}
// Python's floor division and modulo for integers (a zero divisor counts as 1, as in the interpreter)
ANVIL_HD static inline int anvil_idiv(int a, int b) {
    if (b == 0) b = 1;
    int q = a / b, r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}
ANVIL_HD static inline int anvil_imod(int a, int b) {
    if (b == 0) b = 1;
    int r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
ANVIL_HD static inline float anvil_fmod(float a, float b) {
    float r = fmodf(a, b);
    return (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) ? r + b : r;
}
// max/min that propagate nan (as NumPy and NEON fmax do)
ANVIL_HD static inline float anvil_fmax(float a, float b) { return (a != a || b != b) ? NAN : (a > b ? a : b); }
ANVIL_HD static inline float anvil_fmin(float a, float b) { return (a != a || b != b) ? NAN : (a < b ? a : b); }
ANVIL_HD static inline int anvil_imax(int a, int b) { return a > b ? a : b; }
ANVIL_HD static inline int anvil_imin(int a, int b) { return a < b ? a : b; }
ANVIL_HD static inline float anvil_fsign(float a) { return a > 0 ? 1.0f : (a < 0 ? -1.0f : a); }
ANVIL_HD static inline int anvil_isign(int a) { return a > 0 ? 1 : (a < 0 ? -1 : 0); }
ANVIL_HD static inline float anvil_sigmoid(float a) { return 1.0f / (1.0f + expf(-a)); }
ANVIL_HD static inline int anvil_f2i(float a) { return (int)truncf(a); }

// ---------------------------------------------------------------------------- host state
static uint32_t anvil_seed_value = 0;
static uint32_t anvil_stream = 0;
static double anvil_t0 = 0;

static double anvil_now(void) {              // seconds, not counting time the machine slept
#ifdef __APPLE__
    struct timespec ts; clock_gettime(CLOCK_UPTIME_RAW, &ts);
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void anvil_oob(const char *message, long long value) {
    fflush(stdout);
    fprintf(stderr, "anvil: runtime error: %s (got %lld)\n", message, value);
    exit(1);
}

// ---------------------------------------------------------------------------- printing
static void anvil_print_f32(FILE *f, float v) {        // like the interpreter's default format
    double a = fabs((double)v);
    if (v == 0 || v != v || (a >= 1e-4 && a < 1e6)) fprintf(f, "%.4f", (double)v);
    else fprintf(f, "%.4e", (double)v);
}

static void anvil_print_rec(const void *data, int is_int, int rank, const long *dims, const long *strides,
                          long off, int d, int summarize) {
    long n = dims[d];
    putchar('[');
    long idx[7]; int cnt = 0;
    if (summarize && n > 6) { idx[0] = 0; idx[1] = 1; idx[2] = 2; idx[3] = -1; idx[4] = n - 3; idx[5] = n - 2; idx[6] = n - 1; cnt = 7; }
    for (int k = 0; k < (cnt ? cnt : n); k++) {
        long i = cnt ? idx[k] : k;
        if (k) {
            if (d == rank - 1) fputs(", ", stdout);
            else { fputs(",\n", stdout); for (int s = 0; s <= d; s++) putchar(' '); }
        }
        if (i < 0) { fputs("...", stdout); continue; }
        if (d == rank - 1) {
            if (is_int) printf("%d", ((const int *)data)[off + i]);
            else anvil_print_f32(stdout, ((const float *)data)[off + i]);
        } else {
            anvil_print_rec(data, is_int, rank, dims, strides, off + i * strides[d], d + 1, summarize);
        }
    }
    putchar(']');
}

static void anvil_print_tensor(const void *data, int is_int, int rank, const long *dims) {
    long strides[16]; long numel = 1;
    for (int d = rank - 1; d >= 0; d--) { strides[d] = numel; numel *= dims[d]; }
    if (rank == 0) { if (is_int) printf("%d", *(const int *)data); else anvil_print_f32(stdout, *(const float *)data); return; }
    if (numel == 0) { fputs("[]", stdout); return; }
    anvil_print_rec(data, is_int, rank, dims, strides, 0, 0, numel > 1000);
}

static void anvil_print_pick(FILE *f, int k, const char *const *choices, int n) {
    fputs(k >= 0 && k < n ? choices[k] : "?", f);
}

static void anvil_print_chars(const int *codes, long n) {
    for (long i = 0; i < n; i++) putchar(codes[i] & 0xff);
}

static void anvil_show(const int *grid, long rows, long cols, const char *const *glyphs, int n) {
    for (long r = 0; r < rows; r++) {
        for (long c = 0; c < cols; c++) {
            int v = grid[r * cols + c];
            fputs(v >= 0 && v < n ? glyphs[v] : "?", stdout);
        }
        putchar('\n');
    }
}

// ---------------------------------------------------------------------------- data
// Data files are found where they were when the program was compiled, or, if $ANVIL_DATA is set,
// under that directory by their file name (for a .cu file built on one machine and run on another).
static const char *anvil_data_path(const char *path) {
    static char buf[4096];
    const char *dir = getenv("ANVIL_DATA");
    if (!dir || !*dir) return path;
    const char *name = strrchr(path, '/');
    snprintf(buf, sizeof buf, "%s/%s", dir, name ? name + 1 : path);
    return buf;
}

static void anvil_load_idx(const char *path, void *out, long n, long header, int is_int) {
    path = anvil_data_path(path);
    gzFile f = gzopen(path, "rb");
    if (!f) { fflush(stdout); fprintf(stderr, "anvil: cannot open data file %s\n", path); exit(1); }
    unsigned char *raw = (unsigned char *)malloc(n > 0 ? n : 1), skip[64];
    if (gzread(f, skip, (unsigned)header) != (int)header || gzread(f, raw, (unsigned)n) != (int)n) {
        fflush(stdout); fprintf(stderr, "anvil: data file %s does not have the shape this program was compiled for\n", path); exit(1);
    }
    gzclose(f);
    for (long i = 0; i < n; i++) {
        if (is_int) ((int *)out)[i] = raw[i]; else ((float *)out)[i] = raw[i];
    }
    free(raw);
}

static char *anvil_read_file(const char *path, long *size) {
    path = anvil_data_path(path);
    FILE *f = fopen(path, "rb");
    if (!f) { fflush(stdout); fprintf(stderr, "anvil: cannot open data file %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(n + 1);
    n = (long)fread(buf, 1, n, f); buf[n] = 0;
    fclose(f);
    *size = n;
    return buf;
}

static int anvil_number_starts(const char *p) {   // [+-] digit … or [+-] . digit …
    const char *q = p;
    if (*q == '+' || *q == '-') q++;
    if (*q == '.') q++;
    return *q >= '0' && *q <= '9';
}

static void anvil_load_csv(const char *path, float *out, long n, long skip) {
    long size; char *text = anvil_read_file(path, &size), *p = text;
    for (long s = 0; s < skip && *p; p++) if (*p == '\n') s++;
    long got = 0;
    while (got < n && *p) {
        if (anvil_number_starts(p)) { char *end; out[got++] = strtof(p, &end); p = end; }
        else p++;
    }
    free(text);
    if (got < n) { fflush(stdout); fprintf(stderr, "anvil: data file %s does not have the shape this program was compiled for\n", path); exit(1); }
}

static void anvil_load_bytes(const char *path, int *out, long n) {
    long size; char *text = anvil_read_file(path, &size);
    if (size != n) { fflush(stdout); fprintf(stderr, "anvil: data file %s does not have the shape this program was compiled for\n", path); exit(1); }
    for (long i = 0; i < n; i++) out[i] = (unsigned char)text[i];
    free(text);
}

// a NumPy file's n values from byte `offset`, converted to 32 bits (kinds as in _anvil_rt_load_npy)
static void anvil_load_npy(const char *path, void *out, long n, long offset, int kind, int size) {
    path = anvil_data_path(path);
    FILE *f = fopen(path, "rb");
    if (!f) { fflush(stdout); fprintf(stderr, "anvil: cannot open data file %s\n", path); exit(1); }
    fseek(f, offset, SEEK_SET);
    unsigned char *raw = (unsigned char *)malloc((size_t)n * size + 1);
    long got = (long)fread(raw, 1, (size_t)n * size + 1, f);
    fclose(f);
    if (got != n * size) { fflush(stdout); fprintf(stderr, "anvil: data file %s does not have the shape this program was compiled for\n", path); exit(1); }
    for (long i = 0; i < n; i++) {
        unsigned char *p = raw + i * size;
        float fv; double dv; int32_t iv; int64_t lv; int16_t sv; uint16_t uv;
        switch (kind) {
        case 0: memcpy(&fv, p, 4); ((float *)out)[i] = fv; break;
        case 1: memcpy(&dv, p, 8); ((float *)out)[i] = (float)dv; break;
        case 2: { memcpy(&uv, p, 2); int e = (uv >> 10) & 31, m = uv & 1023;   // half precision
                  float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf((float)(m | 1024), e - 25);
                  ((float *)out)[i] = (uv & 0x8000) ? -v : v; break; }
        case 3: memcpy(&iv, p, 4); ((int *)out)[i] = iv; break;
        case 4: memcpy(&lv, p, 8); ((int *)out)[i] = (int)lv; break;
        case 5: ((int *)out)[i] = p[0]; break;
        case 6: ((int *)out)[i] = (signed char)p[0]; break;
        case 7: memcpy(&uv, p, 2); ((int *)out)[i] = uv; break;
        default: memcpy(&sv, p, 2); ((int *)out)[i] = sv; break;
        }
    }
    free(raw);
}

static void anvil_save_npy(const char *path, const unsigned char *header, long hlen, const void *data, long bytes) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "anvil: cannot write %s\n", path); return; }
    fwrite(header, 1, hlen, f);
    fwrite(data, 1, bytes, f);
    fclose(f);
}

struct anvil_ckpt { void *data; uint64_t key; };      // key: the element count (low 32 bits) and a shape hash

static void anvil_save(const char *path, const anvil_ckpt *tab, long n) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "anvil: cannot write %s\n", path); return; }
    uint64_t head = 0x00000002574E4945ull, cnt = (uint64_t)n;          // "EINW", version 2
    fwrite(&head, 8, 1, f); fwrite(&cnt, 8, 1, f);
    for (long i = 0; i < n; i++) fwrite(&tab[i].key, 8, 1, f);
    for (long i = 0; i < n; i++) fwrite(tab[i].data, 4, (long)(tab[i].key & 0xFFFFFFFFull), f);
    fclose(f);
}

static float anvil_load(const char *path, const anvil_ckpt *tab, long n) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint64_t head = 0, cnt, total = 0;
    int ok = fread(&head, 8, 1, f) == 1 && (head == 0x00000002574E4945ull || head == 0x00000001574E4945ull)
             && fread(&cnt, 8, 1, f) == 1 && cnt == (uint64_t)n;
    uint64_t mask = head == 0x00000002574E4945ull ? ~0ull : 0xFFFFFFFFull;   // version 1 stored counts only
    for (long i = 0; ok && i < n; i++) {
        uint64_t c; ok = fread(&c, 8, 1, f) == 1 && c == (tab[i].key & mask); total += tab[i].key & 0xFFFFFFFFull;
    }
    if (ok) { fseek(f, 0, SEEK_END); ok = ftell(f) == (long)(16 + 8 * n + 4 * total); fseek(f, 16 + 8 * n, SEEK_SET); }
    for (long i = 0; ok && i < n; i++) {
        long count = (long)(tab[i].key & 0xFFFFFFFFull);
        ok = (long)fread(tab[i].data, 4, count, f) == count;
    }
    fclose(f);
    return ok ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------- the rest of the runtime
static void anvil_shuffle(int *p, long n) {     // Fisher–Yates, the same draws as every backend
    uint32_t stream = anvil_stream++;
    for (long k = 0; k < n - 1; k++) {
        long i = n - 1 - k;
        long j = anvil_rand_bits((uint32_t)k, stream, 7, anvil_seed_value) % (uint32_t)(i + 1);
        int t = p[i]; p[i] = p[j]; p[j] = t;
    }
}

static void anvil_iota(int *p, long n) { for (long i = 0; i < n; i++) p[i] = (int)i; }

static void anvil_nonzero(const int *flags, long n, int *out, long size) {
    long k = 0;
    for (long i = 0; i < n && k < size; i++) if (flags[i]) out[k++] = (int)i;
    for (; k < size; k++) out[k] = -1;
}

static int anvil_input(float *out) {          // 1 at the end of the input
    char line[256];
    fflush(stdout);
    if (!fgets(line, sizeof line, stdin)) { putchar('\n'); return 1; }
    for (char *p = line; *p; p++) {
        if (anvil_number_starts(p)) {
            char *q = p; if (*q == '+' || *q == '-') q++;
            if (q[0] == '0' && (q[1] == 'x' || q[1] == 'X')) q[1] = 0;     // decimal only
            *out = strtof(p, NULL);
            return 0;
        }
    }
    *out = NAN;
    return 0;
}
