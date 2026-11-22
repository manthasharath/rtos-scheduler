/*
 * rtos_hw.c
 * ---------------------------------------------------------------------------
 * Multi-core EDF scheduling on real hardware (Raspberry Pi 3 / 4 / 5, 4 ARM cores).
 *
 * Runs Partitioned EDF, Global EDF and Clustered EDF on the physical cores of
 * the board with EXACTLY the same rules as the Python simulator, so the
 * hardware schedule can be compared with the simulated one
 * (compare_with_simulator.py on the PC, or the "Raspberry Pi results" page of
 * the website).
 *
 * How it works
 *   * One worker thread per core, pinned to that physical core
 *     (pthread_setaffinity_np), SCHED_FIFO real-time priority (needs sudo).
 *   * Time is divided into ticks (1 tick = 1 time unit, e.g. 10 ms).
 *   * At every tick the dispatcher makes the EDF decision (releases, deadline
 *     misses, choose jobs, assign cores, background aperiodic jobs). Each
 *     worker then really executes its job on its core for that tick and
 *     records the CPU it actually ran on (sched_getcpu) and its start/end time.
 *   * Measured: scheduler decision time, dispatch latency, tick overruns,
 *     jobs on the wrong CPU.  Optional LEDs (per core / per task / both) and a
 *     live Gantt chart that grows tick by tick (tick >= 200 ms).
 *
 * Task types (one table):  P = periodic, S = sporadic, A = aperiodic
 *   S: next release = previous release + T + extra gap, extra gap in
 *      [0, T*gap%/100] from a fixed integer hash of (seed, task, job) - the
 *      same hash as the simulator, so both get identical arrivals.
 *   A: one job (C, Arrival), served in the BACKGROUND: only on cores left idle
 *      by the real-time jobs, oldest arrival first. Never delays P/S jobs.
 *
 * EDF modes
 *   Plain EDF:             earliest absolute deadline first.
 *   Criticality-first EDF: High jobs always before Medium/Low jobs; EDF inside
 *                          each band. Allocation (P, C) in DCDU order: High
 *                          tasks first, then Medium, then Low, each by density.
 *   Tie-break (both): deadline -> higher criticality -> priority number
 *   (1 = highest) -> task order -> job number.
 *   "High tasks guaranteed": density test of the High tasks alone per core
 *   (P), per cluster (C, GFB) or on all cores (G, GFB).
 *
 * Build on the Pi:   make          (= gcc -O2 -o rtos_hw rtos_hw.c -lpthread -lrt -lm)
 * Run (menu):        sudo ./rtos_hw
 * Run (arguments):   sudo ./rtos_hw -f tasks.csv -a ALL -m 4 -k 2 -t 10 [--critfirst]
 * Plain C (gnu89), no extra libraries.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <setjmp.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_TASKS   60
#define MAX_CORES   20
#define MAX_HORIZON 5000
#define EPS         1e-9
#define WORK_FRACTION_PERCENT 80   /* a worker computes for 80 % of a tick */
#define NLED_SLOTS  16

/* ------------------------------------------------------------------ data */
typedef struct {
    char name[32];
    char kind;                    /* 'P', 'S' or 'A' */
    int C, T, D, prio, arrival;   /* prio 0 = automatic (filled in) */
    char crit[8];
    int crit_rank;                /* 1 Low, 2 Medium, 3 High, 0 aperiodic */
    double density;               /* C / min(D, T), 0 for aperiodic */
} Task;

typedef struct {
    int task, index, release, absd, wcet, remaining, prio, crit_rank;
    char kind;
    int start, finish;            /* -1 = not yet */
    int missed, aborted, executed, preemptions, migrations;
    int last_core, last_tick, sel_tick;
    unsigned int cores_used;      /* bit mask of cores used */
    long long work;               /* computation loop iterations on hardware */
} Job;

typedef struct { int cpu; long start_us, end_us; } Slot;

typedef struct { char label[40]; int passed; char detail[200]; } HighCheck;

static Task tasks[MAX_TASKS];
static int  ntasks;
static char task_source[300] = "built-in example";
static int  m = 4, k = 2, H_setting = 0, horizon = 0, tick_ms = 10, abort_policy = 0;
static int  crit_mode = 0, gap_pct = 50, seed = 1;
static int  use_leds = 0;          /* 0 none, 1 per core, 2 per task, 3 both */
static int  live_mode = -1;        /* -1 = automatic (tick >= 200 ms), 0 = off, 1 = on */
static char algs[4] = "PGC";
static int  nalgs = 3, algo_choice = 7;
static int  ncpu_online = 1;
static int  last_run_ok = 0;

/* groups = cores that share one ready queue */
static int ngroups, group_ncores[MAX_CORES], group_core[MAX_CORES][MAX_CORES];
static int task_group[MAX_TASKS], alloc_failed[MAX_TASKS], nfail;
static HighCheck hchk[MAX_CORES];
static int nhchk;
static char analytic_text[200];
static int analytic_ok;

/* jobs (real-time and aperiodic in one array) */
static Job *jobs;
static int  njobs, maxjobs, *ready, nready, next_index[MAX_TASKS], next_release[MAX_TASKS];
static int  oldest[MAX_TASKS];
static int  apjob[MAX_TASKS], nap;  /* job indices of the aperiodic jobs */
static int *sched_tbl;              /* [tick * MAX_CORES + core] -> job or -1 */
static Slot *trace;
static long long *decision_ns, *dispatch_ns;
static int  overruns, wrong_cpu, rt_ok, pin_ok[MAX_CORES];

/* live view */
static int  live_on;
static char ev_text[MAX_CORES][160], miss_text[256], preempt_text[256];

/* shared with worker threads */
static volatile int cur_slot[MAX_CORES];
static volatile int cur_tick, stop_flag;
static volatile long long tick_start_ns;
static pthread_barrier_t bar_start, bar_end;
static volatile unsigned int sink;

/* LEDs: slot i = BCM GPIO led_pin[i]. Core LEDs slots 0-3. Task LEDs: slots 0-15 (task-only
   mode) or slots 4-15 = T1-T12 ("both" mode).  Physical pins: 11 13 15 16 18 22 29 31 33 35 37 36 38 40 32 12 */
static const int led_pin[NLED_SLOTS] = {17, 27, 22, 23, 24, 25, 5, 6, 13, 19, 26, 16, 20, 21, 12, 18};
static int led_fd[NLED_SLOTS];
#define CORE_LEDS (use_leds == 1 || use_leds == 3)
#define TASK_LEDS (use_leds == 2 || use_leds == 3)
static int task_led_slot(int i) { return use_leds == 2 ? i : 4 + i; }
static int task_led_max(void) { return use_leds == 2 ? 16 : 12; }
static int gpio_base = 0;

/* menu navigation: X = main menu, Q = quit, at any prompt */
static jmp_buf menu_jmp;
static int menu_ready = 0;

/* ------------------------------------------------------- built-in examples */
typedef struct { const char *name; int cores, cluster; const char *rows[20]; } Example;
static const Example EXAMPLES[] = {
    {"Automotive controller (P + S + A, 4 cores)", 4, 2, {
        "EngineCtl,P,2,5,5,1,High,",
        "BrakeCtl,P,3,10,10,2,High,",
        "Airbag,P,2,20,10,5,High,",
        "SteeringCtl,P,4,20,20,6,High,",
        "BrakePedal,S,1,10,5,3,High,",
        "CollisionAlert,S,2,20,10,7,High,",
        "CruiseCtl,P,5,20,20,8,Medium,",
        "SensorFusion,P,4,10,10,4,Medium,",
        "BatteryMon,P,5,50,50,9,Medium,",
        "Dashboard,P,10,50,50,10,Low,",
        "DataLogger,P,20,100,100,12,Low,",
        "Infotainment,P,15,50,50,11,Low,",
        "DiagScan,A,6,,,,,12",
        "OTACheck,A,8,,,,,40",
        "FaultLog,A,3,,,,,65",
        NULL}},
    {"Dhall effect: Global EDF fails (2 cores)", 2, 1, {
        "L1,P,2,10,10,1,Low,",
        "L2,P,2,10,10,2,Low,",
        "H,P,10,11,11,3,Medium,",
        NULL}},
    {"Partitioning fails: Global EDF wins (2 cores)", 2, 2, {
        "A,P,3,5,5,1,Medium,",
        "B,P,6,10,10,2,Medium,",
        "C,P,6,10,10,3,Medium,",
        NULL}},
    {"Heavy-task mix: Clustered EDF wins (4 cores)", 4, 2, {
        "Heavy1,P,4,5,5,1,High,",
        "Heavy2,P,7,10,10,5,Medium,",
        "Heavy3,P,3,5,5,2,Low,",
        "Heavy4,P,3,5,5,3,High,",
        "Heavy5,P,5,10,10,6,Medium,",
        "Heavy6,P,2,5,5,4,Low,",
        NULL}},
    {"Overload + criticality (4 cores)", 4, 2, {
        "Sensor1,P,2,5,5,1,Medium,",
        "Logger1,P,3,5,5,2,Low,",
        "Control1,P,3,5,5,3,Medium,",
        "Planner,P,14,20,20,7,Medium,",
        "SafetyMon,P,13,20,20,8,High,",
        "MotorCtl,P,5,10,10,6,High,",
        "Logger2,P,3,5,5,4,Low,",
        "Watchdog,P,3,5,5,5,High,",
        NULL}},
};
#define NEXAMPLES ((int)(sizeof EXAMPLES / sizeof EXAMPLES[0]))

/* ------------------------------------------------------------- helpers */
static long long now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (long long)t.tv_sec * 1000000000LL + t.tv_nsec;
}

static void sleep_until_ns(long long ns)
{
    struct timespec t;
    t.tv_sec = (time_t)(ns / 1000000000LL);
    t.tv_nsec = (long)(ns % 1000000000LL);
    while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &t, NULL) == EINTR)
        ;
}

static const char *alg_name(char a)
{
    return a == 'P' ? "Partitioned EDF" : a == 'G' ? "Global EDF" : "Clustered EDF";
}

static const char *mode_name(void) { return crit_mode ? "Criticality-first EDF" : "Plain EDF"; }

static int is_rt(int i) { return tasks[i].kind != 'A'; }

static int gcd_i(int a, int b) { while (b) { int r = a % b; a = b; b = r; } return a; }

static int hyperperiod(void)
{
    long long h = 1;
    int i;
    for (i = 0; i < ntasks; i++) {
        if (!is_rt(i)) continue;
        h = h / gcd_i((int)(h % 2147483647), tasks[i].T) * tasks[i].T;
        if (h > MAX_HORIZON) return MAX_HORIZON + 1;
    }
    return (int)h;
}

/* identical to sporadic_gap() in scheduler.py */
static int sporadic_gap(int sd, int task_id, int kk, int max_gap)
{
    uint32_t h;
    if (max_gap <= 0) return 0;
    h = (uint32_t)sd * 0x9E3779B1u + (uint32_t)(task_id + 1) * 0x85EBCA77u + (uint32_t)(kk + 1) * 0xC2B2AE3Du;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return (int)(h % (uint32_t)(max_gap + 1));
}

static void leds_close(void);

static void quit_program(void)
{
    leds_close();
    printf("\nBye.\n");
    exit(0);
}

static void trim(char *s)
{
    char *p = s;
    size_t n;
    while (*p && isspace((unsigned char)*p)) p++;
    memmove(s, p, strlen(p) + 1);
    n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = 0;
}

static void read_line(const char *prompt, const char *def, char *buf, int size)
{
    if (def) printf("%s [%s]: ", prompt, def); else printf("%s: ", prompt);
    fflush(stdout);
    if (!fgets(buf, size, stdin)) quit_program();
    trim(buf);
    if (menu_ready && (!strcasecmp(buf, "X") || !strcasecmp(buf, "menu"))) longjmp(menu_jmp, 1);
    if (!strcmp(buf, "Q") || !strcmp(buf, "q")) quit_program();
    if (buf[0] == 0 && def) { strncpy(buf, def, size - 1); buf[size - 1] = 0; }
}

/* blank_ok: Enter returns 0 ("automatic") */
static int ask_int_x(const char *prompt, int lo, int hi, int def, int blank_ok)
{
    char buf[64], d[32], p2[200], *end;
    long v;
    sprintf(d, "%d", def);
    snprintf(p2, sizeof p2, "%s (%d-%d%s)", prompt, lo, hi, blank_ok ? ", Enter = automatic" : "");
    for (;;) {
        read_line(p2, blank_ok ? NULL : d, buf, sizeof buf);
        if (blank_ok && buf[0] == 0) return 0;
        v = strtol(buf, &end, 10);
        if (*end == 0 && buf[0] && v >= lo && v <= hi) return (int)v;
        printf("  ! Enter a whole number from %d to %d.\n", lo, hi);
    }
}

static int ask_int(const char *prompt, int lo, int hi, int def) { return ask_int_x(prompt, lo, hi, def, 0); }

/* one-letter choice from `allowed` (upper case) */
static char ask_letter(const char *prompt, const char *allowed, char def)
{
    char buf[32], d[2];
    d[0] = def; d[1] = 0;
    for (;;) {
        read_line(prompt, d, buf, sizeof buf);
        if (strlen(buf) == 1 && strchr(allowed, toupper((unsigned char)buf[0])))
            return (char)toupper((unsigned char)buf[0]);
        printf("  ! Choose one of: %s\n", allowed);
    }
}

static const char *crit_full(const char *s)
{
    char c = (char)toupper((unsigned char)s[0]);
    if (!s[0]) return NULL;
    return c == 'H' ? "High" : c == 'M' ? "Medium" : c == 'L' ? "Low" : NULL;
}

static void pause_enter(void)
{
    char buf[8];
    read_line("\nPress Enter to continue (X = menu, Q = quit)", "", buf, sizeof buf);
}

/* ------------------------------------------------------------- task table */
static void set_task(Task *t, const char *name, char kind, int C, int T, int D, int p, const char *crit, int arr)
{
    memset(t, 0, sizeof *t);
    strncpy(t->name, name, sizeof t->name - 1);
    t->kind = kind;
    t->C = C;
    if (kind == 'A') {
        t->arrival = arr;
        strcpy(t->crit, "");
        return;
    }
    t->T = T; t->D = D; t->prio = p;
    strncpy(t->crit, crit, sizeof t->crit - 1);
    t->density = (double)C / (double)(D < T ? D : T);
    t->crit_rank = crit[0] == 'H' ? 3 : crit[0] == 'M' ? 2 : 1;
}

/* empty priorities (0) of P/S tasks: rank by period (shorter = higher), ties in table order.
   Identical to fill_priorities() in task_model.py. */
static void fill_priorities(Task *ts, int n)
{
    int order[MAX_TASKS], no = 0, i, j;
    for (i = 0; i < n; i++) if (ts[i].kind != 'A') order[no++] = i;
    for (i = 1; i < no; i++) {
        int x = order[i];
        j = i - 1;
        while (j >= 0 && ts[order[j]].T > ts[x].T) { order[j + 1] = order[j]; j--; }
        order[j + 1] = x;
    }
    for (i = 0; i < no; i++) if (ts[order[i]].prio == 0) ts[order[i]].prio = i + 1;
}

/* split a CSV line on commas, keeping empty fields */
static int split_csv(char *line, char **f, int maxf)
{
    int n = 0;
    char *p = line;
    f[n++] = p;
    while (*p && n < maxf) {
        if (*p == ',') { *p = 0; f[n++] = p + 1; }
        p++;
    }
    { int i; for (i = 0; i < n; i++) trim(f[i]); }
    return n;
}

static int parse_int_field(const char *s, int *out)
{
    char *end;
    double v;
    if (!s || !s[0]) return 0;
    v = strtod(s, &end);
    if (*end != 0 || v != (double)(long)v) return -1;
    *out = (int)v;
    return 1;
}

/* column positions: 0 Name 1 Type 2 C 3 T 4 D 5 Priority 6 Criticality 7 Arrival (-1 = absent) */
static int col[8];
static void default_cols(int nfields)
{
    int i;
    for (i = 0; i < 8; i++) col[i] = -1;
    if (nfields >= 8) { for (i = 0; i < 8; i++) col[i] = i; }
    else if (nfields == 6) { col[0] = 0; col[2] = 1; col[3] = 2; col[4] = 3; col[5] = 4; col[6] = 5; }
    else if (nfields == 5) { col[0] = 0; col[2] = 1; col[3] = 2; col[4] = 3; col[6] = 4; }
}

static int header_cols(char **f, int n)
{
    static const char *names[8] = {"Name", "Type", "C", "T", "D", "Priority", "Criticality", "Arrival"};
    int i, j, found = 0;
    for (i = 0; i < 8; i++) col[i] = -1;
    for (j = 0; j < n; j++)
        for (i = 0; i < 8; i++)
            if (!strcasecmp(f[j], names[i])) { col[i] = j; found++; }
    return found >= 4 && col[0] >= 0;
}

/* parse one data row into *t. Returns NULL if fine, else error text in err */
static int parse_row(char **f, int n, Task *t, char *err, const Task *ts, int nts)
{
    const char *fld[8];
    int i, C = 0, T = 0, D = 0, P = 0, A = 0, r;
    char kind = 'P';
    const char *cf;
    for (i = 0; i < 8; i++) fld[i] = (col[i] >= 0 && col[i] < n) ? f[col[i]] : "";
    if (!fld[0][0]) { strcpy(err, "name is empty"); return 0; }
    if (strchr(fld[0], ' ') || strlen(fld[0]) > 31) { strcpy(err, "name must have no spaces (max 31 chars)"); return 0; }
    for (i = 0; i < nts; i++) if (!strcmp(ts[i].name, fld[0])) { sprintf(err, "duplicate name '%s'", fld[0]); return 0; }
    if (fld[1][0]) {
        char c = (char)toupper((unsigned char)fld[1][0]);
        if (strchr("PSA", c) && (fld[1][1] == 0 || !strcasecmp(fld[1], "periodic") || !strcasecmp(fld[1], "sporadic")
                                 || !strcasecmp(fld[1], "aperiodic"))) kind = c;
        else { strcpy(err, "Type must be P, S or A"); return 0; }
    }
    if ((r = parse_int_field(fld[2], &C)) != 1 || C < 1) { strcpy(err, "C must be a whole number >= 1"); return 0; }
    if (kind == 'A') {
        if (fld[3][0] || fld[4][0] || fld[5][0] || fld[6][0])
            printf("  note: %s (aperiodic): %s%s%s%sremoved - aperiodic jobs have no period, deadline, priority or "
                   "criticality.\n", fld[0], fld[3][0] ? "T " : "", fld[4][0] ? "D " : "", fld[5][0] ? "Priority " : "",
                   fld[6][0] ? "Criticality " : "");
        if ((r = parse_int_field(fld[7], &A)) != 1 || A < 0) { strcpy(err, "aperiodic job needs Arrival >= 0"); return 0; }
        set_task(t, fld[0], 'A', C, 0, 0, 0, "", A);
        return 1;
    }
    if (parse_int_field(fld[3], &T) != 1 || T < 1) { strcpy(err, "T must be a whole number >= 1"); return 0; }
    if (parse_int_field(fld[4], &D) != 1 || D < 1) { strcpy(err, "D must be a whole number >= 1"); return 0; }
    r = parse_int_field(fld[5], &P);
    if (r == -1 || (r == 1 && P < 1)) { strcpy(err, "Priority must be empty or a whole number >= 1"); return 0; }
    if (r == 0) P = 0;
    if ((cf = crit_full(fld[6])) == NULL) { strcpy(err, "Criticality must be High, Medium or Low"); return 0; }
    if (C > D) { sprintf(err, "C=%d is larger than D=%d; every job would miss. Make C <= D.", C, D); return 0; }
    if (D > T) { sprintf(err, "D=%d is larger than T=%d. Deadlines must satisfy D <= T.", D, T); return 0; }
    if (fld[7][0])
        printf("  note: %s (%s): Arrival removed - periodic and sporadic tasks always start at time 0.\n", fld[0],
               kind == 'S' ? "sporadic" : "periodic");
    set_task(t, fld[0], kind, C, T, D, P, cf, 0);
    return 1;
}

static int commit_tasks(Task *ts, int n, const char *source)
{
    int i, nrt = 0;
    for (i = 0; i < n; i++) nrt += ts[i].kind != 'A';
    if (nrt == 0) { printf("  ! Add at least one periodic (P) or sporadic (S) task.\n"); return 0; }
    fill_priorities(ts, n);
    memcpy(tasks, ts, sizeof(Task) * n);
    ntasks = n;
    snprintf(task_source, sizeof task_source, "%s", source);
    last_run_ok = 0;
    return 1;
}

/* CSV: Name,Type,C,T,D,Priority,Criticality,Arrival  (old files Name,C,T,D,Priority,Criticality also work) */
static int load_csv(const char *path)
{
    FILE *f = fopen(path, "r");
    char line[512];
    Task tmp[MAX_TASKS];
    int n = 0, lineno = 0, errors = 0, have_cols = 0;
    if (!f) { printf("  ! Cannot open '%s': %s\n", path, strerror(errno)); return 0; }
    while (fgets(line, sizeof line, f)) {
        char *fld[16], err[160];
        int nf;
        lineno++;
        trim(line);
        if (line[0] == 0 || line[0] == '#') continue;
        nf = split_csv(line, fld, 16);
        if (!have_cols) {
            have_cols = 1;
            if (header_cols(fld, nf)) continue;
            default_cols(nf);
        }
        if (n >= MAX_TASKS) { printf("  ! At most %d tasks.\n", MAX_TASKS); errors++; break; }
        if (!parse_row(fld, nf, &tmp[n], err, tmp, n)) {
            printf("  ! Line %d: %s\n", lineno, err);
            errors++;
            continue;
        }
        n++;
    }
    fclose(f);
    if (errors) { printf("  ! %d error(s); task set NOT loaded.\n", errors); return 0; }
    if (n == 0) { printf("  ! No tasks found in '%s'.\n", path); return 0; }
    return commit_tasks(tmp, n, path);
}

static void load_example(int e)
{
    Task tmp[MAX_TASKS];
    int n = 0, i;
    default_cols(8);
    for (i = 0; EXAMPLES[e].rows[i]; i++) {
        char line[256], *fld[16], err[160];
        int nf;
        strcpy(line, EXAMPLES[e].rows[i]);
        nf = split_csv(line, fld, 16);
        if (parse_row(fld, nf, &tmp[n], err, tmp, n)) n++;
    }
    m = EXAMPLES[e].cores < ncpu_online ? EXAMPLES[e].cores : ncpu_online;
    k = EXAMPLES[e].cluster <= m ? EXAMPLES[e].cluster : m;
    {
        char src[200];
        snprintf(src, sizeof src, "example: %s", EXAMPLES[e].name);
        commit_tasks(tmp, n, src);
    }
}

static void save_csv(const char *path)
{
    FILE *f = fopen(path, "w");
    int i;
    if (!f) return;
    fprintf(f, "Name,Type,C,T,D,Priority,Criticality,Arrival\n");
    for (i = 0; i < ntasks; i++) {
        Task *t = &tasks[i];
        if (t->kind == 'A') fprintf(f, "%s,A,%d,,,,,%d\n", t->name, t->C, t->arrival);
        else fprintf(f, "%s,%c,%d,%d,%d,%d,%s,\n", t->name, t->kind, t->C, t->T, t->D, t->prio, t->crit);
    }
    fclose(f);
}

/* ----------------------------------------------------------- allocation */
static int order_dcdu;
static int alloc_cmp(const void *a, const void *b)
{
    int i = *(const int *)a, j = *(const int *)b;
    if (order_dcdu && tasks[i].crit_rank != tasks[j].crit_rank) return tasks[i].crit_rank > tasks[j].crit_rank ? -1 : 1;
    if (tasks[i].density > tasks[j].density) return -1;
    if (tasks[i].density < tasks[j].density) return 1;
    return i - j;
}

/* GFB test on tasks list[0..n-1] for c cores; total/bound returned */
static int gfb_ok(const int *list, int n, int c, double *tot, double *bnd)
{
    double total = 0.0, dmax = 0.0, bound;
    int i;
    for (i = 0; i < n; i++) {
        total += tasks[list[i]].density;
        if (tasks[list[i]].density > dmax) dmax = tasks[list[i]].density;
    }
    bound = c - (c - 1) * dmax;
    if (tot) *tot = total;
    if (bnd) *bnd = bound;
    return total <= bound + EPS;
}

/* High-only check for group g */
static void high_check(int g, const char *label)
{
    int list[MAX_TASKS], n = 0, i;
    double tot, bnd;
    HighCheck *h = &hchk[nhchk++];
    snprintf(h->label, sizeof h->label, "%s", label);
    for (i = 0; i < ntasks; i++) if (is_rt(i) && task_group[i] == g && tasks[i].crit_rank == 3) list[n++] = i;
    if (n == 0) { h->passed = 1; strcpy(h->detail, "no High tasks"); return; }
    h->passed = gfb_ok(list, n, group_ncores[g], &tot, &bnd);
    if (group_ncores[g] == 1) snprintf(h->detail, sizeof h->detail, "High density %.3f %s 1", tot, h->passed ? "<=" : ">");
    else snprintf(h->detail, sizeof h->detail, "High density %.3f %s GFB bound %.3f", tot, h->passed ? "<=" : ">", bnd);
}

static int high_all_passed(void)
{
    int i;
    for (i = 0; i < nhchk; i++) if (!hchk[i].passed) return 0;
    return 1;
}

static void build_groups(char alg)
{
    int order[MAX_TASKS], nord = 0, i, g, c;
    nfail = 0;
    nhchk = 0;
    memset(alloc_failed, 0, sizeof alloc_failed);
    for (i = 0; i < ntasks; i++) { task_group[i] = -1; if (is_rt(i)) order[nord++] = i; }
    order_dcdu = crit_mode;
    qsort(order, nord, sizeof(int), alloc_cmp);

    if (alg == 'G') {
        double tot, bnd;
        ngroups = 1;
        group_ncores[0] = m;
        for (c = 0; c < m; c++) group_core[0][c] = c;
        for (i = 0; i < nord; i++) task_group[order[i]] = 0;
        analytic_ok = gfb_ok(order, nord, m, &tot, &bnd);
        snprintf(analytic_text, sizeof analytic_text, "GFB test: total density %.3f %s bound %.3f",
                 tot, analytic_ok ? "<=" : ">", bnd);
        high_check(0, "All cores");
        return;
    }
    if (alg == 'P') {
        double load[MAX_CORES];
        ngroups = m;
        for (c = 0; c < m; c++) { group_ncores[c] = 1; group_core[c][0] = c; load[c] = 0.0; }
        for (i = 0; i < nord; i++) {
            int t = order[i], placed = 0, best = 0;
            for (c = 0; c < m && !placed; c++)
                if (load[c] + tasks[t].density <= 1.0 + EPS) {
                    task_group[t] = c; load[c] += tasks[t].density; placed = 1;
                }
            if (!placed) {
                for (c = 1; c < m; c++) if (load[c] < load[best]) best = c;
                task_group[t] = best; load[best] += tasks[t].density;
                alloc_failed[t] = 1; nfail++;
            }
        }
        for (c = 0; c < m; c++) { char lbl[40]; sprintf(lbl, "Core %d", c); high_check(c, lbl); }
    } else {
        int members[MAX_CORES][MAX_TASKS], nmem[MAX_CORES];
        ngroups = 0;
        for (c = 0; c < m; c += k) {
            int j;
            group_ncores[ngroups] = 0;
            for (j = c; j < c + k && j < m; j++) group_core[ngroups][group_ncores[ngroups]++] = j;
            nmem[ngroups] = 0;
            ngroups++;
        }
        for (i = 0; i < nord; i++) {
            int t = order[i], placed = 0, best = 0;
            for (g = 0; g < ngroups && !placed; g++) {
                members[g][nmem[g]] = t;
                if (gfb_ok(members[g], nmem[g] + 1, group_ncores[g], NULL, NULL)) {
                    nmem[g]++; task_group[t] = g; placed = 1;
                }
            }
            if (!placed) {
                double best_spare = -1e18;
                for (g = 0; g < ngroups; g++) {
                    double sum = 0.0, s;
                    int x;
                    for (x = 0; x < nmem[g]; x++) sum += tasks[members[g][x]].density;
                    s = group_ncores[g] - sum;
                    if (s > best_spare) { best_spare = s; best = g; }
                }
                members[best][nmem[best]++] = t;
                task_group[t] = best;
                alloc_failed[t] = 1; nfail++;
            }
        }
        for (g = 0; g < ngroups; g++) {
            char lbl[40];
            sprintf(lbl, "Cluster %d (cores %d-%d)", g + 1, group_core[g][0], group_core[g][group_ncores[g] - 1]);
            high_check(g, lbl);
        }
    }
    analytic_ok = nfail == 0;
    snprintf(analytic_text, sizeof analytic_text, "%s",
             nfail ? "allocation failure: some task did not fit (placed anyway, overloaded)"
                   : (alg == 'P' ? "all tasks fit by FFD, every core density <= 1"
                                 : "all tasks fit, every cluster passes the GFB test"));
}

/* ------------------------------------------------------------ scheduler */
static int edf_cmp(const void *a, const void *b)
{
    const Job *x = &jobs[*(const int *)a], *y = &jobs[*(const int *)b];
    if (x->absd != y->absd) return x->absd < y->absd ? -1 : 1;
    if (x->crit_rank != y->crit_rank) return x->crit_rank > y->crit_rank ? -1 : 1;  /* High first */
    if (x->prio != y->prio) return x->prio < y->prio ? -1 : 1;                      /* 1 = highest */
    if (x->task != y->task) return x->task < y->task ? -1 : 1;
    return x->index < y->index ? -1 : (x->index > y->index);
}

/* criticality-first: High band before the Medium+Low band, EDF inside each band */
static int crit_cmp(const void *a, const void *b)
{
    const Job *x = &jobs[*(const int *)a], *y = &jobs[*(const int *)b];
    int bx = x->crit_rank == 3 ? 0 : 1, by = y->crit_rank == 3 ? 0 : 1;
    if (bx != by) return bx < by ? -1 : 1;
    return edf_cmp(a, b);
}

static void reset_run(void)
{
    int i;
    maxjobs = 0;
    for (i = 0; i < ntasks; i++) maxjobs += is_rt(i) ? horizon / tasks[i].T + 1 : 1;
    free(jobs); free(ready); free(sched_tbl); free(trace); free(decision_ns); free(dispatch_ns);
    jobs = calloc(maxjobs, sizeof(Job));
    ready = malloc(sizeof(int) * maxjobs);
    sched_tbl = malloc(sizeof(int) * horizon * MAX_CORES);
    trace = calloc((size_t)horizon * MAX_CORES, sizeof(Slot));
    decision_ns = calloc(horizon, sizeof(long long));
    dispatch_ns = calloc(horizon, sizeof(long long));
    if (!jobs || !ready || !sched_tbl || !trace || !decision_ns || !dispatch_ns) {
        printf("Out of memory.\n"); exit(1);
    }
    for (i = 0; i < horizon * MAX_CORES; i++) sched_tbl[i] = -1;
    njobs = nready = nap = 0;
    overruns = wrong_cpu = 0;
    memset(next_index, 0, sizeof next_index);
    memset(next_release, 0, sizeof next_release);
    for (i = 0; i < ntasks; i++)                         /* aperiodic jobs exist from the start */
        if (!is_rt(i)) {
            Job *J = &jobs[njobs];
            J->task = i; J->index = 0; J->release = tasks[i].arrival; J->absd = -1;
            J->wcet = J->remaining = tasks[i].C; J->kind = 'A'; J->crit_rank = 0;
            J->start = J->finish = -1; J->last_core = -1; J->last_tick = -10; J->sel_tick = -10;
            apjob[nap++] = njobs++;
        }
}

static void job_label(int j, char *buf)
{
    if (jobs[j].kind == 'A') sprintf(buf, "%s", tasks[jobs[j].task].name);
    else sprintf(buf, "%s.J%d", tasks[jobs[j].task].name, jobs[j].index + 1);
}

static void run_slot(int t, int c, int j)
{
    Job *J = &jobs[j];
    if (live_on) {
        const char *what;
        char mig[40], lbl[48];
        job_label(j, lbl);
        if (J->start < 0) what = "started";
        else if (J->last_core >= 0 && J->last_core != c) { sprintf(mig, "MIGRATED from core %d", J->last_core); what = mig; }
        else if (J->last_tick != t - 1) what = "resumed";
        else what = "running";
        snprintf(ev_text[c], sizeof ev_text[c], "%s  %s%s%s", lbl, what, J->remaining == 1 ? ", finished" : "",
                 J->kind == 'A' ? "  (aperiodic, idle time)" : (crit_mode && J->crit_rank == 3 ? "  [High]" : ""));
    }
    if (J->last_core >= 0 && J->last_core != c) J->migrations++;
    if (J->start < 0) J->start = t;
    J->last_core = c;
    J->cores_used |= 1u << c;
    J->remaining--; J->executed++; J->last_tick = t;
    if (J->remaining == 0) J->finish = t + 1;
}

/* EDF decision for tick t: fills sched_tbl[t][*] and updates the job state */
static void plan_tick(int t)
{
    int i, r, g, c, w;
    int *slot = &sched_tbl[t * MAX_CORES];

    /* 1. releases (periodic every T, sporadic T + extra gap) */
    for (i = 0; i < ntasks; i++)
        if (is_rt(i) && t == next_release[i]) {
            Job *J = &jobs[njobs];
            int kk = next_index[i]++;
            int gap = tasks[i].kind == 'S' ? sporadic_gap(seed, i, kk, tasks[i].T * gap_pct / 100) : 0;
            next_release[i] = t + tasks[i].T + gap;
            memset(J, 0, sizeof *J);
            J->task = i; J->index = kk; J->release = t; J->kind = tasks[i].kind;
            J->absd = t + tasks[i].D; J->wcet = J->remaining = tasks[i].C;
            J->prio = tasks[i].prio; J->start = J->finish = -1;
            J->crit_rank = tasks[i].crit_rank;
            J->last_core = -1; J->last_tick = -10; J->sel_tick = -10;
            ready[nready++] = njobs++;
        }

    /* 2. deadline-miss detection */
    miss_text[0] = preempt_text[0] = 0;
    for (r = 0; r < nready; r++) {
        Job *J = &jobs[ready[r]];
        if (!J->missed && t >= J->absd) {
            J->missed = 1;
            if (live_on && strlen(miss_text) < 200)
                sprintf(miss_text + strlen(miss_text), " %s.J%d", tasks[J->task].name, J->index + 1);
            if (abort_policy) J->aborted = 1;
        }
    }
    if (abort_policy) {
        for (r = w = 0; r < nready; r++) if (!jobs[ready[r]].aborted) ready[w++] = ready[r];
        nready = w;
    }

    /* 3. order (only the oldest pending job of a task may run) */
    for (i = 0; i < ntasks; i++) oldest[i] = 0x7fffffff;
    for (r = 0; r < nready; r++) {
        Job *J = &jobs[ready[r]];
        if (J->index < oldest[J->task]) oldest[J->task] = J->index;
    }
    qsort(ready, nready, sizeof(int), crit_mode ? crit_cmp : edf_cmp);

    /* 4. per group: pick up to (cores) jobs, assign cores (keep previous core, else lowest free) */
    for (g = 0; g < ngroups; g++) {
        int sel[MAX_CORES], acore[MAX_CORES], fc[MAX_CORES], ns = 0, nf;
        for (r = 0; r < nready; r++) {
            Job *J = &jobs[ready[r]];
            if (task_group[J->task] != g) continue;
            if (ns == group_ncores[g]) break;
            if (J->index != oldest[J->task]) continue;
            sel[ns++] = ready[r];
        }
        qsort(sel, ns, sizeof(int), edf_cmp);          /* core assignment always in EDF order */
        nf = group_ncores[g];
        for (c = 0; c < nf; c++) fc[c] = group_core[g][c];
        for (i = 0; i < ns; i++) {
            int lc = jobs[sel[i]].last_core, x;
            acore[i] = -1;
            for (x = 0; lc >= 0 && x < nf; x++)
                if (fc[x] == lc) {
                    acore[i] = lc;
                    memmove(&fc[x], &fc[x + 1], sizeof(int) * (nf - x - 1));
                    nf--;
                    break;
                }
        }
        for (i = 0; i < ns; i++)
            if (acore[i] < 0) {
                acore[i] = fc[0];
                memmove(&fc[0], &fc[1], sizeof(int) * (nf - 1));
                nf--;
            }
        for (i = 0; i < ns; i++) { slot[acore[i]] = sel[i]; jobs[sel[i]].sel_tick = t; }
    }

    /* 5. preemptions: ran in the previous tick, unfinished, not chosen now */
    for (r = 0; r < nready; r++) {
        Job *J = &jobs[ready[r]];
        if (J->last_tick == t - 1 && J->sel_tick != t) {
            J->preemptions++;
            if (live_on && strlen(preempt_text) < 200)
                sprintf(preempt_text + strlen(preempt_text), " %s.J%d", tasks[J->task].name, J->index + 1);
        }
    }

    /* 6. execute one time unit of the real-time jobs */
    for (c = 0; c < m; c++) {
        if (slot[c] < 0) { if (live_on) strcpy(ev_text[c], "idle"); continue; }
        run_slot(t, c, slot[c]);
    }
    for (r = w = 0; r < nready; r++) if (jobs[ready[r]].remaining > 0) ready[w++] = ready[r];
    nready = w;

    /* 7. background service of aperiodic jobs on the cores that are still idle */
    if (nap) {
        int wait[MAX_TASKS], nw = 0, idle[MAX_CORES], ni = 0, acore[MAX_TASKS], x, y;
        for (i = 0; i < nap; i++) {
            Job *J = &jobs[apjob[i]];
            if (J->release <= t && J->remaining > 0) wait[nw++] = apjob[i];
        }
        for (x = 1; x < nw; x++) {                      /* sort by (arrival, task order) */
            int v = wait[x];
            y = x - 1;
            while (y >= 0 && (jobs[wait[y]].release > jobs[v].release
                              || (jobs[wait[y]].release == jobs[v].release && jobs[wait[y]].task > jobs[v].task))) {
                wait[y + 1] = wait[y]; y--;
            }
            wait[y + 1] = v;
        }
        for (c = 0; c < m; c++) if (slot[c] < 0) idle[ni++] = c;
        if (nw > ni) nw = ni;
        for (i = 0; i < nw; i++) {
            int lc = jobs[wait[i]].last_core;
            acore[i] = -1;
            for (x = 0; lc >= 0 && x < ni; x++)
                if (idle[x] == lc) {
                    acore[i] = lc;
                    memmove(&idle[x], &idle[x + 1], sizeof(int) * (ni - x - 1));
                    ni--;
                    break;
                }
        }
        for (i = 0; i < nw; i++)
            if (acore[i] < 0) {
                acore[i] = idle[0];
                memmove(&idle[0], &idle[1], sizeof(int) * (ni - 1));
                ni--;
            }
        for (i = 0; i < nw; i++) { slot[acore[i]] = wait[i]; run_slot(t, acore[i], wait[i]); }
    }
}

static void finish_horizon(void)
{
    int r;
    for (r = 0; r < nready; r++) {
        Job *J = &jobs[ready[r]];
        if (!J->missed && J->absd <= horizon) J->missed = 1;
    }
}

/* ------------------------------------------------------------- LEDs */
static void led_set(int c, int on)
{
    if (use_leds && c < NLED_SLOTS && led_fd[c] >= 0) {
        if (write(led_fd[c], on ? "1" : "0", 1) < 0) { /* ignore */ }
    }
}

static int detect_gpio_base(void)
{
    /* newer kernels number the Pi header GPIOs from a base such as 512 */
    char path[128], label[128];
    int b;
    for (b = 0; b <= 1024; b++) {
        FILE *f;
        sprintf(path, "/sys/class/gpio/gpiochip%d/label", b);
        f = fopen(path, "r");
        if (!f) continue;
        if (fgets(label, sizeof label, f) && (strstr(label, "bcm2835") || strstr(label, "pinctrl-bcm")
                                              || strstr(label, "brcmstb") || strstr(label, "rp1"))) {
            fclose(f);
            return b;
        }
        fclose(f);
    }
    return 0;
}

static int leds_inited = 0;
static void leds_init(void)
{
    int c, n;
    char path[128], num[16];
    for (c = 0; c < NLED_SLOTS; c++) led_fd[c] = -1;
    leds_inited = 1;
    if (!use_leds) return;
    gpio_base = detect_gpio_base();
    n = use_leds == 1 ? (m < 4 ? m : 4)
      : use_leds == 2 ? (ntasks < 16 ? ntasks : 16)
      : 4 + (ntasks < 12 ? ntasks : 12);
    for (c = 0; c < n; c++) {
        int fd, gpio = gpio_base + led_pin[c];
        sprintf(num, "%d", gpio);
        fd = open("/sys/class/gpio/export", O_WRONLY);
        if (fd >= 0) { if (write(fd, num, strlen(num)) < 0) { /* already exported */ } close(fd); }
        usleep(100000);
        sprintf(path, "/sys/class/gpio/gpio%d/direction", gpio);
        fd = open(path, O_WRONLY);
        if (fd >= 0) { if (write(fd, "out", 3) < 0) { /* ignore */ } close(fd); }
        sprintf(path, "/sys/class/gpio/gpio%d/value", gpio);
        led_fd[c] = open(path, O_WRONLY);
        if (led_fd[c] < 0)
            printf("  ! LED %d (BCM GPIO%d) not available: %s  (use sudo)\n", c, led_pin[c], strerror(errno));
    }
    if (CORE_LEDS && m > 4) printf("  Note: core LEDs exist for cores 0-3 only.\n");
    if (TASK_LEDS && ntasks > task_led_max()) printf("  Note: only the first %d tasks have an LED.\n", task_led_max());
}

static void leds_close(void)
{
    int c;
    if (!leds_inited) return;
    for (c = 0; c < NLED_SLOTS; c++)
        if (led_fd[c] >= 0) { led_set(c, 0); close(led_fd[c]); led_fd[c] = -1; }
}

/* ------------------------------------------------------- worker threads */
static void *worker(void *arg)
{
    int c = (int)(long)arg, prev_job = -1;
    cpu_set_t set;
    struct sched_param sp;

    CPU_ZERO(&set);
    CPU_SET(c % ncpu_online, &set);
    pin_ok[c] = pthread_setaffinity_np(pthread_self(), sizeof set, &set) == 0;
    sp.sched_priority = 80;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);

    for (;;) {
        int t, j, cpu;
        long long t0, s, e, end_target;
        pthread_barrier_wait(&bar_start);
        if (stop_flag) break;
        t = cur_tick; j = cur_slot[c]; t0 = tick_start_ns;
        s = now_ns();
        cpu = sched_getcpu();
        end_target = t0 + (long long)tick_ms * 1000000LL * WORK_FRACTION_PERCENT / 100;
        if (j >= 0) {
            unsigned int x = 12345u + (unsigned int)j;
            long long it = 0;
            int q;
            /* core LED: ON while the same job continues; a new job = short OFF blink */
            if (CORE_LEDS && c < 4 && j != prev_job) {
                int gap_ms = tick_ms / 4 < 150 ? tick_ms / 4 : 150;
                led_set(c, 0);
                if (prev_job >= 0 && gap_ms > 0) usleep((useconds_t)gap_ms * 1000);
            }
            if (CORE_LEDS && c < 4) led_set(c, 1);
            while (now_ns() < end_target) {           /* the job's real computation */
                for (q = 0; q < 200; q++) x = x * 1103515245u + 12345u;
                it += 200;
            }
            jobs[j].work += it;
            sink ^= x;
        } else if (CORE_LEDS && c < 4) {
            led_set(c, 0);
        }
        prev_job = j;
        e = now_ns();
        trace[t * MAX_CORES + c].cpu = cpu;
        trace[t * MAX_CORES + c].start_us = (long)((s - t0) / 1000);
        trace[t * MAX_CORES + c].end_us = (long)((e - t0) / 1000);
        pthread_barrier_wait(&bar_end);
    }
    return NULL;
}

static char task_symbol(int i)
{
    const char *s = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    return s[i % 62];
}

static void print_legend(void)
{
    int x;
    printf(" Legend:");
    for (x = 0; x < ntasks; x++) printf(" %c=%s%s", task_symbol(x), tasks[x].name, is_rt(x) ? "" : "(A)");
    printf("  .=idle\n");
}

static void gantt_rows(char alg, int start, int stop)
{
    int c, x;
    printf(" %-12s ", "time");
    for (x = start; x < stop; x++) {
        if (x % 10 == 0) {
            char lbl[12];
            int n = sprintf(lbl, "%d", x);
            if (x + n <= stop) { printf("%s", lbl); x += n - 1; } else putchar('|');
        } else putchar(' ');
    }
    printf("\n");
    for (c = 0; c < m; c++) {
        char label[48];
        if (alg == 'C') sprintf(label, "Core %d C%d", c, c / k + 1); else sprintf(label, "Core %d", c);
        printf(" %-12s ", label);
        for (x = start; x < stop; x++) {
            int j = sched_tbl[x * MAX_CORES + c];
            putchar(j < 0 ? '.' : task_symbol(jobs[j].task));
        }
        printf("\n");
    }
}

/* redraw the screen after tick t: growing Gantt chart + what each core does now */
static void live_render(char alg, int t)
{
    int c, x, done = 0, miss = 0, mig = 0, pre = 0, hmiss = 0;
    for (x = 0; x < njobs; x++) {
        if (jobs[x].kind == 'A') continue;
        done += jobs[x].finish >= 0; miss += jobs[x].missed;
        hmiss += jobs[x].missed && jobs[x].crit_rank == 3;
        mig += jobs[x].migrations; pre += jobs[x].preemptions;
    }
    printf("\033[H\033[2J");
    printf(" LIVE: %s, %s on the Raspberry Pi   tick %d / %d   (%d ms per tick)\n", alg_name(alg),
           mode_name(), t + 1, horizon, tick_ms);
    print_legend();
    printf("\n");
    gantt_rows(alg, t >= 59 ? t - 59 : 0, t + 1);
    printf("\n Now (t = %d):\n", t);
    for (c = 0; c < m; c++)
        if (!CORE_LEDS || c >= 4) printf("   Core %d  %s\n", c, ev_text[c]);
        else printf("   Core %d  LED %-3s  %s\n", c, sched_tbl[t * MAX_CORES + c] >= 0 ? "ON" : "off", ev_text[c]);
    if (TASK_LEDS) {
        printf("   Task LEDs:");
        for (x = 0; x < ntasks && x < task_led_max(); x++) {
            int on = 0;
            for (c = 0; c < m; c++) { int j = sched_tbl[t * MAX_CORES + c]; if (j >= 0 && jobs[j].task == x) on = 1; }
            printf("  %s=%s", tasks[x].name, on ? "ON" : "off");
        }
        printf("\n");
    }
    if (preempt_text[0]) printf("   Preempted:%s\n", preempt_text);
    if (miss_text[0]) printf("   DEADLINE MISSED:%s\n", miss_text);
    printf("\n So far: completed %d | misses %d (High %d) | preemptions %d | migrations %d\n",
           done, miss, hmiss, pre, mig);
    fflush(stdout);
}

/* run one algorithm on the hardware */
static void run_on_hardware(char alg)
{
    pthread_t th[MAX_CORES];
    struct sched_param sp;
    long long t0;
    int t, c;

    build_groups(alg);
    reset_run();
    live_on = live_mode == 1 || (live_mode == -1 && tick_ms >= 200);
    stop_flag = 0;
    pthread_barrier_init(&bar_start, NULL, m + 1);
    pthread_barrier_init(&bar_end, NULL, m + 1);
    for (c = 0; c < m; c++) pthread_create(&th[c], NULL, worker, (void *)(long)c);

    sp.sched_priority = 90;
    rt_ok = pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp) == 0;

    printf("\n Running %s (%s) on the hardware: %d ticks x %d ms = %.1f s ...\n",
           alg_name(alg), mode_name(), horizon, tick_ms, horizon * tick_ms / 1000.0);
    fflush(stdout);
    t0 = now_ns() + 200000000LL;
    for (t = 0; t < horizon; t++) {
        long long ts = t0 + (long long)t * tick_ms * 1000000LL, d0;
        sleep_until_ns(ts);
        d0 = now_ns();
        plan_tick(t);                                  /* the EDF scheduler decision */
        decision_ns[t] = now_ns() - d0;
        for (c = 0; c < m; c++) cur_slot[c] = sched_tbl[t * MAX_CORES + c];
        if (TASK_LEDS) {
            int i, on;
            for (i = 0; i < ntasks && i < task_led_max(); i++) {
                for (on = 0, c = 0; c < m; c++)
                    if (cur_slot[c] >= 0 && jobs[cur_slot[c]].task == i) on = 1;
                led_set(task_led_slot(i), on);
            }
        }
        cur_tick = t;
        tick_start_ns = ts;
        pthread_barrier_wait(&bar_start);
        pthread_barrier_wait(&bar_end);
        if (now_ns() > ts + (long long)tick_ms * 1000000LL) overruns++;
        if (live_on) live_render(alg, t);
        else if (t > 0 && t % 100 == 0) { printf("   tick %d / %d\r", t, horizon); fflush(stdout); }
    }
    stop_flag = 1;
    pthread_barrier_wait(&bar_start);
    for (c = 0; c < m; c++) pthread_join(th[c], NULL);
    pthread_barrier_destroy(&bar_start);
    pthread_barrier_destroy(&bar_end);
    sp.sched_priority = 0;
    pthread_setschedparam(pthread_self(), SCHED_OTHER, &sp);
    finish_horizon();
    if (TASK_LEDS) { int i; for (i = 0; i < ntasks && i < task_led_max(); i++) led_set(task_led_slot(i), 0); }
    if (CORE_LEDS) for (c = 0; c < m && c < 4; c++) led_set(c, 0);

    for (t = 0; t < horizon; t++) {
        long long best = -1;
        for (c = 0; c < m; c++) {
            Slot *s = &trace[t * MAX_CORES + c];
            if (sched_tbl[t * MAX_CORES + c] >= 0 && s->cpu != c % ncpu_online) wrong_cpu++;
            if (s->start_us > best) best = s->start_us;
        }
        dispatch_ns[t] = best * 1000;
    }
}

/* ------------------------------------------------------------- results */
typedef struct {
    int released, completed, evaluated, misses, preempt, migr, ctx, maxrt, aborted;
    double miss_ratio, avgrt, cpu_util, core_util[MAX_CORES], imbalance;
    double dec_avg_us, dec_max_us, disp_avg_us, disp_max_us;
    int lvl_jobs[4], lvl_eval[4], lvl_miss[4];      /* index 3 High, 2 Medium, 1 Low */
    int ap_jobs, ap_done, ap_maxrt;
    double ap_avgrt;
    int high_ok, overruns, wrong_cpu;
} Metrics;

static const char *high_text(const Metrics *M)
{
    return !crit_mode ? "not in plain EDF" : M->high_ok ? "YES" : "NO";
}

static Metrics compute(void)
{
    Metrics M;
    int i, c, t, nrt = 0;
    long long sumrt = 0, busy_total = 0, apsum = 0;
    double umax = 0, umin = 1e9;
    memset(&M, 0, sizeof M);
    for (i = 0; i < njobs; i++) {
        Job *J = &jobs[i];
        if (J->kind == 'A') {
            M.ap_jobs++;
            if (J->finish >= 0) {
                int rt = J->finish - J->release;
                M.ap_done++; apsum += rt;
                if (rt > M.ap_maxrt) M.ap_maxrt = rt;
            }
            continue;
        }
        M.released++;
        if (J->finish >= 0) {
            int rt = J->finish - J->release;
            M.completed++; nrt++; sumrt += rt;
            if (rt > M.maxrt) M.maxrt = rt;
        }
        if (J->finish >= 0 || J->absd <= horizon) { M.evaluated++; M.lvl_eval[J->crit_rank]++; }
        if (J->missed) { M.misses++; M.lvl_miss[J->crit_rank]++; }
        if (J->aborted) M.aborted++;
        M.lvl_jobs[J->crit_rank]++;
        M.preempt += J->preemptions;
        M.migr += J->migrations;
    }
    M.miss_ratio = M.evaluated ? (double)M.misses / M.evaluated : 0.0;
    M.avgrt = nrt ? (double)sumrt / nrt : 0.0;
    M.ap_avgrt = M.ap_done ? (double)apsum / M.ap_done : 0.0;
    for (c = 0; c < m; c++) {
        int busy = 0, last = -1;
        for (t = 0; t < horizon; t++) {
            int j = sched_tbl[t * MAX_CORES + c];
            if (j >= 0) {
                busy++;
                if (j != last) { if (last >= 0) M.ctx++; last = j; }
            }
        }
        busy_total += busy;
        M.core_util[c] = (double)busy / horizon;
        if (M.core_util[c] > umax) umax = M.core_util[c];
        if (M.core_util[c] < umin) umin = M.core_util[c];
    }
    M.cpu_util = (double)busy_total / ((double)m * horizon);
    M.imbalance = umax - umin;
    for (t = 0; t < horizon; t++) {
        double d = decision_ns[t] / 1000.0, p = dispatch_ns[t] / 1000.0;
        M.dec_avg_us += d; M.disp_avg_us += p;
        if (d > M.dec_max_us) M.dec_max_us = d;
        if (p > M.disp_max_us) M.disp_max_us = p;
    }
    M.dec_avg_us /= horizon;
    M.disp_avg_us /= horizon;
    M.high_ok = high_all_passed();
    M.overruns = overruns;
    M.wrong_cpu = wrong_cpu;
    return M;
}

static void print_gantt(char alg)
{
    int start, end = horizon < 120 ? horizon : 120;
    printf("\n Gantt chart (hardware execution), t = 0 to %d%s\n", end,
           end < horizon ? "  (first 120 ticks; full trace in hw_trace CSV)" : "");
    print_legend();
    for (start = 0; start < end; start += 60) {
        printf("\n");
        gantt_rows(alg, start, start + 60 < end ? start + 60 : end);
    }
}

static void print_results(char alg, const Metrics *M)
{
    int c, i, g;
    printf("\n==============================================================================\n");
    printf(" HARDWARE RESULTS - %s, %s  (%d cores, horizon %d, tick %d ms)\n", alg_name(alg),
           mode_name(), m, horizon, tick_ms);
    printf("==============================================================================\n");
    if (alg != 'G') {
        printf(" Allocation%s:\n", crit_mode ? " (DCDU order: High, Medium, Low)" : " (First Fit Decreasing)");
        for (g = 0; g < ngroups; g++) {
            printf("   %s %d (cores", alg == 'P' ? "Core group" : "Cluster", g + 1);
            for (c = 0; c < group_ncores[g]; c++) printf(" %d", group_core[g][c]);
            printf("):");
            for (i = 0; i < ntasks; i++) if (task_group[i] == g) printf(" %s%s", tasks[i].name, alloc_failed[i] ? "(!)" : "");
            printf("\n");
        }
        if (nfail) printf("   (!) = allocation failure: no group passed the test, task placed anyway.\n");
    }
    printf(" Design-time test: %s -> %s\n", analytic_text, analytic_ok ? "PASSED" : "NOT PASSED");
    printf(" High tasks guaranteed (High-only test): %s\n", high_text(M));
    for (i = 0; i < nhchk; i++) printf("   %-24s %-4s %s\n", hchk[i].label, hchk[i].passed ? "ok" : "FAIL", hchk[i].detail);
    printf("\n Schedule result: %s\n", M->misses ? "DEADLINE MISSES occurred" : "no deadline misses in the horizon");
    printf("   Released / completed jobs          %d / %d\n", M->released, M->completed);
    printf("   Deadline misses                    %d   (High %d, Medium %d, Low %d)\n", M->misses,
           M->lvl_miss[3], M->lvl_miss[2], M->lvl_miss[1]);
    printf("   Deadline miss ratio                %.2f %%\n", M->miss_ratio * 100);
    printf("   Average / maximum response time    %.3f / %d ticks\n", M->avgrt, M->maxrt);
    printf("   Preemptions / migrations           %d / %d\n", M->preempt, M->migr);
    printf("   Context switches                   %d\n", M->ctx);
    printf("   CPU utilization                    %.1f %%\n", M->cpu_util * 100);
    printf("   Load imbalance (max-min core)      %.1f %%\n", M->imbalance * 100);
    printf("   Per-core utilization              ");
    for (c = 0; c < m; c++) printf(" C%d %.1f%%", c, M->core_util[c] * 100);
    printf("\n\n By criticality:\n   %-8s %8s %10s %12s\n", "Level", "Jobs", "Misses", "Miss ratio");
    for (i = 3; i >= 1; i--)
        printf("   %-8s %8d %10d %11.2f%%\n", i == 3 ? "High" : i == 2 ? "Medium" : "Low", M->lvl_jobs[i],
               M->lvl_miss[i], M->lvl_eval[i] ? 100.0 * M->lvl_miss[i] / M->lvl_eval[i] : 0.0);
    if (M->ap_jobs) {
        printf("\n Aperiodic jobs (background, idle cores only): %d of %d done, avg response %.3f, max %d\n",
               M->ap_done, M->ap_jobs, M->ap_avgrt, M->ap_maxrt);
        for (i = 0; i < nap; i++) {
            Job *J = &jobs[apjob[i]];
            if (J->finish >= 0)
                printf("   %-16s arrival %-5d start %-5d finish %-5d response %d\n", tasks[J->task].name,
                       J->release, J->start, J->finish, J->finish - J->release);
            else printf("   %-16s arrival %-5d not finished (%d/%d done)\n", tasks[J->task].name, J->release,
                        J->executed, J->wcet);
        }
    }
    printf("\n Measured on the hardware:\n");
    printf("   Scheduler decision time            avg %.1f us, max %.1f us\n", M->dec_avg_us, M->dec_max_us);
    printf("   Dispatch latency (tick -> core)    avg %.1f us, max %.1f us\n", M->disp_avg_us, M->disp_max_us);
    printf("   Tick overruns                      %d of %d\n", overruns, horizon);
    printf("   Jobs on wrong physical CPU         %d  (0 = pinning worked)\n", wrong_cpu);
    printf("   Real-time priority (SCHED_FIFO)    %s\n", rt_ok ? "yes" : "NO - run with sudo for accurate timing");
    if (m > ncpu_online)
        printf("   WARNING: %d cores requested but only %d online; timing is not meaningful.\n", m, ncpu_online);
}

static FILE *open_in(const char *dir, const char *name, const char *mode)
{
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    return fopen(path, mode);
}

static void write_run_info(const char *dir)
{
    FILE *f = open_in(dir, "hw_run_info.csv", "w");
    char tp[300];
    if (f) {
        int i;
        snprintf(tp, sizeof tp, "%s", task_source);
        for (i = 0; tp[i]; i++) if (tp[i] == ',') tp[i] = ';';
        fprintf(f, "Cores,ClusterSize,Horizon,Tick_ms,MissPolicy,TaskFile,CritMode,GapPct,Seed,LEDs\n"
                   "%d,%d,%d,%d,%s,%s,%d,%d,%d,%d\n",
                m, k, horizon, tick_ms, abort_policy ? "abort" : "continue", tp, crit_mode, gap_pct, seed, use_leds);
        fclose(f);
    }
    {
        char path[512];
        snprintf(path, sizeof path, "%s/hw_tasks.csv", dir);
        save_csv(path);
    }
    f = open_in(dir, "hw_summary.csv", "w");     /* new run = new summary */
    if (f) fclose(f);
}

static void write_files(const char *dir, char alg, const Metrics *M)
{
    char name[64];
    FILE *f;
    int t, c, i;
    sprintf(name, "hw_trace_%c.csv", alg);
    if ((f = open_in(dir, name, "w")) != NULL) {
        fprintf(f, "Algorithm,Tick,Core,Job,Task,ActualCPU,Start_us,End_us\n");
        for (t = 0; t < horizon; t++)
            for (c = 0; c < m; c++) {
                int j = sched_tbl[t * MAX_CORES + c];
                Slot *s = &trace[t * MAX_CORES + c];
                if (j >= 0) {
                    char lbl[48];
                    job_label(j, lbl);
                    fprintf(f, "%s,%d,%d,%s,%s,%d,%ld,%ld\n", alg_name(alg), t, c, lbl, tasks[jobs[j].task].name,
                            s->cpu, s->start_us, s->end_us);
                } else
                    fprintf(f, "%s,%d,%d,Idle,Idle,%d,%ld,%ld\n", alg_name(alg), t, c, s->cpu, s->start_us, s->end_us);
            }
        fclose(f);
    }
    sprintf(name, "hw_jobs_%c.csv", alg);
    if ((f = open_in(dir, name, "w")) != NULL) {
        fprintf(f, "Algorithm,Job,Task,Type,Criticality,Release,AbsDeadline,C,Executed,Start,Finish,ResponseTime,"
                   "Missed,Aborted,Preemptions,Migrations,CoresUsed,WorkIterations\n");
        for (i = 0; i < njobs; i++) {
            Job *J = &jobs[i];
            char cores[96] = "", lbl[48];
            for (c = 0; c < m; c++) if (J->cores_used & (1u << c)) {
                char b[8]; sprintf(b, cores[0] ? " %d" : "%d", c); strcat(cores, b);
            }
            job_label(i, lbl);
            fprintf(f, "%s,%s,%s,%c,%s,%d,", alg_name(alg), lbl, tasks[J->task].name, J->kind,
                    tasks[J->task].crit, J->release);
            if (J->kind == 'A') fprintf(f, ","); else fprintf(f, "%d,", J->absd);
            fprintf(f, "%d,%d,", J->wcet, J->executed);
            if (J->start >= 0) fprintf(f, "%d,", J->start); else fprintf(f, ",");
            if (J->finish >= 0) fprintf(f, "%d,%d,", J->finish, J->finish - J->release); else fprintf(f, ",,");
            fprintf(f, "%s,%s,%d,%d,%s,%lld\n", J->missed ? "True" : "False", J->aborted ? "True" : "False",
                    J->preemptions, J->migrations, cores, J->work);
        }
        fclose(f);
    }
    f = open_in(dir, "hw_summary.csv", "a");
    if (f) {
        fseek(f, 0, SEEK_END);
        if (ftell(f) == 0)
            fprintf(f, "Algorithm,Cores,ClusterSize,Horizon,Tick_ms,MissPolicy,EDFMode,Released,Completed,"
                       "DeadlineMisses,HighMisses,MediumMisses,LowMisses,MissRatio,AvgResponse,MaxResponse,"
                       "Preemptions,Migrations,ContextSwitches,CPUUtil,LoadImbalance,AllocationFailures,"
                       "HighGuaranteed,AperiodicJobs,AperiodicCompleted,AperiodicAvgResponse,"
                       "DecisionAvg_us,DecisionMax_us,DispatchAvg_us,DispatchMax_us,TickOverruns,WrongCPU,"
                       "RealTimePriority,GapPct,Seed\n");
        fprintf(f, "%s,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%.4f,%.3f,%d,%d,%d,%d,%.4f,%.4f,%d,%s,%d,%d,%.3f,"
                   "%.1f,%.1f,%.1f,%.1f,%d,%d,%s,%d,%d\n",
                alg_name(alg), m, alg == 'P' ? 1 : alg == 'G' ? m : k, horizon, tick_ms,
                abort_policy ? "abort" : "continue", mode_name(), M->released, M->completed, M->misses,
                M->lvl_miss[3], M->lvl_miss[2], M->lvl_miss[1], M->miss_ratio, M->avgrt, M->maxrt,
                M->preempt, M->migr, M->ctx, M->cpu_util, M->imbalance, nfail, high_text(M),
                M->ap_jobs, M->ap_done, M->ap_avgrt,
                M->dec_avg_us, M->dec_max_us, M->disp_avg_us, M->disp_max_us, overruns, wrong_cpu,
                rt_ok ? "yes" : "no", gap_pct, seed);
        fclose(f);
    }
}

static void print_comparison(const Metrics *all, const char *al, int n)
{
    int i;
    printf("\n==============================================================================\n");
    printf(" COMPARISON ON HARDWARE - %s (same task set, %d cores, cluster size %d)\n", mode_name(), m, k);
    printf("==============================================================================\n");
    printf(" %-28s", "");
    for (i = 0; i < n; i++) printf(" %16s", alg_name(al[i]));
    printf("\n");
#define ROWI(label, expr) do { printf(" %-28s", label); for (i = 0; i < n; i++) printf(" %16d", (expr)); printf("\n"); } while (0)
#define ROWF(label, fmt, expr) do { printf(" %-28s", label); for (i = 0; i < n; i++) printf(fmt, (expr)); printf("\n"); } while (0)
    ROWI("Deadline misses", all[i].misses);
    ROWI("  - High", all[i].lvl_miss[3]);
    ROWI("  - Medium", all[i].lvl_miss[2]);
    ROWI("  - Low", all[i].lvl_miss[1]);
    ROWF("Deadline miss ratio %", " %16.2f", all[i].miss_ratio * 100);
    ROWF("Avg response time", " %16.3f", all[i].avgrt);
    ROWI("Max response time", all[i].maxrt);
    ROWI("Preemptions", all[i].preempt);
    ROWI("Migrations", all[i].migr);
    ROWF("CPU utilization %", " %16.1f", all[i].cpu_util * 100);
    printf(" %-28s", "High tasks guaranteed");
    for (i = 0; i < n; i++) printf(" %16s", high_text(&all[i]));
    printf("\n");
    if (all[0].ap_jobs) ROWF("Aperiodic avg response", " %16.3f", all[i].ap_avgrt);
    ROWF("Decision time avg (us)", " %16.1f", all[i].dec_avg_us);
    ROWF("Dispatch latency max (us)", " %16.1f", all[i].disp_max_us);
    ROWI("Tick overruns", all[i].overruns);
    ROWI("Jobs on wrong CPU", all[i].wrong_cpu);
#undef ROWI
#undef ROWF
}

/* run the chosen algorithms once, write files into dir; returns metrics */
static void run_set(const char *dir, Metrics *all)
{
    int i;
    horizon = H_setting > 0 ? H_setting : hyperperiod();
    if (horizon > MAX_HORIZON) {
        printf(" Hyperperiod exceeds %d; using a horizon of %d ticks.\n", MAX_HORIZON, MAX_HORIZON);
        horizon = MAX_HORIZON;
    }
    write_run_info(dir);
    leds_init();
    for (i = 0; i < nalgs; i++) {
        run_on_hardware(algs[i]);
        all[i] = compute();
        print_results(algs[i], &all[i]);
        print_gantt(algs[i]);
        write_files(dir, algs[i], &all[i]);
    }
    leds_close();
    if (nalgs > 1) print_comparison(all, algs, nalgs);
}

/* ------------------------------------------------------- interactive menu */
static void print_taskset(void)
{
    int i, nrt = 0, nh = 0;
    double U = 0, UH = 0;
    printf("\n Task set (%s)   cores = %d, cluster = %d\n", task_source, m, k);
    printf(" %3s %-16s %4s %5s %5s %5s %8s %-8s %7s %6s\n", "#", "Name", "Type", "C", "T", "D", "Priority",
           "Crit.", "Arrival", "U");
    for (i = 0; i < ntasks; i++) {
        Task *t = &tasks[i];
        if (t->kind == 'A') {
            printf(" %3d %-16s %4s %5d %5s %5s %8s %-8s %7d %6s\n", i + 1, t->name, "A", t->C, "-", "-", "-", "-",
                   t->arrival, "-");
            continue;
        }
        nrt++;
        U += (double)t->C / t->T;
        if (t->crit_rank == 3) { nh++; UH += (double)t->C / t->T; }
        printf(" %3d %-16s %4c %5d %5d %5d %8d %-8s %7s %6.3f\n", i + 1, t->name, t->kind, t->C, t->T, t->D,
               t->prio, t->crit, "-", (double)t->C / t->T);
    }
    printf(" Real-time tasks %d, aperiodic jobs %d.  U = %.3f  ->  %.1f %% of %d cores.  Hyperperiod = %d\n",
           nrt, ntasks - nrt, U, 100.0 * U / m, m, hyperperiod());
    printf(" High tasks: %d (load %.2f)\n", nh, UH);
    if (U > m + EPS) printf(" ! U > number of cores: overloaded, deadline misses are certain.\n");
}

static void set_algorithms(int choice)
{
    static const char *codes[8] = {"", "P", "G", "C", "PG", "PC", "GC", "PGC"};
    algo_choice = choice;
    strcpy(algs, codes[choice]);
    nalgs = (int)strlen(algs);
}

static const char *algo_words(void)
{
    static const char *w[8] = {"", "Partitioned", "Global", "Clustered", "Partitioned + Global",
                               "Partitioned + Clustered", "Global + Clustered", "All three"};
    return w[algo_choice];
}

/* type one task into *t (names already used: ts[0..n-1] except skip) */
static void enter_one_task(Task *t, const Task *ts, int n, int skip, int number)
{
    char buf[64], def[16], name[32];
    const char *cf;
    char kind;
    int C, T, D, P, i, ok;
    printf("\n Task %d:\n", number);
    sprintf(def, "T%d", number);
    for (;;) {
        read_line("   Name (no spaces)", def, buf, sizeof buf);
        ok = buf[0] && !strchr(buf, ' ') && !strchr(buf, ',') && strlen(buf) < 32;
        for (i = 0; ok && i < n; i++) if (i != skip && !strcmp(ts[i].name, buf)) ok = 0;
        if (ok) break;
        printf("  ! Name must be unique, without spaces or commas (max 31 characters).\n");
    }
    strcpy(name, buf);
    kind = ask_letter("   Type: P = periodic, S = sporadic, A = aperiodic", "PSA", 'P');
    if (kind == 'A') {
        C = ask_int("   C = execution time", 1, 1000, 2);
        P = ask_int("   Arrival time", 0, MAX_HORIZON, 0);
        set_task(t, name, 'A', C, 0, 0, 0, "", P);
        return;
    }
    C = ask_int("   C = execution time (WCET)", 1, 1000, 1);
    T = ask_int(kind == 'S' ? "   T = minimum gap between releases" : "   T = period", C, 10000, C > 10 ? C : 10);
    D = ask_int("   D = relative deadline (C <= D <= T)", C, T, T);
    P = ask_int_x("   Priority, 1 = highest (only breaks deadline ties)", 1, 999, 0, 1);
    for (;;) {
        read_line("   Criticality (H/M/L)", "L", buf, sizeof buf);
        if ((cf = crit_full(buf)) != NULL) break;
        printf("  ! Type H, M or L.\n");
    }
    set_task(t, name, kind, C, T, D, P, cf, 0);
}

static double rnd_next(uint32_t *st)
{
    *st ^= *st << 13; *st ^= *st >> 17; *st ^= *st << 5;
    return (double)*st / 4294967296.0;
}

static void random_taskset(void)
{
    Task tmp[MAX_TASKS];
    static const int periods[6] = {10, 20, 25, 40, 50, 100};
    static const char *crits[3] = {"High", "Medium", "Low"};
    int n, na, sp, sd, i, tries, H;
    double U, u[MAX_TASKS];
    uint32_t st;
    char src[120];
    n = ask_int("Number of real-time tasks (P + S)", 2, MAX_TASKS, 8);
    U = ask_int("Total utilization in % of ONE core (e.g. 280 = 2.8 cores)", 10, n * 100, 70 * m) / 100.0;
    sp = ask_int("How many % of them sporadic", 0, 100, 0);
    na = ask_int("Number of aperiodic jobs", 0, MAX_TASKS - n, 0);
    sd = ask_int("Random seed", 0, 1000000, 42);
    st = (uint32_t)sd * 2654435761u + 1u;
    for (tries = 0; tries < 1000; tries++) {          /* UUniFast-Discard */
        double sum = U, nxt;
        int bad = 0;
        for (i = 0; i < n - 1; i++) {
            nxt = sum * pow(rnd_next(&st), 1.0 / (n - 1 - i));
            u[i] = sum - nxt;
            sum = nxt;
        }
        u[n - 1] = sum;
        for (i = 0; i < n; i++) if (u[i] > 1.0) bad = 1;
        if (!bad) break;
    }
    for (i = 0; i < n; i++) {
        char name[16];
        int T = periods[(int)(rnd_next(&st) * 6) % 6], C = (int)(u[i] * T + 0.5);
        if (C < 1) C = 1;
        if (C > T) C = T;
        sprintf(name, "T%d", i + 1);
        {
            char kind = rnd_next(&st) * 100 < sp ? 'S' : 'P';
            int cr = (int)(rnd_next(&st) * 3) % 3;
            set_task(&tmp[i], name, kind, C, T, T, 0, crits[cr], 0);
        }
    }
    {
        long long h = 1;
        for (i = 0; i < n; i++) { h = h / gcd_i((int)h, tmp[i].T) * tmp[i].T; if (h > MAX_HORIZON) h = MAX_HORIZON; }
        H = (int)h;
    }
    for (i = 0; i < na; i++) {
        char name[16];
        sprintf(name, "A%d", i + 1);
        {
            int c2 = 1 + (int)(rnd_next(&st) * 5) % 5, arr = (int)(rnd_next(&st) * H) % H;
            set_task(&tmp[n + i], name, 'A', c2, 0, 0, 0, "", arr);
        }
    }
    snprintf(src, sizeof src, "random (U=%.2f, seed=%d)", U, sd);
    if (commit_tasks(tmp, n + na, src)) { save_csv("tasks_random.csv"); printf(" Saved to tasks_random.csv\n"); }
}

static void menu_tasks(void)
{
    char ch;
    printf("\n 3. ENTER TASKS\n   A. Manual entry\n   B. Load a CSV file\n   C. Random task set\n   D. Built-in example\n");
    ch = ask_letter("How", "ABCD", 'D');
    if (ch == 'A') {
        Task tmp[MAX_TASKS];
        int n = ask_int("How many tasks", 1, MAX_TASKS, 4), i;
        for (i = 0; i < n; i++) enter_one_task(&tmp[i], tmp, i, -1, i + 1);
        if (commit_tasks(tmp, n, "entered manually")) { save_csv("tasks_entered.csv"); printf(" Saved to tasks_entered.csv\n"); }
    } else if (ch == 'B') {
        char buf[256];
        printf(" CSV columns: Name,Type,C,T,D,Priority,Criticality,Arrival   (Priority may be empty)\n"
               "   e.g.  EngineCtl,P,2,5,5,,High,     and     DiagScan,A,6,,,,,12\n");
        for (;;) {
            read_line("CSV file", "tasks.csv", buf, sizeof buf);
            if (load_csv(buf)) break;
        }
    } else if (ch == 'C') {
        random_taskset();
    } else {
        int e;
        for (e = 0; e < NEXAMPLES; e++) printf("   %d. %s\n", e + 1, EXAMPLES[e].name);
        int old_m = m, old_k = k;
        e = ask_int("Example", 1, NEXAMPLES, 1);
        load_example(e - 1);
        if (m != old_m) {
            char yn = ask_letter("This example is made for that many cores. Use them? (Y/N)", "YN", 'Y');
            printf("   (example designed for %d cores)\n", m);
            if (yn == 'N') { m = old_m; k = old_k <= m ? old_k : m; }
        }
        printf(" Cores = %d, cluster size = %d.\n", m, k);
    }
    print_taskset();
}

static void menu_show_edit(void)
{
    for (;;) {
        char ch;
        Task tmp[MAX_TASKS];
        int n = ntasks, i;
        print_taskset();
        printf("\n   C. Values are correct - confirm\n   E. Edit a task\n   A. Add a task\n   D. Delete a task\n");
        ch = ask_letter("Action", "CEAD", 'C');
        if (ch == 'C') { printf(" Confirmed.\n"); return; }
        memcpy(tmp, tasks, sizeof(Task) * n);
        if (ch == 'A') {
            if (n >= MAX_TASKS) { printf("  ! At most %d tasks.\n", MAX_TASKS); continue; }
            enter_one_task(&tmp[n], tmp, n, -1, n + 1);
            n++;
        } else if (ch == 'D') {
            if (n <= 1) { printf("  ! Keep at least one task.\n"); continue; }
            i = ask_int("Row number to delete", 1, n, n) - 1;
            memmove(&tmp[i], &tmp[i + 1], sizeof(Task) * (n - i - 1));
            n--;
        } else {
            i = ask_int("Row number to edit", 1, n, 1) - 1;
            printf(" Enter the new values for row %d:\n", i + 1);
            enter_one_task(&tmp[i], tmp, n, i, i + 1);
        }
        {
            char src[300];
            snprintf(src, sizeof src, "%.280s (edited)", strstr(task_source, "(edited)") ? "edited set" : task_source);
            commit_tasks(tmp, n, src);
        }
    }
}

static void menu_settings(void)
{
    printf("\n 4. SETTINGS\n");
    H_setting = ask_int("Horizon in ticks (0 = one hyperperiod)", 0, MAX_HORIZON, H_setting);
    tick_ms = ask_int("Tick length in ms (10 for measurements; 300-500 to watch LEDs)", 1, 10000, tick_ms);
    abort_policy = ask_int("On deadline miss: 1 = continue late job, 2 = abort job", 1, 2, abort_policy + 1) == 2;
    crit_mode = ask_int("EDF mode: 1 = plain EDF, 2 = criticality-first EDF (High jobs first)", 1, 2, crit_mode + 1) == 2;
    gap_pct = ask_int("Sporadic tasks: max extra gap after T, in % of T", 0, 200, gap_pct);
    seed = ask_int("Seed for sporadic gaps (same as on the PC)", 0, 1000000, seed);
    printf(" LEDs:\n   0. No LEDs (use for measurements)\n   1. One LED per CORE (pins 11, 13, 15, 16)\n"
           "   2. One LED per TASK (up to 16)\n   3. BOTH: core LEDs + task LEDs T1-T12\n");
    use_leds = ask_int("LEDs", 0, 3, use_leds);
    live_mode = ask_int("Live Gantt chart: 0 = off, 1 = on, 2 = automatic (on when tick >= 200 ms)", 0, 2,
                        live_mode < 0 ? 2 : live_mode);
    if (live_mode == 2) live_mode = -1;
    last_run_ok = 0;
}

static void ensure_dir(const char *d) { mkdir(d, 0755); }

static void copy_file(const char *from, const char *to)
{
    FILE *a = fopen(from, "rb"), *b;
    char buf[8192];
    size_t n;
    if (!a) return;
    b = fopen(to, "wb");
    if (!b) { fclose(a); return; }
    while ((n = fread(buf, 1, sizeof buf, a)) > 0) fwrite(buf, 1, n, b);
    fclose(a); fclose(b);
}

static void stamp(char *buf)
{
    time_t now = time(NULL);
    strftime(buf, 32, "%Y%m%d_%H%M%S", localtime(&now));
}

static void menu_compare_modes(void)
{
    Metrics plain[3], crit[3];
    char st[32], d1[96], d2[96];
    int keep = crit_mode, i;
    stamp(st);
    ensure_dir("results");
    sprintf(d1, "results/%s_plain", st);
    sprintf(d2, "results/%s_critfirst", st);
    ensure_dir(d1); ensure_dir(d2);
    crit_mode = 0; run_set(d1, plain);
    crit_mode = 1; run_set(d2, crit);
    crit_mode = keep;
    printf("\n==============================================================================\n");
    printf(" PLAIN EDF vs CRITICALITY-FIRST EDF on the hardware\n");
    printf("==============================================================================\n");
    printf(" %-18s %-22s %6s %6s %6s %6s %10s %9s\n", "Algorithm", "EDF mode", "High", "Med", "Low", "Total",
           "High guar.", "Avg resp");
    for (i = 0; i < nalgs; i++) {
        int c2;
        for (c2 = 0; c2 < 2; c2++) {
            Metrics *M = c2 ? &crit[i] : &plain[i];
            printf(" %-18s %-22s %6d %6d %6d %6d %10s %9.3f\n", alg_name(algs[i]),
                   c2 ? "Criticality-first EDF" : "Plain EDF", M->lvl_miss[3], M->lvl_miss[2], M->lvl_miss[1],
                   M->misses, c2 ? (M->high_ok ? "YES" : "NO") : "-", M->avgrt);
        }
    }
    printf("\n Files: %s/ and %s/  (copy both folders to the PC)\n", d1, d2);
}

static void menu_export(void)
{
    static const char *files[] = {"hw_run_info.csv", "hw_tasks.csv", "hw_summary.csv", "hw_trace_P.csv",
                                  "hw_trace_G.csv", "hw_trace_C.csv", "hw_jobs_P.csv", "hw_jobs_G.csv",
                                  "hw_jobs_C.csv", NULL};
    char st[32], dir[96], to[200];
    int i, n = 0;
    if (!last_run_ok) { printf("  ! Run menu 7 first (the last run is exported).\n"); return; }
    stamp(st);
    ensure_dir("results");
    sprintf(dir, "results/%s_%s", st, crit_mode ? "critfirst" : "plain");
    ensure_dir(dir);
    for (i = 0; files[i]; i++) {
        if (access(files[i], F_OK) != 0) continue;
        sprintf(to, "%s/%s", dir, files[i]);
        copy_file(files[i], to);
        n++;
    }
    printf(" Copied %d files to %s/\n Copy this folder to the PC and run:  python compare_with_simulator.py %s\n",
           n, dir, dir);
}


static void print_settings_line(void)
{
    printf(" Settings: horizon %s, tick %d ms, on miss %s, %s, sporadic gap %d %%, seed %d, LEDs %s, live view %s\n",
           H_setting ? "custom" : "= hyperperiod", tick_ms, abort_policy ? "abort" : "continue", mode_name(),
           gap_pct, seed, use_leds == 0 ? "off" : use_leds == 1 ? "cores" : use_leds == 2 ? "tasks" : "cores+tasks",
           live_mode == 1 ? "on" : live_mode == 0 ? "off" : "auto");
}

/* the normal way to use the program: all steps in order, Enter keeps the value in [ ] */
static void guided_flow(void)
{
    Metrics all[3];
    int full = 1, exported = 0;
    for (;;) {
        char next;
        if (full) {
            printf("\n================ GUIDED RUN  (Enter = keep value in [ ],  X = full menu,  Q = quit) ================\n");
            printf("\n STEP 1/6  Cores\n");
            m = ask_int("Number of cores", 1, MAX_CORES, m);
            if (m > ncpu_online) printf(" WARNING: only %d cores online; cores will share CPUs (test only).\n", ncpu_online);
            if (k > m) k = m;
            printf("\n STEP 2/6  Cluster size (Clustered EDF)\n");
            k = ask_int("Cores per cluster", 1, m, k);
            printf("\n STEP 3/6  Tasks\n");
            menu_tasks();
            printf("\n STEP 4/6  Check the values\n");
            menu_show_edit();
            printf("\n STEP 5/6  Algorithms\n   1. Partitioned\n   2. Global\n   3. Clustered\n   4. Partitioned + Global\n"
                   "   5. Partitioned + Clustered\n   6. Global + Clustered\n   7. All three\n");
            set_algorithms(ask_int("Run which", 1, 7, algo_choice));
        }
        printf("\n STEP 6/6  Settings\n");
        print_settings_line();
        if (ask_letter("Change settings? (Y/N)", "YN", 'N') == 'Y') menu_settings();
        run_set(".", all);
        last_run_ok = 1;
        exported = 0;
        printf("\n Files written here: hw_*.csv\n");
        for (;;) {
            printf("\n WHAT NEXT?\n   R. Run again (same tasks, change settings if you like)\n"
                   "   N. New run from step 1\n   C. Compare plain EDF vs criticality-first EDF (same tasks)\n"
                   "   E. Export this run to a results folder\n   X. Full menu\n   Q. Quit\n");
            next = ask_letter("Choice", "RNCEXQ", exported ? 'R' : 'E');
            if (next == 'C') { menu_compare_modes(); continue; }
            if (next == 'E') { menu_export(); exported = 1; continue; }
            if (next == 'X') longjmp(menu_jmp, 1);
            if (next == 'Q') quit_program();
            break;
        }
        full = next == 'N';
    }
}

static void usage(void)
{
    printf("Usage: sudo ./rtos_hw                    (menu)\n"
           "       sudo ./rtos_hw -f tasks.csv [-a P|G|C|PG|PC|GC|ALL] [-m cores] [-k cluster]\n"
           "                      [-H horizon] [-t tick_ms] [--abort] [--critfirst] [--gap pct] [--seed n]\n"
           "                      [--leds|--task-leds|--both-leds] [--live|--no-live]\n"
           "  --critfirst (or --crit) = criticality-first EDF (High jobs always first)\n");
}

int main(int argc, char **argv)
{
    int i;
    Metrics all[3];

    setvbuf(stdout, NULL, _IOLBF, 0);
    ncpu_online = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu_online < 1) ncpu_online = 1;
    if (m > ncpu_online) m = ncpu_online;
    if (k > m) k = m;
    load_example(0);
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) { /* needs sudo; only affects timing */ }

    printf("==============================================================================\n");
    printf(" MULTI-CORE EDF ON HARDWARE  -  Partitioned / Global / Clustered EDF\n");
    printf(" Online CPU cores on this board: %d\n", ncpu_online);
    printf("==============================================================================\n");

    if (argc > 1) {                                   /* command-line run */
        char file[256] = "tasks.csv";
        for (i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "-f") && i + 1 < argc) snprintf(file, sizeof file, "%s", argv[++i]);
            else if (!strcmp(argv[i], "-a") && i + 1 < argc) {
                const char *a = argv[++i];
                if (!strcasecmp(a, "ALL")) set_algorithms(7);
                else if (!strcasecmp(a, "P")) set_algorithms(1);
                else if (!strcasecmp(a, "G")) set_algorithms(2);
                else if (!strcasecmp(a, "C")) set_algorithms(3);
                else if (!strcasecmp(a, "PG")) set_algorithms(4);
                else if (!strcasecmp(a, "PC")) set_algorithms(5);
                else if (!strcasecmp(a, "GC")) set_algorithms(6);
                else { usage(); return 1; }
            }
            else if (!strcmp(argv[i], "-m") && i + 1 < argc) m = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-k") && i + 1 < argc) k = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-H") && i + 1 < argc) H_setting = atoi(argv[++i]);
            else if (!strcmp(argv[i], "-t") && i + 1 < argc) tick_ms = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--gap") && i + 1 < argc) gap_pct = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--abort")) abort_policy = 1;
            else if (!strcmp(argv[i], "--crit") || !strcmp(argv[i], "--critfirst")) crit_mode = 1;
            else if (!strcmp(argv[i], "--leds")) use_leds = 1;
            else if (!strcmp(argv[i], "--task-leds")) use_leds = 2;
            else if (!strcmp(argv[i], "--both-leds")) use_leds = 3;
            else if (!strcmp(argv[i], "--live")) live_mode = 1;
            else if (!strcmp(argv[i], "--no-live")) live_mode = 0;
            else { usage(); return 1; }
        }
        if (m < 1 || m > MAX_CORES) { printf("Cores must be 1-%d.\n", MAX_CORES); return 1; }
        if (k < 1 || k > m) { printf("Cluster size must be 1-%d.\n", m); return 1; }
        if (tick_ms < 1 || gap_pct < 0) { printf("Bad tick or gap value.\n"); return 1; }
        if (!load_csv(file)) return 1;
        print_taskset();
        if (m > ncpu_online)
            printf(" WARNING: %d cores requested, only %d online. Cores will share CPUs (test only).\n", m, ncpu_online);
        run_set(".", all);
        printf("\n Files written: hw_summary.csv, hw_trace_*.csv, hw_jobs_*.csv, hw_tasks.csv, hw_run_info.csv\n");
        return 0;
    }

    {
        static volatile int started = 0;
        setjmp(menu_jmp);
        menu_ready = 1;
        if (!started) { started = 1; guided_flow(); }
    }
    for (;;) {
        int ch;
        printf("\n------------------------------------------------------------------------------\n");
        printf(" MAIN MENU   cores=%d  cluster=%d  tasks=%d  algorithms=%s  mode=%s\n"
               "             tick=%d ms  LEDs=%s\n", m, k, ntasks, algo_words(), mode_name(), tick_ms,
               use_leds == 0 ? "off" : use_leds == 1 ? "cores" : use_leds == 2 ? "tasks" : "cores+tasks");
        printf("------------------------------------------------------------------------------\n");
        printf("  1. Number of cores\n  2. Cluster size\n  3. Enter tasks  (A manual / B CSV / C random / D example)\n"
               "  4. Settings  (horizon, tick, miss policy, EDF mode, sporadic gap, LEDs, live view)\n"
               "  5. Show tasks, check values, edit / add / delete\n"
               "  6. Choose algorithms  (P / G / C / P+G / P+C / G+C / all three)\n"
               "  7. Run on the hardware: results + Gantt charts\n"
               "  8. Compare plain EDF vs criticality-first EDF on the hardware\n"
               "  9. Export the last run to a results folder\n"
               " 10. Guided run (all steps in order, like at the start)\n"
               "  0. Exit\n  (At any question: X = main menu, Q = quit)\n");
        ch = ask_int("Choose", 0, 10, 10);
        switch (ch) {
        case 0: quit_program(); break;
        case 1:
            m = ask_int("Number of cores", 1, MAX_CORES, m);
            if (m > ncpu_online) printf(" WARNING: only %d cores online; cores will share CPUs (test only).\n", ncpu_online);
            if (k > m) k = m;
            last_run_ok = 0;
            break;
        case 2:
            k = ask_int("Cores per cluster (Clustered EDF)", 1, m, k);
            last_run_ok = 0;
            break;
        case 3: menu_tasks(); break;
        case 4: menu_settings(); break;
        case 5: menu_show_edit(); break;
        case 6:
            printf("   1. Partitioned\n   2. Global\n   3. Clustered\n   4. Partitioned + Global\n"
                   "   5. Partitioned + Clustered\n   6. Global + Clustered\n   7. All three\n");
            set_algorithms(ask_int("Run which", 1, 7, algo_choice));
            last_run_ok = 0;
            break;
        case 7:
            print_taskset();
            run_set(".", all);
            last_run_ok = 1;
            printf("\n Files written here: hw_*.csv  (menu 9 copies them to a results folder)\n");
            pause_enter();
            break;
        case 8: menu_compare_modes(); pause_enter(); break;
        case 9: menu_export(); break;
        case 10: guided_flow(); break;
        }
    }
    return 0;
}
