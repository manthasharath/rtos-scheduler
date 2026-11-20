#!/bin/bash
# collect_data.sh - runs every measurement needed for the report, without LEDs.
# Usage on the Pi:   sudo ./collect_data.sh
# Result: one folder results/report_<date>/ with a sub-folder per run, plus a .tar.gz of it.
# Takes about 5 minutes. Do not touch the Pi while it runs (it disturbs the timing).

cd "$(dirname "$0")" || exit 1
if [ "$(id -u)" -ne 0 ]; then echo "Please run with sudo:  sudo ./collect_data.sh"; exit 1; fi
make -s || exit 1
OUT="results/report_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT"
BIN="$(pwd)/rtos_hw"
EX="$(pwd)/example_csv"

run() {   # run <folder name> <csv> <cores> <cluster> <tick ms> [extra options]
    local name=$1 csv=$2 m=$3 k=$4 t=$5; shift 5
    mkdir -p "$OUT/$name"
    echo ">>> $name"
    (cd "$OUT/$name" && "$BIN" -f "$csv" -a ALL -m "$m" -k "$k" -t "$t" --no-live "$@" > console.txt 2>&1)
    grep -E "Decision time avg|Dispatch latency max|Tick overruns|Jobs on wrong CPU" "$OUT/$name/console.txt" | head -4
}

# 1. Overhead vs tick length (automotive example, both EDF modes)
for t in 400 10 2 1; do
    run "tick_${t}ms_plain"     "$EX/ex1_automotive.csv" 4 2 "$t"
    run "tick_${t}ms_critfirst" "$EX/ex1_automotive.csv" 4 2 "$t" --critfirst
done

# 2. The five built-in examples, both EDF modes, 10 ms tick
run ex1_automotive_plain        "$EX/ex1_automotive.csv"      4 2 10
run ex1_automotive_critfirst    "$EX/ex1_automotive.csv"      4 2 10 --critfirst
run ex2_dhall_plain             "$EX/ex2_dhall.csv"           2 1 10
run ex3_partition_fails_plain   "$EX/ex3_partition_fails.csv" 2 2 10
run ex4_heavy_mix_plain         "$EX/ex4_heavy_mix.csv"       4 2 10
run ex4_heavy_mix_critfirst     "$EX/ex4_heavy_mix.csv"       4 2 10 --critfirst
run ex5_overload_plain          "$EX/ex5_overload_crit.csv"   4 2 10
run ex5_overload_critfirst      "$EX/ex5_overload_crit.csv"   4 2 10 --critfirst
run ex5_overload_critfirst_abort "$EX/ex5_overload_crit.csv"  4 2 10 --critfirst --abort

# 3. Board information for the report
{ echo "date: $(date)"; uname -a; grep -m1 -i "model" /proc/cpuinfo; nproc; cat /proc/device-tree/model 2>/dev/null; echo; } > "$OUT/board_info.txt"

tar czf "$OUT.tar.gz" -C results "$(basename "$OUT")"
chown -R "${SUDO_USER:-pi}":"${SUDO_USER:-pi}" results 2>/dev/null
echo
echo "Done. All runs are in:  $OUT/"
echo "Copy to the PC (in PowerShell on the PC):"
echo "   scp ${SUDO_USER:-pi}@$(hostname -I | awk '{print $1}'):$(pwd)/$OUT.tar.gz ."
