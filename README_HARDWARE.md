# Raspberry Pi version – step-by-step guide (new board, from scratch)

`rtos_hw.c` runs Partitioned, Global and Clustered EDF on the **real 4 cores** of a Raspberry Pi (3, 4 or 5) with the
same rules as the PC simulator. It has the same menu as the terminal version, optional LEDs (per core, per task, or
both), a live Gantt chart that grows tick by tick, and it **measures** the real scheduling overhead.

What you need: Raspberry Pi + power supply, micro-SD card (16 GB+), monitor + HDMI cable + USB keyboard (or SSH),
network cable or Wi-Fi, the lab PC, and for the LED demo: breadboard, LEDs, 330 Ω resistors, jumper wires.

---

## Part A – Prepare the SD card (on the lab PC)

1. Install and open **Raspberry Pi Imager** (raspberrypi.com/software).
2. **Choose device**: your board (e.g. Raspberry Pi 4). **Choose OS**: *Raspberry Pi OS* (the recommended one).
   **Choose storage**: the SD card.
3. Click **Next → Edit settings**:
   * General: hostname `rtospi`, username `pi`, a password you remember; Wi-Fi only if you use Wi-Fi; your time zone.
   * Services: **Enable SSH → use password authentication**.
4. **Save → Yes → Yes**. Wait until it says *Write successful*. Take the card out.

## Part B – First start of the Pi

1. Put the card in the Pi, connect monitor, keyboard, mouse and network cable, then power.
2. Wait for the desktop (first start takes 2–3 minutes). Open **Terminal**.
3. Find the IP address:  `hostname -I`   → write down the first number, e.g. `192.168.1.23`.
4. Make sure the compiler is there:  `gcc --version`
   (if it says *not found*: `sudo apt update && sudo apt install -y build-essential`).
5. Check the cores:  `nproc`  → must print **4**.

## Part C – Copy the program to the Pi

**Way 1 – network (PowerShell on the lab PC, in the folder with the zip):**
```
scp rtos_hardware_pi.zip pi@192.168.1.23:~
```
(type `yes` the first time, then the Pi password; nothing is shown while typing the password).

**Way 2 – no network, no pen drive (SD card):** shut the Pi down (`sudo poweroff`), put the SD card in the PC,
open the drive called **bootfs**, copy `rtos_hardware_pi.zip` into it, eject, put the card back, start the Pi, then:
```
cp /boot/firmware/rtos_hardware_pi.zip ~
```

**Then on the Pi (Terminal):**
```
cd ~
unzip rtos_hardware_pi.zip
cd rtos_hardware_pi
make
```
`make` prints nothing on success. `ls` must now show `rtos_hw`.

## Part D – First run (no LEDs)

```
sudo ./rtos_hw
```
The menu appears with the default example (automotive controller, 4 cores, all three algorithms):
```
1 Number of cores          6 Choose algorithms
2 Cluster size             7 Run on the hardware: results + Gantt charts
3 Enter tasks              8 Compare plain EDF vs criticality-first EDF on the hardware
4 Settings                 9 Export the last run to a results folder
5 Show / check / edit      0 Exit        (at any question: X = main menu, Q = quit)
```
Type **7** and Enter. After ~3 s you get, for each algorithm: allocation, design-time test, *High tasks guaranteed*,
misses per criticality, aperiodic jobs, Gantt chart and the **measured** decision time, dispatch latency, tick overruns
and *Jobs on wrong physical CPU* (must be 0). Then a comparison table.

* Always start with `sudo` (needed for real-time priority, CPU pinning and LEDs). Without it the result says
  *Real-time priority: NO*.
* Task input (menu 3): **A** type them, **B** CSV file (e.g. `example_csv/ex5_overload_crit.csv`), **C** random,
  **D** one of the 5 examples.
* CSV format (same as the PC): `Name,Type,C,T,D,Priority,Criticality,Arrival` – e.g. `EngineCtl,P,2,5,5,,High,` and
  `DiagScan,A,6,,,,,12`. Priority may be empty (automatic).
* Command line (no menu), e.g.: `sudo ./rtos_hw -f example_csv/ex1_automotive.csv -a ALL -m 4 -k 2 -t 10 --critfirst`
  (options: `-a P|G|C|PG|PC|GC|ALL  -H horizon  -t tick_ms  --abort  --critfirst  --gap pct  --seed n
  --leds|--task-leds|--both-leds  --live|--no-live`).

## Part E – LED wiring (demo only)

For every LED: **GPIO pin → 330 Ω resistor → LED long leg (+)**, **LED short leg (−) → GND** (pin 6, 9, 14, 20, 25, 30,
34 or 39). Use the breadboard's blue (−) rail as common ground, connected to pin 6 and pin 39.
Power the Pi **off** while wiring.

| LED | BCM GPIO | Header pin | Mode 1: per core | Mode 2: per task | Mode 3: both |
|---|---|---|---|---|---|
| 1 | 17 | 11 | Core 0 | Task 1 | Core 0 |
| 2 | 27 | 13 | Core 1 | Task 2 | Core 1 |
| 3 | 22 | 15 | Core 2 | Task 3 | Core 2 |
| 4 | 23 | 16 | Core 3 | Task 4 | Core 3 |
| 5 | 24 | 18 | | Task 5 | Task 1 |
| 6 | 25 | 22 | | Task 6 | Task 2 |
| 7 | 5 | 29 | | Task 7 | Task 3 |
| 8 | 6 | 31 | | Task 8 | Task 4 |
| 9 | 13 | 33 | | Task 9 | Task 5 |
| 10 | 19 | 35 | | Task 10 | Task 6 |
| 11 | 26 | 37 | | Task 11 | Task 7 |
| 12 | 16 | 36 | | Task 12 | Task 8 |
| 13 | 20 | 38 | | Task 13 | Task 9 |
| 14 | 21 | 40 | | Task 14 | Task 10 |
| 15 | 12 | 32 | | Task 15 | Task 11 |
| 16 | 18 | 12 | | Task 16 | Task 12 |

Task number = row number in the task table (aperiodic rows included). With many LEDs on, 470 Ω–1 kΩ is also fine.

**Test each LED** (Pi on): `pinctrl set 17 op dh` (on) and `pinctrl set 17 op dl` (off); replace 17 by each GPIO.
If an LED stays dark, turn it round (long leg to the resistor side).

**Demo run with LEDs and the live Gantt chart:** menu **4** → horizon `0`, tick **400** ms, miss policy 1, EDF mode as
you like, gap 50, seed 1, LEDs **3** (both), live view **2** (automatic) → menu **7**. The Gantt chart grows on the
screen tick by tick while the LEDs show which core and which task is running. Core LED: ON = same job continues,
short blink = a different job starts, OFF = idle. Use the **Overload + criticality** example (menu 3 → D → 5) to show
High tasks protected in criticality-first mode.

LEDs and live view slow the program down: never use them for the measurements below.

## Part F – Collect all data for the report (one command, ~5 minutes)

```
cd ~/rtos_hardware_pi
sudo ./collect_data.sh
```
It runs, **without LEDs**, all three algorithms for:
1. tick length **400, 10, 2, 1 ms** (automotive example, plain and criticality-first) → overhead vs tick size;
2. the **5 examples** at 10 ms (plain and criticality-first where it matters, plus overload with *abort*);
3. writes the board information.

Everything goes into `results/report_<date>/` (one sub-folder per run, each with `hw_*.csv` and `console.txt`) and a
single file `results/report_<date>.tar.gz`. Do not use the Pi while it runs.

**Copy to the PC** (PowerShell on the PC, in the project folder `multicore_rtos_simulator`; the script prints the exact
command):
```
scp pi@192.168.1.23:~/rtos_hardware_pi/results/report_<date>.tar.gz .
tar -xzf report_<date>.tar.gz
python compare_with_simulator.py report_<date>
```
The last line compares **every** run with the simulator, prints MATCH / DIFFERENT for each and writes
`report_<date>/all_hw_vs_simulator.csv` (all runs in one table: misses, response times, decision time, dispatch
latency, overruns – ready for the report).

You can also open the website → **Raspberry Pi results** and drag in the `hw_*.csv` files of one run folder: it shows
the Pi Gantt chart next to the simulator Gantt chart, MATCH per algorithm and the overhead charts (good screenshots).

## Part G – Remaining data from the PC (website)

1. **6 · Batch experiments**: cores **4**, cluster 2, tasks 8, levels 0.3–0.9, 50 sets per level, EDF mode **Both**,
   seed 42 → Run. Then the same with cores **8**, tasks 12. Screenshot the line charts (schedulability, miss ratio,
   response time, migrations, High miss ratio).
2. **7 · Export** → *Download everything (Excel)*.
3. For each example: **1 · Setup** → load → **3 · Run** (all three) → screenshots of **4 · Results** (Gantt) and
   **5 · Compare** (bars, *Run both modes*).

## Checklist for the report

| Item | From |
|---|---|
| Gantt charts of the 5 examples, plain and criticality-first | website 4 · Results |
| Comparison bars (misses, response, migrations, preemptions, utilization) | website 5 · Compare |
| Schedulability / miss ratio / response / migrations vs U/m at 4 and 8 cores | website 6 · Batch + Excel |
| High-criticality miss ratio, plain vs criticality-first | website 6 · Batch |
| Pi vs simulator: MATCH table | `all_hw_vs_simulator.csv` |
| Measured decision time and dispatch latency vs tick (400/10/2/1 ms) | `all_hw_vs_simulator.csv`, Pi results page |
| Tick overruns, wrong-CPU count, real-time priority = yes | same |
| Photo/video of the LED demo with the live Gantt chart | Part E |
| Board information | `board_info.txt` |

## How the program works (for the viva)

* One worker thread per core, pinned to that physical core (`pthread_setaffinity_np`), `SCHED_FIFO` priority 80; the
  dispatcher thread runs at priority 90. Memory is locked (`mlockall`).
* Every tick: the dispatcher makes the EDF decision (releases, deadline misses, choose jobs, assign cores, then
  aperiodic jobs on idle cores), the workers start together (barrier), each really computes for 80 % of the tick on its
  core, records `sched_getcpu()` and its start/end time, and waits at the second barrier.
* **Decision time** = time of the scheduler decision; **dispatch latency** = tick start → slowest core starts its job;
  **tick overrun** = the tick took longer than its length; **wrong CPU** = a job ran on a different physical CPU.
* Simulator vs Pi: same decisions (MATCH), but the simulator assumes zero overhead while the Pi measures it.

## AppCOE (MapuSoft) build, if required

Create a C project, add `rtos_hw.c`, target your Pi's ARM Linux toolchain (e.g. `arm-linux-gnueabihf-gcc` for 32-bit
or `aarch64-linux-gnu-gcc` for 64-bit Raspberry Pi OS), linker flags `-lpthread -lrt -lm`, build, copy the binary to
the Pi and run it with `sudo`. The code is plain C (gnu89) with no extra libraries.

## Verified before delivery

Compiled with `-std=gnu89 -Wall -Wextra` without warnings. Compared with the Python simulator on 480 algorithm runs
(random P/S/A task sets, plain and criticality-first EDF, continue and abort, 2 and 4 cores): **all identical**.
Tested on a Linux PC; on the Pi itself, run Part F and check MATCH.
