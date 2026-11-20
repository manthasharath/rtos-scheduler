rtos_hw: rtos_hw.c
	gcc -O2 -o rtos_hw rtos_hw.c -lpthread -lrt -lm
	@chmod +x collect_data.sh 2>/dev/null || true

clean:
	rm -f rtos_hw hw_*.csv tasks_entered.csv tasks_random.csv
