import time

count = 0

while True:
    print(f"Loop iteration: {count}")
    count += 1
    # Pause the script for 2 seconds
    time.sleep(2)

print("Loop finished.")