
import time
time1 = time.clock()
for i in range(1000):
    i += i**5
    for j in range(1000):
        i *= j
        for k in range(1000):
            i /= (k + 1)
print(time.clock() - time1)
