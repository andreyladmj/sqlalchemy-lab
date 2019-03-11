a = 16
v0 = 20
s = 0

def get_speed(time=1):
    return v0 + a * time


for i in range(20):
    s += get_speed()
    print(i, a, s)
    a -= 4