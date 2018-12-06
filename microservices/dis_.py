def myfunc(data):
    for value in data:
        yield value + 1

if __name__ == '__main__':
    import dis
    dis.dis(myfunc)