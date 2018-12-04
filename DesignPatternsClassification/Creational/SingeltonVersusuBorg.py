class Borg:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

if __name__ == '__main__':
    t = Borg()
    t.name = 'tetete'
    g = Borg()
    print(g.name)
    print(t == g)