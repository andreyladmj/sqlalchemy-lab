class Singelton:
    __instance = None
    def __new__(cls, val=None):
        if Singelton.__instance is None:
            Singelton.__instance = object.__new__(cls)
        Singelton.__instance = val
        return Singelton.__instance