from copy import deepcopy


class Car:
    def __init__(self):
        self.__wheels = list()
        self.__engine = None
        self.__body = None

    def setBody(self, body):
        self.__body = body

    def attachWheels(self, wheel):
        self.__wheels.append(wheel)

    def setEngine(self, engine):
        self.__engine = engine

    def specification(self):
        print('Body: ', self.__body.shape)
        print('engine horsepower: ', self.__engine.horsepower)
        print('tire size: ', self.__wheels[0].size)

    def clone(self):
        return deepcopy(self)

class Wheel:
    size = None

class Engine:
    horsepower = None

class Body:
    shape = None

class Director:
    __builder = None

    def setBuilder(self, builder):
        self.__builder = builder

    def getCar(self):
        car = Car()

        body = self.__builder.getBody()
        car.setBody(body)

        engine = self.__builder.getEngine()
        car.setEngine(engine)

        i = 0
        while i < 4:
            wheel = self.__builder.getWheel()
            car.attachWheels(wheel)
            i += 1

        return car

class BuilderInterface:
    def getWheel(self): pass
    def getEngine(self): pass
    def getBody(self): pass

class JeepBuilder(BuilderInterface):
    def getWheel(self):
        wheel = Wheel()
        wheel.size = 22
        return wheel

    def getEngine(self):
        engine = Engine()
        engine.horsepower = 400
        return engine

    def getBody(self):
        body = Body()
        body.shape = 'SUV'
        return body

class NissanBuilder(BuilderInterface):
    def getWheel(self):
        wheel = Wheel()
        wheel.size = 16
        return wheel

    def getEngine(self):
        engine = Engine()
        engine.horsepower = 100
        return engine

    def getBody(self):
        body = Body()
        body.shape = 'hatchback'
        return body

if __name__ == '__main__':
    d = Director()
    d.setBuilder(JeepBuilder())
    print(d.getCar())
    d.getCar().specification()

    d.setBuilder(NissanBuilder())
    print(d.getCar())
    d.getCar().specification()