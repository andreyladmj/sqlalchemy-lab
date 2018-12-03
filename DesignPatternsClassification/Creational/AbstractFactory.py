class Shape2DInterface:
    def draw(self): pass

class Shape3DInterface:
    def build(self): pass

class Circle(Shape2DInterface):
    def draw(self):
        print('Circle.Draw (Shape2DInterface)')

class Square(Shape2DInterface):
    def draw(self):
        print('Square.Draw (Shape2DInterface)')

class Sphere(Shape3DInterface):
    def build(self):
        print('Sphere.Build (Shape3DInterface)')

class Cube(Shape3DInterface):
    def build(self):
        print('Cube.Build (Shape3DInterface)')
        
class ShapeFactoryInterface:
    def getShape(sides): pass
    

class Shape2DFactory(ShapeFactoryInterface):
    @staticmethod
    def getShape(sides):
        if sides == 1:
            return Circle()

        if sides == 4:
            return Square()

        assert 0, "Bad 2D shape creation"

class Shape3DFactory(ShapeFactoryInterface):
    @staticmethod
    def getShape(sides):
        """here, sides refers to hte number of faces"""
        if sides == 1:
            return Sphere()

        if sides == 6:
            return Cube()

        assert 0, "Bad 3d shape creating"


if __name__ == '__main__':
    s2 = Shape2DFactory()
    s3 = Shape3DFactory()
    print(s2.getShape(1))
    s2.getShape(1).draw()
    print(s3.getShape(1))
    s3.getShape(1).build()