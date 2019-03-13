from itertools import chain
from threading import Timer
from time import time

import cocos
import pyglet
from cocos.actions import MoveBy, FadeOut
from cocos.batch import BatchNode
from pyglet.window import key
import cocos.collision_model as cm

# from Global import CurrentKeyboard, CollisionManager
# from handlers.BulletMovingHandlers import BulletMovingHandlers
# from handlers.UserTankMovingHandlers import UserTankMovingHandlers
# # from helper.CalcCoreHelper import CalcCoreHelper
# from layers.TankNodeLayer import TankNodeLayer, ObjectsNodeLayer
# from objects.Explosion import Explosion


class MainSceneLayer(cocos.layer.ScrollableLayer, pyglet.event.EventDispatcher):
    is_event_handler = True

    def __init__(self):
        super(MainSceneLayer, self).__init__()
        self.schedule(self.update)
        self.backgroundLayer = ObjectsNodeLayer()
        self.tanksLayer = TankNodeLayer()
        self.objectsLayer = ObjectsNodeLayer()
        self.bulletsLayer = ObjectsNodeLayer()
        self.additionalLayer = ObjectsNodeLayer()
        self.globalPanel = cocos.layer.Layer()
        self.add(self.backgroundLayer, z=1)
        self.add(self.objectsLayer, z=2)
        self.add(self.tanksLayer, z=3)
        self.add(self.bulletsLayer, z=4)
        self.add(self.additionalLayer, z=5)
        self.add(self.globalPanel, z=5)

        self.backgroundLayer.add(cocos.sprite.Sprite('assets/background.png'))
        # self.calc_core = CalcCoreHelper(self.tanksLayer, self.objectsLayer, self.bulletsLayer)

    s = 0
    def update(self, dt):
        self.checkCollisions()




    def checkCollisions(self):
        for bullet in self.bulletsLayer.get_children():
            bullet.cshape = cm.AARectShape(bullet.position, 2, 2)
            collisions = CollisionManager.objs_colliding(bullet)

            if collisions:
                items = chain(self.objectsLayer.get_children(), self.tanksLayer.get_children())

                for item in items:
                    if item in collisions and item != bullet.fired_tank:
                        explosion = Explosion(bullet)
                        explosion.checkDamageCollisions()
                        bullet.destroy()
                        # self.bulletsLayer.remove(bullet)
                        # CollisionManager.remove_tricky(bullet)

            if bullet.exceededTheLengthLimit():
                explosion = Explosion(bullet)
                explosion.checkDamageCollisions()
                bullet.destroy()
            # if Collisions.checkWithWalls(bullet) \
            #         or Collisions.checkWithObjects(bullet, bullet.parent_id) \
            #         or bullet.exceededTheLengthLimit():
            #     bullet.destroy()

    def on_clicked(self, clicks):
        print('on_clicked', clicks)


    def resize(self, width, height):
        self.viewPoint = (width // 2, height // 2)
        self.currentWidth = width
        self.currentHeight = height

