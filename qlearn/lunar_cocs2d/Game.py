from threading import Thread
from time import sleep

from cocos import director
from cocos import scene
#
# from Global import CurrentKeyboard, set_main_layer
from .MainSceneLayer import MainSceneLayer
# from objects.Tank import Tank


def main():
    createInterface()

def createInterface():
    director.director.init(width=3000, height=960, autoscale=True, resizable=True)


    MainLayer = MainSceneLayer()
    # set_main_layer(MainLayer)
    main_scene = scene.Scene(MainLayer)
    main_scene.schedule(MainLayer.buttonsHandler)
    MainLayer.register_event_type('on_clicked')

    MainLayer.register_event_type('add_tank')
    MainLayer.register_event_type('add_animation')
    MainLayer.register_event_type('add_bullet')
    MainLayer.register_event_type('remove_animation')


    MainLayer.dispatch_event('on_clicked', '12314124')
    director.director.on_resize = MainLayer.resize
    director.director.window.push_handlers(CurrentKeyboard)
    director.director.run(main_scene)



if __name__ == '__main__':
    main()
