import asyncio
import contextlib
import logging

async def coro1():
    while True:
        for i in range(100):
            await asyncio.sleep(0.1)
        print('Coro 1')

async def coro2():
    for i in range(25):
        await asyncio.sleep(.5)
        print('coro 2')

    #loop = asyncio.get_event_loop()
    #loop.stop()

logging.getLogger('asyncio').setLevel(logging.DEBUG)

asyncio.ensure_future(coro1())
f = asyncio.ensure_future(coro2())

#loop = asyncio.get_event_loop()
#loop.run_forever()
#loop.close()

with contextlib.closing(asyncio.get_event_loop()) as loop:
    loop.run_until_complete(f)
