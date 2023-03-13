import logging
import time
import os

import pynvml


logger = logging.getLogger('monitor')
logger.setLevel(logging.INFO) 
formatter = logging.Formatter("%(asctime)s: %(message)s")
os.makedirs('./monitorLog',exist_ok=True)
fileHandler = logging.FileHandler(os.path.join('./monitorLog', "monitor.log"))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)
commandHandler = logging.StreamHandler()
commandHandler.setLevel(logging.INFO)
commandHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.addHandler(commandHandler)

while True:
    pynvml.nvmlInit()
    for i in range(8):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i) # i表示显卡标号
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        logger.info( f'cuda:{i} memory: {str( meminfo.used/1024**2)}' )
    time.sleep(1)