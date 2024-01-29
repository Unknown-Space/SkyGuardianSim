import numpy as np
import cv2

def render(drones, targets, camera):
    # render the drones and targets in the simulation
    # drones: list of drones
    # targets: list of targets
    # return: rendered image
    for drone in drones:
        drone.position