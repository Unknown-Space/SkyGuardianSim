import numpy as np
import cv2
from time import perf_counter

class Camera:
    def __init__(self, fov, resolution):
        self.fov = fov
        self.resolution = resolution
        # Convert fov to radians and calculate focal length
        fov_rad = np.radians(fov)
        f = 1 / np.tan(fov_rad / 2)  # Assuming near plane at 1

        # Aspect ratio
        aspect_ratio = resolution[0] / resolution[1]

        # Intrinsic Matrix K
        self.K = np.array([[f, 0, resolution[0] / 2],
                           [0, f * aspect_ratio, resolution[1] / 2],
                           [0, 0, 1]])
        self.position = np.zeros(3)
        self.rotation = np.eye(3)
        self.E = np.eye(4)

    def reset(self, position, rotation):
        self.position = position
        self.rotation = rotation
        # Extrinsic Matrix E
        t = -rotation @ position
        self.E = np.eye(4)
        self.E[:3, :3] = rotation
        self.E[:3, 3] = t

    def world_to_pixel(self, world_coord):
        # Convert world coordinate to homogenous coordinate
        world_homog = np.append(world_coord, 1)

        # Apply extrinsic and intrinsic matrices
        camera_coord = self.E @ world_homog
        image_coord = self.K @ camera_coord[:3]

        # Normalize by w and convert to pixel coordinates
        pixel_coord = (image_coord / image_coord[2])[:2]

        return int(pixel_coord[0]), int(pixel_coord[1])

    def render(self, drones, targets):
        # render the drones and targets in the simulation
        # drones: list of drones
        # targets: list of targets
        # return: rendered image
        image = np.zeros(self.resolution)
        for drone in drones:
            pixel_coord = self.world_to_pixel(drone.position)
            if 0 <= pixel_coord[0] < self.resolution[0] and 0 <= pixel_coord[1] < self.resolution[1]:
                dist = np.linalg.norm(drone.position - self.position)
                drone.calc_rangefinder_position()
                #TODO: draw rangefinder laser
                rangefinder_pixel_coord = self.world_to_pixel(drone.rangefinder_position)
                rangefinder_intensity = np.clip(drone.rangefinder_intensity * (1000 / dist)**2, 0, 255)


                # calculate the radius of the drone in the image by the distance to the drone and the fov
                radius = 0.1 * dist * np.tan(np.radians(drone.radius) / 2) / np.tan(np.radians(self.fov) / 2)
                radius = max(radius, 1) # drone size cannot be smaller than 1 pixel
                # the intensity of the drone is proportional to the distance squared because of the inverse square law
                # yet, given the intensity from 1000m away is drone.intensity we can calculate the intensity at any distance
                intensity = np.clip(drone.intensity * (1000 / dist)**2, 0, 255)
                # draw the drone
                cv2.circle(image, pixel_coord, int(radius), int(intensity), -1)
        return image

class Drone:
    def __init__(self, radius, drag_coefficient, intensity=200, mass=1, max_actuator=1, dim=2):
        self.mass = mass
        self.dim = dim
        self.friction = drag_coefficient
        self.max_actuator = max_actuator
        self.max_altitude_rate = 1
        self.radius = radius
        self.intensity = intensity # intensity is the brightness of the drone from 1000m away
        self.prev_time = 0
        self.position = np.zeros(dim)
        self.velocity = np.zeros(dim)
        self.acceleration = np.zeros(dim)
        self.actuator = np.zeros(dim)
        self.max_angle = 25

        # rangefinder
        self.rangefinder_intensity = 2 # intensity of the rangefinder from 1000m away
        self.rangefinder_fov = 10 # rangefinder field of view in degrees
        self.rangefinder_position = np.zeros(dim)
        self.rangefinder_radius = 0.0

        # takeoff
        self.is_takeoff = False
        self.takeoff_altitude = 10
        self.takeoff_ema = 0.5
        self.takeoff_deadband_percentage = 0.1
        self.takeoff_start_altitude = 0
        self.takeoff_prev_altitude = 0

    def reset(self, position):
        self.position = position
        self.velocity = np.zeros(self.dim)
        self.acceleration = np.zeros(self.dim)
        self.prev_time = perf_counter()
        self.is_takeoff = False

    def update(self):
        current_time = perf_counter()
        dt = current_time - self.prev_time
        self.prev_time = 1.0 * current_time
        if self.is_takeoff:
            self.takeoff()
        else:
            self.acceleration[0] = self.actuator[0] / self.mass
            self.acceleration[1] = self.actuator[1] / self.mass
            # self.acceleration[2] = -9.81 + self.actuator[2] / self.mass
            self.acceleration[2] = 0
            self.velocity = (1-self.friction) * self.velocity + self.acceleration * dt
            self.velocity[2] = np.clip(self.actuator[2], -self.max_actuator, self.max_actuator) * self.max_altitude_rate
            self.position += self.velocity * self.dt
        return self.position

    def step(self, action):
        self.actuator[0] = np.clip(action[0], -self.max_actuator, self.max_actuator)
        self.actuator[1] = np.clip(action[1], -self.max_actuator, self.max_actuator)

    def takeoff(self):
        if not self.is_takeoff:
            self.is_takeoff = True
            self.takeoff_start_altitude = 1.0 * self.position[2]
        if self.position[2] < self.takeoff_start_altitude + self.takeoff_altitude - self.takeoff_deadband_percentage * self.takeoff_altitude:
            self.position[2] = self.takeoff_ema * self.position[2] + (1-self.takeoff_ema) * (self.takeoff_start_altitude + self.takeoff_altitude)
        else:
            self.is_takeoff = False

    def calc_rangefinder_position(self):
        tilt_angle = np.deg2rad(self.max_angle * self.actuator / self.max_actuator)
        # the range finder is sending a laser from the minus z direction of the drone to the ground.
        # the laser is tilted by the tilt_angle
        # we calculate the position of the laser on the ground
        # the laser is a line from the drone position to the ground
        # given the drones position, the tilt angle in the roll and pitch axes and the altitude of the drone
        # the equation of using these parameters is: x = x0 + z * tan(tilt_angle)
        # where x0 is the x position of the drone, z is the altitude of the drone and x is the x position of the laser on the ground
        # we do the same for the y axis
        # the equation is: y = y0 + z * tan(tilt_angle)
        # where y0 is the y position of the drone, z is the altitude of the drone and y is the y position of the laser on the ground
        self.rangefinder_position[0] = self.position[0] + self.position[2] * np.tan(tilt_angle[0])
        self.rangefinder_position[1] = self.position[1] + self.position[2] * np.tan(tilt_angle[1])
        # rangefinder_position is where the laser hits the ground
        # the rangefinder radius size depends on the fov and the distance of the drone from where the laser hits the ground
        # it is like a isosceles triangle where the rangefinder radius is the base and the fov is the angle between the base and the sides
        # the distance of the drone from where the laser hits the ground is the height of the triangle
        # the base of the triangle is dist of drone of where the laser hits the ground * tan(fov/2)
        # the rangefinder radius is the base of the triangle
        self.rangefinder_radius = np.linalg.norm(self.position - self.rangefinder_position) * np.tan(
            np.deg2rad(self.rangefinder_fov / 2))

