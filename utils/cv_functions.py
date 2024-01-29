import numpy as np
import cv2


def world_to_pixel(world_coord, cam_pos, cam_rot, fov, resolution):
    # Convert fov to radians and calculate focal length
    fov_rad = np.radians(fov)
    f = 1 / np.tan(fov_rad / 2)  # Assuming near plane at 1

    # Aspect ratio
    aspect_ratio = resolution[0] / resolution[1]

    # Intrinsic Matrix K
    K = np.array([[f, 0, resolution[0] / 2],
                  [0, f * aspect_ratio, resolution[1] / 2],
                  [0, 0, 1]])

    # Extrinsic Matrix E
    t = -cam_rot @ cam_pos
    E = np.eye(4)
    E[:3, :3] = cam_rot
    E[:3, 3] = t

    # Convert world coordinate to homogenous coordinate
    world_homog = np.append(world_coord, 1)

    # Apply extrinsic and intrinsic matrices
    camera_coord = E @ world_homog
    image_coord = K @ camera_coord[:3]

    # Normalize by w and convert to pixel coordinates
    pixel_coord = (image_coord / image_coord[2])[:2]

    return int(pixel_coord[0]), int(pixel_coord[1])


def triangulate_point(pixel_coords, camera_matrices):
    # Convert pixel_coords and camera_matrices to correct format
    points_2d = np.array(pixel_coords).transpose(1, 0, 2)
    camera_matrices = np.array(camera_matrices)

    # Perform triangulation
    homog_3d_point = cv2.triangulatePoints(camera_matrices[0, :3], camera_matrices[1, :3],
                                           points_2d[0], points_2d[1])

    # Convert from homogeneous to 3D coordinates
    world_coord = homog_3d_point[:3] / homog_3d_point[3]

    return world_coord.ravel()

if __name__ == '__main__':
    # World to pixel: example usage
    cam_pos = np.array([x, y, z])  # Camera position
    cam_rot = np.array(R)  # Camera rotation matrix
    world_coord = np.array([x_world, y_world, z_world])  # World coordinate
    resolution = (w, h)  # Camera resolution

    pixel_coord = world_to_pixel(world_coord, cam_pos, cam_rot, fov, resolution)

    # triangulate_point: example usage
    pixel_coords = [[x1, y1], [x2, y2], [x3, y3]]  # Pixel coordinates in each camera
    camera_matrices = [P1, P2, P3]  # Projection matrices (intrinsic * extrinsic) for each camera

    world_coord = triangulate_point(pixel_coords, camera_matrices)