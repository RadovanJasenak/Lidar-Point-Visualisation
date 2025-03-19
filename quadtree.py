import numpy as np
import laspy
import os
import shutil
from matplotlib import pyplot as plt
from collections import namedtuple

class PointCloud:
    def __init__(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist")
        self.file_path = file_path
        with laspy.open(self.file_path) as file:
            header = file.header
            self.global_encoding = header.global_encoding
            self.generating_software = header.generating_software
            self.creation_date = header.creation_date
            self.version = header.version
            self.point_format = header.point_format
            self.point_count = header.point_count
            self.point_format_standard_dims = header.point_format.standard_dimensions
            self.point_format_stdandard_dim_names = header.point_format.standard_dimension_names
            self.point_format_dim_names = header.point_format.dimension_names
            self.mins = header.mins
            self.maxs = header.maxs
            self.scales = header.scales
            self.offsets = header.offsets
            self.vlrs = header.vlrs
            self.evlrs = header.evlrs

    def get_bounds(self):
        """
        Returns bounds of point cloud for use with quadtree
        X Y as middle point
        width height at half-length
        """
        x_min = self.mins[0]
        x_max = self.maxs[0]
        y_min = self.mins[1]
        y_max = self.maxs[1]

        half_width = (x_max - x_min) / 2.0
        half_height = (y_max - y_min) / 2.0
        x = x_min + half_width
        y = y_min + half_height

        return x, y, half_width+1, half_height+1

    def get_info(self):
        print(f"Generating Software: {self.generating_software}")
        print(f"Creation Date: {self.creation_date}")
        print(f"Global Encoding: {self.global_encoding}")
        print(f"Version: {self.version}")
        print(f"Point Format: {self.point_format}")
        print(f"Point Count: {self.point_count}")
        print(f"Point Format Standard Dimensions: {self.point_format_standard_dims}")
        print(f"Point Format Standard Dimension Names: {self.point_format_stdandard_dim_names}")
        print(f"Point Format Dimension Names: {self.point_format_dim_names}")
        print(f"Mins: {self.mins}")
        print(f"Maxs: {self.maxs}")
        print(f"Scales: {self.scales}")
        print(f"Offsets: {self.offsets}")
        print(f"VLRs: {self.vlrs}")
        print(f"EVLRs: {self.evlrs}")

    def read_points(self, chunk_size=50_000):
        """
        Reads a .laz file in chunks and yields each chunk
        Args:
            chunk_size (int): Number of points to read in each chunk.
        Yields:
            A chunk of the point cloud.
        """
        with laspy.open(self.file_path) as file:
            for chunk in file.chunk_iterator(chunk_size):
                yield chunk

    def convert_chunk_to_array(self, chunk):
        """
        Process a chunk of point cloud data
        """
        x_coords = [chunk.x]
        y_coords = [chunk.y]
        z_coords = [chunk.z]
        intensity = [chunk.intensity]
        points_array = np.vstack((x_coords, y_coords, z_coords), dtype=np.float32).transpose()

        if hasattr(chunk, 'red') and hasattr(chunk, 'green') and hasattr(chunk, 'blue'):
            colours = np.vstack((chunk.red, chunk.green, chunk.blue), dtype=np.float32).transpose()
            colours = colours / 65535.0
            points_array = np.hstack((points_array, colours))
        else:
            grayscale = np.vstack((intensity, intensity, intensity), dtype=np.float32).transpose()
            grayscale = grayscale / 65535.0
            points_array = np.hstack((points_array, grayscale))
        return points_array

class QuadTree:
    """
    quadtree class
    """
    def __init__(self, boundary, capacity):
        self.Boundary = namedtuple("Boundary", ["x", "y", "width", "height"])
        self.boundary = self.Boundary(*boundary)
        self.capacity = capacity
        self.points = []
        self.divided = False
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None
        self.number_of_points = 0
        self.redistribute_saved_points = False
        self.points_to_redistribute = []

    def get_range(self):
        x, y, width, height = self.boundary
        return f"{x - width} - {x + width}, {y - height} - {y + height}"

    def insert(self, point):
        if not self.includes(point):
            return False

        if self.divided:
            self.northeast.insert(point)
            self.northwest.insert(point)
            self.southeast.insert(point)
            self.southwest.insert(point)
            return True

        if self.number_of_points < self.capacity:
            self.points.append(point)
            self.number_of_points = self.number_of_points + 1
            return True
        else:
            if not self.divided:
                self.subdivide()
                for pt in self.points:
                    self.northeast.insert(pt)
                    self.northwest.insert(pt)
                    self.southeast.insert(pt)
                    self.southwest.insert(pt)
                self.points = []
                self.number_of_points = 0

            self.northeast.insert(point)
            self.northwest.insert(point)
            self.southeast.insert(point)
            self.southwest.insert(point)

    def insert_chunk(self, chunk):
        for point in chunk:
            self.insert(point)

    def subdivide(self):
        """
        divide a rectangle into 4 rectangles
        """
        x = self.boundary.x
        y = self.boundary.y
        width = self.boundary.width
        height = self.boundary.height
        ne = self.Boundary(x + width / 2.0, y + height / 2.0, width / 2.0, height / 2.0)
        self.northeast = QuadTree(ne, self.capacity)
        nw = self.Boundary(x - width / 2.0, y + height / 2.0, width / 2.0, height / 2.0)
        self.northwest = QuadTree(nw, self.capacity)
        se = self.Boundary(x + width / 2.0, y - height / 2.0, width / 2.0, height / 2.0)
        self.southeast = QuadTree(se, self.capacity)
        sw = self.Boundary(x - width / 2.0, y - height / 2.0, width / 2.0, height / 2.0)
        self.southwest = QuadTree(sw, self.capacity)

        self.divided = True
        self.redistribute_saved_points = True

    def save_chunk(self, path):
        os.makedirs(path, exist_ok=True)
        points_file = os.path.join(path, "points.txt")
        with open(points_file, "a") as f:
            for point in self.points:
                f.write(f"{point[0]},{point[1]},{point[2]},{point[3]},{point[4]},{point[5]}\n")
            self.points = []
        if self.divided:
            self.northeast.save_chunk(os.path.join(path, self.northeast.get_range()))
            self.northwest.save_chunk(os.path.join(path, self.northwest.get_range()))
            self.southeast.save_chunk(os.path.join(path, self.southeast.get_range()))
            self.southwest.save_chunk(os.path.join(path, self.southwest.get_range()))

    def redistribute_points(self, path):
        if self.redistribute_saved_points:
            points_file = os.path.join(path, "points.txt")
            with open(points_file, "r") as f:
                for line in f:
                    point = list(map(float, line.strip().split(",")))
                    self.points_to_redistribute.append(point)
            self.insert_chunk(self.points_to_redistribute)
            self.save_chunk(path)
            self.points_to_redistribute = []
            with open(points_file, "w") as f:
                pass
        self.redistribute_saved_points = False
        if self.divided:
            self.northeast.redistribute_points(os.path.join(path, self.northeast.get_range()))
            self.northwest.redistribute_points(os.path.join(path, self.northwest.get_range()))
            self.southeast.redistribute_points(os.path.join(path, self.southeast.get_range()))
            self.southwest.redistribute_points(os.path.join(path, self.southwest.get_range()))

    def includes(self, point):
        x, y, width, height = self.boundary
        return x - width <= point[0] <= x + width and y - height <= point[1] <= y + height

    def __repr__(self):
        return f"Quadtree(boundary={self.boundary}, divided={self.divided})"

def visualize(point_cloud):
    # doesn't work
    # point cloud is now stored in files, not kept in memory
    x_coords = point_cloud.points[:, 0]
    y_coords = point_cloud.points[:, 1]
    colours = point_cloud.points[:, 3:]
    plt.figure(figsize=(8, 7))
    plt.scatter(x_coords, y_coords, c=colours, s=1, marker='o')
    plt.title("Point Cloud Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def create_directory(path):
    # create a directory to save quadtree
    # make it empty in case a previous quadtree was saved here
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

pc = PointCloud("data/Velky_Biel_32634_WGS84-TM34_sample.laz")
point_count = pc.point_count
print(f"Nuber of points: {point_count}")

create_directory("Quadtree")
qt = QuadTree(pc.get_bounds(), 90_000)

chunk_size = 50_000
total_number_of_chunks = point_count//chunk_size
chunk_number = 0

for chunk in pc.read_points(chunk_size):
    print(f"Processing chunk {chunk_number}/{total_number_of_chunks}")
    chunk_arr = pc.convert_chunk_to_array(chunk)
    qt.insert_chunk(chunk_arr)
    qt.save_chunk(f"QuadTree/{qt.get_range()}")
    qt.redistribute_points(f"QuadTree/{qt.get_range()}")
    chunk_number += 1
