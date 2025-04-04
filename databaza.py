import hashlib
import geohash2
from matplotlib import pyplot as plt
from pymongo import MongoClient, GEOSPHERE
import pyproj
import laspy
import os
import numpy as np
from tqdm import tqdm
import math

class PointCloud:
    """
    Point Cloud class
    reads a .las or .laz file
    """
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

        return x, y, half_width + 1, half_height + 1

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
        """
        with laspy.open(self.file_path) as file:
            for chunk in file.chunk_iterator(chunk_size):
                yield chunk

    def convert_chunk_to_array(self, chunk):
        """
        Process a chunk of point cloud data
        """
        intensity = [chunk.intensity]
        x_coords = np.array(chunk.x, dtype=np.float64)
        y_coords = np.array(chunk.y, dtype=np.float64)
        z_coords = np.array(chunk.z, dtype=np.float64)
        points_array = np.vstack((x_coords, y_coords, z_coords), dtype=np.float64).transpose()

        if hasattr(chunk, 'red') and hasattr(chunk, 'green') and hasattr(chunk, 'blue'):
            colours = np.vstack((chunk.red, chunk.green, chunk.blue), dtype=np.float64).transpose()
            colours = colours / 65535.0
            points_array = np.hstack((points_array, colours))
        else:
            grayscale = np.vstack((intensity, intensity, intensity), dtype=np.float64).transpose()
            grayscale = grayscale / 65535.0
            points_array = np.hstack((points_array, grayscale))
        return points_array


class Database:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/",
                 db_name="testDB"):
        self.client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[db_name]
        self.files_collection = self.db["files"]

    def close(self):
        self.client.close()

    def insert_to_db(self, points_array, file_name, collection_name):
        collection = self.db[collection_name]
        # Create geospatial index
        collection.create_index([("location", GEOSPHERE)])
        points = []
        x, y, z = points_array[:, 0], points_array[:, 1], points_array[:, 2]
        r, g, b = points_array[:, 3], points_array[:, 4], points_array[:, 5]
        x = np.around(x, 2)
        y = np.around(y, 2)
        z = np.around(z, 2)

        for i in range(len(points_array)):
            lon, lat = convert_to_lat_lon(x[i], y[i])
            geohash = geohash2.encode(lat, lon, precision=5)  # Generate Geohash
            points.append({
                "file_name": file_name,
                "geohash": geohash,
                "location": {"type": "Point", "coordinates": [lon, lat]},
                "original_x": float(x[i]),
                "original_y": float(y[i]),
                "original_z": float(z[i]),
                "r": float(r[i]),
                "g": float(g[i]),
                "b": float(b[i])
            })
        collection.insert_many(points)

    def get_stored_files(self):
        return self.files_collection.distinct("file_name")

    def remove_file(self, file_name):
        """
        Remove a file from files collection and drop its points collection.
        """
        doc = self.files_collection.find_one({"file_name": file_name})
        if not doc:
            raise Exception("File not found in database.")
        collection_name = doc.get("collection_name")
        self.db.drop_collection(collection_name)
        self.files_collection.delete_one({"file_name": file_name})

    def file_exists(self, file_hash):
        """Check if a file with the given hash already exists in the database"""
        return self.files_collection.find_one({"file_hash": file_hash})

    def find_near_points(self, file_name, x, y, max_distance_meters):
        """
        Find LiDAR points within max_distance_meters of a point
        """
        doc = self.files_collection.find_one({"file_name": file_name})
        if not doc:
            raise Exception("File not found in database.")
        collection = self.db[doc.get("collection_name")]
        lon, lat = convert_to_lat_lon(x, y)
        query = {
            "location": {
                "$near": {
                    "$geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]  # GeoJSON format: [lon, lat]
                    },
                    "$maxDistance": max_distance_meters
                }
            }
        }
        results = list(collection.find(query))
        return np.array([
            [point["original_x"],
             point["original_y"],
             point["original_z"],
             point["r"],
             point["g"],
             point["b"]]
            for point in results
        ], dtype=np.float32)

    def find_middle_point(self, file_name):
        """
        Find the center of the point cloud
        """
        doc = self.files_collection.find_one({"file_name": file_name})
        if not doc:
            raise Exception("File not found in database.")
        collection = self.db[doc.get("collection_name")]
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avgX": {"$avg": "$original_x"},
                    "avgY": {"$avg": "$original_y"},
                    "avgZ": {"$avg": "$original_z"}
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "x": "$avgX",
                    "y": "$avgY",
                    "z": "$avgZ"
                }
            }
        ]
        result = list(collection.aggregate(pipeline))
        if result:
            return [result[0]['x'], result[0]['y'], result[0]['z']]
        else:
            return None

def visualize(x_coords, y_coords, colours, center):
    plt.figure(figsize=(8, 7))
    plt.scatter(x_coords, y_coords, c=colours, s=1, marker='o')
    plt.scatter(center[0], center[1], color='red', s=20)
    plt.title("Search Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# Function to get the hash of a file
def get_file_hash(file_path):
    """Generate a hash for the .laz file to track it uniquely"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def convert_to_lat_lon(x, y):
    transformer = pyproj.Transformer.from_crs(
        "EPSG:32634",  # UTM Zone 34N
        "EPSG:4326",   # WGS84 (lat/lon)
        always_xy=True
    )
    lon, lat = transformer.transform(x, y)
    return lon, lat

def save_pc_to_db(pc, db):
    file_hash = get_file_hash(pc.file_path)
    file_name = pc.file_path
    collection_name = "points_" + file_hash
    if not db.file_exists(file_hash):
        db.files_collection.insert_one({
            "file_hash": file_hash,
            "file_name": file_name,
            "collection_name": collection_name
        })
        for chunk in pc.read_points(50_000):
            chunk_arr = pc.convert_chunk_to_array(chunk)
            db.insert_to_db(chunk_arr, file_name, collection_name)

def save_pc_to_db_path(path, db):
    pc = PointCloud(path)
    file_hash = get_file_hash(pc.file_path)
    file_name = os.path.basename(path)
    collection_name = "points_" + file_hash
    if not db.file_exists(file_hash):
        db.files_collection.insert_one({
            "file_hash": file_hash,
            "file_name": file_name,
            "collection_name": collection_name
        })
        total_points = pc.point_count
        total_chunks = math.ceil(total_points / 50000)
        progress = tqdm(pc.read_points(50_000), total=total_chunks, desc="Uploading point cloud to database")
        # progress = tqdm(pc.read_points(50_000), desc="Saving chunks")
        for chunk in progress:
            chunk_arr = pc.convert_chunk_to_array(chunk)
            db.insert_to_db(chunk_arr, file_name, collection_name)

def example_query(file_name):
    print(f"Number of points in point cloud: {pc.point_count}")
    center = database.find_middle_point(file_name)
    result = database.find_near_points(file_name, center[0], center[1], 30)
    print(f"Found {len(result)} points")
    x_coords = result[:, 0]
    y_coords = result[:, 1]
    colours = result[:, 3:]
    visualize(x_coords, y_coords, colours, center)

if __name__ == "__main__":
    pc = PointCloud("data/Velky_Biel_32634_WGS84-TM34_sample.laz")
    database = Database()
    save_pc_to_db(pc, database)
    mid = database.find_middle_point()

    print(mid)

    # res = database.find_near_gps(17.360368637676316, 48.21594493483842, 40)

    # find points in 40m radius around middle point
    res = database.find_near_points(mid[0], mid[1], 40)
    print(len(res))

    #example_query()

    database.close()
