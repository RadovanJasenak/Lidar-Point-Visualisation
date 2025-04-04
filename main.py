import os
import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GLU import gluUnProject
import numpy as np
import pyrr
import time
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import queue
import laspy
from databaza import Database, save_pc_to_db_path

last_cam_pos = None
latest_view = None
latest_projection = None
latest_points = None
file_name = None
camera_pos = None
camera_front = None
camera_up = None
cpu_points = None
camera_speed = 100.0

picked_points = []  # store clicked points to save them later as output
object_queue = queue.Queue()

database = Database()
RADIUS = 500
CHECK_DISTANCE = 300

# Ray line data
ray_line_active = False
ray_line_vao = None
ray_line_vbo = None
line_shader_program = None

picked_line_vao = None
picked_line_vbo = None

#Vertex shader applies point transformations
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 vertexColor;
out vec3 fragColor;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
    gl_PointSize = 3.0;
    fragColor = vertexColor;
}
"""

# Fragment shader outputs vertex color.
fragment_shader_source = """
#version 400 core
in vec3 fragColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(fragColor, 1.0);
}
"""

line_vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main() {
    gl_Position = projection * view * model * vec4(position, 1.0);
}
"""

line_fragment_shader_source = """
#version 330 core
out vec4 FragColor;
void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0); 
}
"""

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        error = glGetShaderInfoLog(shader).decode()
        glDeleteShader(shader)
        raise RuntimeError(f"Shader compilation error: {error}")
    return shader

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_shader(vertex_source, GL_VERTEX_SHADER)
    fragment_shader = compile_shader(fragment_source, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)
    if not glGetProgramiv(program, GL_LINK_STATUS):
        error = glGetProgramInfoLog(program).decode()
        glDeleteProgram(program)
        raise RuntimeError(f"Shader linking error: {error}")
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return program

def process_keyboard(window, camera_pos, camera_front, camera_up, speed=1.0, delta_time=0.016):
    # delta time (frame time) for smoother movement
    movement = speed * delta_time
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos += movement * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos -= movement * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        left = np.cross(camera_up, camera_front)
        left /= np.linalg.norm(left)
        camera_pos += movement * left
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        right = np.cross(camera_front, camera_up)
        right /= np.linalg.norm(right)
        camera_pos += movement * right
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        camera_pos += movement * camera_up
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        camera_pos -= movement * camera_up
    return camera_pos

def load_new_points():
    # perform a database query for new points
    new_result = database.find_near_points(file_name, camera_pos[0], camera_pos[1], RADIUS)
    new_points = np.ascontiguousarray(new_result, dtype=np.float32)
    cpu_points = np.ascontiguousarray(new_result, dtype=np.float64)
    object_queue.put((new_points, cpu_points))

def compute_picking_ray(mouse_x, mouse_y, view, projection):
    # gluUnProject to get ray in double precision
    viewport = glGetIntegerv(GL_VIEWPORT)  # [x, y, width, height]
    # convert coordinates to OpenGL system - 0, 0 in left bottom corner
    win_x = mouse_x
    win_y = viewport[3] - mouse_y - 1
    # unproject near/far planes
    near_point = gluUnProject(win_x, win_y, 0.0, view, projection, viewport)
    far_point = gluUnProject(win_x, win_y, 1.0, view, projection, viewport)
    ray_origin = np.array(near_point, dtype=np.float64)
    ray_dir = np.array(far_point, dtype=np.float64) - ray_origin
    ray_dir /= np.linalg.norm(ray_dir)
    return ray_origin, ray_dir

def pick_point_along_ray(ray_origin, ray_dir, points_cpu, threshold=2.0):
    """
    Returns the nearest point in points_cpu threshold distance of the ray
    """
    points_xyz = points_cpu[:, :3].astype(np.float64)
    diff = points_xyz - ray_origin
    # t = distance along ray_dir to each point
    t = np.dot(diff, ray_dir)
    # only consider points in front of the ray origin
    valid = t > 0
    if not np.any(valid):
        return None
    proj_points = ray_origin + np.outer(t, ray_dir)
    dists = np.linalg.norm(points_xyz - proj_points, axis=1)
    dists[~valid] = np.inf
    min_idx = np.argmin(dists)
    if dists[min_idx] < threshold:
        return points_xyz[min_idx]
    return None

def save_selected_points(default_color=(65530, 65530, 65530)):
    output_folder = "Output"
    os.makedirs(output_folder, exist_ok=True)
    header = laspy.LasHeader(point_format=2, version="1.2")
    header.scale = [0.01, 0.01, 0.01]
    header.offset = [0.0, 0.0, 0.0]

    points = np.array(picked_points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    n_points = points.shape[0]
    r = np.full(n_points, default_color[0], dtype=np.uint16)
    g = np.full(n_points, default_color[1], dtype=np.uint16)
    b = np.full(n_points, default_color[2], dtype=np.uint16)

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z
    las.red   = r
    las.green = g
    las.blue  = b

    las.update_header()
    las_file_path = os.path.join(output_folder, "selected_points.las")
    las.write(las_file_path)

    txt_file_path = os.path.join(output_folder, "selected_points.txt")
    np.savetxt(txt_file_path, points, delimiter=", ", header="X Y Z", comments="")

    obj_file_path = os.path.join(output_folder, "selected_points.obj")
    with open(obj_file_path, "w") as f:
        f.write("# OBJ file containing vertex data only\n")
        for x, y, z in points:
            f.write(f"v {x} {y} {z}\n")
        f.write("p")
        for i in range(1, len(points) + 1):
            f.write(f" {i}")
        f.write("\n")

    print("Point selection saved")

def toggle_pick_point(new_point, threshold=1e-3):
    """check if picked point is already in picked_points array"""
    global picked_points
    for i, pt in enumerate(picked_points):
        if np.linalg.norm(np.array(pt) - np.array(new_point)) < threshold:
            picked_points.pop(i)
            return False
    picked_points.append(new_point)
    return True

def mouse_button_callback(window, button, action, mods):
    global camera_pos, camera_front, camera_up
    global latest_view, latest_projection, latest_points
    global picked_points, ray_line_active, ray_line_vbo

    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:

        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        mouse_x *= dpi_scale_x
        mouse_y *= dpi_scale_y

        ray_origin, ray_dir = compute_picking_ray(mouse_x, mouse_y, latest_view, latest_projection)
        # print("Ray Origin:", ray_origin)
        # print("Ray Direction:", ray_dir)

        # use cpu-side points array of float64
        picked = pick_point_along_ray(ray_origin, ray_dir, cpu_points, threshold=0.5)
        if picked is not None:
            toggle_pick_point(picked, threshold=0.5)
            update_picked_line_data()
            # if toggle_pick_point(picked, threshold=0.5):
            #     print("Picked point:", picked)
            # else:
            #     print("Unpicked point:", picked)

        far_distance = 500
        ray_end = ray_origin + ray_dir * far_distance
        # convert to float32 for GPU
        line_vertices = np.array([ray_origin, ray_end], dtype=np.float32).flatten()
        glBindBuffer(GL_ARRAY_BUFFER, ray_line_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, line_vertices.nbytes, line_vertices)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        ray_line_active = True

def key_callback(window, key, scancode, action, mods):
    global camera_speed
    if key == glfw.KEY_ENTER and action == glfw.PRESS:
        save_selected_points()
    if key == glfw.KEY_BACKSPACE and action == glfw.PRESS:
        if picked_points:
            picked_points.pop()
            update_picked_line_data()
    if key == glfw.KEY_R and action == glfw.PRESS:
        camera_speed += 10
    if key == glfw.KEY_F and action == glfw.PRESS:
        camera_speed = max(10.0, camera_speed - 10.0)

def update_picked_line_data():
    global picked_points, picked_line_vbo
    if not picked_points:
        return
    line_data = []
    for pt in picked_points:
        line_data.extend([pt[0], pt[1], pt[2], 1.0, 0.0, 0.0])
    line_data = np.array(line_data, dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, picked_line_vbo)
    glBufferData(GL_ARRAY_BUFFER, line_data.nbytes, line_data, GL_DYNAMIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)

def init_picked_line_resources():
    global picked_line_vao, picked_line_vbo
    picked_line_vao = glGenVertexArrays(1)
    picked_line_vbo = glGenBuffers(1)
    glBindVertexArray(picked_line_vao)
    glBindBuffer(GL_ARRAY_BUFFER, picked_line_vbo)
    glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
    # Attribute 0: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # Attribute 1: color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

def main(File_name):
    # Initialize GLFW
    global camera_pos, camera_front, camera_up
    global latest_view, latest_projection, latest_points, last_cam_pos
    global ray_line_vao, ray_line_vbo, line_shader_program, ray_line_active
    global cpu_points, file_name, dpi_scale_x, dpi_scale_y
    global camera_speed
    file_name = File_name

    center_of_pc = database.find_middle_point(file_name)
    result = database.find_near_points(file_name, center_of_pc[0], center_of_pc[1], RADIUS)

    if not glfw.init():
        raise Exception("GLFW could not be initialized!")

    work_width = 1280
    work_height = 720

    # glfw.window_hint(glfw.COCOA_RETINA_FRAMEBUFFER, glfw.FALSE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)

    window = glfw.create_window(work_width, work_height, "LiDAR Points Visualization", None, None)
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed!")
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)

    win_width, win_height = glfw.get_window_size(window)
    fb_width, fb_height = glfw.get_framebuffer_size(window)
    dpi_scale_x = fb_width / win_width
    dpi_scale_y = fb_height / win_height
    #glViewport(0, 0, fb_width, fb_height)
    print(f"X and Y dpi scales: {dpi_scale_x}, {dpi_scale_y}")
    print(f"window size W: {win_width}, H: {win_height}")
    print(f"Framebuffer size W: {fb_width}, H: {fb_height}")

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_key_callback(window, key_callback)

    pointcloud_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    line_shader_program = create_shader_program(line_vertex_shader_source, line_fragment_shader_source)

    # points array
    points = np.ascontiguousarray(result, dtype=np.float32)
    cpu_points = np.ascontiguousarray(result, dtype=np.float64)

    # VAO and VBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)

    # attribute 0: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # attribute 1: color
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # create VAO/VBO for the ray line
    ray_line_vao = glGenVertexArrays(1)
    ray_line_vbo = glGenBuffers(1)
    glBindVertexArray(ray_line_vao)
    glBindBuffer(GL_ARRAY_BUFFER, ray_line_vbo)
    glBufferData(GL_ARRAY_BUFFER, 2 * 3 * 4, None, GL_DYNAMIC_DRAW)  # space for 2 points
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    init_picked_line_resources()

    projection = pyrr.matrix44.create_perspective_projection_matrix(
        fovy=45.0, aspect=work_width / work_height, near=0.1, far=1000.0, dtype=np.float64
    )

    # initialize camera
    camera_pos = np.array(center_of_pc + np.array((-10., 0., 10.), dtype=np.float64), dtype=np.float64)
    camera_yaw = 0.0
    camera_pitch = 0.0
    # initial camera_front from yaw and pitch.
    front = np.array([
        np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
        np.sin(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
        np.sin(np.radians(camera_pitch))
    ], dtype=np.float32)
    camera_front = front / np.linalg.norm(front)
    camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    model = pyrr.matrix44.create_identity(dtype=np.float32)

    model_loc_pc = glGetUniformLocation(pointcloud_program, "model")
    view_loc_pc = glGetUniformLocation(pointcloud_program, "view")
    proj_loc_pc = glGetUniformLocation(pointcloud_program, "projection")

    # locations for ray
    model_loc_line = glGetUniformLocation(line_shader_program, "model")
    view_loc_line = glGetUniformLocation(line_shader_program, "view")
    proj_loc_line = glGetUniformLocation(line_shader_program, "projection")

    glEnable(GL_DEPTH_TEST)
    glPointSize(5.0)

    # Variables for FPS calculation.
    last_time = time.time()
    frame_count = 0

    # Variables for mouse rotation.
    last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
    mouse_sensitivity = 0.1
    last_frame_time = 0

    last_cam_pos = camera_pos.copy()
    latest_points = points
    latest_projection = projection

    background_thread = None

    # render loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        # --- Keyboard movement ---
        camera_pos = process_keyboard(window, camera_pos, camera_front, camera_up, speed=camera_speed, delta_time=delta_time)

        # --- Mouse rotation ---
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            current_mouse_x, current_mouse_y = glfw.get_cursor_pos(window)
            offset_x = current_mouse_x - last_mouse_x
            offset_y = last_mouse_y - current_mouse_y
            last_mouse_x, last_mouse_y = current_mouse_x, current_mouse_y

            offset_x *= mouse_sensitivity
            offset_y *= mouse_sensitivity

            camera_yaw -= offset_x
            camera_pitch += offset_y

            # prevent flipping
            if camera_pitch > 89.0:
                camera_pitch = 89.0
            if camera_pitch < -89.0:
                camera_pitch = -89.0

            front = np.array([
                np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
                np.sin(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
                np.sin(np.radians(camera_pitch))
            ], dtype=np.float32)
            camera_front = front / np.linalg.norm(front)
        else:
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)

        distance = np.linalg.norm(camera_pos[:2] - last_cam_pos[:2])
        if distance > CHECK_DISTANCE:
            if background_thread is None or not background_thread.is_alive():
                last_cam_pos = camera_pos.copy()
                background_thread = threading.Thread(target=load_new_points, daemon=True)
                background_thread.start()
                #print("thread started")

        try:
            new_pts, cpu_pts = object_queue.get_nowait()
            #new_pts, cpu_pts = object_queue.get(timeout=0.01)
            points = latest_points = new_pts
            cpu_points = cpu_pts
            glBindBuffer(GL_ARRAY_BUFFER, VBO)
            glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        except queue.Empty:
            pass

        view = pyrr.matrix44.create_look_at(
            eye=camera_pos,
            target=camera_pos + camera_front,
            up=camera_up,
            dtype=np.float64
        )

        latest_view = view
        latest_projection = projection

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw point cloud
        glUseProgram(pointcloud_program)
        glUniformMatrix4fv(model_loc_pc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc_pc, 1, GL_FALSE, view.astype(np.float32))
        glUniformMatrix4fv(proj_loc_pc, 1, GL_FALSE, projection.astype(np.float32))
        glBindVertexArray(VAO)
        glDrawArrays(GL_POINTS, 0, len(points))
        glBindVertexArray(0)

        # draw ray line
        # if ray_line_active:
        #     glUseProgram(line_shader_program)
        #     glUniformMatrix4fv(model_loc_line, 1, GL_FALSE, model)
        #     glUniformMatrix4fv(view_loc_line, 1, GL_FALSE, view.astype(np.float32))
        #     glUniformMatrix4fv(proj_loc_line, 1, GL_FALSE, projection.astype(np.float32))
        #     glBindVertexArray(ray_line_vao)
        #     glDrawArrays(GL_LINES, 0, 2)
        #     glBindVertexArray(0)

        # draw picked points
        if picked_points:
            highlight_data = []
            for pt in picked_points:
                highlight_data.extend([pt[0], pt[1], pt[2], 1.0, 0.0, 0.0])
            highlight_data = np.array(highlight_data, dtype=np.float32)
            highlight_VAO = glGenVertexArrays(1)
            highlight_VBO = glGenBuffers(1)
            glBindVertexArray(highlight_VAO)
            glBindBuffer(GL_ARRAY_BUFFER, highlight_VBO)
            glBufferData(GL_ARRAY_BUFFER, highlight_data.nbytes, highlight_data, GL_DYNAMIC_DRAW)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)

            glBindVertexArray(highlight_VAO)
            glPointSize(20.0)
            glDrawArrays(GL_POINTS, 0, len(picked_points))
            glPointSize(5.0)
            glBindVertexArray(0)
            # Clean up
            glDeleteBuffers(1, [highlight_VBO])
            glDeleteVertexArrays(1, [highlight_VAO])

        if len(picked_points) >= 2:
            glUseProgram(line_shader_program)
            glUniformMatrix4fv(model_loc_line, 1, GL_FALSE, model)
            glUniformMatrix4fv(view_loc_line, 1, GL_FALSE, view.astype(np.float32))
            glUniformMatrix4fv(proj_loc_line, 1, GL_FALSE, projection.astype(np.float32))
            glBindVertexArray(picked_line_vao)
            glDrawArrays(GL_LINE_LOOP, 0, len(picked_points))
            glBindVertexArray(0)

        glfw.swap_buffers(window)

        # FPS calculation.
        frame_count += 1
        current_time = time.time()
        elapsed = current_time - last_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            # print(f"FPS: {fps:.2f}")
            frame_count = 0
            last_time = current_time

    glDeleteVertexArrays(1, (VAO,))
    glDeleteBuffers(1, (VBO,))
    glDeleteVertexArrays(1, (ray_line_vao,))
    glDeleteBuffers(1, (ray_line_vbo,))
    glfw.terminate()

if __name__ == "__main__":

    def refresh_file_list():
        file_list.delete(0, tk.END)
        files = database.get_stored_files()
        for file in files:
            file_list.insert(tk.END, file)

    def visualize_selected_file():
        selected_index = file_list.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Select a file from the list.")
            return
        file_name = file_list.get(selected_index[0])
        main(file_name)

    def upload_selected_file():
        save_pc_to_db_path(filedialog.askopenfilename(), database)
        refresh_file_list()

    def remove_selected_file():
        selected_index = file_list.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Select a file to remove.")
            return
        file_name = file_list.get(selected_index[0])
        database.remove_file(file_name)
        refresh_file_list()

    # TK setup
    root = tk.Tk()
    root.title("LiDAR File Manager")
    root.geometry("500x400")

    # upload file button
    upload_button = tk.Button(root, text="Upload file", command=upload_selected_file)
    upload_button.pack(pady=10)

    # label
    status_label = tk.Label(root, text="Select a file")
    status_label.pack()

    # listbox
    file_list = tk.Listbox(root, width=50, height=10)
    file_list.pack(pady=10)
    refresh_file_list()

    # create a frame to hold the remove and load buttons
    button_frame = tk.Frame(root)
    button_frame.pack(pady=5)

    # remove button
    remove_button = tk.Button(button_frame, text="Remove", command=remove_selected_file)
    remove_button.pack(side=tk.LEFT, padx=5)

    # load selected file
    load_button = tk.Button(button_frame, text="Load & Visualize", command=visualize_selected_file)
    load_button.pack(side=tk.LEFT, pady=5)

    root.mainloop()
    database.close()
