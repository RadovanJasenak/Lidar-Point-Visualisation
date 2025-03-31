import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
import numpy as np
import pyrr
import time
from databaza import PointCloud, Database, save_pc_to_db_path

#latest_camera_pos = None
last_cam_pos = None
latest_view = None
latest_projection = None
latest_points = None
window_width = None
window_height = None
picked_points = []  # store clicked points to save them later as output

# pc = PointCloud("data/Velky_Biel_32634_WGS84-TM34_sample.laz")
#
database = Database()
# save_pc_to_db(pc, database)

center_of_pc = database.find_middle_point()
RADIUS = 40
CHECK_DISTANCE = 30

result = database.find_near_points(center_of_pc[0], center_of_pc[1], RADIUS)
#print(f"Found {len(result)} points")

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
    # delta time (frame time) for smooth movement
    movement = speed * delta_time
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        camera_pos += movement * camera_front
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        camera_pos -= movement * camera_front
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        left = np.cross(camera_up, camera_front)
        left /= np.linalg.norm(left)
        camera_pos += movement * left
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        right = np.cross(camera_front, camera_up)
        right /= np.linalg.norm(right)
        camera_pos += movement * right
    return camera_pos

def check_distance(camera_pos, last_cam_pos, VBO, threshold=500):
    # check if camera has moved a certain distance from its last position
    # threshold is in meters
    distance = np.linalg.norm(camera_pos[:2] - last_cam_pos[:2])
    if distance > threshold:
        # print(f"camera moved {distance} meters from its last position, load more points")
        new_points = load_new_points(camera_pos, VBO, radius=RADIUS)
        return camera_pos.copy(), new_points
    return last_cam_pos, None

def load_new_points(camera_pos, VBO, radius):
    # perfrom a database query for new points
    # update VBO with new points
    new_result = database.find_near_points(camera_pos[0], camera_pos[1], radius)
    new_points = np.ascontiguousarray(new_result, dtype=np.float32)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, new_points.nbytes, new_points, GL_STATIC_DRAW)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    return new_points

def mouse_pick(mouse_x, mouse_y, window_width, window_height, camera_pos, view, projection, points, threshold=0.5):
    # threshold = see if a point is within <threshold> distance of where I clicked
    # @ operator is matrix multiplication between numpy matrices

    # Convert clicked mouse position to NDC normalized device coordinates
    x_ndc = (2.0 * mouse_x / window_width) - 1.0
    y_ndc = 1.0 - (2.0 * mouse_y / window_height)
    ray_clip = np.array([x_ndc, y_ndc, -1.0, 1.0])

    # Unproject the projection matrix
    inv_projection = np.linalg.inv(projection)
    ray_eye = inv_projection @ ray_clip
    ray_eye = np.array([ray_eye[0], ray_eye[1], -1.0, 0.0])

    # Unproject to world space
    inv_view = np.linalg.inv(view)
    ray_world = inv_view @ ray_eye
    ray_world = ray_world[:3]
    ray_world = ray_world / np.linalg.norm(ray_world)

    # Compute distances from the ray to each point
    # exclude points which couldn't be clicked due to not being visible
    points_xyz = points[:, :3]
    diff = points_xyz - camera_pos
    t = np.dot(diff, ray_world)
    valid = t > 0  # only points in front of the camera
    if not np.any(valid):
        return None

    proj_points = camera_pos + np.outer(t, ray_world)
    dists = np.linalg.norm(points_xyz - proj_points, axis=1)
    dists[~valid] = np.inf
    min_idx = np.argmin(dists)
    if dists[min_idx] < threshold:
        return points_xyz[min_idx]
    return None

def mouse_button_callback(window, button, action, mods):
    global last_camera_pos, latest_view, latest_projection, latest_points, window_width, window_height, picked_points
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        mouse_x, mouse_y = glfw.get_cursor_pos(window)
        picked = mouse_pick(mouse_x, mouse_y, window_width, window_height,
                            last_cam_pos, latest_view, latest_projection, latest_points, threshold=0.5)
        if picked is not None:
            print("Picked point:", picked)
            picked_points.append(picked)


def main():
    # Initialize GLFW
    global last_cam_pos, latest_view, latest_projection, latest_points, window_width, window_height
    if not glfw.init():
        raise Exception("GLFW could not be initialized!")
    monitor = glfw.get_primary_monitor()
    workarea = glfw.get_monitor_workarea(monitor)  # returns (x, y, width, height)
    x, y, work_width, work_height = workarea
    window_width = work_width
    window_height = work_height

    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, GLFW_CONSTANTS.GLFW_TRUE)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE)

    window = glfw.create_window(
        work_width, work_height, "LiDAR Points Visualization", None, None
    )
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed!")
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    glfw.set_mouse_button_callback(window, mouse_button_callback)

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)

    # NumPy array with shape (N, 6) (x, y, z, r, g, b).
    points = np.ascontiguousarray(result, dtype=np.float32)

    # Create VAO and VBO.
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(1)

    glBindVertexArray(VAO)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, points.nbytes, points, GL_STATIC_DRAW)

    # Attribute 0: position.
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    # Attribute 1: color.
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    projection = pyrr.matrix44.create_perspective_projection_matrix(
        fovy=45.0, aspect=work_width / work_height, near=0.1, far=10000.0, dtype=np.float32
    )

    # Initialize camera
    camera_pos = np.array(center_of_pc + np.array((-10., 0., 10.), dtype=np.float64), dtype=np.float64)
    # Initial yaw and pitch
    camera_yaw = 0.0
    camera_pitch = 0.0
    # Compute initial camera_front from yaw and pitch.
    front = np.array([
        np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
        np.sin(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
        np.sin(np.radians(camera_pitch))
    ], dtype=np.float32)
    camera_front = front / np.linalg.norm(front)
    camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    model = pyrr.matrix44.create_identity(dtype=np.float32)

    model_loc = glGetUniformLocation(shader_program, "model")
    view_loc = glGetUniformLocation(shader_program, "view")
    projection_loc = glGetUniformLocation(shader_program, "projection")

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

    # Render loop.
    while not glfw.window_should_close(window):
        glfw.poll_events()
        # At the beginning of your main loop:
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        # --- Keyboard movement ---
        camera_pos = process_keyboard(window, camera_pos, camera_front, camera_up, speed=100.0, delta_time=delta_time)

        # --- Mouse rotation ---
        # Only update camera rotation if the right mouse button is pressed.
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            current_mouse_x, current_mouse_y = glfw.get_cursor_pos(window)
            # Calculate offset.
            offset_x = current_mouse_x - last_mouse_x
            offset_y = last_mouse_y - current_mouse_y  # Reversed: y-coordinates go from top to bottom.
            last_mouse_x, last_mouse_y = current_mouse_x, current_mouse_y

            offset_x *= mouse_sensitivity
            offset_y *= mouse_sensitivity

            camera_yaw -= offset_x
            camera_pitch += offset_y

            # Clamp the pitch angle to prevent flipping.
            if camera_pitch > 89.0:
                camera_pitch = 89.0
            if camera_pitch < -89.0:
                camera_pitch = -89.0

            # Update camera_front vector.
            front = np.array([
                np.cos(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
                np.sin(np.radians(camera_yaw)) * np.cos(np.radians(camera_pitch)),
                np.sin(np.radians(camera_pitch))
            ], dtype=np.float32)
            camera_front = front / np.linalg.norm(front)
        else:
            # If the right button is not pressed, update last mouse positions.
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)

        last_cam_pos, new_points = check_distance(camera_pos, last_cam_pos, VBO, CHECK_DISTANCE)
        if new_points is not None:
            points = new_points
            latest_points = new_points


        # Update view matrix.
        view = pyrr.matrix44.create_look_at(
            eye=camera_pos,
            target=camera_pos + camera_front,
            up=camera_up,
            dtype=np.float32
        )

        latest_view = view  # update global
        latest_projection = projection  # update global

        glClearColor(0.1, 0.1, 0.1, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader_program)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(projection_loc, 1, GL_FALSE, projection)

        glBindVertexArray(VAO)
        glDrawArrays(GL_POINTS, 0, len(points))
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
    database.close()
    glfw.terminate()


if __name__ == "__main__":

    import tkinter as tk
    from tkinter import filedialog, messagebox


    # refresh file listbox when adding a file
    def refresh_file_list():
        file_list.delete(0, tk.END)
        files = database.get_stored_files()
        for file in files:
            file_list.insert(tk.END, file)


    # function to open GLFW window with OpenGL and selected point cloud
    def visualize_selected_file():
        selected_index = file_list.curselection()
        if not selected_index:
            messagebox.showwarning("Warning", "Please select a file from the list.")
            return

        file_name = file_list.get(selected_index[0])
        main()


    # TK setup
    root = tk.Tk()
    root.title("LiDAR File Manager")
    root.geometry("500x400")

    # upload file button
    upload_button = tk.Button(root, text="Upload .LAZ File",
                              command=lambda: save_pc_to_db_path(filedialog.askopenfilename(), database))
    upload_button.pack(pady=10)

    # label
    status_label = tk.Label(root, text="Select a file")
    status_label.pack()

    # listbox
    file_list = tk.Listbox(root, width=50, height=10)
    file_list.pack(pady=10)
    refresh_file_list()  # refresh list to show saved files

    # load selected file
    load_button = tk.Button(root, text="Load & Visualize", command=visualize_selected_file)
    load_button.pack(pady=5)


    root.mainloop()
