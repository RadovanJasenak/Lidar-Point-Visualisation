import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
import numpy as np
import pyrr
import time
from databaza import PointCloud
from databaza import Database
from databaza import save_pc_to_db

pc = PointCloud("../data/Velky_Biel_32634_WGS84-TM34_sample.laz")
database = Database()
save_pc_to_db(pc, database)

# result = database.find_near_points(229655.09375, 5346709, 50)
center_of_pc = database.find_middle_point()
result = database.find_near_points(center_of_pc[0], center_of_pc[1], 30)
minmax = database.find_min_max()
print(minmax['x'])
print(f"Found {len(result)} points")


# Vertex shader: accepts vertex positions and colors.
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

# Fragment shader: outputs the interpolated vertex color.
fragment_shader_source = """
#version 330 core
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
    # Multiply speed by delta_time for smoother, frame-independent movement.
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


def main():
    # Initialize GLFW.
    if not glfw.init():
        raise Exception("GLFW could not be initialized!")
    monitor = glfw.get_primary_monitor()
    workarea = glfw.get_monitor_workarea(monitor)  # returns (x, y, width, height)
    x, y, work_width, work_height = workarea

    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT,
        GLFW_CONSTANTS.GLFW_TRUE
    )
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_PROFILE,
        GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
    )

    window = glfw.create_window(
        work_width, work_height, "LiDAR Points Visualization", None, None
    )
    if not window:
        glfw.terminate()
        raise Exception("GLFW window creation failed!")
    glfw.window_hint(glfw.DECORATED, glfw.TRUE)

    glfw.make_context_current(window)
    glfw.swap_interval(1)  # Enable vsync

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)

    # NumPy array with shape (N, 6) (x, y, z, r, g, b).
    points = np.ascontiguousarray(result, dtype=np.float32)
    # Assuming points is an (N, 3) array for positions.
    center = np.mean(points[:, :3], axis=0)
    min_vals = np.min(points[:, :3], axis=0)
    max_vals = np.max(points[:, :3], axis=0)
    scale_factor = 1.0 / np.linalg.norm(max_vals - min_vals)  # Choose a factor that fits your scene.

    # Apply transformation to center and scale the points:
    transformed_positions = (points[:, :3] - center) * scale_factor
    # Reassemble your points array (if you have colors, concatenate them back).
    points[:, :3] = transformed_positions
    middle = np.mean(points[:, :3], axis=0)

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
        fovy=45.0, aspect=work_width / work_height, near=0.1, far=100.0, dtype=np.float32
    )

    # Initialize camera parameters.
    camera_pos = np.array(middle + np.array([-0.2, 0.0, 0.0]), dtype=np.float32)
    # Initial yaw and pitch are chosen so that the camera looks at the origin.
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
    glPointSize(6.0)

    # Variables for FPS calculation.
    last_time = time.time()
    frame_count = 0

    # Variables for mouse rotation.
    last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
    mouse_sensitivity = 0.1
    last_frame_time = 0
    # Render loop.
    while not glfw.window_should_close(window):
        glfw.poll_events()
        # At the beginning of your main loop:
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time

        # --- Keyboard movement ---
        camera_pos = process_keyboard(window, camera_pos, camera_front, camera_up, speed=1.0, delta_time=delta_time)

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

        # Update view matrix.
        view = pyrr.matrix44.create_look_at(
            eye=camera_pos,
            target=camera_pos + camera_front,
            up=camera_up,
            dtype=np.float32
        )

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
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            last_time = current_time

    glDeleteVertexArrays(1, (VAO,))
    glDeleteBuffers(1, (VBO,))
    database.close()
    glfw.terminate()


if __name__ == "__main__":
    main()
