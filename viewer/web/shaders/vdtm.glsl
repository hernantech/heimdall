// View-Dependent Texture Mapping (VDTM) Fragment Shader
// Projects camera video feeds onto mesh geometry on the client GPU.
// Eliminates server-side atlas generation and texture blending.

// Camera uniforms (set per frame from calibration data)
#define MAX_CAMERAS 8

uniform int u_num_cameras;
uniform mat4 u_camera_vp[MAX_CAMERAS];       // view-projection per camera
uniform sampler2D u_camera_textures[MAX_CAMERAS]; // video feed textures
uniform vec3 u_camera_positions[MAX_CAMERAS];
uniform vec2 u_camera_resolution[MAX_CAMERAS];

// Viewer
uniform vec3 u_viewer_position;

// Fragment inputs
varying vec3 v_world_position;
varying vec3 v_world_normal;

float compute_camera_weight(int cam_idx) {
    vec3 to_cam = normalize(u_camera_positions[cam_idx] - v_world_position);
    vec3 to_viewer = normalize(u_viewer_position - v_world_position);
    vec3 normal = normalize(v_world_normal);

    // Cosine of angle between surface normal and camera direction
    float ndotc = max(dot(normal, to_cam), 0.0);

    // Penalize cameras viewing from very different angle than viewer
    float view_similarity = max(dot(to_cam, to_viewer), 0.0);

    // Penalize grazing angles (< 15 degrees from surface tangent)
    float grazing_penalty = smoothstep(0.0, 0.26, ndotc);

    return ndotc * view_similarity * grazing_penalty;
}

vec2 project_to_camera(int cam_idx) {
    vec4 clip = u_camera_vp[cam_idx] * vec4(v_world_position, 1.0);
    vec3 ndc = clip.xyz / clip.w;

    // NDC [-1,1] → UV [0,1]
    return ndc.xy * 0.5 + 0.5;
}

void main() {
    vec3 blended_color = vec3(0.0);
    float total_weight = 0.0;

    for (int i = 0; i < u_num_cameras; i++) {
        float w = compute_camera_weight(i);
        if (w < 0.01) continue;

        vec2 uv = project_to_camera(i);

        // Skip if projection is outside camera frustum
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) continue;

        vec3 cam_color = texture2D(u_camera_textures[i], uv).rgb;
        blended_color += cam_color * w;
        total_weight += w;
    }

    if (total_weight > 0.0) {
        blended_color /= total_weight;
    }

    gl_FragColor = vec4(blended_color, 1.0);
}
