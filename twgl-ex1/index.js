var fs = `#version 300 es

precision mediump float;

uniform vec2 resolution;
uniform float time;
uniform vec2 mouse_pos;

out vec4 o_FragColor;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution;
  
  float x_diff = uv.x - mouse_pos.x;
  float y_diff = uv.y - mouse_pos.y;
  float distanceFromCenter = sqrt((x_diff * x_diff) + (y_diff * y_diff));
  float color = 0.0;
  // lifted from glslsandbox.com

  color += sin( distanceFromCenter * cos( time / 2.76 ) * 60.0 ) + cos( uv.y * cos( time / 2.80 ) * 10.0 );
  color += sin( distanceFromCenter * sin( time / 1.66 ) * 40.0 ) + cos( uv.y* sin( time / 1.70 ) * 40.0 );
  color += sin( uv.x * sin( time / 0.88 ) * 10.0 ) + sin( distanceFromCenter * sin( time / 3.50 ) * 50.0 );
  color *= sin( time / 10.0 ) * 0.5;

  o_FragColor = vec4( vec3( color * 0.5, sin( color ) * 0.75, color ), 1.0 );
}
`

var vs = `#version 300 es

in vec4 position;

void main() {
  gl_Position = position;
}
`;

var p_vs = `#version 300 es

in vec4 position;
in vec2 texcoord;

out vec2 v_texCoord;

void main() {
  v_texCoord = texcoord;

  gl_Position = position;
}
`;

var p_fs = `#version 300 es
precision highp float;

uniform sampler2D u_Texture;

in vec2 v_texCoord;

out vec4 o_FragColor;

void main() {
  o_FragColor = texture(u_Texture, v_texCoord);
  // o_FragColor = vec4(1,0,1,1);
}
`