var vs = `#version 300 es

precision mediump float;

in vec2 i_Position;
uniform vec2 u_Origin;
uniform vec2 u_Screen;

void main() {
  vec2 displacement = vec2(0.0);
  vec2 aspect = vec2(u_Screen.y / u_Screen.x, 1.0);
  vec2 diff = i_Position - u_Origin;
  if ( length(diff) < 0.25 ) {
    displacement = -normalize(diff) * sin(length(diff) * 4.0) * 0.22;
  }

  gl_PointSize = 2.0;
  gl_Position = vec4(aspect * (0.5 * i_Position + displacement), 0.0, 1.0);
}
`

var fs = `#version 300 es

precision mediump float;

out vec4 o_FragColor;

vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

void main() {
  o_FragColor = vec4(0.0,0.0,0.0,1.0);
}
`;