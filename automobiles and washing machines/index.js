var vs = `#version 300 es

precision mediump float;

in vec2 i_Position;

uniform vec2 u_Origin;
uniform vec2 u_TextureSize;
uniform sampler2D u_Texture;
uniform float u_Time;

out vec2 vPUv;
out vec2 diff;

// Simplex 2D noise
//
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

void main() {
  vec2 puv = i_Position * 1.0;
  vPUv = vec2(puv.x, -puv.y);

  vec2 displacement = vec2(0.0);
  displacement += vec2(
    rand(sin(u_Time) * vPUv * 0.45 - 0.5) / 20.0 + (snoise(vPUv) - 0.1) / 1.0, 
    snoise( 
      (vPUv / (sin(u_Time / 1.2) * 2.0)) + sin(u_Time / 1.0) / 100.0)
       - 0.5);
  displacement *= 0.05 * cos(u_Time * 1.5) / sin(u_Time * 1.5);
       //* cos(u_Time / 5.0) / (sin(u_Time * 2.5) 
  diff = i_Position - u_Origin;
  if ( length(diff) < 0.25 ) {
    displacement = normalize(diff) * sin(length(diff) * 7.0) * max(0.16, sin(u_Time) * 0.45);
    gl_PointSize = 2.0;
    gl_Position = vec4(0.6 * i_Position + displacement, 0.5, 1.0);  
  } else {
    gl_PointSize = 2.0 - 4.0 * length(displacement);
    gl_Position = vec4(0.6 * i_Position + displacement, 0.6, 1.0);  
  }
}
`

var fs = `#version 300 es

precision mediump float;

in vec2 vPUv;
in vec2 diff;

uniform sampler2D u_Texture;
uniform float u_Time;

out vec4 o_FragColor;

// Simplex 2D noise
//
vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

float snoise(vec2 v){
  const vec4 C = vec4(0.211324865405187, 0.366025403784439,
           -0.577350269189626, 0.024390243902439);
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);
  vec2 i1;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;
  i = mod(i, 289.0);
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
  + i.x + vec3(0.0, i1.x, 1.0 ));
  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
    dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;
  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float rand(vec2 n) { 
	return fract(sin(dot(n, vec2(12.9898, 4.1414))) * 43758.5453);
}

vec4 getGlitchColor(vec2 uv) {
  vec2 r = vec2(
    rand(vec2(ceil(u_Time * 20.0), 0.0)) * 2.0 - 1.0,
    rand(vec2(0.0, ceil(u_Time * 20.0))) * 2.0 - 1.0
  );
  vec2 noiseUv = uv + r * 0.001;
  vec2 pos = uv * vec2(0.2, 0.4) * r;
  float mask = smoothstep(
    length(vec3(1.0)) - 0.004,
    length(vec3(1.0)),
    length(vec3(pos, 1.0))
    );
  vec4 texColor = vec4(snoise(noiseUv + r * 0.01) * (1.0 - mask));
  vec4 texColorDiff = vec4(snoise(noiseUv + r) * mask);
  return texColor + texColorDiff;
}

float noise(vec2 p){
	vec2 ip = floor(p);
	vec2 u = fract(p);
	u = u*u*(3.0-2.0*u);
	
	float res = mix(
		mix(rand(ip),rand(ip+vec2(1.0,0.0)),u.x),
		mix(rand(ip+vec2(0.0,1.0)),rand(ip+vec2(1.0,1.0)),u.x),u.y);
	return res*res;
}

vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

void main() {
	// pixel color
  vec4 colA = texture(u_Texture, vPUv);
  
  if (length(diff) < 0.25) {
    o_FragColor = colA;
  } else { 
    vec4 colB = colA + 0.4 * vec4(palette(
      sin(u_Time * vPUv.x),
      vec3(0.5,0.5,0.5) * vPUv.y,
      vec3(1.0,0.5,1.0) * vPUv.x,
      vec3(0.3,0.5,0.8),
      vec3(0.5,0.5,0.5)
    ), 1.0);

    // RGB Shift
    float texColorR = texture(u_Texture, vPUv).r;
    float texColorG = texture(u_Texture, vPUv - 0.01).g;
    float texColorB = texture(u_Texture, vPUv + 0.01).b;
    vec4 colC = vec4(vec3(texColorR, texColorG, texColorB), 1.0);

    o_FragColor = 0.8 * colC + 0.2 * colB;
  }
}
`;
