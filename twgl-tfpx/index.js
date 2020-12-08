var tf_vs = `#version 300 es

precision highp float;

/* Number of seconds (possibly fractional) that has passed since the last
   update step. */
uniform float u_TimeDelta;
uniform float u_Time;

/* A texture with just 2 channels (red and green), filled with random values.
   This is needed to assign a random direction to newly born particles. */
uniform sampler2D u_RgNoise;

/* This is the gravity vector. It's a force that affects all particles all the
   time.*/
uniform vec2 u_Gravity;

/* This is the point from which all newborn particles start their movement. */
uniform vec2 u_Origin;

/* Theta is the angle between the vector (1, 0) and a newborn particle's
   velocity vector. By setting u_MinTheta and u_MaxTheta, we can restrict it
   to be in a certain range to achieve a directed "cone" of particles.
   To emit particles in all directions, set these to -PI and PI. */
uniform float u_MinTheta;
uniform float u_MaxTheta;

/* The min and max values of the (scalar!) speed assigned to a newborn
   particle.*/
uniform float u_MinSpeed;
uniform float u_MaxSpeed;

uniform sampler2D u_ForceField;

uniform vec2 u_Screen;

/* Inputs. These reflect the state of a single particle before the update. */

/* Where the particle is. */
in vec2 i_Position;

/* Age of the particle in seconds. */
in float i_Age;

/* How long this particle is supposed to live. */
in float i_Life;

/* Which direction it is moving, and how fast. */ 
in vec2 i_Velocity;

/* Outputs. These mirror the inputs. These values will be captured
   into our transform feedback buffer! */
out vec2 v_Position;
out float v_Age;
out float v_Life;
out vec2 v_Velocity;

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

void main() {
  /* First, choose where to sample the random texture. I do it here
      based on particle ID. It means that basically, you're going to
      get the same initial random values for a given particle. The result
      still looks good. I suppose you could get fancier, and sample
      based on particle ID *and* time, or even have a texture where values
      are not-so-random, to control the pattern of generation. */
  ivec2 noise_coord = ivec2(gl_VertexID % 1024, gl_VertexID / 1024);
  
  /* Get the pair of random values. */
  vec2 rand = texelFetch(u_RgNoise, noise_coord, 0).rg;

  if (i_Age >= i_Life) {
    /* Particle has exceeded its lifetime! Time to spawn a new one
       in place of the old one, in accordance with our rules.*/

    /* Decide the direction of the particle based on the first random value.
       The direction is determined by the angle theta that its vector makes
       with the vector (1, 0).*/
    float theta = u_MinTheta + rand.r*(u_MaxTheta - u_MinTheta);

    /* Derive the x and y components of the direction unit vector.
       This is just basic trig. */
    float x = cos(theta);
    float y = sin(theta);

    /* Return the particle to a random position. */
    v_Position = (rand * 2.0) - 1.0;

    /* It's new, so age must be set accordingly.*/
    v_Age = 0.0;
    v_Life = i_Life;

    /* Generate final velocity vector. We use the second random value here
       to randomize speed. */
    v_Velocity =
      vec2(x, y) * (u_MinSpeed + rand.g * (u_MaxSpeed - u_MinSpeed));

  } else {
    /* Update parameters according to our simple rules.*/
    v_Position = i_Position + i_Velocity * u_TimeDelta;

    /* wrap around the screen */
    if (v_Position.x > 1.0) {
      v_Position.x = v_Position.x - 2.0;
    } else if (v_Position.x < -1.0) { 
      v_Position.x = v_Position.x + 2.0;
    }

    if (v_Position.y > 1.0) {
      v_Position.y = v_Position.y - 2.0;
    } else if (v_Position.y < -1.0) { 
      v_Position.y = v_Position.y + 2.0;
    }

    v_Age = i_Age + u_TimeDelta;
    v_Life = i_Life;

    rand = rand * 2.0 - vec2(1.0);
    vec2 force = vec2(0);
    if (i_Age < 2.0) { 
      force = 2.5 * (texture(u_ForceField, i_Position) - i_Age * vec4(0.5)).rg;
    } else { 
      if ( u_Time < 10000.0 ) { 
        force = 1.5 * (texture(u_ForceField, i_Position) - vec4(1.0, -1.0,0,0)).rg;
        // force = snoise(v_Position) * vec2(1.0);
      }
      else { 
        force = snoise(v_Position) * (0.5 * vec2(-1.0, 1.0) + 0.5 * vec2(sin());
        // force = 3.5 * (texture(u_ForceField, i_Position).rg - i_Age * vec2(0.0, 1.0)).rg;
      }
    }
    
    // force += vec2(sin(i_Age), cos(i_Age))
    // this is the old center-y line one
    // vec2 force = 1.5 * (texture(u_ForceField, i_Position) - vec4(0.5)).rg;
    // force = 0.5 * (force - 0.5 * vec2(rand.r)); 
    v_Velocity = i_Velocity + force * u_TimeDelta;

    if (length(v_Velocity) > u_MaxSpeed) { 
      v_Velocity = v_Velocity * 0.8;
    }
  }
}
`

var tf_fs = `#version 300 es

precision highp float;

out vec4 o;

void main() {
  o = vec4(0);
}
`
var vs = `#version 300 es

precision highp float;

in vec2 i_Position;
in float i_Age;
in float i_Life;
in vec2 i_Velocity;

out float v_Age;
out float v_Life;

void main() {
  v_Age = i_Age;
  v_Life = i_Life;

  gl_PointSize = 1.0 + 3.0 * (1.0 - i_Age/i_Life);
  gl_Position = vec4(i_Position, 0.0, 1.0);
}
`

var fs = `#version 300 es

precision highp float;

out vec4 o_FragColor;

vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

in float v_Age;
in float v_Life;

void main() {
  float t =  v_Age/v_Life;
  o_FragColor = 1.00 * vec4(
    palette(t,
            vec3(0.5,0.5,0.4),
            vec3(0.1,0.4,0.2),
            vec3(1.0,1.0,1.5),
            vec3(0.0,0.10,0.10)), max(0.1, t));
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
uniform float u_offsets[49];
uniform float u_kernel[49];
uniform float u_kernelWeight;
uniform int u_kernelSize;
uniform bool u_horizontal;
uniform float u_Time;

in vec2 v_texCoord;

out vec4 o_FragColor;

vec4 gaussianBlur(sampler2D u_Texture, vec2 uv) {
  vec4 sum = vec4(0.0);                           
  for (int i = 0; i< u_kernelSize; i++)             
  {     
    vec4 tmp = vec4(0.0);
    if (u_horizontal) { 
      tmp = texture( u_Texture,  uv + vec2(u_offsets[i], 0.0) );  
      sum += tmp * u_kernel[i] / u_kernelWeight;  
    }                                          
    else { 
      tmp = texture( u_Texture,  uv + vec2(0.0, u_offsets[i]) );
      sum += tmp * u_kernel[i] / u_kernelWeight;  
    }
    
    // sum += tmp * u_kernel[i] / (u_kernelWeight / 1000.0);               
  }
  return sum;                                                  
}

float stepping(float t){
  if(t<0.)return -1.+pow(1.+t,2.);
  else return 1.-pow(1.-t,2.);
}

void main() {
  vec4 color = gaussianBlur(u_Texture, v_texCoord);
  o_FragColor = min(vec4(1.0), color);
}
`

var combine_fs = `#version 300 es
precision highp float;

uniform sampler2D u_First;
uniform sampler2D u_Second;
uniform sampler2D u_Third;

in vec2 v_texCoord;

out vec4 o_FragColor;

void main() {
  vec4 firColor = texture(u_First, v_texCoord);
  vec4 secColor = texture(u_Second, v_texCoord);
  vec4 thirdCol = texture(u_Third, v_texCoord);

  // make the 'atmosphere' more blue
  thirdCol.rgb /= thirdCol.a;
  thirdCol.rgb += vec3(0.02,0.02,0.4);
  thirdCol.rgb *= thirdCol.a;

  secColor.rgb /= secColor.a;
  secColor.rgb += vec3(0.02,0.02,0.06);
  secColor.rgb *= secColor.a;

  // modify color of non-blurred texture
  firColor.rgb /= firColor.a;
  firColor.rgb = ((firColor.rgb - 0.55) * 1.75) + 0.55;
  firColor.rgb -= vec3(-0.23, 0.055, 0.17) / 1.2;
  firColor.rgb *= firColor.a;
  vec4 col = vec4(0);

  // perceived luminance = (0.299*R + 0.587*G + 0.114*B)
  if (firColor.r * 0.299 + firColor.g * 0.587 + firColor.b * 0.114 > 0.25) { 
    col += 0.35 * firColor;
    col += 0.31 * secColor;
    col += 0.3 * thirdCol;
    // col *= 0.98;
  } else { 
    col += 0.7 * thirdCol;
    col += 0.125 * secColor;
    col += 0.375 * firColor;
  }

  o_FragColor = min(vec4(1.0), col);
}
`