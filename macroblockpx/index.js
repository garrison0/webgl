var tf_vs = `#version 300 es

precision highp float;

uniform sampler2D u_RgNoise;
uniform sampler2D u_YGradient;
uniform sampler2D u_XGradient;
uniform float u_TimeDelta;
uniform float u_Time;
uniform vec2 u_Screen;

in vec2 i_Position;
in vec2 i_Size;
in vec2 i_Velocity;

/* Outputs. These mirror the inputs. These values will be captured
   into our transform feedback buffer! */
out vec2 v_Position;
out vec2 v_Size;
out vec2 v_Velocity;

void main() {
  ivec2 noise_coord = ivec2(gl_VertexID % 512, gl_VertexID / 512);
  
  vec2 rand = texelFetch(u_RgNoise, noise_coord, 0).rg;

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

  vec2 texture_Coords = ((i_Position + 1.0) / 2.0) * vec2(1.0, -1.0);
  rand = rand * 2.0 - vec2(1.0);
  vec4 x_val = texture(u_XGradient, texture_Coords);
  // 0.299*R + 0.587*G + 0.114*B
  float x_dir = x_val.r * 0.299 + 0.587 * x_val.g + 0.114 * x_val.b;
  vec4 y_val = texture(u_YGradient, texture_Coords);
  float y_dir = y_val.r * 0.299 + 0.587 * y_val.g + 0.114 * y_val.b;
  vec2 force = vec2(x_dir, y_dir);
  vec2 fixed_direction = ( vec2(x_dir, y_dir) - vec2(0.25, 0.3) ) ;

  if (length(fixed_direction) > 0.3) {
    v_Velocity = i_Velocity + 2.0 * ((0.3 * rand + fixed_direction) * u_TimeDelta);
  } else { 
    v_Velocity = i_Velocity + 0.5 * rand * u_TimeDelta;
    v_Velocity *= 0.98;
  }

  // don't accelerate to infinity
  if (length(v_Velocity) > 0.2) { 
    v_Velocity = v_Velocity * 0.9;
  }

  v_Size = i_Size;
  // this expects that BOX_SIZE is never > 64
  v_Size = min(vec2(64.0), max(vec2(4.0), i_Size - 0.1 * rand + 0.1 * v_Velocity));
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
in vec2 i_TexCoord;

out vec2 v_Position;
flat out int vertexId;

void main() {
  vertexId = gl_VertexID;
  v_Position = i_Position;
  // v_TexCoord = i_TexCoord;
  gl_Position = vec4(i_Position, 0.0, 1.0);
}
`

var fs = `#version 300 es

precision highp float;

out vec4 o_FragColor;

in vec2 v_Position;
flat in int vertexId;

uniform sampler2D u_RgNoise;
uniform sampler2D u_Frame;
uniform float u_Time;

vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

void main() {
  vec2 uv = vec2(1.0, -1.0) * (v_Position + 1.0) / 2.0;
  o_FragColor = texture(u_Frame, uv);
}
`;

var default_vs = `#version 300 es

in vec4 position;
in vec2 texcoord;

out vec2 v_texCoord;

void main() {
  v_texCoord = texcoord;

  gl_Position = position;
}
`;

var flipped_vs = `#version 300 es

in vec4 position;
in vec2 texcoord;

out vec2 v_texCoord;

void main() {
  v_texCoord = texcoord * vec2(1, -1);

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
    }                                          
    else { 
      tmp = texture( u_Texture,  uv + vec2(0.0, u_offsets[i]) ); 
    }
    sum += tmp * u_kernel[i] / 4.0; 
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

  vec4 fragColor = vec4(0);
  vec2 uv = v_texCoord;
  float iTime = 0.04;
  for(int i=0;i<12;i++){
      float t = iTime + float(i)*3.141592/12.*(5.+1.*stepping(sin(iTime*3.)));
      vec2 p = vec2(cos(t),sin(t));
      p *= cos(iTime + float(i)*3.141592*cos(iTime/8.));
      vec3 col = cos(vec3(0,1,-1)*3.141592*2./3.+3.141925*(iTime/2.+float(i)/5.)) * 0.4;
      fragColor += vec4(0.01/length(uv-p*0.9)*col,1.0);
  }
  fragColor.xyz = pow(fragColor.xyz,vec3(3.));
  fragColor.w = 0.05;

  o_FragColor = color + vec4(0,0,0.01,0) + 0.01 * fragColor;

  // o_FragColor = min(vec4(1.0), color);
  // o_FragColor = min(vec4(1.0), fragColor + color);
  // vec3 brightened = color.rgb * 1.1;
  // o_FragColor = vec4(brightened, max(1.0, color.a * 1.2));
  // o_FragColor = vec4(texture(u_Texture, v_texCoord).rgb, 1.0);
}
`

// find image gradient
var n_fs = `#version 300 es
precision highp float;

uniform sampler2D u_Texture;
uniform float u_sobelX[9];
uniform float u_sobelY[9];
uniform float u_offsets[18];
uniform vec2 u_screen;
uniform bool u_isVerticle;
in vec2 v_texCoord;

out vec4 o_FragColor;

void main() {
  vec4 col = vec4(0);
  for (int i = 0; i < 9; i++) {
    vec2 tmp = v_texCoord + ( vec2( u_offsets[i*2], u_offsets[i*2 + 1] ) / u_screen );
    if (u_isVerticle) { 
      col += u_sobelY[i] * (1.0 / 128.0) * texture(u_Texture, tmp);
    } else { 
      col += u_sobelX[i] * (1.0 / 128.0) * texture(u_Texture, tmp);
    }
  }
  // 0.299*R + 0.587*G + 0.114*B = luminance perceived
  // float lum = 0.299 * col.r + 0.587 * col.g + 0.114 * col.b;
  // lum = 0.5 + lum;
  // lum = lum * 20.0;
  // if (lum < 0.23) { 
  //   lum = lum * 20.5;
  // } else {
  //   lum = lum * 0.9;
  // }
  o_FragColor = clamp(vec4(vec3(0.5) + col.rgb, 1.0), 0.0, 1.0);
  // o_FragColor = vec4(lum, lum, lum, 1.0);
  // o_FragColor = texture(u_Texture, v_texCoord); //* vec4(0,1,0,1);
}
`

var combine_fs = `#version 300 es
precision highp float;

uniform sampler2D u_First;
uniform sampler2D u_Second;
uniform sampler2D u_Blocks;
uniform sampler2D u_Frame;

in vec2 v_texCoord;

out vec4 o_FragColor;

void main() {
  vec4 col = texture(u_Blocks, v_texCoord);
  if (col.r > 0.1) { 
    o_FragColor = col;
  } else { 
    o_FragColor = 0.5 * texture(u_First, v_texCoord);
  }
  // o_FragColor = texture(u_Blocks, v_texCoord);
  // o_FragColor = texture(u_Frame, v_texCoord * vec2(1, -1)); 
  // vec4 blockCol = texture(u_Blocks, v_texCoord);
  
  // if (blockCol.g >= 0.23) {
  //   o_FragColor = texture(u_Frame, v_texCoord * vec2(1, -1));
  // } 
  //   o_FragColor = texture(u_Frame, v_texCoord * vec2(1, -1)); 
  // } 
  //   o_FragColor = texture(u_Blocks, v_texCoord) + 0.5 * texture(u_First, v_texCoord) + 0.5 * texture(u_Second, v_texCoord);
  // } else { 
  //   o_FragColor = texture(u_Frame, v_texCoord * vec2(1, -1));
  // }

  // o_FragColor = texture(u_Blocks, v_texCoord) + 0.5 * texture(u_First, v_texCoord) + 0.5 * texture(u_Second, v_texCoord); 
  // o_FragColor = 1.0 * texture(u_Blocks, v_texCoord);
  // o_FragColor = 0.5 * texture(u_First, v_texCoord) + 0.5 * texture(u_Second, v_texCoord);
  // vec2 onePixel = vec2(1) / vec2(textureSize(u_Second, 0));
  // o_FragColor = texture(u_Second, v_texCoord);
  // o_FragColor = 1.0 * texture(u_First, v_texCoord) + 1.0 * texture(u_Second, v_texCoord) + 1.0 * texture(u_Third, v_texCoord);
}
`