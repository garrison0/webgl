#version 300 es
#define AA 2 //# of anti-aliasing passes

precision highp float;

in vec2 vUv;
uniform vec2 uResolution;
uniform float uTime;
uniform sampler2D uWaterTexture;
uniform sampler2D uGrassTexture;
uniform sampler2D uSandTexture;

out vec4 o_FragColor;

// "globals"
float sunsetAmt(float time) { return clamp(time*0.05, 0., 1.); }
vec3 islandCenter() { return vec3(0.0,-5.7,-25.0); }

// crunch the tree, add one type of noise to the sky color
float glitchAmtOne(float time) { 
    float a = mod(time + 1.8, 56.11);
    float ap = smoothstep(0.0, 0.08, a) - smoothstep(2.2, 2.3, a);
    return ap;
    // return 0.0;
}

// slowly 'dissolves' the tree
float glitchAmtTwo(float time) { 
    float a = mod(time + 21.45, 68.1331);
    float ap = smoothstep(0.0, 16.0, a) - smoothstep( 20.4, 20.75, a);
    return ap;
    // return 0.0;
}

//messes up the ground texture, adds another noise to the sky 
float glitchAmtThree(float time) { 
    float a = mod(time + 14.2, 91.61); 
    float ap = smoothstep(0.0, 0.1, a) - smoothstep(5.2, 6.4, a);
    return ap;
}

float glitchAmtFour(float time) { 
    float a = mod(time + 14.2, 31.61); 
    float ap = smoothstep(0.0, 1.4, a) - smoothstep(6.2, 6.8, a);
    return ap;
}

// noise stuff - none of which is original
float mod289(float x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 mod289(vec4 x){return x - floor(x * (1.0 / 289.0)) * 289.0;}
vec4 perm(vec4 x){return mod289(((x * 34.0) + 1.0) * x);}

float noise(vec3 p){
    vec3 a = floor(p);
    vec3 d = p - a;
    d = d * d * (3.0 - 2.0 * d);

    vec4 b = a.xxyy + vec4(0.0, 1.0, 0.0, 1.0);
    vec4 k1 = perm(b.xyxy);
    vec4 k2 = perm(k1.xyxy + b.zzww);

    vec4 c = k2 + a.zzzz;
    vec4 k3 = perm(c);
    vec4 k4 = perm(c + 1.0);

    vec4 o1 = fract(k3 * (1.0 / 41.0));
    vec4 o2 = fract(k4 * (1.0 / 41.0));

    vec4 o3 = o2 * d.z + o1 * (1.0 - d.z);
    vec2 o4 = o3.yw * d.x + o3.xz * (1.0 - d.x);

    return o4.y * d.y + o4.x * (1.0 - d.y);
}

///// 2d FBM ///////////
float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                            vec2(12.9898,78.233)))*
        43758.5453123);
}

float noise2 (in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

#define OCTAVES 6
float fbm2 (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise2(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

//// 3d SIMPLEX NOISE /////
vec3 mod289(vec3 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}

vec4 taylorInvSqrt(vec4 r)
{
    return 1.79284291400159 - 0.85373472095314 * r;
}

float snoise(vec3 v)
{ 
    const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
    const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 =   v - i + dot(i, C.xxx) ;

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    //   x0 = x0 - 0.0 + 0.0 * C.xxx;
    //   x1 = x0 - i1  + 1.0 * C.xxx;
    //   x2 = x0 - i2  + 2.0 * C.xxx;
    //   x3 = x0 - 1.0 + 3.0 * C.xxx;
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy; // 2.0*C.x = 1/3 = C.y
    vec3 x3 = x0 - D.yyy;      // -1.0+3.0*C.x = -0.5 = -D.y

    // Permutations
    i = mod289(i); 
    vec4 p = permute( permute( permute( 
                i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
            + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    // Gradients: 7x7 points over a square, mapped onto an octahedron.
    // The ring size 17*17 = 289 is close to a multiple of 49 (49*6 = 294)
    float n_ = 0.142857142857; // 1.0/7.0
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    //vec4 s0 = vec4(lessThan(b0,0.0))*2.0 - 1.0;
    //vec4 s1 = vec4(lessThan(b1,0.0))*2.0 - 1.0;
    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    //Normalise gradients
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.5 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 105.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                    dot(p2,x2), dot(p3,x3) ) );
}

/// 3d FBM VARIANTS /////
const mat2 m2 = mat2( 0.60, -0.80, 0.80, 0.60 );

const mat3 m3 = mat3( 0.00,  0.80,  0.60,
                     -0.80,  0.36, -0.48,
                     -0.60, -0.48,  0.64 );

float fbm( in vec3 p ) {
    float f = 0.0;
    f += 0.5000*noise( p ); p = m3*p*2.02;
    f += 0.2500*noise( p ); p = m3*p*2.03;
    f += 0.1250*noise( p ); p = m3*p*2.01;
    f += 0.0625*noise( p );
    return f/0.9375;
}

float turbulence( vec3 p ) {
    float t = -.5;

    for (float f = 1.0 ; f <= 10.0 ; f++ ){
        float power = pow( 2.0, f );
        t += abs( snoise( vec3( power * p ) ) / power );
    }

    return t;
}

vec3 hash3( vec2 p ){
    vec3 q = vec3( dot(p,vec2(127.1,311.7)), 
				   dot(p,vec2(269.5,183.3)), 
				   dot(p,vec2(419.2,371.9)) );
	return fract(sin(q)*43758.5453);
}

// u = 1.0, v = 0.5 
// voronoi like noise
float vnoise( in vec2 x, float u, float v )
{
    vec2 p = floor(x);
    vec2 f = fract(x);

    float k = 1.0 + 63.0*pow(1.0-v,4.0);
    float va = 0.0;
    float wt = 0.0;
    for( int j=-2; j<=2; j++ )
    for( int i=-2; i<=2; i++ )
    {
        vec2  g = vec2( float(i), float(j) );
        vec3  o = hash3( p + g )*vec3(u,u,1.0);
        vec2  r = g - f + o.xy;
        float d = dot(r,r);
        float w = pow( 1.0-smoothstep(0.0,1.414,sqrt(d)), k );
        va += w*o.z;
        wt += w;
    }

    return va/wt;
}

float vnoiseOctaves (in vec2 x, float u, float v ) {
    float t = -.5;

    for (float f = 1.0 ; f <= 10.0 ; f++ ){
        float power = pow( 2.0, f );
        t += abs( vnoise( power * x, u, v ) / power );
    }

    return t;
}

// --------- DISTANCE FUNCTIONS ---------- //
// ---- UTILITIES --- // 

vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }


vec3 rotatePoint(vec3 p, vec3 n, float theta) { 
    vec4 q = vec4(cos(theta / 2.0), sin (theta / 2.0) * n);
    vec3 temp = cross(q.xyz, p) + q.w * p;
    vec3 rotated = p + 2.0*cross(q.xyz, temp);
    return rotated;
}

vec2 opU( vec2 d1, vec2 d2 )
{
	return (d1.x<d2.x) ? d1 : d2;
}

vec2 opSmoothU( vec2 d1, vec2 d2, float k) 
{ 
    float colorSmoothness = k * 4.0;
    float interpo = clamp( 0.5 + 0.5 * (d1.x - d2.x) / colorSmoothness, 0.0, 1.0 );
    float h = max( k - abs(d1.x - d2.x), 0.0) / k;
    float diff = h*h*h*k*(1.0/6.0);
    return vec2( min(d1.x, d2.x) - diff,
                 mix(d1.y, d2.y, interpo) - k * interpo * ( interpo - 1.0) );
}

/////-----------  SDF ---------/////////
float sdfSphere(vec3 p, float r) {
    return length( p ) - r;
}

float sdTorus( vec3 p, vec2 t )
{
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdRoundBox( vec3 p, vec3 b, float r )
{
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdbEllipsoid( in vec3 p, in vec3 r )
{
    float k1 = length(p/r);
    float k2 = length(p/(r*r));
    return k1*(k1-1.0)/k2;
}

vec2 sdCapsule( vec3 p, vec3 a, vec3 b )
{
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return vec2( length( pa - ba*h ) - 0.01, h );
}

float sdTrunk ( vec3 p, float h, float r, float time ) { 
    vec3 pCopy = p;
    float treeY = clamp(pCopy.y, 0.0, h);
    p.y -= treeY;
    p.x += 0.833 * cos( 0.7 * (1.66666 + 0.722 * pCopy.y) ); // tilting shape

    float noiseAmount = 1.0 - smoothstep(0.0, 1.45, treeY / h);
    // glitch: spike the noise amount (up from 0.3 to 1.5 ish)
    float noise = (1.3 * glitchAmtOne(time) + 0.3) * noiseAmount * clamp(fbm2( pCopy.xy ), -1.0, 1.0);

    // 0.422
    p.x -= 0.222 * noise;

    float ipy = h - pCopy.y;
    float bigger = 0.22 * clamp(ipy * ipy * 0.725, 0.0, h * 1.1) / h; // make the bottom bigger

    float bumps = 0.18 * noiseAmount * clamp (fbm(pCopy), 0.0, 1.0);

    // glitch: 0.422 * uTime (or, some glitch spike amount)
    p.x -= (0.222 + 110.0*glitchAmtTwo(time)) * bumps;

    return length(p) - (r + bigger) + noise;
}

// ---------- MAPPING SUB FUNCTIONS ------ //
// type = 0, 1
vec2 mapBranch ( in vec3 p, float l, vec2 cur, float type, float time )
{
    // float theta = 0.654 + 2.755 * 0.2 * type;
    float theta = 0.754 + 2.455 * 0.2 * type;
    vec3 rq = vec3(0.0,1.0,0.0);
    p = rotatePoint(p, rq, theta);
    vec3 glitchDisp = (glitchAmtOne(time) + glitchAmtTwo(time)) * vec3(18.0, 0.0, -10.0);
    p += glitchDisp;

    // main stem/branch
    float cos45 = .70710678;
    float sin45 = cos45;
    float branchRad = 0.03;

    vec3 a = vec3(0);
    float y = 0.0;
    
    if (type < 1.0) { 
        y = 0.3 * cos(0.3 * length(p.xz));
    } else if (type < 2.0) { 
        y = -0.8 * cos(0.3 + 0.5 * length(p.xz));
    } else { 
        y = 1.75 * cos(0.2 * length(p.xz));
    }
    
    vec3 b = l * vec3(1,y + (10.0*glitchAmtTwo(time) + 0.1)*fbm(vec3(y+time*0.25)),0);

    float angle = 0.0;
    if (type < 1.0) { 
        angle = 6.283185 / 4.0;
    } else { 
        angle = 6.283185 / 3.0;
    }
    
    float quadrant = round(atan(p.z,p.x) / angle); // angle / (pi/2) => 0,1,2,3
    float an = quadrant * angle;

    mat2 rot = mat2(cos(an), -sin(an), sin(an), cos(an));
    p.xz = rot * p.xz;

    float branchDist = sdCapsule(p, a, b).x - 0.005; 

    float m = 20.0;
    vec2 res = opU( cur, vec2(branchDist, m));

    // leaves
    p.z = abs(p.z);
    p *= (1.0 - 0.9*glitchAmtOne(time));
    vec3 c = vec3(branchRad * 2.0, 0, 0);
    vec3 l2 = vec3(36,0,0);
    vec3 n = clamp(round(p/c),-l2,l2);
    vec3 q = p-c*n;
    float nx = abs(n.x);
    float howCloseToEnd = nx / l2.x;
    float howMuchDisp = cos(0.814543 * quadrant + 2.2 * type + time);
    float quadrantNoise = noise2(vec2(0.748243*quadrant,0.12325*quadrant));
    float nxNoise = noise2(vec2(0.12323*nx + quadrantNoise, 0.557*nx + quadrantNoise));
    float constMovement = (11.0*glitchAmtTwo(time) + 0.1) * fbm( vec3(nxNoise * time) );
    howMuchDisp = constMovement + 0.03 * max(0.0, howMuchDisp);
    float dispY = 0.123 + ((-0.6 * type * (1.0-howCloseToEnd*howCloseToEnd)) + 0.2 * nxNoise) + howMuchDisp * cos(0.5 * (quadrantNoise + nx + 4.0 * time));
    a = vec3(0.0, howCloseToEnd * b.y, 0.0);
    b = vec3(0.0, howCloseToEnd * b.y + dispY, clamp(2.0 * (howCloseToEnd * (1.1 - howCloseToEnd)), 0.0, 0.65 - 0.1 * type));
    m = 25.0;

    res = opU( res, vec2( sdCapsule(q, a, b).x - 0.01, m ) );

    // res.x *= 0.5; 

    return res;
}

vec2 map (vec3 p, float time) { 
    vec2 res = vec2(1e10, 0.0);

    // move the island around
    p = p - islandCenter();

    // island
    float z = 0.0;
    vec3 islandP = p + vec3(0,5.0, z+1.75);
    // glitch amt offset the size
    res = vec2( sdbEllipsoid (islandP, (1.0 - (sin(glitchAmtOne(time) + sin(1.8*time*glitchAmtOne(time))) + 0.75 * glitchAmtOne(time))) * vec3(7.6, 2.15, 7.0)), 10.0 );
    res = opSmoothU ( vec2( sdBox (islandP, vec3(30.,0.5,30.)), 10.0), res, 3.0 );

    // palm tree w/ bounding volumes
    float treeX = -1.75;
    float dToBound = 1e10;

    dToBound = sdfSphere( p + vec3(treeX - 0.45, 0.0, z), 4.8 );
    // res = opU ( vec2 (dToBound, 0.0), res );
    if (dToBound < 0.001) { 
        // trunk
        res = opSmoothU( vec2( sdTrunk( p + vec3(treeX, 2.5, z), 4.8, 0.13, time ), 15.0 ), res, 1.2 );

        // coconuts
        vec3 glitchDisp = glitchAmtTwo(time) * vec3(0.0, 0.0, -100.0);
        res = opU( vec2( sdfSphere(p + glitchDisp + vec3(treeX-0.65, -2.22, z), 0.08), 30.0), res );
        res = opU( vec2( sdfSphere(p - glitchDisp + vec3(treeX-0.85, -2.26, z), 0.08), 30.0), res );

        // branches
        dToBound = sdfSphere( p + vec3(treeX - 0.75, -2.22, z), 4.5 );
        // res = opU ( vec2 (dToBound, 0.0), res );
        if (dToBound < 0.001) { 
            res = mapBranch( p + vec3(treeX-0.75, -2.22, z), 2.2, res, 1.0, time ); //top
            res = mapBranch( p + vec3(treeX-0.75, -2.22, z), 2.2, res, 0.0, time ); //mid
            // res = mapBranch( p + vec3(treeX-0.75, -1.22, z), 1.2, res, 2.0 ); //bot
        }
    }
    
    
    return res;
}

vec3 calcNormal( in vec3 p, float time )
{
    const float eps = 0.0001; 
    const vec2 h = vec2(eps,0);
    return normalize( vec3(map(p+h.xyy, time).x - map(p-h.xyy, time).x,
                        map(p+h.yxy, time).x - map(p-h.yxy, time).x,
                        map(p+h.yyx, time).x - map(p-h.yyx, time).x ) );
}

float calcAO( in vec3 pos, in vec3 nor, float time )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.01 + 0.12*float(i)/4.0;
        float d = map( pos + h*nor, time ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
        if( occ>0.35 ) break;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}

float calcSoftshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k, float time )
{
    float res = 1.0;
    for( float t=mint; t<maxt; )
    {
        float h = map(ro + rd*t, time).x;
        if( h< mint )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

vec3 skyColor( in vec3 ro, in vec3 rd, in vec3 sunLig, float time )
{
    // 64 is a nice base
    // 30, 40 quite nice

    // figure out a way to make the sky not move in jitters
    // plus project the sky onto a sphere so it's more convincing
    // rd.x *= 10.0;
    float realTime = time;
    time = 127.0 + 0.1 * time;// + 5.432*(time * 0.0001 * glitchAmtFour(time)); // * sunsetAmt(time); //// + time * 0.00001;
    time = time;// + 0.25 * (0.5 + 0.5 * sin(0.545444*time));
    
    vec3 col = vec3(0.3,0.4,0.56)*0.3 - 0.3*rd.y;

    float t = (1000.0-ro.y)/rd.y;
    // float t = (100.0 - length(ro)) / length(rd);
    if( t>0.0 )
    {
        vec3 vPos = (ro + t * rd) - vec3(0,0,1000.0*time);
        vec3 pos = 100000.*rd.y*normalize(rd * t);

        float sunFacing = 0.1 + 0.9*dot(sunLig,rd);
        // float tNoise = 2.5 * turbulence( 0.00001*vec3(1000.+1.1*pos.x + 0.0*time, rd.y *0.1 + 0.0 * time, 1000. -0.6*pos.z + 0.0 * time ) );
        float tNoise = 2.18 * turbulence( 0.000063*vec3(0.844*vPos.x + 0.0051*time, 0.38*vPos.z, rd.y *0.1 + 0.006 * time ) );

        // get a 3d noise using the position, low frequency
        // float b = -2.6*snoise( 0.0001 * vec3( pos.x + 0.2* time, rd.y * 0.001 * time, 10000. - pos.z *0.62+ 0.2 * time ) );
        float b = -2.67*snoise( 0.00024 * ( time*0.00007*vec3(888.11, 1000.0, -888.8) + 1.8 * vec3( 0.009*vPos.x, rd.y, vPos.z * 0.0022 ) ) );
        
        // compose both noises
        float displacement = 2.2 * (0.8 * tNoise + 0.4 * b);

        float fbmNoise = 0.07*fbm( 0.033 * vec3(pos.x/100.+vPos.x + time*0.004, vPos.z/100. + time*0.004, vPos.z));

        // make a second noise with domain warped noise, b
        float displacementFBM = 7.5 * ( 0.2 * fbmNoise + 0.3 * b );

        // interpolate between the two 
        // float scalar = (0.5 + 0.5 * sin(realTime + sin(realTime))) / 3.0;
        // float scalar2 = (0.5 + 0.5 * cos(sin(realTime - cos(realTime)))) / 3.0;
        // float scalar3 = (0.5 + 0.5 * cos(realTime + cos(realTime))) / 3.0;

        // vPos.xz *= 0.00019;

        float vNoise = vnoiseOctaves(0.01*vec2(vPos.x * 0.089, vPos.z * 0.0121), 0.99, 0.994);
        vNoise = pow(vNoise, 2.0);
        float pNoise = (0.333) * tNoise + (0.333) * fbmNoise + (0.233) * b + 0.13 * vNoise;
        
        float ratio = smoothstep(16500.0, 30500., -vPos.z);
        ratio = 0.0;
        // pNoise = (1.0 - ratio) * (0.7 * pNoise + 0.3 * vNoise) + (ratio) * (0.5*vNoise + 0.5 * pNoise * vNoise);
        // pNoise = 0.35 * pNoise + 0.15 * vNoise;
        // pNoise *= pNoise;

        // col = palette( -0.1 * time + 0.25 * (displacement+ displacementFBM),
        //     vec3(0.5,0.5,0.5),
        //     vec3(0.5,0.5,0.5),
        //     vec3(2.0,1.0,0.0),
        //     vec3(0.5,0.2,0.25));
        vec2 uv = (ro+t*rd).xz;
        float cl = 1.0 * sin( (uv.y + 0.1 * sin(time) * abs(0.5 - uv.x)) * (0.0005 + 0.001 * time) );

        col = palette( pNoise * 1.766,
                    vec3(0.5,0.5,0.5),
                    vec3(0.5,0.5,0.5),
                    vec3(2.0,1.0,0.0),
                    vec3(0.5,0.2,0.05) );

        // col = mix( palette( pNoise + time * 0.25,
        //             vec3(0.5,0.5,0.5),
        //             vec3(0.5,0.5,0.5),
        //             vec3(2.0,1.0,0.0),
        //             vec3(0.45,0.15,0.05)), col, 0.01*rd.y);
        
        // col = palette( 1.666*pNoise,
        //             vec3(0.5,0.5,0.5),
        //             vec3(0.5,0.5,0.5),
        //             vec3(2.0,1.0,0.0),
        //             vec3(0.6,0.5,0.25) );

        // if (vPos.z < -13500.) { 
        //     col = vec3(1.0,1.0,1.0);
        // }
        // col.rg *= 0.9;
        col.r *= 0.81;
        col.g *= 1.1;
        col.b *= 1.05;
        // col.rgb *= 1.6 - (0.6 * sunsetAmt(realTime));
        // col.b *= 0.45 + (0.55 * sunsetAmt(realTime));

        // col *= 1.2 - 0.5 * sunsetAmt(realTime);
        // col = 0.3 * col + normalize(col) * pow(length(col), 0.2) * 0.8;

        col = mix( col, vec3(0.05,0.05,0.3), rd.y * 0.1 * cl);
        // for glitch:
        // col = mix( col, vec3(0.3,0.2,0.1), glitchAmtThree(uTime) * time * cl * uv.x * 0.1 );
        // col = mix(mix(sin(col*time), col, rd.y * time * sin(cos(time * uv.y - uv.x))), col, 1.0-glitchAmtOne(uTime));
    }
    
    float sd = pow( clamp( 0.04 + 0.96*dot(sunLig,rd), 0.0, 1.0 ), 4.0 );
    
    // over time:
    // set to -abs((60-55*sd))
    // col = mix( col, vec3(0.2,0.25,0.30)*0.7, exp(-40.0*rd.y) ) ;

    col = mix( col, vec3(1.0,0.30,0.05), sd*exp(-abs((16.0-(12.05*sunsetAmt(realTime)*sd)))*rd.y) ) ;
    col = mix( col, vec3(0.2,0.25,0.34)*0.7, exp((-40.*sunsetAmt(realTime)-10.0)*rd.y) ) ;

    return col;
}

float waterMap ( in vec2 p, float time ) { 
    vec2 pm = p * m2;
    float a = 1.0 * (pow(fbm ( vec3(0.099 * p, time * 0.22) ), 2.0));
    float b = 0.5 * abs ( fbm ( vec3(0.099 * p, uTime * 0.22) ) - 0.5 ) ;
    float c = pow(0.5 * ( fbm ( vec3(0.099 * p, uTime * 0.22) ) + 1.00 ), 1.4) ;
    return c;
}

vec3 getWaterNormal ( vec3 pos, float time ) { 
    float BUMP_DISTANCE = 650.0;
    float bump_scale = 1.0 - smoothstep(0.0, BUMP_DISTANCE, abs(pos.z));

    vec3 normal = vec3(0,0.5,0);
    float eps = 0.001;
    vec2 dx = vec2(eps, 0.0);
    vec2 dz = vec2(0.0, eps);
    normal.x = -bump_scale * 0.25 * (waterMap( pos.xz + dx, time ) - waterMap( pos.xz - dx, time )) / ( 2.0 * eps );
    normal.z = -bump_scale * 0.25 * (waterMap( pos.xz + dz, time ) - waterMap( pos.xz - dz, time )) / ( 2.0 * eps );
    normal = normalize(normal);
    return normal;
}

vec3 waterColor (in vec3 pos, vec3 rd, vec3 sunLig, float time) {
    vec3 texCol = 0.3 * texture(uWaterTexture, (1. / pow(2., 0.5)) * (vec2(0.7,0.3) + pos.zx * 0.009)).rgb;
    texCol += 0.4 * texture(uWaterTexture, (1. / pow(2., 0.5)) * (vec2(0.233,0.233) + pos.zx * 0.025)).rgb;
    texCol += 0.2 * texture(uWaterTexture, m2 * pos.xz * 0.05).rgb;
    texCol += 0.1 * texture(uWaterTexture, m2 * vec2(0.24,0.74) + pos.xz * 0.03).rgb;
    texCol *= 0.5 * 0.5;

    float TEXTURE_DISTANCE = 1000.0;
    float texture_scale = 1.0 - smoothstep(0.0, TEXTURE_DISTANCE, abs(pos.z));

    vec3 textureBlue = 0.5 * 0.5 * texture(uWaterTexture, vec2(0.5,0.5)).rgb;
    vec3 col = texture_scale * texCol + (1.0 - texture_scale) * textureBlue;

    col = vec3(0.5);

    vec3 normal = getWaterNormal(pos, time);
    float shadow = calcSoftshadow(pos, sunLig, 0.01, 100.0, 1.0, time); // from the island
    shadow = pow(shadow, 1.2);

    // manually add gradient around the island shadow because water had a strange cutoff on the shoreline
    shadow = min(1.0, (shadow) + smoothstep(13.0, 15.0, length(pos - (islandCenter() - vec3(8.0, 0.0, -13.0)))));

    float ndotr = dot(normal, rd);
    float fresnel = pow(1.0-abs(ndotr),5.);

    vec3 skyReflect = skyColor(pos, reflect(rd, normal), sunLig, time);
    skyReflect = skyReflect * skyReflect * (3.5 - 2.0 * skyReflect);

    col = shadow * (0.15 + .3 * sunsetAmt(time)) * col + (0.3 + 0.35 * sunsetAmt(time)) * fresnel * skyReflect;
    return col * (0.35 + 0.65 * dot(normal, sunLig));
}

vec3 groundNormal (in vec3 pos, out vec3 material, out vec2 K, out float percentWater, float time) { 
    vec3 pFromCenter = vec3(0.78,1.0,1.02)*pos - (vec3(0.0,0,-0.75) + islandCenter());
    float distFromCenter = length(pFromCenter);
    float waterLineDisp = 0.144 * cos( time + ( -pFromCenter.x - pFromCenter.y + pFromCenter.z) );
    float waterLineDisp2 = -0.232 * cos( cos( 4.326316* atan(5.34948*pFromCenter.x, 4.75333*pFromCenter.y) ) );
    float grassLineDisp = -0.732 * cos( cos( 5.326316* atan(4.34948*pFromCenter.x, 3.35333*pFromCenter.y) ) );
    float waterLine = 7.15 + waterLineDisp;
    float grassLine = 5.35 + grassLineDisp;

    percentWater = smoothstep(waterLine - 0.1, waterLine + 0.75, distFromCenter);
    float interp = smoothstep( grassLine - 0.5, grassLine + 1.0, distFromCenter );
    float darken = smoothstep( -4.3, -3.6, waterLineDisp2 + pFromCenter.y);
    // darken = pow(darken, 2.0);

    // interp b/w grass and sand
    material = ((1.0 - interp) * 1.1 * vec3(0.11, 0.21, 0.11)) + (interp * 0.8 * vec3(0.76, 0.69, 0.50)); 
    material *= (0.4 + 0.6 * darken);
    // material *= darken;

    if (distFromCenter > waterLine ) { 
        K = vec2(0.8, 16.0);
        material *= (1.0 - (0.2 + 0.6 * percentWater));
    } else { 
        K = vec2(0.3, 16.0);
    }

    // return normal of sand texture
    vec3 sandNor = normalize(texture(uSandTexture, pos.xz * 0.39).rgb);
    vec3 grassNor = normalize(texture(uGrassTexture, pos.xz * 0.05).rgb);
    vec3 gp = pos * 0.1;
    return (interp * sandNor) + ((1.0 - interp) * grassNor) + vec3(fbm(time*gp), fbm(time*gp + fbm(gp)), cos(fbm(time*0.01*gp))) * glitchAmtThree(time);
}

vec2 raycast (in vec3 ro, in vec3 rd, float time){
    vec2 res = vec2(-1.0,-1.0);

    float tmin = 0.001;
    float tmax = 100.0;
    
    // raytrace floor plane (the water)
    float tp1 = (-10.0-ro.y)/rd.y;
    if( tp1 > 0.0 )
    {
        tmax = min( tmax, tp1 );
        res = vec2( tp1, 0.0 );
    }

    float eps = 0.00015;
    float t = tmin;
    for( int i = 0; i < 128 && t < tmax; i++) {
        vec2 h = map( ro + rd*t, time );

        if( abs(h.x) < eps){
            res = vec2(t, h.y);
            break;
        } 

        t += h.x;
    }

    return res;
}

vec3 render(in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy, float time) { 
    // background - will probably be overwritten
    vec3 col = vec3(0.0);
    vec3 sunLig = vec3(6.0, 1.7, -6.0) + sunsetAmt(time) * vec3(2.0, -0.75, -2.0);

    col = skyColor(ro, rd, sunLig, time);

    vec2 res = raycast(ro,rd, time);
    float t = res.x;
    float m = res.y;

    // MATERIALS
    // ISLAND SAND/GRASS = 10
    // TREE TRUNK = 15
    // TREE BRANCH = 20
    // TREE LEAF = 25
    // COCONUT = 30
    // WATER = 0 

    vec3 material = vec3(0);
    vec2 K = vec2(0,1); // amount, power

    // water
    if ( m > -0.5 )
    {
        if (m < 0.5) { 
            vec3 pos = ro + rd*t;
            col = waterColor(pos, rd, sunLig, time);
        } 
        else if (m < 10.5) { 
            // island sand
            K = vec2(0.2, 20.0);
            material = vec3(0.22, 0.19, 0.16);
        }
        else if ( m < 15.5 ) { 
            // tree trunk
            K = vec2(0.25, 10.0);
            material = vec3(0.2, 0.18, 0.16);
            col = vec3(1);
        }
        else if ( m < 20.5 ) { 
            // tree branch
            K = vec2(0.25, 10.0);
            material = vec3(0.2, 0.18, 0.16);
        }
        else if ( m < 25.5 ) { 
            // tree leaf
            K = vec2(0.95, 8.0);
            material = 1.2 * vec3(0.12, 0.2, 0.065);
        }
        else if ( m < 30.5 ) { 
            // coconut
            K = vec2(0.25, 10.0);
            material = vec3(0.2, 0.09, 0.062);
        }
    }     

    if ( m > 9.5 ) { 
        // all the tree stuff
        vec3 pos = ro + rd*t;
        vec3 nor = vec3(0,0,0);
        float percentWater = 0.0;
        if ( m < 15.0 ) {
            nor = groundNormal(pos, material, K, percentWater, time); // uses textures etc
            nor = 0.6 * nor + 0.5 * calcNormal(pos, time);
        } else { 
            nor = calcNormal(pos, time);
        }
        
        percentWater = pow( percentWater, 1.0 / 2.33 );

        nor = (1.0 - percentWater) * nor + percentWater * getWaterNormal(pos, time);

        float sunsetScalar = sunsetAmt(time);

        float occ = calcAO(pos, nor, time);
        vec3 hal  = normalize(rd - sunLig);
        float fre = clamp(1.0 + dot(nor, rd), 0.0, 1.0);
        float shadowSharpness = smoothstep(-8.0, -4.0, pos.y);
        float shadow = calcSoftshadow(pos, sunLig, (shadowSharpness*0.0029) + 0.0001, 100.0, (shadowSharpness*13.75) + ((1.0 - sunsetScalar) * 0.75 + 1.45), time);
        float bou = clamp( 0.3-0.7*nor.y, 0.0, 1.0 );

        vec3 ref = reflect( rd, nor );
        
        vec3 lin = vec3(0);
        // sun
        {
            float dif = clamp(dot(nor, hal), 0.0, 1.0);
            dif *= shadow;
            float spe = K.x * pow( clamp(dot(nor, hal), 0.0, 1.0), K.y );
            spe *= dif;
            spe *= 0.04+0.96*pow(clamp(1.0-dot(hal,sunLig),0.0,1.0),5.0); // fresnel
            lin += material * (6.20 + 3. * sunsetScalar) * dif * vec3(2.1,1.1,0.6);
            lin += material * (30.70 + 30. * sunsetScalar) * spe * vec3(2.1,1.1,0.6);
        }
        // sky / reflections
        {
            float dif = occ * sqrt(clamp(0.2 + 0.8 * nor.y, 0.0, 1.0));
            vec3 skyCol = skyColor(pos, ref, sunLig, time);
            lin += material * 0.6 * dif * (0.55 * clamp(skyCol, 0.0, 0.6));

            float spe = smoothstep( -0.2, 0.2, ref.y );
            spe *= dif;
            spe *= occ * occ * K.x * 0.04 + 0.96 * pow( fre, 5.0 );
            spe *= shadow; 
            lin += 0.45 * spe * skyCol;
        }
        // ground / bounce light
        {
            lin += 4.0 * material * vec3(0.5,0.21,0.39) * bou * occ;
        }
        // back
        {
        	float dif = clamp( dot( nor, normalize(vec3(-0.5,0.0,-0.75))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
                  dif *= occ;
        	lin += material*0.55*dif*vec3(0.35,0.25,0.35);
        }
        // // sss
        {
            float dif = clamp(dot(nor, hal), 0.0, 1.0);
            lin += ((15.0 * sunsetScalar) + 15.0) * fre * fre * (0.2 + 0.8 * dif * occ * shadow) * material;
        }

        if (percentWater > 0.001) { 
            col = percentWater * waterColor(pos, rd, sunLig, time) + ( (1.0 - percentWater) * ((0.04 * material) + lin) ); // ambient + diffuse/spec
        } else { 
            col = (0.04 * material) + lin; // ambient + diffuse/spec
        }
    }
    
    return vec3( clamp(col, 0.0, 1.0) );
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr )
{
    vec3 cw = normalize(ta-ro);
    vec3 cp = vec3(sin(cr), cos(cr),0.0);
    vec3 cu = normalize( cross(cw,cp) );
    vec3 cv =          ( cross(cu,cw) );
    return mat3( cu, cv, cw );
}

void main() {
    // camera
    // the water is at -10
    vec3 ta = vec3( 3.5, -6.46, -1.0);
    vec3 ro = vec3( 3.5, -6.5, 0.0);

    // see it from the side
    // ro = vec3(6.5, -6.5, 0.0);
    // ta = vec3(0, -10.5, -40.0) - ro;

    // vec3 ta = vec3(0, -4.48, 0);
    // vec3 ro = vec3( 3.5 * cos (0.15 * uTime), -2.48, 3.5 * sin (0.15 * uTime));
    // vec3 ta = ro * 0.1;
    // vec3 ta = vec3(0.0,-5.5,-40.0) - ro;
    // ta.y = ro.y - (3.0 * (1.0 + sin (uTime * 0.075)));
    mat3 ca = setCamera(ro, ta, 0.0);

    float aspect = uResolution.x / uResolution.y;

    vec3 total = vec3(0.0);
#if AA>1
    for (int m=0; m < AA; m++)
    for (int n=0; n < AA; n++) { 
        vec2 o = (vec2(float(m), float(n)) / uResolution) / float(AA);
        vec2 p = vec2(aspect, 1.0) * ( (vUv+o) - vec2(0.5));
        float time = uTime + 10.0 * (1.0/48.0) * float(m * n * AA) / float(AA*AA);
#else
        vec2 p = vec2(aspect, 1.0) * (vUv - vec2(0.5));
        float time = uTime;
#endif

        // ray direction
        vec3 rd = ca * normalize( vec3(p, 2.2) );

        // ray differentials 
        vec2 px =  vec2(aspect, 1.0) * ( (vUv+vec2(1.0,0.0)) - vec2(0.5));
        vec2 py =  vec2(aspect, 1.0) * ( (vUv+vec2(0.0,1.0)) - vec2(0.5));
        vec3 rdx = ca * normalize( vec3(px, 2.5));
        vec3 rdy = ca * normalize( vec3(py, 2.5));

        vec3 color = render( ro, rd, rdx, rdy, time );

        color *= 1.0 - 0.385 * dot(p, p); // vignette
        color = pow(color, vec3(0.5245));

        total += color;
#if AA>1
    }
    total /= float(AA*AA);
#endif
    
    // color correction / post processing
    total = min(total, 1.0);

    // s-curve contrast
    // float n = 1.5;
    // total.x = pow(total.x, n) / (pow(total.x, n ) + pow( 1.0 - total.x, n ));
    // total.y = pow(total.y, n) / (pow(total.y, n ) + pow( 1.0 - total.y, n ));
    // total.z = pow(total.z, n) / (pow(total.z, n ) + pow( 1.0 - total.z, n ));

    // little bit more red
    // float d = total.r * 0.121;
    // total.r += d;
    // total.g -= d;

    total = min(total, 1.0);

    o_FragColor = vec4( total, 1.0 );
}