#version 300 es
#define AA 1

precision highp float;

in vec2 vUv;
uniform vec2 uResolution;
uniform float uTime;
uniform sampler2D uMetalColor;
uniform sampler2D uMetalRough;
uniform sampler2D uMetalNormal;
uniform sampler2D uPlasticColor;
uniform sampler2D uPlasticRough;
uniform sampler2D uPlasticNormal;

out vec4 o_FragColor;

float signalOne(float time) {
    float length = 60.0; 
    float a = mod(time, length);
    float ap = 30.0*(0.5 - 0.5*cos( ( 2.0*3.141593653 / length ) * a ));
    return ap;
}

float signalTwo(float time) { 
    float length = 60.0; 
    float a = mod(time, length);
    float ap = 30.0 * (0.5 + 0.5*cos( ( 2.0*3.141593653 / length ) * a + 3.141592653/4.0 ));
    return ap;
}

float sizeSignal(float time) { 
    float length = 60.0; 
    float a = mod(time, length);
    float ap = clamp(0.475 + 0.5 * sin( (3.0*3.141592653/2.0) + ( 2.0*3.141593653 / length ) * a ), 0.0, 1.0);
    return ap;
}

float introSignalTime(float time) { 
    float introLength = 3.0;  
    float k = time < 0.5 * introLength ? 0.0 : 1.0;
    float v = time < introLength ? introLength * pow(time/introLength - 0.5, 2.0) : time - 0.75*introLength;
    return k * v;
}

float introSmokeSignal(float time) { 
    float introLength = 3.0;  
    // clamp(0.475 + 0.5 * sin( (3.0*3.141592653/2.0) + ( 2.0*3.141593653 / length ) * a ), 0.0, 1.0);
    return smoothstep(introLength - 1.0, introLength, time);
}

/*
 
                                                    
                                                    
 `7MN.   `7MF' .g8""8q. `7MMF' .M"""bgd `7MM"""YMM  
   MMN.    M .dP'    `YM. MM  ,MI    "Y   MM    `7  
   M YMb   M dM'      `MM MM  `MMb.       MM   d    
   M  `MN. M MM        MM MM    `YMMNq.   MMmmMM    
   M   `MM.M MM.      ,MP MM  .     `MM   MM   Y  , 
   M     YMM `Mb.    ,dP' MM  Mb     dM   MM     ,M 
 .JML.    YM   `"bmmd"' .JMML.P"Ybmmd"  .JMMmmmmMMM 
                                                    
                                                    
 
*/
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

vec3 hash3( vec2 p ){
    vec3 q = vec3( dot(p,vec2(127.1,311.7)), 
				   dot(p,vec2(269.5,183.3)), 
				   dot(p,vec2(419.2,371.9)) );
	return fract(sin(q)*43758.5453);
}

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

/*
 
                                       
                                       
  .M"""bgd `7MM"""Yb. `7MM"""YMM       
 ,MI    "Y   MM    `Yb. MM    `7       
 `MMb.       MM     `Mb MM   d ,pP"Ybd 
   `YMMNq.   MM      MM MM""MM 8I   `" 
 .     `MM   MM     ,MP MM   Y `YMMMa. 
 Mb     dM   MM    ,dP' MM     L.   I8 
 P"Ybmmd"  .JMMmmmdP' .JMML.   M9mmmP' 
                                       
                                       
 
*/

vec2 opU( vec2 d1, vec2 d2 )
{
	return (d1.x<d2.x) ? d1 : d2;
}

vec2 opSmoothU( vec2 d1, vec2 d2, float k) 
{ 
    float colorSmoothness = k*.075;
    float interpo = clamp( 0.5 + 0.5 * (d1.x - d2.x) / colorSmoothness, 0.0, 1.0 );
    float h = max( k - abs(d1.x - d2.x), 0.0) / k;
    float diff = h*h*h*k*(1.0/6.0);
    return vec2( min(d1.x, d2.x) - diff,
                 mix(d1.y, d2.y, interpo) - k * interpo * ( interpo - 1.0) );
}

float opSmoothSubtraction( float d1, float d2, float k ) {
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h); }

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

float sdCappedCone(vec3 p, vec3 a, vec3 b, float ra, float rb)
{
    float rba  = rb-ra;
    float baba = dot(b-a,b-a);
    float papa = dot(p-a,p-a);
    float paba = dot(p-a,b-a)/baba;
    float x = sqrt( papa - paba*paba*baba );
    float cax = max(0.0,x-((paba<0.5)?ra:rb));
    float cay = abs(paba-0.5)-0.5;
    float k = rba*rba + baba;
    float f = clamp( (rba*(x-ra)+paba*baba)/k, 0.0, 1.0 );
    float cbx = x-ra - f*rba;
    float cby = paba - f;
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s*sqrt( min(cax*cax + cay*cay*baba,
                       cbx*cbx + cby*cby*baba) );
}

// vertical
float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

/*
 
                                                                                      
                                                                                      
 `7MMM.     ,MMF'      db      `7MM"""Mq.`7MM"""Mq.`7MMF'`7MN.   `7MF' .g8"""bgd      
   MMMb    dPMM       ;MM:       MM   `MM. MM   `MM. MM    MMN.    M .dP'     `M      
   M YM   ,M MM      ,V^MM.      MM   ,M9  MM   ,M9  MM    M YMb   M dM'       `      
   M  Mb  M' MM     ,M  `MM      MMmmdM9   MMmmdM9   MM    M  `MN. M MM               
   M  YM.P'  MM     AbmmmqMA     MM        MM        MM    M   `MM.M MM.    `7MMF'    
   M  `YM'   MM    A'     VML    MM        MM        MM    M     YMM `Mb.     MM      
 .JML. `'  .JMML..AMA.   .AMMA..JMML.    .JMML.    .JMML..JML.    YM   `"bmmmdPY      
                                                                                      
                                                                                      
 
*/

vec3 rotatePoint(vec3 p, vec3 n, float theta) { 
    vec4 q = vec4(cos(theta / 2.0), sin (theta / 2.0) * n);
    vec3 temp = cross(q.xyz, p) + q.w * p;
    vec3 rotated = p + 2.0*cross(q.xyz, temp);
    return rotated;
}

vec2 mapPot (vec3 p, float time) { 
    // p = rotatePoint(p, vec3(0,1,0), -3.14159/6.0);
    float rotateValue = -3.14159/8.0;

    p = rotatePoint(p, vec3(0,1,0), -3.14159/8.0 + 0.05*time*0.85*0.65*sin(-.89));
    p.y = -p.y;

    // material: 15 = metal body, 16 = coarse plastic for handles
    // body - potentially make it a clay glaze color (for object-ness)
    vec2 res = vec2(sdfSphere(p, 1.0), 15.0);
    res.x = max(res.x, -p.y);
    res.x -= 0.35;
    res.x = opSmoothSubtraction(sdfSphere(p - vec3(0,1.15,0), 0.6), res.x, 0.15);

    // base (little torus)
    res = opU(res, vec2(sdTorus(p + vec3(0,0.375,0), vec2(1.295, 0.05)),15.0));

    // spout
    res = opSmoothU(res, vec2(sdCappedCone(p + vec3(1.0,0.38,0.0), vec3(0,1,0), vec3(-0.5, 1.38, 0.0), 0.3, 0.19), 14.0),0.1);

    // lid (base)
    float d = sdfSphere(p - vec3(0, 0.345, 0), 1.035);
    res = opU(res, vec2(max(d, 1.0 - p.y), 15.0));

    // torus around lid
    res = opSmoothU(res, vec2(sdTorus(p - vec3(0,1.18,0), vec2(0.61, 0.02)),15.0), 0.03);

    // lid handle
    d = sdCappedCylinder( p - vec3(0, 1.1, 0), 0.1, 0.4 );
    res = opU(res, vec2(d, 14.0) );

    d = sdfSphere( p - vec3(0, 1.6, 0), 0.165 );
    res = opSmoothU(res, vec2(d, 16.0), 0.01 );

    // main handle
    d = sdCapsule(p - vec3(1.1, 0.5, 0.2), vec3(0,0,0), vec3(0.8*sin(3.141592653/3.0 + 1.05*p.y), 1.85, 0)).x - 0.05;
    float d2 = sdCapsule(p - vec3(1.1, 0.3, 0.2), vec3(0,0,0), 0.15*vec3(0.85, 3.15, 0)).x - 0.03;
    
    d = opSmoothU( vec2(d, 15.0), vec2(d2, 15.0), 0.2 ).x;

    res = opU(res, vec2(d, 14.0));

    d = sdCapsule(p - vec3(1.1, 0.5, -0.2), vec3(0,0,0), vec3(0.8*sin(3.141592653/3.0 + 1.05*p.y), 1.85, 0)).x - 0.05;
    d2 = sdCapsule(p - vec3(1.1, 0.3, -0.2), vec3(0,0,0), 0.15*vec3(0.85, 3.15, 0)).x - 0.03;
    
    d = opSmoothU( vec2(d, 15.0), vec2(d2, 15.0), 0.2 ).x;
    
    res = opU(res, vec2(d, 14.0));

    d = sdRoundBox(p - vec3(0.0, 1.825 + 0.75*cos(p.x), 0.0), vec3(0.75, 0.025, 0.25), 0.085);
    res = opSmoothU(res, vec2(d, 16.5), 0.01);

    // noise 
    vec3 p2 = p;

    float signal = 1.0;
    float localTime = uTime * 3.45;
    // if (signal > 0.00001) { 
        // time *= signal;
    float ss = 0.8 + 0.2*1.0;
    // ss += 0.15;
    ss *= signal;
    // ss *= 2.0;
    // ss *= introSmokeSignal(uTime);
    float NoiseScale = .79; // 1.092382;
    NoiseScale = 0.914;

        
        // if(signal < 0.001) { 
        //     NoiseScale = 0.0;
        // } else { 
        //     NoiseScale = NoiseScale * 1.2 - (NoiseScale * 0.2 * signal);
        // }
        // NoiseScale = 1.2 - NoiseScale * 
        // NoiseScale = max(0.5, signal);

        // ??? color balls? - a nice blue?
        // NoiseScale = 1.0 * max(0.1,ss);

        // ???
        // ss = 0.8;
        // NoiseScale = max(0.15,cos(sin(0.01*time + cos(time*0.077))));

        // ss = 1.0;
    float NoiseIsoline = (0.04 + 0.23839919) * (smoothstep(3.9*ss, 2.2*ss, length(vec3(0,-0.5,0) + p)));
        // float NoiseIsoline = 0.419 * (smoothstep(6.0, 2.0, p.y));
        // float NoiseIsoline = 0.419;
    p = signal*(p / NoiseScale) - localTime * vec3(0,0.15,0);// - localTime;// * vec3(0,0.1,0);
        // float noise = NoiseScale * (fbm(p) - NoiseIsoline);

    float noise = NoiseScale * (fbm(p + 0.422*fbm( p + 0.05*vec3(1.99821,2.003,1.552)*localTime )) - NoiseIsoline);
        // p.x -= noise;

        // float d = sdCapsule(p, vec3(-2.0, 0.0, 0.0), vec3(-2.0,1.0,0.0)).x - ( 0.5 + 0.5 * ( sizeSignal(time)) ); // capsule // two capsules alternating?
        // return opSmoothU(res, vec2(noise, 15.0), 0.6); // return with metal texture
    res = opSmoothU(res, vec2(noise, 40.0), 0.001 + signal*(0.55)); // return with metal texture
        
    // works -- but not so good looking
    // {
    //     float signal = introSmokeSignal(uTime);
    //     float ss = sizeSignal(time);
    //     // ss = 1.0;
    //     signal = 1.0;
    //     // ss *= signal;

    //     float NoiseScale = 0.755 ; // 1.092382;
    //     // NoiseScale = 0.914;
    //     float NoiseIsoline = (0.04 + 0.2839919) * (smoothstep(-1.0, 0.0,p.z)) * (smoothstep(1.0, 0.0,p.z));
    //     NoiseIsoline *= (smoothstep(7.5, 6.1, length(vec3(0,0.0,0) + p)));
    //     // NoiseIsoline *= smoothstep
    //     p = signal*(p / NoiseScale) - time * vec3(0,0.05,0);

    //     float noise = NoiseScale * (fbm(0.9*p + 0.422*fbm( 0.9*p + 0.05*vec3(1.99821,2.003,1.552)*time )) - NoiseIsoline);

    //     res = opSmoothU(res, vec2(noise, 40.0), 0.8);
    // }

    // res.x = opSmoothSubtraction(sdCappedCone(p2 + vec3(1.0,0.38,0.0), vec3(0,1,0), 1.1*vec3(-0.5, 1.28, 0.0), 0.275, 0.165), res.x, 0.01);

    return res;
}


vec2 mapOutlineBox (vec3 p, float time) { 
    vec2 res = vec2(1e10, -1.0);
    // define the box shape
    // rotate xy then xz
    vec3 new_p = p + vec3(0.0, -2.65, 20.0);
    float new_time = time*0.025;
    float new_time_2 = new_time * 0.88;
    new_p.xy = vec2(new_p.x*cos(new_time) - new_p.y*sin(new_time), new_p.x*sin(new_time) + new_p.y*cos(new_time));
    new_p.xz = vec2(new_p.x*cos(new_time_2) - new_p.z*sin(new_time_2), new_p.x*sin(new_time_2) + new_p.z*cos(new_time_2));
    float dToBound = sdBox(new_p, 1.125*vec3(1.75, 1.75, 1.425));
    // float dToBound = sdfSphere(p + vec3(0.0, -2.5, 20.0), 2.80);
    res = opU(res, vec2(dToBound, 0.0));

    return res;
}

vec2 mapBoxScene (vec3 p, float time) { 
    vec2 res = vec2(1e10, -1.0);

    res = opU(res, mapPot(p + vec3(0.0, -1.65, 20.0), time));
    
    return res;
}

vec2 raycastOutlineBox (in vec3 ro, in vec3 rd, float time){
    vec2 res = vec2(1e10, -1.0);

    float tmin = 0.001;
    float tmax = 100.0;

    float eps = 0.00015;
    float t = tmin;
    for( int i = 0; i < 256 && t < tmax; i++) {
        vec2 h = mapOutlineBox( ro + rd*t, time );

        if( abs(h.x) < eps){
            res = vec2(t, h.y);
            break;
        } 

        t += h.x;
    }

    return res;
}

vec2 raycastBoxScene(in vec3 ro, in vec3 rd, float time){
    vec2 res = vec2(1e10,-1.0);

    float tmin = 0.001;
    float tmax = 100.0;

    float eps = 0.00015;
    float t = tmin;
    for( int i = 0; i < 800 && t < tmax; i++) {
        vec2 h = mapBoxScene( ro + rd*t, time );

        if( abs(h.x) < eps){
            res = vec2(t, h.y);
            break;
        } 

        t += h.x * 0.6;
    }

    return res;
}

/*
 
                                                                                             
                                                                                             
 `7MMF'   `7MF'MMP""MM""YMM `7MMF'`7MMF'      `7MMF'MMP""MM""YMM `7MMF'`7MM"""YMM   .M"""bgd 
   MM       M  P'   MM   `7   MM    MM          MM  P'   MM   `7   MM    MM    `7  ,MI    "Y 
   MM       M       MM        MM    MM          MM       MM        MM    MM   d    `MMb.     
   MM       M       MM        MM    MM          MM       MM        MM    MMmmMM      `YMMNq. 
   MM       M       MM        MM    MM      ,   MM       MM        MM    MM   Y  , .     `MM 
   YM.     ,M       MM        MM    MM     ,M   MM       MM        MM    MM     ,M Mb     dM 
    `bmmmmd"'     .JMML.    .JMML..JMMmmmmMMM .JMML.   .JMML.    .JMML..JMMmmmmMMM P"Ybmmd"  
                                                                                             
                                                                                             
 
*/
vec3 palette( in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d )
{  return a + b*cos( 6.28318*(c*t+d) ); }

vec3 calcNormal( in vec3 p, float time )
{
    const float eps = 0.001; 
    const vec2 h = vec2(eps,0);
    return normalize( vec3(mapOutlineBox(p+h.xyy, time).x - mapOutlineBox(p-h.xyy, time).x,
                        mapOutlineBox(p+h.yxy, time).x - mapOutlineBox(p-h.yxy, time).x,
                        mapOutlineBox(p+h.yyx, time).x - mapOutlineBox(p-h.yyx, time).x ) );
}

/*
 
                                                                                                            
                                                                                                            
 `7MM"""Mq.  `7MM"""YMM  `7MN.   `7MF'`7MM"""Yb. `7MM"""YMM  `7MM"""Mq.  `7MMF'`7MN.   `7MF' .g8"""bgd      
   MM   `MM.   MM    `7    MMN.    M    MM    `Yb. MM    `7    MM   `MM.   MM    MMN.    M .dP'     `M      
   MM   ,M9    MM   d      M YMb   M    MM     `Mb MM   d      MM   ,M9    MM    M YMb   M dM'       `      
   MMmmdM9     MMmmMM      M  `MN. M    MM      MM MMmmMM      MMmmdM9     MM    M  `MN. M MM               
   MM  YM.     MM   Y  ,   M   `MM.M    MM     ,MP MM   Y  ,   MM  YM.     MM    M   `MM.M MM.    `7MMF'    
   MM   `Mb.   MM     ,M   M     YMM    MM    ,dP' MM     ,M   MM   `Mb.   MM    M     YMM `Mb.     MM      
 .JMML. .JMM..JMMmmmmMMM .JML.    YM  .JMMmmmdP' .JMMmmmmMMM .JMML. .JMM..JMML..JML.    YM   `"bmmmdPY      
                                                                                                            
                                                                                                            
    lighting, raymarching. 
*/

vec3 render(in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy, float time) { 
    vec3 col = vec3(1.0); //background color

    vec2 res = raycastOutlineBox(ro,rd,time);
    float t = res.x;
    float m = res.y;


    // if m == 0.0 (we hit the box, i.e.)
    if (m > -1.0) { 
        vec3 new_ro = ro+rd*t;
        res = raycastBoxScene(new_ro,rd,time);

        t = res.x;
        m = res.y;

        vec3 pos = new_ro + rd*t;

        if (m < 1.0) { 
            // didn't hit anything -- should return vec3(1.0);
            return vec3(1.0);
        } 
        else {
            vec3 nor = calcNormal(new_ro, time);
            // we hit something 
            vec3 lightOne = normalize(vec3(0.1, -0.8, 0.5));
            vec3 lightTwo = normalize(vec3(-0.2, 0.5, -0.23));

            vec3 col = 0.05*vec3(0.2);
            vec3 clrOne = palette(length(0.55*new_ro.xyz + time*0.04*vec3(-0.6,1,0.22)), vec3(0.5), vec3(0.5),	vec3(1.0, 0.7, 0.4), vec3(0.00, 0.15, 0.2));
            vec3 clrTwo = palette(length(pos.xyz*0.4 + time*0.00888*vec3(0.33,.111,2.22)), vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.00, 0.1, 0.2));
            vec3 clrThree = palette(length(pos.z + time*0.00132), vec3(0.5, 0.5, 0.5), vec3(0.5, 0.5, 0.5), vec3(1.0, 1.0, 1.0), vec3(0.30, 0.20, 0.20));
            // normal --
            // col += normalize(vec3(0.5, 0.633, 0.9)) * 0.45 * clamp(dot(lightOne,nor), 0.0, 1.0);
            // col += normalize(vec3(1.0, 0.6, 0.5)) * 0.45 * clamp(dot(lightTwo,nor), 0.0, 1.0);

            // fancy? 
            vec3 sunsetClr = normalize(vec3(1.0, 0.6, 0.5));
            col += (0.6*clrOne + 0.5*sunsetClr + 0.12 * clrThree) * 0.91 * clamp(dot(lightOne,nor), 0.0, 1.0);
            col += (0.5*sunsetClr + clrTwo * 0.8) * clamp(dot(lightTwo,nor), 0.0, 1.0);
            col *= 0.5;
            return clamp(col, 0.0, 1.0);
            // return vec3(0.01) * dot(nor, normalize(vec3(0.1, -0.8, 0.5)));
        } 

    } else {
        return vec3(0.0);
    }
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
    vec3 ro = vec3( 0.0, 4.5, 1.5);
    vec3 ta = vec3( 0.0, 4.41, 0.5);
    // vec3 ta = vec3(0.0,0.0,40.0) - ro;

    mat3 ca = setCamera(ro, ta, 0.0);

    float aspect = uResolution.x / uResolution.y;

    vec3 total = vec3(0.0);
#if AA>1
    for (int m=0; m < AA; m++)
    for (int n=0; n < AA; n++) { 
        vec2 o = (vec2(float(m), float(n)) / uResolution) / float(AA);
        vec2 p = vec2(aspect, 1.0) * ( (vUv+o) - vec2(0.5));
        float time = uTime + 1.0 * (150.0/4.0) * float(m * n * AA) / float(AA*AA);
#else
        vec2 p = vec2(aspect, 1.0) * (vUv - vec2(0.5));
        float time = uTime * 50.0;
#endif

        // slow 
        time *= 0.2;

        // fast
        // time *= 2.5;

        // ray direction
        vec3 rd = ca * normalize( vec3(p, 2.2) );

        // ray differentials 
        vec2 px =  vec2(aspect, 1.0) * ( (vUv+vec2(1.0,0.0)) - vec2(0.5));
        vec2 py =  vec2(aspect, 1.0) * ( (vUv+vec2(0.0,1.0)) - vec2(0.5));
        vec3 rdx = ca * normalize( vec3(px, 2.5));
        vec3 rdy = ca * normalize( vec3(py, 2.5));

        vec3 color = render( ro, rd, rdx, rdy, time );

        color = pow(color, vec3(0.4545));

        total += color;
#if AA>1
    }
    total /= float(AA*AA);
#endif
    
    // color correction / post processing
    total = min(total, 1.0);

    o_FragColor = vec4( total, 1.0 );
}