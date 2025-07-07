#version 300 es
#define AA 2

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
    float introLength = 10.0;  
    float k = time < 0.5 * introLength ? 0.0 : 1.0;
    float v = time < introLength ? introLength * pow(time/introLength - 0.5, 2.0) : time - 0.75*introLength;
    return k * v;
}

float introSmokeSignal(float time) { 
    float introLength = 10.0;  
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

    p = rotatePoint(p, vec3(0,1,0), -3.14159/8.0 + introSignalTime(uTime)*1.7*7.8*sin(-.89));
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
    // if i subtract the inside of the cone last, does that fix it?
    // res.x = opSmoothSubtraction(sdCappedCone(p + vec3(1.0,0.38,0.0), vec3(0,1,0), vec3(-0.5, 1.38, 0.0), 0.28, 0.17), res.x, 0.01);

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

    // time *= 0.2;
    // time *= 0.5;
    // float noise = (1.3 * (signalOne(time)*0.6)) * clamp(fbm( p*2.0 ), -1.0, 1.0) * (1.0 - smoothstep(0.0, 6.45,-p.x+p.y));
    // p -= vec3(0.43, 0.23, -0.35) * 0.5 * noise;

    // noise = (1.3 * (signalTwo(time)*0.6)) * clamp(fbm( p2*1.333 ), -1.0, 1.0) * (1.0 - smoothstep(0.0, 6.45,-p2.x+p2.y));
    // p2 -= vec3(0.43, 0.23, -0.35) * 0.5 * noise;

    // d = sdCapsule(p, vec3(-0.6, 0.0, 0.0), vec3(0.6,0.6,0.0)).x - (0.60*(0.6 + 0.4*sizeSignal(time))); // capsule // two capsules alternating?
    // res = opSmoothU(res, vec2(d, 15.0), 0.1);

    // ver 1: full pot
    {
        
        // pos -= vec3(0,4.5,0);
        // nice neutral/fake smoke:
        
        float signal = introSmokeSignal(uTime);
        // if (signal > 0.00001) { 
            // time *= signal;
            float ss = sizeSignal(time);
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
            // NoiseScale = max(0.5, introSmokeSignal(uTime));

            // ??? color balls? - a nice blue?
            // NoiseScale = 1.0 * max(0.1,ss);

            // ???
            // ss = 0.8;
            // NoiseScale = max(0.15,cos(sin(0.01*time + cos(time*0.077))));

            // ss = 1.0;
            float NoiseIsoline = (0.04 + 0.23839919) * (smoothstep(3.9*ss, 0.75*ss, length(vec3(0,-2.25,0) + p)));
            // float NoiseIsoline = 0.419 * (smoothstep(6.0, 2.0, p.y));
            // float NoiseIsoline = 0.419;
            p = signal*(p / NoiseScale) - time * vec3(0,0.15,0);// - time;// * vec3(0,0.1,0);
            // float noise = NoiseScale * (fbm(p) - NoiseIsoline);

            float noise = NoiseScale * (fbm(p + 0.422*fbm( p + 0.05*vec3(1.99821,2.003,1.552)*time )) - NoiseIsoline);
            // p.x -= noise;

            // float d = sdCapsule(p, vec3(-2.0, 0.0, 0.0), vec3(-2.0,1.0,0.0)).x - ( 0.5 + 0.5 * ( sizeSignal(time)) ); // capsule // two capsules alternating?
            // return opSmoothU(res, vec2(noise, 15.0), 0.6); // return with metal texture
            res = opSmoothU(res, vec2(noise, 40.0), 0.001 + signal*(0.55)); // return with metal texture
                // add an extra .4-.6 to the K over the course of the song?? vary by ss???
        // }
        
    }

    // ver 2: line towards the side of the pot (original intention)
    {
        // float NoiseScale = 0.5391;
        // // pos -= vec3(0,4.5,0);
        // float ss = 1.0;//sizeSignal(time);
        // vec3 flowDir = normalize(vec3(2.5,4.5,-8));
        // float distToLine = sdCapsule(p, flowDir, 5.0*flowDir).x;
        // float NoiseIsoline = (0.0 + 0.49419 * (smoothstep(4.9*ss, 2.35*ss, length(vec3(0,-1,0) + p))) * (smoothstep(1.0*ss, 0.1*ss, distToLine)));
        // // float NoiseIsoline = 0.419 * (smoothstep(6.0, 2.0, p.y));
        // // float NoiseIsoline = 0.419;
        // p = (p / NoiseScale) - time * 0.2*flowDir;// - time;// * vec3(0,0.1,0);
        // // float noise = NoiseScale * (fbm(p) - NoiseIsoline);

        // float noise = NoiseScale * (fbm(p + 0.722*fbm( p + 0.3*flowDir*vec3(1.99821,2.003,1.552)*time )) - NoiseIsoline);
        // // p.x -= noise;

        // // float d = sdCapsule(p, vec3(-2.0, 0.0, 0.0), vec3(-2.0,1.0,0.0)).x - ( 0.5 + 0.5 * ( sizeSignal(time)) ); // capsule // two capsules alternating?
        // return opSmoothU(res, vec2(noise, 15.0), 0.2);
    }

    // d = sdCapsule(p2, vec3(-0.6, 0.0, 0.0), vec3(0.6,0.6,0.0)).x - (0.75*(0.6 + 0.4*(1.0 - sizeSignal(time)))); // capsule // two capsules alternating?
    // res = opSmoothU(res, vec2(d, 15.0), 0.1);

    res.x = opSmoothSubtraction(sdCappedCone(p2 + vec3(1.0,0.38,0.0), vec3(0,1,0), 1.1*vec3(-0.5, 1.28, 0.0), 0.275, 0.165), res.x, 0.01);

    // put a ball or plate inside the spout to block the inner sphere?
    res = opSmoothU(res, vec2(sdfSphere(p2 + vec3(1.0, -0.65, 0.0), 0.3), 50.0), 0.25);
    
    return res;
}

vec2 map (vec3 p, float time) { 
    vec2 res = vec2(1e10, 0.0);
    float boxDistance = 5.0;

    // walls, ground
    // res = vec2(sdBox(rotatePoint(p + vec3(boxDistance, 0.0, 20.0), vec3(0,1,0), 3.141592653/4.0), vec3(0.5, 10.0, 20.0)), 25.0);
    // res = opU(res, vec2(sdBox(rotatePoint(p + vec3(-boxDistance, 0.0, 20.0), vec3(0,1,0), -3.141592653/4.0), vec3(0.5, 10.0, 20.0)), 26.0));
    // res = opU(res, vec2(sdBox(p + vec3(0, 0.0, 0.0), vec3(10.0, .1, 25.0)), 30.0));
    
    // kettle
    // time*=2.5;
    // time*=0.1;
    // radius of 4 around vec3(0,-1,0) + whatever is given as pot position
    // p = rotatePoint(p, vec3(0,1,0), -3.14159/8.0 + 0.65*sin(0.12*time));
    // p.y = -p.y;

    // float dToBound = sdfSphere(p + vec3(0.0, -1.15, 6.5), 8.80);
    float dToBound = sdfSphere(p + vec3(0.0, -3.15, 15.0), 3.80);
    // res = opU(res, vec2(dToBound, 30.0));
    if (dToBound > 0.001) {
        res = opU(res, vec2(dToBound, 30.0));
    } else {  
        res = opU(res, mapPot(p + vec3(0.0, -1.15, 15.0), time));
    }
    
    return res;
}

vec2 raycast (in vec3 ro, in vec3 rd, float time){
    vec2 res = vec2(-1.0,-1.0);

    float tmin = 0.001;
    float tmax = 100.0;
    
    // raytrace floor plane
    // float tp1 = (-ro.y)/rd.y;
    // if( tp1 > 0.0 )
    // {
    //     tmax = min( tmax, tp1 );
    //     res = vec2( tp1, 0.0 );
    // }

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


vec3 skyColor( in vec3 ro, in vec3 rd, in vec3 sunLig, float time )
{
    vec3 col = 0.95*vec3(.948,.9529,.9373);
    // float t = (20.0-ro.y)/rd.y;
    // if( t>0.0 )
    // {
    //     vec3 pos = ro + t * rd;
    //     pos = pos * 0.003; 
    //     col = vec3(vnoiseOctaves(vec2(0.0, -uTime/50.) + pos.xz, 1.0, 0.5));
    //     col = 0.1 + 0.6 * col + 0.3 * vec3(fbm(0.2*pos + vec3(-uTime/100., -uTime / 75., uTime/25.)));
    //     col *= (1.0 / (pos.z*pos.z));
    // }
    
    return col;
}

vec3 render(in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy, float time) { 
    vec3 col = vec3(0.0); //background color

    // col = 0.95*vec3(.948,.9529,.9373);

    if (rd.x < -0.13){
        return col;
    } else if (rd.x > 0.13) { 
        return col;
    } else if (rd.y < -0.26) { 
        return col;
    }

    vec2 res = raycast(ro,rd, time);
    float t = res.x;
    float m = res.y;

    vec3 pos = ro + rd*t;
    vec3 nor = calcNormal(pos, time);

    float angleBetweenXY = asin(nor.x) / 3.141592653;
    float angleBetweenXZ = asin(nor.y) / 3.141592653;
    vec2 texCoords = 3.25 * vec2(angleBetweenXY, angleBetweenXZ); // close enough for sphere-like object

    texCoords.x += introSignalTime(uTime)*1.7*7.8*sin(-.89);
    // 0.65*sin(0.12*time)

    vec3 material = vec3(0);
    vec2 K = vec2(0,1); // amount, power for specular
    vec3 f0 = vec3(0.05);
    float rou = 1.0; // roughness
    vec3 hNor = vec3(0); // microfacet normal - is getting it from texture better than 'hal'? 
    float a = 1.0;

    // MATERIALS
    // 25 / 26 - walls
    // 30 - ground

    // 14 - noise (silicon-ish)
    // kettle - metal
    if (m < 16.) {
        if (m < 15.) { 
            // spout - add detail 
            angleBetweenXY = asin(nor.x) / 3.141592653;
            angleBetweenXZ = asin(nor.y) / 3.141592653;
            texCoords = 0.8 * vec2(angleBetweenXY, angleBetweenXZ);
            material = 0.6 *texture(uMetalColor, 0.7 * pos.xy + 0.7*texCoords).rgb
                    +  0.4 *texture(uMetalColor, 0.62 * pos.yz + 0.6*texCoords).rgb;
            hNor = texture(uMetalNormal, 0.82 * pos.xy + 0.65*texCoords).rgb;
            rou = texture(uMetalRough, 0.82 * pos.xy + 0.65*texCoords).r;
        } else { 
            material = 0.65*texture(uMetalColor, 0.35 * pos.xz + texCoords).rgb
                +  0.35*texture(uMetalColor, 0.212 * pos.xy + texCoords).rgb;
            hNor = texture(uMetalNormal, 0.7*texCoords).rgb;
            rou = texture(uMetalRough, 0.7*texCoords).r;
        }

        K = vec2(0.85, 24.0);
        f0 = vec3(.972, .961, .915);
        a = 16.0 * (0.65 + 0.35 * (1.0 - rou));
    // kettle - plastic
    } else if (m < 17.) { 
        K = vec2(0.65, 12.0);
        material = 0.7*texture(uPlasticColor, pos.xy * 1.1).rgb
                +  0.3*texture(uPlasticColor, pos.yz * 1.25).rgb;
        hNor = texture(uPlasticNormal, pos.xy * 1.5).rgb;
        rou = texture(uPlasticRough, pos.xy * 1.5).r;

        a = 12.0 * (1.0 - rou);
        f0 = vec3(.042);
    // wall - left
    } else if (m < 26.) { 
        K = vec2(0.05, 2.0);
        material = vec3(.9608, .9686, .949);
    // wall - right
    } else if (m < 27.) { 
        K = vec2(0.05, 2.0);
        material = vec3(.9608, .9686, .949);
    // floor
    } else if (m < 31.) { 
        K = vec2(0.5, 16.0);
        material = vec3(0.8, 0.749, 0.7019);
    } else if (m < 41.) {
    // noise (when set to non-metal)
        // fake smoke/steam (either one)
        K = vec2(0.05, 1.0);
        material = 2.6*vec3(0.98, 0.98, 0.98019);
        // return vec3(1.0);
        // color balls?
        // K = vec2(0.05, 1.0);
        // material = 1.5*vec3(0.68, 0.5, 0.99019);
    } else if (m < 51.) {
        // dark ball inside kettle spot to make it look better
        material = texture(uMetalColor, 0.85 * pos.xy + texCoords).rgb;
        return material * vec3(0.05);
    }

    // lighting
    if ( m > 0. ) { 
        material *= 0.72 * material;
        
        float occ = calcAO(pos, nor, time);
        a = 0.5*a + 0.5*((2.0 / pow(rou, 2.0)) - 2.0);
        
        float bou = clamp( 0.3-0.7*nor.y, 0.0, 1.0 );
        vec3 ref = reflect( rd, nor );
        vec3 lin = vec3(0);

        // indoor lighting 
        // top - BRDF
        {
            vec3 lig = normalize(vec3(0.2,1.0,0.05));

            float dif = clamp(dot(lig,nor), 0.0, 1.0);
            dif *= occ;

            vec3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
            // fre = 0.05 * fre + 0.95 * clamp(1.0 - dot(lig, hNor), 0.0, 1.0); // i like both qualities
            float shadow = calcSoftshadow(pos, lig, 0.000001, 14.0, 6.0, time);
            (m == 16.5) ? shadow = 1.0 : shadow = shadow;

            vec3 clr = normalize(vec3(0.5, 0.633, 0.9));
            // float speBias = smoothstep(0.3, 0.42, ref.y); // to make it seem more like an area light
            // float speBias = 1.0; // or not
            
            // fresnel
            vec3 fSch = f0 + (vec3(1) - f0)*pow(fre, 5.0);  
            
            // distribution
            // float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(nor, hNor), 0.0, 1.0), a ); // more fake, 90s
            float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(nor, hNor), 0.0, 1.0), a / 6.0); // more accurate - K.y is normally a

            // full spectral light addition
            vec3 spe = (fSch * dBlinnPhong) / 2.0;
            lin += K.x * clr * 0.65 * spe * dif; // spec - add material, or not? shadow, or not?

            lin += (1.0 - K.x) * 2.5 * dif * clr * shadow * shadow * material; // dif
        }
        //side 
        {
            vec3 lig = normalize(vec3(-0.5, 0.2, 0.1));
            float dif = 0.1 + 0.9 * clamp(dot(lig, nor), 0.0, 1.0);
            float shadow = calcSoftshadow(pos, lig, 0.000001, 14.0, 4.0, time);
            (m == 16.5) ? shadow = 1.0 : shadow = shadow;
            // shadow = pow(shadow, 0.5);
            vec3 clr = vec3(1.0, 0.6, 0.5);
            dif *= occ;

            vec3 hal  = normalize(lig - rd);
            float fre = clamp(1.0 - dot(lig, nor), 0.0, 1.0); 

            vec3 spe = vec3(1)*(pow(clamp(dot(nor,hal), 0.0, 1.0), a / 2.0));
            // spe *= 

            vec3 fSch = f0 + (vec3(1) - f0)*pow(fre, 5.0);   
            spe *= fSch;

            // float dBlinnPhong = ((a + 2.0) / (2.0*3.141592653)) * pow(clamp(dot(nor, hal), 0.0, 1.0), a); 

            // vec3 spe = (fSch * dBlinnPhong) / (4.0);
            lin += K.x * 0.75 * clr * spe;

            lin += (1.0 - K.x) * 3.0 * dif * shadow * shadow * material * clr;
        }
        // back (?)
        // below - bounce
        {
            float spoutFix = clamp( dot(nor, normalize(vec3(-0.3, -0.5, 0.1))), 0.0, 1.0 );
            lin += 4.5 * material * vec3(0.8,0.4,0.45) * spoutFix * occ + 2.0 * material * vec3(0.5,0.41,0.39) * bou * occ;
        }
        // sss
        {
            vec3 lig = normalize(vec3(0.2,1.0,0.05));
            vec3 hal  = normalize(lig - rd);
            float dif = clamp(dot(nor, hal), 0.0, 1.0);
            float fre = clamp(1.0 - dot(lig, hal), 0.0, 1.0);
            vec3 fSch = f0 + (vec3(1) - f0)*pow(fre, 5.0);   
            lin += 3.5 * (1.0 - K.x) * fre * fre * (0.2 + 0.8 * dif * occ) * material;
        }
        
        // sun
        // {
        //     float dif = clamp(dot(nor, hal), 0.0, 1.0);
        //     dif *= shadow;
        //     float spe = K.x * pow( clamp(dot(nor, hal), 0.0, 1.0), K.y );
        //     spe *= dif;
        //     spe *= 0.04+0.96*pow(clamp(1.0-dot(hal,lig),0.0,1.0),5.0); // fresnel
        //     lin += material * (3.20) * dif * vec3(2.1,1.1,0.6);
        //     lin += material * (3.) * spe * vec3(2.1,1.1,0.6);
        // }
        // // sky / reflections
        // {
        //     float dif = occ * sqrt(clamp(0.2 + 0.8 * nor.y, 0.0, 1.0));
        //     vec3 skyCol = skyColor(pos, ref, lig, time);
        //     lin += material * 0.6 * dif * (0.55 * clamp(skyCol, 0.0, 0.6));

        //     float spe = smoothstep( -0.2, 0.2, ref.y );
        //     spe *= dif;
        //     spe *= occ * occ * K.x * 0.04 + 0.96 * pow( fre, 5.0 );
        //     spe *= shadow; 
        //     lin += 4.0 * spe * skyCol;
        // }
        // // ground / bounce light
        // {
        //     lin += 2.0 * material * vec3(0.5,0.41,0.39) * bou * occ;
        // }
        // // back
        // {
        // 	float dif = clamp( dot( nor, normalize(vec3(-0.5,0.0,-0.75))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
        //           dif *= occ;
        // 	lin += 2.0*material*0.55*dif*vec3(0.35,0.25,0.35);
        // }
        // // sss
        // {
        //     float dif = clamp(dot(nor, hal), 0.0, 1.0);
        //     lin += 3.5 * fre * fre * (0.2 + 0.8 * dif * occ * shadow) * material;
        // }

        col = (0.05 * material) + (0.95 * lin);
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
        float time = uTime + 1.0 * (1.0/4.0) * float(m * n * AA) / float(AA*AA);
#else
        vec2 p = vec2(aspect, 1.0) * (vUv - vec2(0.5));
        float time = uTime * 50.0;
#endif

        // slow 
        time *= 0.0;

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