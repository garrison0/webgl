#version 300 es
#define AA 1 //# of anti-aliasing passes (1 for low quality, 2 for high quality)

precision highp float;

in vec2 vUv;
uniform vec2 uResolution;
uniform float uTime;

out vec4 o_FragColor;

// noise stuff
float random (in vec2 st) {
    return fract(sin(dot(st.xy,
                            vec2(12.9898,78.233)))*
        43758.5453123);
}

float noise (in vec2 st) {
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
float fbm (in vec2 st) {
    // Initial values
    float value = 0.0;
    float amplitude = .5;
    float frequency = 0.;
    //
    // Loop of octaves
    for (int i = 0; i < OCTAVES; i++) {
        value += amplitude * noise(st);
        st *= 2.;
        amplitude *= .5;
    }
    return value;
}

// noise stuff
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

const mat2 m2 = mat2( 0.60, -0.80, 0.80, 0.60 );

const mat3 m3 = mat3( 0.00,  0.80,  0.60,
                     -0.80,  0.36, -0.48,
                     -0.60, -0.48,  0.64 );

float fbm3( in vec3 p ) {
    float f = 0.0;
    f += 0.5000*noise( p ); p = m3*p*2.02;
    f += 0.2500*noise( p ); p = m3*p*2.03;
    f += 0.1250*noise( p ); p = m3*p*2.01;
    f += 0.0625*noise( p );
    return f/0.9375;
}

// ---- UTILITIES --- // 
vec3 rotatePoint(vec3 p, vec3 n, float theta) { 
    vec4 q = vec4(cos(theta / 2.0), sin (theta / 2.0) * n);
    vec3 temp = cross(q.xyz, p) + q.w * p;
    vec3 rotated = p + 2.0*cross(q.xyz, temp);
    return rotated;
}

// ------ MAPPING / LIGHTING ----- // 
vec2 opU( vec2 d1, vec2 d2 )
{
	return (d1.x<d2.x) ? d1 : d2;
}

vec2 opSmoothU( vec2 d1, vec2 d2, float k, float colorSmoothness ) 
{ 
    float interpo = clamp( 0.5 + 0.5 * (d1.x - d2.x) / colorSmoothness, 0.0, 1.0 );
    float h = max( k - abs(d1.x - d2.x), 0.0) / k;
    float diff = h*h*h*k*(1.0/6.0);
    return vec2( min(d1.x, d2.x) - diff,
                 mix(d1.y, d2.y, interpo) - k * interpo * ( interpo -  1.0));
}

// --------- DISTANCE FUNCTIONS ---------- //
float sdfSphere(vec3 p, float r) {
    return length( p ) - r;
}

float sdTorus( vec3 p, vec2 t )
{
    vec2 q = vec2(length(p.xz)-t.x,p.y);
    return length(q)-t.y;
}

float sdRoundBox( vec3 p, vec3 b, float r )
{
    vec3 q = abs(p) - b;
    return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

float sdIsland(vec3 p, float r) { 
    float y = abs(p.y);
    float h = 0.99; // peak/max of cosh function
    float disp = (h - cosh( y ));

    return length( p ) - r - disp; 
}

float sdbEllipsoid( in vec3 p, in vec3 r )
{
    float k1 = length(p/r);
    float k2 = length(p/(r*r));
    return k1*(k1-1.0)/k2;
}

vec2 sdCapsule( vec3 p, vec3 a, vec3 b )
{
//   vec3 pa = p - a, ba = b - a;
//   float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
//   return length( pa - ba*h ) - r;
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return vec2( length( pa - ba*h ) - 0.01, h );
}

float sdBeach(vec3 p, float r) { 
    return 1.0;
}

float sdTrunk ( vec3 p, float h, float r ) { 
    vec3 pCopy = p;
    float treeY = clamp(pCopy.y, 0.0, h);
    p.y -= treeY;
    p.x += 0.833 * cos( 0.7 * (1.66666 + 0.722 * pCopy.y) ); // tilting shape

    // add some noise (like holes in the tree bark) 
    // check out how inigo did the lines on the elephants!!
    float noiseAmount = 1.0 - smoothstep(0.0, 1.45, treeY / h);
    float noise = 0.3 * noiseAmount * clamp(fbm( pCopy.xy ), -1.0, 1.0);

    p.x -= 0.422 * noise;

    float ipy = h - pCopy.y;
    float bigger = 0.27 * clamp(ipy * ipy * 0.9, 0.0, h) / h; // make the bottom bigger

    // float bumps = 0.18 * noiseAmount * clamp (fbm3(pCopy), 0.0, 1.0);

    // p.x -= 0.422 * bumps;

    return length(p) - (r + bigger) + noise;
}

// ---------- MAPPING SUB FUNCTIONS ------ //
// type = 0, 1
vec2 mapBranch ( vec3 p, float l, vec2 cur, float type )
{
    // p.y -= 0.08;
    // p.z -= 4.0;
    // p.y -= 1.0;
    float theta = 0.754 + 2.355 * type;// * 0.2 * uTime;
    vec3 rq = vec3(0.0,1.0,0.0);
    p = rotatePoint(p, rq, theta);
    // p.y = - p.y;
    // scale and rotate
    // p.xz = m2 * p.xz;
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
    
    // y = y + 0.5 * cos(0.9 * uTime);
    vec3 b = l * vec3(1,y + 0.1*fbm3(vec3(y+uTime*0.25)),0);

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

    // vec3 box = vec3(b.x, 0.05, 0);
    // float branchDist = sdRoundBox(p, box, 0.05);
    float branchDist = sdCapsule(p, a, b).x - 0.005; 

    float m = 74.0;
    vec2 res = opU( cur, vec2(branchDist, m));

    // leaves
    // figure out how far along the branch we are

    // finite repetition
    // p-c*clamp(round(p/c),-l,l);

    p.z = abs(p.z);
    vec3 c = vec3(branchRad * 2.0, 0, 0);
    vec3 l2 = vec3(36,0,0);
    vec3 n = clamp(round(p/c),-l2,l2);
    vec3 q = p-c*n;
    float nx = abs(n.x);
    float howCloseToEnd = nx / l2.x;
    float howMuchDisp = cos(0.814543 * quadrant + 2.2 * type + uTime);
    float quadrantNoise = noise(vec2(0.748243*quadrant,0.12325*quadrant));
    float nxNoise = noise(vec2(0.12323*nx + quadrantNoise, 0.557*nx + quadrantNoise));
    // float constMovement = 0.1 * (1.0 + cos(nxNoise * 1.2 * type * uTime));
    float constMovement = 0.1 * fbm3( vec3(nxNoise * uTime) );
    howMuchDisp = constMovement + 0.03 * max(0.0, howMuchDisp);
    float dispY = 0.123 + ((-0.6 * type * (1.0-howCloseToEnd*howCloseToEnd)) + 0.2 * nxNoise) + howMuchDisp * cos(0.5 * (quadrantNoise + nx + 4.0 * uTime));
    a = vec3(0.0, howCloseToEnd * b.y, 0.0);
    b = vec3(0.0, howCloseToEnd * b.y + dispY, clamp(2.0 * (howCloseToEnd * (1.1 - howCloseToEnd)), 0.0, 0.65 - 0.1 * type));
    m = 10.0;
    res = opU( res, vec2( sdCapsule(q, a, b).x - 0.01, m ) );

    res.x *= 0.5;
    return res;
}

// --------------- RENDERING FUNCTIONS ------------- // 
vec2 map (vec3 p) { 
    vec2 res = vec2(1e10, 0.0);

    // move the primitives around
    p = p - vec3(0,3.65,0);

//    res = opSmoothU( vec2(sdfSphere(p + vec3(0,0.5,0), 1.0), 20.3), res, 2.0 );
    // res = opU( vec2(sdIsland (p, 3.0), 20.0), res );

    float z = 2.0;
    res = vec2( sdbEllipsoid (p + vec3(0,5.0, z ), vec3(9.6, 2.25, 8.0)), 50.0);
    // res = opSmoothU( vec2( sdbEllipsoid (p + vec3(0,4.45, z + 0.75), vec3(9, 0.15, 9.0)), 0.0), res, 2.0, 5.0 );

    // palm tree
    // trunk
    // res = opSmoothU( vec2( sdTrunk( p + vec3(0, 2.5, z), 2.6, 0.08 ), 20.0 ), res, 0.7, 1.4 );
    res = opU( vec2( sdTrunk( p + vec3(0, 2.5, z), 4.8, 0.07 ), 20.0 ), res );

    // branches
    res = mapBranch( p + vec3(-0.75, -2.22, z), 2.2, res, 1.0 ); //top
    res = mapBranch( p + vec3(-0.75, -2.22, z), 2.2, res, 0.0 ); //mid
    // res = mapBranch( p + vec3(-0.75, -2.22, z), 1.2, res, 2.0 ); //bot

    // coconuts
    res = opU( vec2( sdfSphere(p + vec3(-0.65, -2.22, z), 0.08), 30.0), res );
    res = opU( vec2( sdfSphere(p + vec3(-0.85, -2.26, z), 0.08), 30.0), res );
    // res = opSmoothU( vec2(sdRoundBox(p, vec3(3.0, 0.1, 3.0), 0.0)), res, 3.5 );

    // --v bowditch thing
    // for (int i = 0; i<8; i++) { 
    //     float t = float(-2 * (i%2));
    //     float d = float(i) * 1.0 / 2.0;
    //     vec3 trans = vec3(sin(uTime) * 2.6 * sin(4. * 0.5 * uTime + d + 3.14159 / 2.0),
    //                       cos(uTime) * 2.6 * sin(5. * 0.5 * uTime + d),
    //                       sin(uTime) * 2.6 * sin(0.5 * uTime / 4.0 + d));
    //     res = opSmoothU( vec2(sdfSphere(p + trans, max(0.2, min(0.485,sin(float(i) * 0.05 * uTime * 0.2)))), 11.5), res, 0.5 );
    //  }

    // vec3 n = normalize(vec3(0,0.5,0.5));
    // float theta = uTime * 0.4 * 6.2222;
    
    // vec3 n2 = normalize(vec3(0.5,0.3,0.3));
    // float theta2 = uTime * 0.33 * 6.554;
    
    // vec3 p2 = rotatePoint(p, n2, theta2);
    // vec3 p3 = rotatePoint(p, n, theta);
    // res = opSmoothU( vec2(sdTorus( p2, vec2(1.75, 0.25)), 50.5), res, 0.72 + abs(sin(uTime * 0.2)) * 1.7 );
    // res = opSmoothU( vec2(sdTorus( p3, vec2(1.75, 0.25)), 35.5), res, 0.72 + abs(sin(uTime * 0.2)) * 1.9 ); 

    return res;
}

vec3 calcNormal( in vec3 p )
{
    const float eps = 0.0001; 
    const vec2 h = vec2(eps,0);
    return normalize( vec3(map(p+h.xyy).x - map(p-h.xyy).x,
                        map(p+h.yxy).x - map(p-h.yxy).x,
                        map(p+h.yyx).x - map(p-h.yyx).x ) );
}

float calcAO( in vec3 pos, in vec3 nor )
{
	float occ = 0.0;
    float sca = 1.0;
    for( int i=0; i<5; i++ )
    {
        float h = 0.01 + 0.12*float(i)/4.0;
        float d = map( pos + h*nor ).x;
        occ += (h-d)*sca;
        sca *= 0.95;
        if( occ>0.35 ) break;
    }
    return clamp( 1.0 - 3.0*occ, 0.0, 1.0 ) * (0.5+0.5*nor.y);
}

float calcSoftshadow( in vec3 ro, in vec3 rd, float mint, float maxt, float k )
{
    float res = 1.0;
    for( float t=mint; t<maxt; )
    {
        float h = map(ro + rd*t).x;
        if( h<0.001 )
            return 0.0;
        res = min( res, k*h/t );
        t += h;
    }
    return res;
}

vec2 raycast (in vec3 ro, in vec3 rd){
    vec2 res = vec2(1.0,-1.0);

    float tmin = 0.00001;
    float tmax = 75.0;
    
    // raytrace floor plane
    // float tp1 = (-1.0-ro.y)/rd.y;
    // if( tp1 > 0.0 )
    // {
    //     tmax = min( tmax, tp1 );
    //     res = vec2( tp1, 0.0 );
    // }

    // raycast the primitives
    float eps = 0.00015;
    for( int i = 0; i < 256; i++) {
        vec2 h = map( ro + rd*res.x );

        if( h.x < (eps * res.x) || res.x > tmax ){
            break;
        } 

        res.x += h.x;
        res.y = h.y;
    }

    return res;
}

vec3 render(in vec3 ro, in vec3 rd, in vec3 rdx, in vec3 rdy) { 
    // background - will probably be overwritten
    vec3 col = vec3(0.4,0.4,0.7) - max(rd.y, 0.0) * 0.3;

    vec2 res = raycast(ro,rd);
    float t = res.x;
    float m = res.y;

    // i.e., if given some float to make color with
    if (m > -0.5) { 
        vec3 pos = ro + rd*t;
        vec3 nor = (m<1.5) ? vec3(0.0,1.0,0.0) : calcNormal(pos);
        vec3 ref = reflect( rd, nor );
    
        //col = vec3(t);

        col = 0.15 + 0.0025 * m * sin(vec3(0.1,0.5,1.));
        float ks = 1.0;

        // could add whatever for the floor

        float occ = calcAO( pos, nor );
        
        vec3 lin = vec3(0.0);
        // sun 
        {
            vec3  lig = normalize( vec3(-0.4, 0.8, 0.5) );
            vec3  hal = normalize( lig-rd );
            float dif = clamp( dot( nor, lig ), 0.0, 1.0 );
        	      dif *= calcSoftshadow( pos, lig, 0.02, 2.5, 16.0 );
			float spe = pow( clamp( dot( nor, hal ), 0.0, 1.0 ),16.0);
                  spe *= dif;
                  spe *= 0.04+0.96*pow(clamp(1.0-dot(hal,lig),0.0,1.0),5.0);
            lin += col*2.20*dif*vec3(1.30,1.00,0.70);
            lin +=     5.00*spe*vec3(1.30,1.00,0.70)*ks;
        }
        // sky / reflections
        {
            float dif = sqrt(clamp( 0.5+0.5*nor.y, 0.0, 1.0 ));
                  dif *= occ;
            float spe = smoothstep( -0.2, 0.2, ref.y );
                  spe *= dif;
                  spe *= 0.04+0.96*pow(clamp(1.0+dot(nor,rd),0.0,1.0), 5.0 );
                  spe *= calcSoftshadow( pos, ref, 0.02, 2.5, 16.0 );
            lin += col*0.60*dif*vec3(0.40,0.60,1.15);
            lin +=     2.00*spe*vec3(0.40,0.60,1.30)*ks;
        }
        // // back
        // {
        // 	float dif = clamp( dot( nor, normalize(vec3(0.5,0.0,0.6))), 0.0, 1.0 )*clamp( 1.0-pos.y,0.0,1.0);
        //           dif *= occ;
        // 	lin += col*0.55*dif*vec3(0.25,0.25,0.25);
        // }
        // // sss
        // {
        //     float dif = pow(clamp(1.0+dot(nor,rd),0.0,1.0),2.0);
        //           dif *= occ;
        // 	lin += col*0.25*dif*vec3(1.00,1.00,1.00);
        // }

        col = lin;

        col = mix( col, vec3(0.6,0.6,0.9), 1.0-exp( -0.0001*t*t*t ) );
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
    vec3 ta = vec3( 0, 2.9, -3.0);
    vec3 ro = vec3(0.0, 10.5, 10.5);
    // vec3 ro = vec3( 8.5 * cos (0.25 * uTime), 6.2, 8.5 * sin (0.25 * uTime));
    mat3 ca = setCamera(ro, ta, 0.0);

    float aspect = uResolution.x / uResolution.y;

    vec3 total = vec3(0.0);
#if AA>1
    for (int m=0; m < AA; m++)
    for (int n=0; n < AA; n++) { 
        vec2 o = (vec2(float(m), float(n)) / uResolution) / float(AA);
        vec2 p = vec2(aspect, 1.0) * ( (vUv+o) - vec2(0.5));
#else
        vec2 p = vec2(aspect, 1.0) * (vUv - vec2(0.5));
#endif

        // ray direction
        vec3 rd = ca * normalize( vec3(p, 1.1) );

        // ray differentials 
        vec2 px =  vec2(aspect, 1.0) * ( (vUv+vec2(1.0,0.0)) - vec2(0.5));
        vec2 py =  vec2(aspect, 1.0) * ( (vUv+vec2(0.0,1.0)) - vec2(0.5));
        vec3 rdx = ca * normalize( vec3(px, 2.5));
        vec3 rdy = ca * normalize( vec3(py, 2.5));

        vec3 color = render( ro, rd, rdx, rdy );

        color = pow(color, vec3(0.566));

        total += color;
#if AA>1
    }
    total /= float(AA*AA);
#endif
    
    o_FragColor = vec4( total, 1.0 );
}