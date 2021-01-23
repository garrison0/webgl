import * as twgl from './node_modules/twgl.js/dist/4.x/twgl-full.module.js';

const gl = document.querySelector("#c").getContext("webgl2");
var state = { mousePos: [0.5, 0.5] };

function init(preloaded, waterImage, grassImage, sandImage)
{
    state.programInfo = twgl.createProgramInfo(gl, [preloaded['vs'], preloaded['fs']]);

    state.bufferInfo = twgl.primitives.createXYQuadBufferInfo(gl);

    state.waterTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, state.waterTexture);
    const level = 0;
    const internalFormat = gl.RGB;
    const srcFormat = gl.RGB;
    const srcType = gl.UNSIGNED_BYTE;
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
                srcFormat, srcType, waterImage);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    state.grassTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, state.grassTexture);
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
                srcFormat, srcType, grassImage);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    state.sandTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, state.sandTexture);
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
                srcFormat, srcType, sandImage);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.0,0.0,0.0,1.0);

    requestAnimationFrame(render);
}

function render(time) { 
    time *= 0.001;

    twgl.resizeCanvasToDisplaySize(gl.canvas, 1.0);
    gl.viewport(0, 0, gl.canvas.clientWidth, gl.canvas.clientHeight);

    const uniforms = {
        uTime: time,
        uResolution: [gl.canvas.clientWidth, gl.canvas.clientHeight],
        uWaterTexture: state.waterTexture,
        uGrassTexture: state.grassTexture,
        uSandTexture: state.sandTexture
    }

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(state.programInfo.program);
    twgl.setUniforms(state.programInfo, uniforms);
    twgl.setBuffersAndAttributes(gl, state.programInfo, state.bufferInfo); 
    twgl.drawBufferInfo(gl, state.bufferInfo);

    requestAnimationFrame(render);
}


// -------- HANDLE BROWSER EVENTS / INTERACTIVITY             --------- //
document.onmousemove = (event) => { 
    const rect = gl.canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / gl.canvas.width * 2 - 1;
    const y = (event.clientY - rect.top ) / gl.canvas.height * 2 - 1;
    state.mousePos = [2 * x,2 * -y];
}

// -------- FETCH FILES ( SHADERS, IMAGES FOR TEXTURES, ETC ) --------- //
// keys correspond to the file names of each shader
var files = {'fs': '', 'vs': ''};
var count = 0;
var imageWater = new Image();
var imageGrass = new Image();
var imageSand = new Image();
let numFiles = 5;

for (const key of Object.keys(files)) { 
  fetch('./shader/' + key + '.glsl')
    .then(response => response.text())
    .then(text => {
      files[key] = text;
      count = count + 1;
      if (count >= numFiles) 
        init(files, imageWater, imageGrass, imageSand);
    });
};

imageWater.src = './assets/water.png';
imageWater.onload = function () {
  count = count + 1;
  if (count >= numFiles) 
    init(files, imageWater, imageGrass, imageSand);
}

imageGrass.src = './assets/grass.jpg';
imageGrass.onload = function () {
  count = count + 1;
  if (count >= numFiles) 
    init(files, imageWater, imageGrass, imageSand);
}

imageSand.src = './assets/sand.jpg';
imageSand.onload = function () {
  count = count + 1;
  if (count >= numFiles) 
    init(files, imageWater, imageGrass, imageSand);
}