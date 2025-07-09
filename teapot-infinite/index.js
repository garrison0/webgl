import * as twgl from './node_modules/twgl.js/dist/4.x/twgl-full.module.js';
// var twgl = require ('./node_modules/twgl.js/dist/4.x/twgl-full.js');
// import makeLostContextSimulatingCanvas from './node_modules/webgl-debug/index.js';
// var WebGLDebugUtils = require('./node_modules/webgl-debug/');

var canvas = document.querySelector("#c");
// canvas = WebGLDebugUtils.makeLostContextSimulatingCanvas(canvas);
var gl = canvas.getContext("webgl2");
var state = { mousePos: [0.5, 0.5], images: {}, textures: {} };
var files = {'fs': '', 'vs': ''};
var ext = gl.getExtension("WEBGL_lose_context"); 
var currentFrame = 0;
var timeString = ''; // captured so far in frames

var capturer;
var id;

function download(data, filename) { 
  var file = new Blob([data]);
  var a = document.createElement("a"),
      url = URL.createObjectURL(file);
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(function() {
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);  
      window.location.reload();
  }, 0); 
};

function init()
{
  capturer = new CCapture( { format: 'webm', 
                            framerate: 24, 
                            name: "startTime-"+timeString,
                            verbose: true, 
                            startTime: Number(timeString)*24,
                            timeLimit: 10000} );
                                // set to length of music video in seconds
                                // - Number(timeString)*24} 
  let imgs = state.images;
  state = { mousePos: [0.5, 0.5], images: {}, textures: {} };
  state.images = imgs;

  state.programInfo = twgl.createProgramInfo(gl, [files['vs'], files['fs']]);
  state.bufferInfo = twgl.primitives.createXYQuadBufferInfo(gl);

  for (const [fileName,image] of Object.entries(state.images)) { 
    state.textures[fileName] = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, state.textures[fileName]);
    const level = 0;
    const internalFormat = gl.RGB;
    const srcFormat = gl.RGB;
    const srcType = gl.UNSIGNED_BYTE;
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat,
                srcFormat, srcType, image);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  }

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.clearColor(0.0,0.0,0.0,1.0);

  id = requestAnimationFrame(render);
  capturer.start();
}

function render(time) { 
  time *= 0.001;
  currentFrame += 1;

  twgl.resizeCanvasToDisplaySize(gl.canvas, 1.0);
  gl.viewport(0, 0, gl.canvas.clientWidth, gl.canvas.clientHeight);

  var uniforms = {
      uTime: time,
      uResolution: [gl.canvas.clientWidth, gl.canvas.clientHeight],
  }

  for (const [textureName, texture] of Object.entries(state.textures)) { 
    let uniformName = 'u' + textureName.charAt(0).toUpperCase() + textureName.slice(1);
    uniforms[uniformName] = texture;
  }

  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  gl.useProgram(state.programInfo.program);
  twgl.setUniforms(state.programInfo, uniforms);
  twgl.setBuffersAndAttributes(gl, state.programInfo, state.bufferInfo); 
  twgl.drawBufferInfo(gl, state.bufferInfo);

  id = requestAnimationFrame(render);
  capturer.capture( gl.canvas );
}


// -------- HANDLE BROWSER EVENTS / INTERACTIVITY             --------- //
document.onmousemove = (event) => { 
    const rect = gl.canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / gl.canvas.width * 2 - 1;
    const y = (event.clientY - rect.top ) / gl.canvas.height * 2 - 1;
    state.mousePos = [2 * x,2 * -y];
}

function handleContextLost(event) {
  event.preventDefault();
  cancelAnimationFrame(id);
  capturer.stop();
  capturer.save();
  setTimeout(() => {
    download( (Number(timeString) + currentFrame).toString(), 'time.txt' );
  }, 500);
}

// function handleContextRestored(event) { 
//   ext.restoreContext();
//   gl = document.querySelector("#c").getContext("webgl2");
//   init();
// }

window.addEventListener("mousedown", function() {
  gl.getExtension('WEBGL_lose_context').loseContext();
});

// window.addEventListener("keydown", function() { 
//   handleContextRestored();
// });

{
  gl.canvas.addEventListener('webglcontextlost', handleContextLost, false);
}

// -------- FETCH FILES ( SHADERS, IMAGES FOR TEXTURES, ETC ) --------- //
// keys correspond to the file names of each shader
var imageFiles = ['metalColor.png', 'metalNormal.png', 'metalRough.png', 
                  'plasticColor.png', 'plasticNormal.png', 'plasticRough.png'];
var count = 0;
let numFiles = 2 + imageFiles.length;

fetch('./time.txt')
  .then(response => response.text())
  .then(text => {
    timeString = text;
    if (count >= numFiles && time !== '') 
      init();
});

imageFiles.forEach(file => { 
  let fileName = file.split('.')[0];
  state.images[fileName] = new Image();
  state.images[fileName].src = './assets/' + file;
  state.images[fileName].onload = function() { 
    count = count + 1;
    if (count >= numFiles && timeString !== '') 
      init();
  }
}); 

// todo: simplify this, ideally it is not separate
for (const key of Object.keys(files)) { 
  fetch('./shader/' + key + '.glsl')
    .then(response => response.text())
    .then(text => {
      files[key] = text;
      count = count + 1;
      if (count >= numFiles && timeString !== '') 
        init();
    });
};
