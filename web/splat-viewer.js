/**
 * Splat-Viewer: WebGL2-Renderer fuer 3D-Gaussian-Splats im 3DGS-PLY-Format.
 *
 * Warum selbst geschrieben: der Renderer muss ohne CUDA und ohne externe
 * Bibliothek laufen, echtes Tiefensortieren machen (keine Verformung beim
 * Drehen) und im Zweifel offline funktionieren.
 *
 * Verfahren: pro Splat ein Billboard-Quadrat, das aus der 2D-Kovarianz
 * (Eigenzerlegung) aufgespannt wird. Sortierung hinten->vorne auf der CPU,
 * premultiplied Alpha-Blending auf der GPU.
 */

const VS = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
layout(location = 2) in float aOpacity;
layout(location = 3) in vec3 aU0;
layout(location = 4) in vec3 aU1;
layout(location = 5) in vec3 aU2;

uniform mat4 uView;
uniform mat4 uProj;
uniform vec2 uViewport;
uniform vec2 uFocal;

out vec3 vColor;
out float vAlpha;
out vec3 vConic;
out vec2 vOffset;

void main() {
  // Ecken des Billboards aus dem Vertex-Index: (-1,-1) (1,-1) (-1,1) (1,1)
  vec2 corner = vec2(float(gl_VertexID & 1), float((gl_VertexID >> 1) & 1)) * 2.0 - 1.0;

  vec3 p = (uView * vec4(aPos, 1.0)).xyz;
  // Nur echt hinter der Kamera verwerfen. Frueher stand hier 0.15 -- das hat
  // beim Hineinzoomen alle Splats vor der Kamera geloescht (Loecher, die mit
  // dem Blickwinkel wandern).
  if (p.z <= 0.02) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  // 3D-Kovarianz im Kameraraum:  C = (R_view * U) * (R_view * U)^T
  mat3 U = mat3(aU0, aU1, aU2);
  mat3 M = mat3(uView) * U;
  mat3 C = M * transpose(M);

  float iz = 1.0 / p.z;
  float iz2 = iz * iz;

  // Jacobi der Perspektivprojektion (z positiv nach vorne, y nach oben)
  vec3 j0 = vec3(uFocal.x * iz, 0.0, -uFocal.x * p.x * iz2);
  vec3 j1 = vec3(0.0, uFocal.y * iz, -uFocal.y * p.y * iz2);

  float a = dot(j0, C * j0);
  float b = dot(j0, C * j1);
  float c = dot(j1, C * j1);

  // Anti-Aliasing-Epsilon wie in 3DGS
  a += 0.3;
  c += 0.3;

  float det = a * c - b * b;
  if (det <= 1e-9) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  // Eigenzerlegung -> Hauptachsen des Ellipsoids
  float mid = 0.5 * (a + c);
  float rad = sqrt(max(0.25 * (a - c) * (a - c) + b * b, 0.0));
  float l1 = mid + rad;
  float l2 = max(mid - rad, 0.0);
  vec2 e1 = (abs(b) > 1e-9) ? normalize(vec2(b, l1 - a)) : vec2(1.0, 0.0);
  vec2 e2 = vec2(-e1.y, e1.x);

  // Obergrenze pro Splat: ein einzelner Riesen-Splat darf nicht das halbe
  // Bild ueberdecken (erzeugt sonst Schlieren beim Drehen)
  float rmax = 0.5 * min(uViewport.x, uViewport.y);
  float r1 = min(3.0 * sqrt(l1), rmax);
  float r2 = min(max(3.0 * sqrt(l2), 0.7), rmax);

  vec2 delta = e1 * (corner.x * r1) + e2 * (corner.y * r2);

  vec4 clip = uProj * vec4(p, 1.0);
  clip.xy += delta * (2.0 / uViewport) * clip.w;

  gl_Position = clip;
  vColor = aColor;
  vAlpha = aOpacity;
  vConic = vec3(c, -b, a) / det;   // inverse 2x2 Kovarianz
  vOffset = delta;
}`;

const FS = `#version 300 es
precision highp float;

in vec3 vColor;
in float vAlpha;
in vec3 vConic;
in vec2 vOffset;
out vec4 outColor;

void main() {
  float p = -0.5 * (vConic.x * vOffset.x * vOffset.x
                  + 2.0 * vConic.y * vOffset.x * vOffset.y
                  + vConic.z * vOffset.y * vOffset.y);
  if (p > 0.0) discard;
  float a = exp(p) * vAlpha;
  if (a < 0.003) discard;
  outColor = vec4(vColor * a, a);   // premultiplied
}`;

const STRIDE = 16;           // floats pro Splat
const BYTES = STRIDE * 4;

export class SplatViewer {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl2', {
      antialias: true,
      alpha: false,
      premultipliedAlpha: true,
      powerPreference: 'high-performance',
    });
    if (!this.gl) throw new Error('WebGL2 nicht verfuegbar');

    this.bg = opts.background || [0.945, 0.953, 0.965];
    this.azimuth = opts.azimuth ?? 0.6;
    this.elevation = opts.elevation ?? 0.18;
    this.distance = null;
    this.target = [0, 0, 0];
    this.autoRotate = false;
    this.dirty = true;
    this.count = 0;
    this.meta = {};
    this._raf = null;
    this._order = null;
    this._depth = null;

    this._initGL();
    this._initInput();
    this._resize();
    this._onResize = () => { this._resize(); this.dirty = true; };
    window.addEventListener('resize', this._onResize);
    if (window.ResizeObserver) {
      this._ro = new ResizeObserver(() => { this._resize(); this.dirty = true; });
      this._ro.observe(canvas);
    }
  }

  _initGL() {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        throw new Error('Shader: ' + gl.getShaderInfoLog(s));
      }
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error('Link: ' + gl.getProgramInfoLog(prog));
    }
    this.prog = prog;
    this.u = {
      view: gl.getUniformLocation(prog, 'uView'),
      proj: gl.getUniformLocation(prog, 'uProj'),
      viewport: gl.getUniformLocation(prog, 'uViewport'),
      focal: gl.getUniformLocation(prog, 'uFocal'),
    };

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);

    const layout = [
      [0, 3, 0], [1, 3, 12], [2, 1, 24],
      [3, 3, 28], [4, 3, 40], [5, 3, 52],
    ];
    for (const [loc, size, off] of layout) {
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, BYTES, off);
      gl.vertexAttribDivisor(loc, 1);
    }
    gl.bindVertexArray(null);

    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  }

  _resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  _initInput() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, y: e.clientY, az: this.azimuth, el: this.elevation };
      c.setPointerCapture(e.pointerId);
      c.style.cursor = 'grabbing';
    });
    c.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      this.azimuth = drag.az - dx * 0.008;
      this.elevation = Math.max(-1.45, Math.min(1.45, drag.el + dy * 0.008));
      this.dirty = true;
    });
    c.addEventListener('pointerup', (e) => {
      drag = null;
      c.style.cursor = 'grab';
      try { c.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      // nicht naeher als knapp an die Oberflaeche: sonst steht die Kamera im
      // Modell und die Sortierung kann keine sinnvolle Reihenfolge mehr bilden
      const minD = Math.max(0.35, (this.meta.radius || 0.5) * 1.15);
      this.distance = Math.max(minD, Math.min(20, this.distance * (1 + Math.sign(e.deltaY) * 0.08)));
      this.dirty = true;
    }, { passive: false });

    // Doppelklick: zurueck zur Startansicht
    c.addEventListener('dblclick', () => this.reset());
    c.style.cursor = 'grab';
  }

  reset() {
    this.azimuth = 0.6;
    this.elevation = 0.18;
    this.distance = this._fitDistance();
    this.dirty = true;
  }

  setAutoRotate(on) { this.autoRotate = !!on; this.dirty = true; }

  _fitDistance() {
    const r = this.meta.radius || 1;
    return Math.max(1.0, r * 3.0);
  }

  async load(url, onProgress) {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Modell konnte nicht geladen werden (' + res.status + ')');
    const buf = await res.arrayBuffer();
    this._parse(buf);
    if (onProgress) onProgress(1);
  }

  _parse(buffer) {
    const bytes = new Uint8Array(buffer);
    // Header liegt am Anfang, "end_header" beendet ihn
    const headSlice = bytes.subarray(0, Math.min(bytes.length, 1 << 16));
    const head = new TextDecoder('latin1').decode(headSlice);
    const endMarker = 'end_header';
    const markerAt = head.indexOf(endMarker);
    if (markerAt < 0) throw new Error('Kein PLY-Header');
    let dataStart = markerAt + endMarker.length;
    while (dataStart < head.length && (head[dataStart] === '\r' || head[dataStart] === '\n')) dataStart++;

    const lines = head.slice(0, markerAt).split(/\r?\n/);
    let n = 0;
    const props = [];
    for (const line of lines) {
      const t = line.trim();
      if (t.startsWith('element vertex')) n = parseInt(t.split(/\s+/)[2], 10);
      else if (t.startsWith('property')) {
        const p = t.split(/\s+/);
        props.push({ name: p[p.length - 1], type: p[1] === 'double' ? 'f8' : (p[1] === 'uchar' ? 'u1' : 'f4') });
      }
    }
    if (!n || !props.length) throw new Error('PLY ohne Vertex-Daten');

    const sizes = { f4: 4, f8: 8, u1: 1 };
    let off = 0;
    const idx = {};
    for (const p of props) {
      idx[p.name] = { off, type: p.type };
      off += sizes[p.type];
    }
    const stride = off;
    const dv = new DataView(buffer, dataStart);

    const get = (i, name, dflt = 0) => {
      const p = idx[name];
      if (!p) return dflt;
      const at = i * stride + p.off;
      return p.type === 'f8' ? dv.getFloat64(at, true)
        : p.type === 'u1' ? dv.getUint8(at)
        : dv.getFloat32(at, true);
    };

    const out = new Float32Array(n * STRIDE);
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let cx = 0, cy = 0, cz = 0;

    const SH_C0 = 0.28209479177387814;
    for (let i = 0; i < n; i++) {
      const x = get(i, 'x'), y = get(i, 'y'), z = get(i, 'z');
      const op = 1 / (1 + Math.exp(-get(i, 'opacity', 10)));
      let r = Math.min(1, Math.max(0, SH_C0 * get(i, 'f_dc_0') + 0.5));
      let g = Math.min(1, Math.max(0, SH_C0 * get(i, 'f_dc_1') + 0.5));
      let b = Math.min(1, Math.max(0, SH_C0 * get(i, 'f_dc_2') + 0.5));
      const sx = Math.exp(get(i, 'scale_0')), sy = Math.exp(get(i, 'scale_1')), sz = Math.exp(get(i, 'scale_2'));

      // Quaternion (w,x,y,z) -> Rotationsmatrix * scale = U
      let qw = get(i, 'rot_0', 1), qx = get(i, 'rot_1'), qy = get(i, 'rot_2'), qz = get(i, 'rot_3');
      const ql = Math.hypot(qw, qx, qy, qz) || 1;
      qw /= ql; qx /= ql; qy /= ql; qz /= ql;
      const R00 = 1 - 2 * (qy * qy + qz * qz), R01 = 2 * (qx * qy - qw * qz), R02 = 2 * (qx * qz + qw * qy);
      const R10 = 2 * (qx * qy + qw * qz), R11 = 1 - 2 * (qx * qx + qz * qz), R12 = 2 * (qy * qz - qw * qx);
      const R20 = 2 * (qx * qz - qw * qy), R21 = 2 * (qy * qz + qw * qx), R22 = 1 - 2 * (qx * qx + qy * qy);

      const o = i * STRIDE;
      out[o] = x; out[o + 1] = y; out[o + 2] = z;
      out[o + 3] = r; out[o + 4] = g; out[o + 5] = b;
      out[o + 6] = op;
      out[o + 7] = R00 * sx; out[o + 8] = R10 * sx; out[o + 9] = R20 * sx;
      out[o + 10] = R01 * sy; out[o + 11] = R11 * sy; out[o + 12] = R21 * sy;
      out[o + 13] = R02 * sz; out[o + 14] = R12 * sz; out[o + 15] = R22 * sz;

      cx += x; cy += y; cz += z;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }

    this.data = out;
    this.count = n;
    // Zielpunkt = Mitte der Bounding-Box (nicht Schwerpunkt: der liegt bei
    // Modellen mit Sockel zu tief und schneidet oben ab)
    this.target = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
    this.meta = {
      count: n,
      radius: 0.5 * Math.max(maxX - minX, maxY - minY, maxZ - minZ),
      bounds: { minX, minY, minZ, maxX, maxY, maxZ },
    };
    if (this.distance === null) this.distance = this._fitDistance();

    this._order = new Uint32Array(n);
    this._depth = new Float32Array(n);
    this._sorted = new Float32Array(n * STRIDE);

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, this._sorted.byteLength, gl.DYNAMIC_DRAW);
    this.dirty = true;
  }

  _viewMatrix() {
    const ce = Math.cos(this.elevation), se = Math.sin(this.elevation);
    const ca = Math.cos(this.azimuth), sa = Math.sin(this.azimuth);
    const d = this.distance;
    const t = this.target;
    const eye = [t[0] + d * ce * sa, t[1] + d * se, t[2] + d * ce * ca];

    // forward zeigt ins Ziel, z positiv nach vorne (OpenCV-Konvention)
    let fx = t[0] - eye[0], fy = t[1] - eye[1], fz = t[2] - eye[2];
    const fl = Math.hypot(fx, fy, fz) || 1; fx /= fl; fy /= fl; fz /= fl;
    let sx = fy * 0 - fz * 1, sy = fz * 0 - fx * 0, sz = fx * 1 - fy * 0;
    const sl = Math.hypot(sx, sy, sz) || 1; sx /= sl; sy /= sl; sz /= sl;
    const ux = sy * fz - sz * fy, uy = sz * fx - sx * fz, uz = sx * fy - sy * fx;

    // column-major
    return new Float32Array([
      sx, ux, fx, 0,
      sy, uy, fy, 0,
      sz, uz, fz, 0,
      -(sx * eye[0] + sy * eye[1] + sz * eye[2]),
      -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
      -(fx * eye[0] + fy * eye[1] + fz * eye[2]),
      1,
    ]);
  }

  _sortAndUpload(view) {
    const d = this.data, n = this.count, order = this._order, depth = this._depth;
    // Zeile 3 der View-Matrix = Vorwärtsachse (column-major: Indizes 2, 6, 10, 14)
    const m2 = view[2], m6 = view[6], m10 = view[10], m14 = view[14];

    for (let i = 0; i < n; i++) {
      const o = i * STRIDE;
      depth[i] = m2 * d[o] + m6 * d[o + 1] + m10 * d[o + 2] + m14;
      order[i] = i;
    }
    // hinten -> vorne (absteigende Tiefe)
    const ord = this._ordArr || (this._ordArr = new Array(n));
    for (let i = 0; i < n; i++) ord[i] = order[i];
    ord.sort((a, b) => depth[b] - depth[a]);

    const out = this._sorted;
    let w = 0;
    for (let k = 0; k < n; k++) {
      const src = ord[k] * STRIDE;
      for (let j = 0; j < STRIDE; j++) out[w++] = d[src + j];
    }

    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, out);
  }

  _frame() {
    this._raf = requestAnimationFrame(() => this._frame());
    if (this.autoRotate) {
      this.azimuth += 0.006;
      this.dirty = true;
    }
    if (!this.dirty || !this.count) return;
    this.dirty = false;

    const gl = this.gl;
    const W = this.canvas.width, H = this.canvas.height;
    gl.viewport(0, 0, W, H);
    gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const view = this._viewMatrix();
    const fovy = 0.857;                     // ~49.1 Grad, wie LGM
    const f = 1 / Math.tan(fovy / 2);       // Projektionsskala (dimensionslos)
    const focalPx = (H / 2) * f;            // gleiche Skala in Pixeln
    const near = 0.02, far = 100;
    const aspect = W / H;
    const A = (far + near) / (far - near);
    const B = -(2 * near * far) / (far - near);
    const proj = new Float32Array([
      f / aspect, 0, 0, 0,
      0, f, 0, 0,
      0, 0, A, 1,
      0, 0, B, 0,
    ]);

    this._sortAndUpload(view);

    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.u.view, false, view);
    gl.uniformMatrix4fv(this.u.proj, false, proj);
    gl.uniform2f(this.u.viewport, W, H);
    gl.uniform2f(this.u.focal, focalPx, focalPx);
    gl.bindVertexArray(this.vao);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, this.count);
    gl.bindVertexArray(null);
  }

  start() {
    if (!this._raf) this._frame();
  }

  stop() {
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = null;
  }

  dispose() {
    this.stop();
    window.removeEventListener('resize', this._onResize);
    if (this._ro) this._ro.disconnect();
  }
}
