"""Face attendance - website + installable app (Hugging Face Space version).

  /      landing website
  /app   the attendance app (installable on phone/desktop as a PWA)

The visitor's own browser opens the camera and sends small JPEG frames to this
server. The server detects faces (Haar cascade) and recognises them (LBPH).
"""
import json
import re
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request

THRESHOLD = 70        # lower distance = better match (visitor can tune it)
SHOTS = 40            # photos captured per enrolment run
MAX_PHOTOS = 120      # hard cap per person
MAX_PEOPLE = 5        # per visitor
MAX_SESSIONS = 20     # whole server
SESSION_TTL = 30 * 60 # seconds of inactivity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024      # one frame is ~30 KB
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
detector_alt = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")   # better with tilted faces
lock = threading.Lock()
cv_lock = threading.Lock()   # OpenCV cascades are not safe to share between threads
sessions = {}
SID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


class Session:
    def __init__(self):
        self.people = {}     # id -> name
        self.faces = {}      # id -> list of 200x200 grayscale crops
        self.model = None
        self.present = {}    # name -> epoch seconds of first check-in
        self.next_id = 1
        self.touched = time.time()


def get_session(create=True):
    """Return this visitor's session (call with `lock` held)."""
    sid = request.args.get("sid", "")
    if not SID_RE.match(sid):
        abort(400)
    now = time.time()
    for k in [k for k, s in sessions.items() if now - s.touched > SESSION_TTL]:
        del sessions[k]
    s = sessions.get(sid)
    if s is None:
        if not create:
            return None
        if len(sessions) >= MAX_SESSIONS:
            abort(503, "Demo is full. Try again in a few minutes.")
        s = sessions[sid] = Session()
    s.touched = now
    return s


def find_faces(gray):
    """Even out the lighting, then look for faces with two detectors.
    Returns (equalised image, boxes). The same image is used for training and recognition."""
    eq = cv2.equalizeHist(gray)
    with cv_lock:
        for det, scale, neighbours, smallest in ((detector, 1.1, 4, (50, 50)), (detector_alt, 1.1, 3, (40, 40))):
            boxes = det.detectMultiScale(eq, scale, neighbours, minSize=smallest)
            if len(boxes):
                return eq, boxes
    return eq, []


def train(s):
    imgs, labels = [], []
    for pid, crops in s.faces.items():
        for im in crops:
            imgs.append(im)
            labels.append(pid)
    if not imgs:
        s.model = None
        return 0
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(imgs, np.array(labels, dtype=np.int32))
    s.model = model
    return len(set(labels))


def people_json(s):
    return [{"id": pid, "name": name, "photos": len(s.faces.get(pid, []))}
            for pid, name in sorted(s.people.items())]


@app.get("/health")
def health():
    return jsonify(ok=True, sessions=len(sessions))


@app.post("/api/frame")
def frame():
    started = time.time()
    img = cv2.imdecode(np.frombuffer(request.get_data(), np.uint8),
                       cv2.IMREAD_GRAYSCALE)
    if img is None:
        return jsonify(error="bad image"), 400
    mode = request.args.get("mode", "watch")
    try:
        thr = float(request.args.get("thr", THRESHOLD))
    except ValueError:
        thr = THRESHOLD
    img, boxes = find_faces(img)
    faces, photos = [], 0
    with lock:
        s = get_session()
        pid = None
        if mode == "enroll":
            try:
                pid = int(request.args.get("pid", ""))
            except ValueError:
                pid = None
            if pid not in s.people:
                pid = None
        if pid is not None and len(boxes):
            # enrolling: only the biggest face, saved as a 200x200 crop
            x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
            crops = s.faces.setdefault(pid, [])
            if len(crops) < MAX_PHOTOS:
                crops.append(cv2.resize(img[y:y + h, x:x + w], (200, 200)))
            faces.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                          "name": "Enrolling " + s.people[pid],
                          "state": "enroll", "conf": None})
        else:
            for (x, y, w, h) in boxes:
                f = {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                     "name": "Unknown", "state": "bad", "conf": None}
                if s.model is None:
                    f["name"] = "Face (train the model)"
                    f["state"] = "warn"
                else:
                    roi = cv2.resize(img[y:y + h, x:x + w], (200, 200))
                    label, conf = s.model.predict(roi)
                    f["conf"] = round(float(conf), 1)
                    if conf < thr and label in s.people:
                        f["name"] = s.people[label]
                        f["state"] = "ok"
                        s.present.setdefault(f["name"], int(time.time()))
                faces.append(f)
        if pid is not None:
            photos = len(s.faces.get(pid, []))
        out = {"faces": faces, "photos": photos, "trained": s.model is not None,
               "present": s.present, "w": int(img.shape[1]), "h": int(img.shape[0]),
               "ms": int((time.time() - started) * 1000)}
    return jsonify(out)


@app.get("/api/state")
def state():
    with lock:
        s = get_session()
        return jsonify(people=people_json(s), trained=s.model is not None,
                       present=s.present, shots=SHOTS)


@app.post("/api/person")
def add_person():
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()[:40]
    if not name:
        return jsonify(ok=False, msg="Type a name first."), 400
    with lock:
        s = get_session()
        if len(s.people) >= MAX_PEOPLE:
            return jsonify(ok=False, msg=f"Demo limit: {MAX_PEOPLE} people."), 400
        pid = s.next_id
        s.next_id += 1
        s.people[pid] = name
        s.faces[pid] = []
    return jsonify(ok=True, id=pid)


@app.delete("/api/person/<int:pid>")
def delete_person(pid):
    with lock:
        s = get_session()
        name = s.people.pop(pid, None)
        s.faces.pop(pid, None)
        s.present.pop(name, None)
        train(s)
    return jsonify(ok=True)


@app.post("/api/train")
def train_route():
    with lock:
        s = get_session()
        n = train(s)
    if not n:
        return jsonify(ok=False, msg="No photos yet. Capture someone first.")
    return jsonify(ok=True, msg=f"Training complete ({n} people).")


@app.post("/api/clear")
def clear():
    with lock:
        get_session().present.clear()
    return jsonify(ok=True)


@app.post("/api/reset")
def reset():
    with lock:
        sessions.pop(request.args.get("sid", ""), None)
    return jsonify(ok=True)


PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face attendance</title>
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#2457d6">
<link rel="apple-touch-icon" href="/icon-192.png">
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#16181d;--mut:#667085;--line:#e3e6eb;--acc:#2457d6;--ok:#2e9e4f;--bad:#d6403a;--warn:#e8890c}
@media(prefers-color-scheme:dark){:root{--bg:#101216;--card:#181b21;--ink:#eceef2;--mut:#98a2b3;--line:#2a2f38;--acc:#6b93f5}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:18px 20px 6px}h1{margin:0;font-size:22px}header p{margin:4px 0 0;color:var(--mut)}
main{display:grid;gap:16px;padding:16px 20px 28px;max-width:1100px;margin:auto;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr)}
@media(max-width:820px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
.card h2{margin:0 0 10px;font-size:16px}
.stage{position:relative;background:#000;border-radius:12px;overflow:hidden;min-height:240px}
.stage video{display:block;width:100%;height:auto}
.stage canvas{position:absolute;inset:0;width:100%;height:100%}
.ph{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#cbd2dc;text-align:center;padding:20px}
[hidden]{display:none!important}
button{font:inherit;border:0;border-radius:9px;padding:9px 14px;background:var(--acc);color:#fff;cursor:pointer}
button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
button.danger{background:transparent;color:var(--bad);border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:default}button.sm{padding:5px 9px;font-size:13px}
input[type=text]{width:100%;font:inherit;padding:9px 11px;border-radius:9px;border:1px solid var(--line);background:var(--bg);color:var(--ink)}
.row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center}
.bar{height:8px;background:var(--line);border-radius:6px;overflow:hidden;margin-top:10px}.bar span{display:block;height:100%;background:var(--warn);width:0}
ul{list-style:none;margin:0;padding:0}li{display:flex;gap:10px;align-items:center;padding:8px 0;border-top:1px solid var(--line)}li:first-child{border-top:0}
.nm{flex:1;min-width:0;overflow-wrap:anywhere}time,.sub{color:var(--mut);font-size:13px}.empty{color:var(--mut);font-size:14px}
.av{width:32px;height:32px;border-radius:50%;background:var(--acc);color:#fff;display:flex;align-items:center;justify-content:center;font-size:12px;flex:none}
.pill{display:inline-block;padding:3px 10px;border:1px solid var(--line);border-radius:99px;font-size:13px;color:var(--mut);margin-right:6px}
.banner{background:var(--bad);color:#fff;border-radius:10px;padding:10px 14px;margin:10px 0 0;font-size:15px}
.note{font-size:13px;color:var(--mut)}#msg{margin:10px 0 0;font-size:14px;color:var(--mut)}
</style></head>
<body>
<header><h1>Face attendance</h1>
<p>Your camera stays in your browser tab. <a href="/">Home</a> <button class="ghost sm" id="installBtn" hidden>Install app</button></p>
<p class="note" id="iosHint" hidden>On iPhone: tap the Share button, then <b>Add to Home Screen</b>.</p>
<div class="banner" id="banner" role="alert" hidden></div></header>
<main>
<section>
  <div class="card">
    <div class="stage">
      <video id="video" playsinline muted></video><canvas id="overlay"></canvas>
      <div class="ph" id="ph">Press <b>&nbsp;Start camera&nbsp;</b> and allow camera access.</div>
    </div>
    <div class="row">
      <button id="camBtn">Start camera</button>
      <span class="pill" id="camPill">Camera off</span><span class="pill" id="modelPill">Model not trained</span>
    </div>
    <div class="row note"><label for="thr">Match strictness (lower = stricter):</label>
      <input id="thr" type="range" min="40" max="110" value="70"><b id="thrVal">70</b></div>
    <div class="row note" id="chip">No face in view</div>
    <div class="note" id="diag">Waiting for the first picture from the camera...</div>
  </div>
</section>
<aside>
  <div class="card"><h2>Add a person</h2>
    <input type="text" id="name" maxlength="40" placeholder="Full name" autocomplete="off">
    <div class="bar" id="bar" hidden><span id="barFill"></span></div>
    <div class="row"><button id="enrollBtn">Start capture</button><button class="ghost" id="trainBtn">Train model</button></div>
    <p id="msg" role="status">Start the camera, capture 40 photos, then train.</p></div>
  <div class="card"><h2>Checked in <span class="pill" id="count">0</span></h2>
    <ul id="roster"></ul>
    <div class="row"><button class="ghost sm" id="clearBtn">Clear list</button><button class="ghost sm" id="csvBtn">Download CSV</button></div></div>
  <div class="card"><h2>People <span class="pill" id="peopleCount">0</span></h2>
    <ul id="people"></ul>
    <div class="row"><button class="danger sm" id="resetBtn">Delete my data</button></div></div>
</aside>
</main>
<script>
const $ = id => document.getElementById(id);
const newId = () => (window.crypto && crypto.randomUUID) ? crypto.randomUUID() : (Date.now() + "-" + String(Math.random()).slice(2));
let sid;
try { sid = localStorage.getItem("fa_sid"); if (!sid) { sid = newId(); localStorage.setItem("fa_sid", sid); } }
catch (e) { sid = sid || newId(); }
let fails = 0, lastTrained = null;
function banner(msg) { const b = $("banner"); b.textContent = msg || ""; b.hidden = !msg; }
const api = (path, opts) => fetch(path + (path.includes("?") ? "&" : "?") + "sid=" + sid, opts).then(r => r.json());
const post = (path, body) => api(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
const initials = n => n.split(/\s+/).map(w => w[0] || "").slice(0, 2).join("").toUpperCase();
const video = $("video");
const MAXW = 640;
let running = false, busy = false, enrollId = null, startPhotos = 0, shots = 40, present = {}, people = [];

$("thr").oninput = () => $("thrVal").textContent = $("thr").value;

$("camBtn").onclick = async () => {
  if (running) { stopCam(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480, facingMode: "user"}, audio: false});
    video.srcObject = stream; await video.play();
    running = true; $("ph").hidden = true;
    $("camBtn").textContent = "Stop camera"; $("camPill").textContent = "Camera live";
  } catch (e) { $("msg").textContent = "Could not open the camera: " + e.message; }
};
function stopCam() {
  running = false; stopCapture();
  (video.srcObject ? video.srcObject.getTracks() : []).forEach(t => t.stop());
  video.srcObject = null; $("ph").hidden = false;
  $("camBtn").textContent = "Start camera"; $("camPill").textContent = "Camera off";
  const c = $("overlay"); c.getContext("2d").clearRect(0, 0, c.width, c.height);
}

const off = document.createElement("canvas");
let sent = 0;
const diag = text => { $("diag").textContent = text; };
async function tick() {
  if (!running || busy || !video.videoWidth) return;
  busy = true;
  try {
    const W = MAXW, H = Math.round(MAXW * video.videoHeight / video.videoWidth);
    off.width = W; off.height = H;
    off.getContext("2d").drawImage(video, 0, 0, W, H);
    const blob = await new Promise(r => off.toBlob(r, "image/jpeg", 0.7));
    if (!blob) throw new Error("could not read a picture from the camera");
    const q = new URLSearchParams({sid, mode: enrollId ? "enroll" : "watch", pid: enrollId || "", thr: $("thr").value});
    const ctrl = new AbortController(), timer = setTimeout(() => ctrl.abort(), 8000);
    let res;
    try { res = await fetch("/api/frame?" + q, {method: "POST", body: blob, signal: ctrl.signal}); }
    finally { clearTimeout(timer); }
    sent++;
    if (res.status === 503) { banner("The demo is full right now. Try again in a few minutes."); diag("The server is full (HTTP 503)."); }
    else if (!res.ok) { throw new Error("the server answered HTTP " + res.status); }
    else {
      const d = await res.json(); fails = 0; banner("");
      diag("Server received a " + d.w + "x" + d.h + " picture and found " + d.faces.length + " face(s) in " + d.ms + " ms. Pictures sent: " + sent + ".");
      if (d.faces) handle(d, W);
    }
  } catch (e) {
    diag("Problem: " + (e.name === "AbortError" ? "no reply from the server within 8 seconds" : e.message));
    if (++fails >= 3) banner("Lost connection to the server. If the app is restarting or waking up from sleep, wait a minute and reload the page.");
  }
  busy = false;
}
setInterval(tick, 300);

function handle(d, W) {
  if (lastTrained !== null && d.trained !== lastTrained) loadState();
  lastTrained = d.trained;
  const c = $("overlay"); c.width = video.clientWidth; c.height = video.clientHeight;
  const k = c.width / W, g = c.getContext("2d");
  g.lineWidth = 3; g.font = "600 15px system-ui,sans-serif";
  const col = {ok: "#2e9e4f", bad: "#d6403a", warn: "#e8890c", enroll: "#e8890c"};
  d.faces.forEach(f => {
    g.strokeStyle = g.fillStyle = col[f.state];
    g.strokeRect(f.x * k, f.y * k, f.w * k, f.h * k);
    const ty = Math.max(f.y * k, 22), tw = g.measureText(f.name).width + 10;
    g.fillRect(f.x * k, ty - 22, tw, 22);
    g.fillStyle = "#fff"; g.fillText(f.name, f.x * k + 5, ty - 6);
  });
  const f0 = d.faces[0];
  $("chip").textContent = f0 ? f0.name + (f0.conf !== null ? "  (distance " + f0.conf + ")" : "") : "No face in view";
  $("modelPill").textContent = d.trained ? "Model ready" : "Model not trained";
  present = d.present; drawRoster();
  if (enrollId) {
    const done = Math.max(0, d.photos - startPhotos);
    $("barFill").style.width = Math.min(100, done / shots * 100) + "%";
    const got = Math.min(done, shots) + " of " + shots + " photos";
    if (!d.faces.length) $("msg").textContent = "No face found (" + got + "). Look straight at the camera, move closer, and put some light on your face.";
    else if (d.faces[0].w / W < 0.14) $("msg").textContent = "Move a little closer (" + got + ").";
    else $("msg").textContent = "Capturing photos: " + got + ". Turn your head slightly.";
    if (done >= shots) {
      stopCapture();
      $("msg").textContent = "Capture finished. Now click Train model."; loadState();
    }
  }
}

function drawRoster() {
  const ul = $("roster"); ul.replaceChildren();
  const e = Object.entries(present); $("count").textContent = e.length;
  if (!e.length) { ul.append(li("empty", "No one has checked in yet.")); return; }
  e.reverse().forEach(([n, t]) => {
    const row = document.createElement("li");
    row.append(div("av", initials(n)), span("nm", n), Object.assign(document.createElement("time"), {textContent: new Date(t * 1000).toLocaleTimeString()}));
    ul.append(row);
  });
}
const span = (cls, text) => Object.assign(document.createElement("span"), {className: cls, textContent: text});
const div = (cls, text) => Object.assign(document.createElement("div"), {className: cls, textContent: text});
const li = (cls, text) => { const x = document.createElement("li"); x.append(span(cls, text)); return x; };

async function loadState() {
  let s;
  try { s = await api("/api/state"); }
  catch (e) { banner("Could not reach the server. If the app is waking up from sleep, wait a minute and reload the page."); return; }
  if (!s.people) { banner("The server answered with an error. Try again in a moment."); return; }
  banner("");
  people = s.people; shots = s.shots; present = s.present;
  $("modelPill").textContent = s.trained ? "Model ready" : "Model not trained";
  const ul = $("people"); ul.replaceChildren(); $("peopleCount").textContent = people.length;
  if (!people.length) ul.append(li("empty", "No one enrolled yet."));
  people.forEach(p => {
    const add = Object.assign(document.createElement("button"), {className: "ghost sm", textContent: "Add photos"});
    const del = Object.assign(document.createElement("button"), {className: "danger sm", textContent: "Delete"});
    add.onclick = () => startCapture(p.id, p.photos);
    del.onclick = async () => { if (confirm("Delete " + p.name + " and their photos?")) { await api("/api/person/" + p.id, {method: "DELETE"}); loadState(); } };
    const row = document.createElement("li");
    const info = document.createElement("div"); info.className = "nm";
    info.append(span("", p.name), document.createElement("br"), span("sub", p.photos + " photos"));
    row.append(div("av", initials(p.name)), info, add, del);
    ul.append(row);
  });
  drawRoster();
}

function startCapture(id, existing) {
  if (!running) { $("msg").textContent = "Start the camera first."; return; }
  enrollId = id; startPhotos = existing; $("bar").hidden = false; $("barFill").style.width = "0";
  $("enrollBtn").textContent = "Stop capture";
}
function stopCapture(note) {
  enrollId = null; $("bar").hidden = true; $("enrollBtn").textContent = "Start capture";
  if (note) $("msg").textContent = note;
}
$("enrollBtn").onclick = async () => {
  if (enrollId) { stopCapture("Capture stopped."); return; }
  const name = $("name").value.trim();
  if (!running) { $("msg").textContent = "Start the camera first."; return; }
  if (!name) { $("msg").textContent = "Type a name first."; $("name").focus(); return; }
  const r = await post("/api/person", {name});
  if (!r.ok) { $("msg").textContent = r.msg || "Could not add this person."; return; }
  $("name").value = ""; startCapture(r.id, 0); loadState();
};
$("trainBtn").onclick = async () => {
  if (enrollId) { $("msg").textContent = "Wait until the photos are captured (or press Stop capture), then click Train model."; return; }
  const r = await post("/api/train"); $("msg").textContent = r.msg; loadState(); };
$("clearBtn").onclick = async () => { await post("/api/clear"); present = {}; drawRoster(); };
$("resetBtn").onclick = async () => {
  if (!confirm("Delete all your photos, names and the model?")) return;
  stopCapture();
  await post("/api/reset"); present = {}; loadState(); $("msg").textContent = "Your data was deleted.";
};
$("csvBtn").onclick = () => {
  const safe = x => { x = String(x); if (/^[=+\-@]/.test(x)) x = "'" + x; return '"' + x.replace(/"/g, '""') + '"'; };
  const rows = [["Name", "Date", "Time"]].concat(Object.entries(present).map(([n, t]) => {
    const d = new Date(t * 1000); return [n, d.toLocaleDateString(), d.toLocaleTimeString()]; }));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.map(r => r.map(safe).join(",")).join("\n")], {type: "text/csv"}));
  a.download = "attendance.csv"; a.click();
};
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
let deferredInstall = null;
window.addEventListener("beforeinstallprompt", e => { e.preventDefault(); deferredInstall = e; $("installBtn").hidden = false; });
$("installBtn").onclick = async () => { if (!deferredInstall) return; deferredInstall.prompt(); await deferredInstall.userChoice; deferredInstall = null; $("installBtn").hidden = true; };
if (/iphone|ipad/i.test(navigator.userAgent) && !navigator.standalone) $("iosHint").hidden = false;
loadState();
</script></body></html>"""


LANDING = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Face Attendance - check in with your camera</title>
<meta name="description" content="Take attendance with face recognition, right in your browser.">
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#2457d6">
<link rel="apple-touch-icon" href="/icon-192.png">
<style>
:root{--bg:#f5f6f8;--card:#fff;--ink:#16181d;--mut:#667085;--line:#e3e6eb;--acc:#2457d6}
@media(prefers-color-scheme:dark){:root{--bg:#101216;--card:#181b21;--ink:#eceef2;--mut:#98a2b3;--line:#2a2f38;--acc:#6b93f5}}
*{box-sizing:border-box}[hidden]{display:none!important}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:960px;margin:auto;padding:0 20px}
nav{display:flex;justify-content:space-between;align-items:center;padding:16px 0}nav b{font-size:18px}
a.btn,button.btn{display:inline-block;font:inherit;text-decoration:none;border:0;border-radius:10px;padding:12px 20px;background:var(--acc);color:#fff;cursor:pointer}
.ghost{background:transparent!important;color:var(--ink)!important;border:1px solid var(--line)!important}
.hero{padding:44px 0 28px}.hero h1{font-size:clamp(30px,6vw,46px);line-height:1.12;margin:0 0 14px}
.hero p{font-size:19px;color:var(--mut);max-width:620px;margin:0 0 22px}.hero .row{display:flex;gap:10px;flex-wrap:wrap}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}.card h3{margin:0 0 6px;font-size:17px}.card p{margin:0;color:var(--mut);font-size:15px}
h2{margin:34px 0 10px;font-size:24px}ol{padding-left:22px}ol li{margin:8px 0}
.note{color:var(--mut);font-size:14px}footer{padding:30px 0 40px;color:var(--mut);font-size:14px;border-top:1px solid var(--line);margin-top:36px}
</style></head>
<body><div class="wrap">
<nav><b>Face Attendance</b><a class="btn" href="/app">Open the app</a></nav>
<section class="hero">
  <h1>Take attendance with a camera and nothing else.</h1>
  <p>Add people once, then just stand in front of the camera. Names and check-in times appear on the list
  automatically, and you can download the list as a spreadsheet file.</p>
  <div class="row"><a class="btn" href="/app">Start now</a>
  <button class="btn ghost" id="installBtn" hidden>Install on this device</button></div>
  <p class="note" id="iosHint" hidden style="margin-top:12px">On iPhone: tap the Share button, then <b>Add to Home Screen</b>.</p>
</section>
<div class="grid">
  <div class="card"><h3>Works in the browser</h3><p>No software to install. Open the link on a laptop or a phone and allow the camera.</p></div>
  <div class="card"><h3>Recognises faces live</h3><p>Faces are found and matched as people walk by, with a coloured box and a name.</p></div>
  <div class="card"><h3>Download the list</h3><p>One button saves everyone who checked in, with the date and time, as a CSV file.</p></div>
</div>
<h2>How it works</h2>
<ol>
  <li>Open the app and press <b>Start camera</b>.</li>
  <li>Type a name and press <b>Start capture</b>. The app takes 40 photos while you turn your head slightly.</li>
  <li>Press <b>Train model</b>. This takes a second or two.</li>
  <li>Stand in front of the camera. You are checked in and the time is recorded.</li>
</ol>
<h2>Install it like an app</h2>
<p><b>Android (Chrome):</b> open the app, then use the menu and choose <b>Install app</b>.<br>
<b>iPhone (Safari):</b> tap Share, then <b>Add to Home Screen</b>.<br>
<b>Laptop (Chrome or Edge):</b> click the install icon at the right end of the address bar.</p>
<p class="note">Your list of people lives in the server's memory only. It is cleared after 30 minutes without use, and whenever the app restarts or wakes up from sleep.</p>
<footer>Face Attendance &middot; built with Python, OpenCV and Flask</footer>
</div>
<script>
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
let deferredInstall = null;
const ib = document.getElementById("installBtn");
window.addEventListener("beforeinstallprompt", e => { e.preventDefault(); deferredInstall = e; ib.hidden = false; });
ib.onclick = async () => { if (!deferredInstall) return; deferredInstall.prompt(); await deferredInstall.userChoice; deferredInstall = null; ib.hidden = true; };
if (/iphone|ipad/i.test(navigator.userAgent) && !navigator.standalone) document.getElementById("iosHint").hidden = false;
</script></body></html>"""

SW = r"""
const OFFLINE = `<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1">
<body style="font-family:system-ui,sans-serif;max-width:520px;margin:40px auto;padding:0 20px;line-height:1.5">
<h2>The server is offline</h2>
<p>The app may be waking up from sleep or restarting. Wait about a minute, then reload this page.</p>
<p><button onclick="location.reload()" style="padding:10px 18px;font-size:16px">Reload</button></p>`;
const offline = () => new Response(OFFLINE, {headers: {'Content-Type': 'text/html; charset=utf-8'}});
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));
self.addEventListener('fetch', e => {
  if (e.request.mode !== 'navigate') return;
  e.respondWith(fetch(e.request).then(r => (r.status >= 500 ? offline() : r)).catch(offline));
});
"""


def make_icon(size):
    """Draw the app icon (blue square, white scan corners, head and shoulders)."""
    img = np.full((size, size, 3), (214, 87, 36), np.uint8)   # #2457d6 in BGR
    white, c = (255, 255, 255), size // 2
    t, m, n = max(2, size // 24), int(size * 0.2), int(size * 0.16)
    for x, y, dx, dy in [(m, m, 1, 1), (size - m, m, -1, 1), (m, size - m, 1, -1), (size - m, size - m, -1, -1)]:
        cv2.line(img, (x, y), (x + dx * n, y), white, t)
        cv2.line(img, (x, y), (x, y + dy * n), white, t)
    cv2.circle(img, (c, int(size * 0.42)), int(size * 0.12), white, -1)
    cv2.ellipse(img, (c, int(size * 0.74)), (int(size * 0.2), int(size * 0.16)), 0, 180, 360, white, -1)
    return cv2.imencode(".png", img)[1].tobytes()


ICONS = {192: make_icon(192), 512: make_icon(512)}


@app.get("/")
def index():
    return LANDING


@app.get("/app")
def app_page():
    return PAGE


@app.get("/manifest.webmanifest")
def manifest():
    data = {"name": "Face Attendance", "short_name": "Attendance", "start_url": "/app", "scope": "/",
            "display": "standalone", "background_color": "#f5f6f8", "theme_color": "#2457d6",
            "icons": [{"src": f"/icon-{n}.png", "sizes": f"{n}x{n}", "type": "image/png",
                       "purpose": "any maskable"} for n in ICONS]}
    return Response(json.dumps(data), mimetype="application/manifest+json")


@app.get("/sw.js")
def service_worker():
    r = Response(SW, mimetype="application/javascript")
    r.headers["Service-Worker-Allowed"] = "/"
    r.headers["Cache-Control"] = "no-cache"
    return r


@app.get("/icon-<int:size>.png")
def icon(size):
    if size not in ICONS:
        abort(404)
    r = Response(ICONS[size], mimetype="image/png")
    r.headers["Cache-Control"] = "public, max-age=86400"
    return r


def run(port=7860):
    app.run(host="0.0.0.0", port=port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    run()
