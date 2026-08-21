const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let backendProcess = null;

function startBackend() {
  // Локальный бэкенд поднимается как дочерний процесс — пользователь
  // не видит консоли, для него это просто "приложение открылось".
  //
  // ВАЖНО: используем python.exe именно из venv бэкенда, а не системную
  // команду 'python' — иначе Electron находит первый попавшийся Python
  // в PATH (может быть версия без установленных зависимостей проекта:
  // uvicorn, coqui-tts, insightface и т.д.) и падает с "No module named".
  const backendDir = path.join(__dirname, '..', 'backend');
  const venvPython = process.platform === 'win32'
    ? path.join(backendDir, 'venv', 'Scripts', 'python.exe')
    : path.join(backendDir, 'venv', 'bin', 'python');

  backendProcess = spawn(venvPython, ['-m', 'uvicorn', 'main:app', '--port', '8420'], {
    cwd: backendDir,
  });
  backendProcess.stdout.on('data', (data) => {
    console.log(`[backend] ${data}`);
  });
  backendProcess.stderr.on('data', (data) => {
    console.error(`[backend] ${data}`);
  });
  backendProcess.on('error', (err) => {
    console.error(`[backend] Не удалось запустить: ${err.message}`);
    console.error(`[backend] Ожидаемый путь к python: ${venvPython}`);
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 1080,
    minHeight: 680,
    backgroundColor: '#14120e',
    titleBarStyle: 'hiddenInset',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, 'src', 'index.html'));
}

ipcMain.handle('dialog:openFile', async (_event, filters) => {
  const result = await dialog.showOpenDialog({ properties: ['openFile'], filters });
  return result.canceled ? null : result.filePaths[0];
});

// Файлы читаются с диска здесь, в main-процессе (у renderer с
// contextIsolation нет доступа к fs) — и сразу отправляются на бэкенд
// как multipart. Renderer работает только с готовым JSON-ответом.
const fs = require('fs');

async function postFileToBackend(endpoint, filePath, fields) {
  const buffer = fs.readFileSync(filePath);
  const fileName = path.basename(filePath);

  const form = new FormData();
  form.append('file', new Blob([buffer]), fileName);
  for (const [key, value] of Object.entries(fields)) {
    form.append(key, String(value));
  }

  const res = await fetch(`http://127.0.0.1:8420${endpoint}`, {
    method: 'POST',
    body: form,
  });

  const body = await res.json();
  if (!res.ok) {
    throw new Error(body.detail || `Backend error ${res.status}`);
  }
  return body;
}

ipcMain.handle('backend:createAvatar', async (_event, { filePath, name, consentConfirmed }) => {
  return postFileToBackend('/avatar/create', filePath, {
    name,
    consent_confirmed: consentConfirmed,
  });
});

ipcMain.handle('backend:cloneVoice', async (_event, { filePath, name, consentConfirmed }) => {
  return postFileToBackend('/voice/clone', filePath, {
    name,
    consent_confirmed: consentConfirmed,
  });
});

ipcMain.handle('backend:generateSegment', async (_event, payload) => {
  // Локальная генерация видео на CPU может занимать 5+ минут на чанк —
  // дольше стандартного таймаута fetch на ожидание ответа. AbortSignal
  // здесь встроен в Node.js/Electron и не требует внешних пакетов.
  const res = await fetch('http://127.0.0.1:8420/segment/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal: AbortSignal.timeout(20 * 60 * 1000), // 20 минут
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || `Backend error ${res.status}`);
  return body;
});

ipcMain.handle('backend:getJob', async (_event, jobId) => {
  const res = await fetch(`http://127.0.0.1:8420/jobs/${jobId}`);
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || `Backend error ${res.status}`);
  return body;
});

ipcMain.handle('backend:stitchProject', async (_event, payload) => {
  const res = await fetch('http://127.0.0.1:8420/project/stitch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const body = await res.json();
  if (!res.ok) throw new Error(body.detail || `Backend error ${res.status}`);
  return body;
});

app.whenReady().then(() => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (backendProcess) backendProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});