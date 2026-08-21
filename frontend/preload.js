const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('avatarStudio', {
  openFile: (filters) => ipcRenderer.invoke('dialog:openFile', filters),
  backendUrl: 'http://127.0.0.1:8420',

  // Все эти вызовы читают файл и ходят к бэкенду из main-процесса —
  // renderer никогда не видит сырой FormData с путём к файлу на диске.
  createAvatar: (args) => ipcRenderer.invoke('backend:createAvatar', args),
  cloneVoice: (args) => ipcRenderer.invoke('backend:cloneVoice', args),
  generateSegment: (payload) => ipcRenderer.invoke('backend:generateSegment', payload),
  getJob: (jobId) => ipcRenderer.invoke('backend:getJob', jobId),
  stitchProject: (payload) => ipcRenderer.invoke('backend:stitchProject', payload),
});
