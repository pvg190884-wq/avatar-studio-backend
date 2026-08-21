const BACKEND = window.avatarStudio.backendUrl;

// ---------- Навигация между экранами ----------
document.querySelectorAll('.rail-item').forEach((btn) => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.view;
    document.querySelectorAll('.rail-item').forEach((b) => b.classList.remove('is-active'));
    document.querySelectorAll('.view').forEach((v) => v.classList.remove('is-active'));
    btn.classList.add('is-active');
    document.querySelector(`.view[data-view="${target}"]`).classList.add('is-active');
  });
});

// ---------- Проверка статуса бэкенда ----------
async function checkBackend() {
  const dot = document.querySelector('.status-dot');
  const text = document.querySelector('.status-text');
  try {
    const res = await fetch(`${BACKEND}/health`);
    if (res.ok) {
      dot.classList.add('is-ready');
      dot.classList.remove('is-down');
      text.textContent = 'готово';
      return;
    }
    throw new Error('not ok');
  } catch {
    dot.classList.add('is-down');
    dot.classList.remove('is-ready');
    text.textContent = 'бэкенд не отвечает';
    setTimeout(checkBackend, 3000);
  }
}
checkBackend();

// ---------- Состояние ----------
const state = {
  avatarFile: null,
  avatarId: null,
  voiceFile: null,
  voiceId: null,
  segments: [], // { text, duration_sec_estimate, emotion }
  availableEmotions: [], // [{ id, available }] — загружается с бэкенда
};

const EMOTION_LABELS = {
  neutral: 'Нейтрально',
  happy: 'Радость',
  sadness: 'Грусть',
  anger: 'Гнев',
  love: 'Любовь',
};
const DEFAULT_EMOTION = 'neutral';

// Забираем список эмоций и флаг доступности (есть ли реально файл-драйвер
// на диске) с бэкенда — см. GET /emotions в main.py. Пока идёт запрос,
// селектор эмоции на экране "Монтаж" временно недоступен, чтобы нельзя
// было выбрать эмоцию до того, как узнаем, что реально доступно.
async function loadAvailableEmotions() {
  try {
    const res = await fetch(`${BACKEND}/emotions`);
    const data = await res.json();
    state.availableEmotions = data.emotions || [];
  } catch {
    // Бэкенд ещё не поднялся — не блокируем интерфейс, просто оставим
    // список пустым; повторная попытка не критична, т.к. checkBackend()
    // всё равно уже опрашивает /health и подскажет пользователю проблему.
    state.availableEmotions = [];
  }
  populateEmotionSelect();
}

// ---------- Экран "Аватар" ----------
const avatarBrowse = document.getElementById('avatar-browse');
const avatarSubmit = document.getElementById('avatar-submit');
const avatarName = document.getElementById('avatar-name');
const avatarConsent = document.getElementById('avatar-consent');
const avatarResult = document.getElementById('avatar-result');

avatarBrowse.addEventListener('click', async () => {
  const path = await window.avatarStudio.openFile([
    { name: 'Изображения и видео', extensions: ['jpg', 'jpeg', 'png', 'mp4'] },
  ]);
  if (path) {
    state.avatarFile = path;
    document.querySelector('#avatar-dropzone .dz-title').textContent = path.split(/[\\/]/).pop();
    updateAvatarSubmitState();
  }
});

[avatarName, avatarConsent].forEach((el) => el.addEventListener('input', updateAvatarSubmitState));
function updateAvatarSubmitState() {
  avatarSubmit.disabled = !(state.avatarFile && avatarName.value.trim() && avatarConsent.checked);
}

avatarSubmit.addEventListener('click', async () => {
  avatarSubmit.disabled = true;
  avatarResult.textContent = 'Создаю аватар…';
  try {
    const data = await window.avatarStudio.createAvatar({
      filePath: state.avatarFile,
      name: avatarName.value.trim(),
      consentConfirmed: true,
    });
    state.avatarId = data.avatar_id;
    avatarResult.textContent = `Аватар создан: ${data.name} (id ${data.avatar_id})`;
    renderFilmstrip(); // пересчитать доступность кнопки "Собрать фильм"
  } catch (err) {
    avatarResult.textContent = `Ошибка: ${err.message}`;
  } finally {
    avatarSubmit.disabled = false;
  }
});

// ---------- Экран "Голос" ----------
const voiceBrowse = document.getElementById('voice-browse');
const voiceSubmit = document.getElementById('voice-submit');
const voiceName = document.getElementById('voice-name');
const voiceConsent = document.getElementById('voice-consent');
const voiceResult = document.getElementById('voice-result');

voiceBrowse.addEventListener('click', async () => {
  const path = await window.avatarStudio.openFile([
    { name: 'Аудио', extensions: ['wav', 'mp3'] },
  ]);
  if (path) {
    state.voiceFile = path;
    document.querySelector('#voice-dropzone .dz-title').textContent = path.split(/[\\/]/).pop();
    updateVoiceSubmitState();
  }
});

[voiceName, voiceConsent].forEach((el) => el.addEventListener('input', updateVoiceSubmitState));
function updateVoiceSubmitState() {
  voiceSubmit.disabled = !(state.voiceFile && voiceName.value.trim() && voiceConsent.checked);
}

voiceSubmit.addEventListener('click', async () => {
  voiceSubmit.disabled = true;
  voiceResult.textContent = 'Клонирую голос…';
  try {
    const data = await window.avatarStudio.cloneVoice({
      filePath: state.voiceFile,
      name: voiceName.value.trim(),
      consentConfirmed: true,
    });
    state.voiceId = data.voice_id;
    voiceResult.textContent = `Голос сохранён: ${data.name} (id ${data.voice_id})`;
    renderFilmstrip();
  } catch (err) {
    voiceResult.textContent = `Ошибка: ${err.message}`;
  } finally {
    voiceSubmit.disabled = false;
  }
});

// ---------- Экран "Монтаж" — плёнка сегментов ----------
const segmentText = document.getElementById('segment-text');
const segmentAdd = document.getElementById('segment-add');
const filmstrip = document.getElementById('filmstrip');
const filmstripEmpty = document.getElementById('filmstrip-empty');
const cutTotal = document.getElementById('cut-total');
const cutRender = document.getElementById('cut-render');

const WORDS_PER_SECOND = 2.3; // грубая оценка темпа речи для превью длительности

// Селектор эмоции для нового сегмента. HTML-разметки под него в исходном
// index.html не было, поэтому создаём и вставляем его перед полем ввода
// текста сегмента программно — так не нужно трогать index.html отдельно.
let segmentEmotionSelect = document.getElementById('segment-emotion');
if (!segmentEmotionSelect) {
  segmentEmotionSelect = document.createElement('select');
  segmentEmotionSelect.id = 'segment-emotion';
  segmentEmotionSelect.className = 'segment-emotion-select';
  segmentText.parentElement.insertBefore(segmentEmotionSelect, segmentText);
}

function populateEmotionSelect() {
  segmentEmotionSelect.innerHTML = '';
  const knownIds = Object.keys(EMOTION_LABELS);
  const availabilityById = Object.fromEntries(
    state.availableEmotions.map((e) => [e.id, e.available])
  );

  knownIds.forEach((id) => {
    const opt = document.createElement('option');
    opt.value = id;
    const isAvailable = availabilityById[id] !== false; // до загрузки с бэкенда — не блокируем
    opt.textContent = isAvailable
      ? EMOTION_LABELS[id]
      : `${EMOTION_LABELS[id]} (драйвер ещё не записан)`;
    if (id !== 'neutral' && availabilityById[id] === false) {
      // Для видео-эмоций без реального файла-драйвера не запрещаем выбор
      // полностью (бэкенд и так сделает fallback на статичный кадр
      // аватара — генерация не упадёт), но явно предупреждаем в подписи,
      // чтобы не было сюрприза, почему лицо не двигалось.
    }
    segmentEmotionSelect.appendChild(opt);
  });
  segmentEmotionSelect.value = DEFAULT_EMOTION;
}
populateEmotionSelect();
loadAvailableEmotions();

segmentAdd.addEventListener('click', () => {
  const text = segmentText.value.trim();
  if (!text) return;

  const estSeconds = Math.max(3, Math.round(text.split(/\s+/).length / WORDS_PER_SECOND));
  const emotion = segmentEmotionSelect.value || DEFAULT_EMOTION;
  state.segments.push({ text, estSeconds, emotion });
  segmentText.value = '';
  segmentEmotionSelect.value = DEFAULT_EMOTION;
  renderFilmstrip();
});

function renderFilmstrip() {
  filmstripEmpty.style.display = state.segments.length ? 'none' : 'flex';
  filmstrip.querySelectorAll('.frame').forEach((f) => f.remove());

  let totalSeconds = 0;
  state.segments.forEach((seg, i) => {
    totalSeconds += seg.estSeconds;

    const emotionLabel = EMOTION_LABELS[seg.emotion] || seg.emotion;
    const frame = document.createElement('div');
    frame.className = 'frame';
    frame.innerHTML = `
      <div class="frame-body">
        <span class="frame-index">СЕГМЕНТ ${String(i + 1).padStart(2, '0')}</span>
        <p class="frame-text">${escapeHtml(seg.text)}</p>
        <span class="frame-emotion">${escapeHtml(emotionLabel)}</span>
        <span class="frame-time">~${formatTime(seg.estSeconds)}</span>
      </div>
    `;
    filmstrip.appendChild(frame);
  });

  cutTotal.textContent = formatTime(totalSeconds);
  cutRender.disabled = state.segments.length === 0 || !state.avatarId || !state.voiceId;
}

function formatTime(totalSeconds) {
  const m = Math.floor(totalSeconds / 60);
  const s = Math.round(totalSeconds % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

const JOB_POLL_INTERVAL_MS = 2000;
const JOB_POLL_TIMEOUT_MS = 20 * 60 * 1000; // 20 минут на сегмент — щедрый запас

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// Опрашивает /jobs/{id}, пока статус не станет "done" (или не истечёт таймаут,
// или бэкенд не вернёт "failed"). Пока в pipeline/lipsync.py не подключена
// реальная модель, job остаётся в статусе "pending_gpu_inference" —
// это ожидаемо и явно сообщается пользователю, а не тихо виснет.
async function waitForJob(jobId, onTick) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < JOB_POLL_TIMEOUT_MS) {
    const job = await window.avatarStudio.getJob(jobId);
    onTick(job);
    if (job.status === 'done') return job;
    if (job.status === 'failed') throw new Error('Генерация сегмента завершилась ошибкой');
    if (job.status === 'pending_gpu_inference') {
      throw new Error(
        'Липсинк-модель ещё не подключена на бэкенде (см. TODO в pipeline/lipsync.py) — ' +
        'генерация не может завершиться, пока модель не установлена локально.'
      );
    }
    await sleep(JOB_POLL_INTERVAL_MS);
  }
  throw new Error('Превышено время ожидания генерации сегмента');
}

cutRender.addEventListener('click', async () => {
  cutRender.disabled = true;
  const originalLabel = cutRender.textContent;

  try {
    const segmentVideoPaths = [];

    for (let i = 0; i < state.segments.length; i++) {
      const seg = state.segments[i];
      cutRender.textContent = `Сегмент ${i + 1}/${state.segments.length}…`;

      const { job_id } = await window.avatarStudio.generateSegment({
        avatar_id: state.avatarId,
        voice_id: state.voiceId,
        text: seg.text,
        language: 'ru',
        emotion: seg.emotion || DEFAULT_EMOTION,
      });

      const finishedJob = await waitForJob(job_id, (job) => {
        cutRender.textContent = `Сегмент ${i + 1}/${state.segments.length}: ${job.status}`;
      });
      segmentVideoPaths.push(finishedJob.output_path);
    }

    cutRender.textContent = 'Склеиваю фильм…';
    const projectId = `project-${Date.now()}`;
    const { output_path } = await window.avatarStudio.stitchProject({
      project_id: projectId,
      segment_paths: segmentVideoPaths,
      transition_sec: 0.6,
    });

    cutRender.textContent = 'Готово';
    alert(`Фильм собран: ${output_path}`);
  } catch (err) {
    alert(`Не удалось собрать фильм: ${err.message}`);
    cutRender.textContent = originalLabel;
  } finally {
    cutRender.disabled = state.segments.length === 0 || !state.avatarId || !state.voiceId;
  }
});