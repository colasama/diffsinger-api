// Run with `node --test tests/test_ui.cjs` with playwright on NODE_PATH.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { chromium } = require('playwright');

function silentWav() {
  const data = Buffer.alloc(44 + 8000 * 2 * 30);
  data.write('RIFF'); data.writeUInt32LE(data.length - 8, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(8000, 24); data.writeUInt32LE(16000, 28);
  data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write('data', 36); data.writeUInt32LE(data.length - 44, 40);
  return data;
}

test('delete confirms, handles errors, clears selection and protects active tasks', { timeout: 30000 }, async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const html = readFileSync(join(__dirname, '../src/dsapi/static/index.html'), 'utf8');
    let tasks = ['failed', 'done', 'running', 'queued'].map(status => ({
      task_id: status, status, spec: {}, timings: {}, outputs: {}, song: { name: status },
      alignment: [{ line: '过', backend: 'hubertfa', confidence: null, syllables: 1, notes: 1 }]
    }));
    let deletes = 0, reject = false, accept = false;
    page.on('dialog', dialog => dialog.type() === 'confirm' && !accept ? dialog.dismiss() : dialog.accept());
    await page.route('http://dsapi.test/**', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/ui/') return route.fulfill({ contentType: 'text/html', body: html });
      if (url.pathname === '/tasks') return route.fulfill({ json: tasks });
      if (route.request().method() === 'DELETE') {
        deletes++;
        assert.equal(url.searchParams.get('key'), 'secret');
        if (reject) return route.fulfill({ status: 409, body: 'task busy' });
        tasks = tasks.filter(t => t.task_id !== url.pathname.split('/').pop());
        return route.fulfill({ json: { deleted: true } });
      }
      if (url.pathname === '/query_result') return route.fulfill({ json: tasks.find(t => t.task_id === url.searchParams.get('task_id')) });
      return route.fulfill({ status: 404 });
    });
    await page.goto('http://dsapi.test/ui/');
    await page.locator('#apikey').fill('secret');
    const row = id => page.locator(`#tasks li[data-id="${id}"]`);
    await row('failed').waitFor();
    assert.equal(await row('running').locator('button').isDisabled(), true);
    assert.equal(await row('queued').locator('button').isDisabled(), true);
    await row('failed').locator('button').click();
    assert.equal(deletes, 0);
    assert.match(await page.locator('#detail').textContent(), /选择或提交/);
    await row('failed').locator('span').first().click();
    await page.locator('#taskStatus').waitFor();
    assert.equal(await page.locator('#taskExtras table tr').nth(1).locator('td').nth(1).textContent(), '—');
    accept = true;
    reject = true;
    await row('failed').locator('button').click();
    await page.waitForFunction(() => !document.querySelector('[data-id="failed"] button').disabled);
    assert.equal(deletes, 1);
    assert.equal(await page.locator('#detail').getAttribute('data-task-id'), 'failed');
    reject = false;
    await row('failed').locator('button').click();
    await row('failed').waitFor({ state: 'detached' });
    assert.match(await page.locator('#detail').textContent(), /任务已删除/);
    assert.equal(await page.locator('#taskPlayers').count(), 0);
    assert.equal(await page.evaluate(() => current === null && elapsedClock === null), true);
    await row('done').locator('button').click();
    await row('done').waitFor({ state: 'detached' });
    assert.equal(await page.locator('#tasks li').count(), 2);
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
});

test('audio survives polling and completion; elapsed time advances and freezes', { timeout: 30000 }, async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const html = readFileSync(join(__dirname, '../src/dsapi/static/index.html'), 'utf8');
    const wav = silentWav();
    const started = Date.now();
    let polls = 0;
    let failPoll = false;
    let runtime = { configured: 'auto', priority: ['cuda', 'webgpu', 'cpu'], sessions: [] };
    let state = { task_id: 'partial', status: 'running', stage: 'render_score',
      spec: { modes: ['mimic', 'score'] }, timings: { render_mimic: 10 },
      outputs: { mimic: '/v1/audio?task_id=partial&kind=mimic' } };
    await page.route('http://dsapi.test/**', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/ui/') return route.fulfill({ contentType: 'text/html', body: html });
      if (url.pathname === '/tasks') return route.fulfill({ json: [state] });
      if (url.pathname === '/runtime') return route.fulfill({ json: { synthesis: runtime } });
      if (url.pathname === '/query_result') {
        polls++;
        if (failPoll) return route.fulfill({ status: 503, body: 'temporary failure' });
        return route.fulfill({ json: { ...state, elapsed_seconds: state.status === 'running'
          ? 10 + (Date.now() - started) / 1000 : state.elapsed_seconds } });
      }
      if (url.pathname === '/v1/audio') return route.fulfill({ contentType: 'audio/wav', body: wav });
      return route.fulfill({ status: 404 });
    });
    await page.goto('http://dsapi.test/ui/');
    await page.waitForFunction(() => document.querySelector('#runtime').textContent.includes('待加载'));
    assert.match(await page.locator('#runtime').textContent(), /GPU \(CUDA\) → WebGPU → CPU/);
    runtime.sessions = [{ model: 'acoustic', backend: 'webgpu',
      providers: ['WebGpuExecutionProvider', 'CPUExecutionProvider'], fallback_reasons: [] }];
    await page.locator('#refresh').click();
    await page.waitForFunction(() => document.querySelector('#runtime').textContent === '合成后端：WebGPU');
    assert.match(await page.locator('#runtime').getAttribute('title'), /WebGpuExecutionProvider/);
    await page.locator('#tasks li').click();
    await page.locator('#taskPlayers audio').waitFor();
    await page.evaluate(async () => {
      window.originalAudio = document.querySelector('#taskPlayers audio');
      originalAudio.muted = true;
      await originalAudio.play();
    });
    const elapsed = () => page.locator('#taskStatus').textContent();
    const before = await elapsed();
    await page.waitForFunction(() => originalAudio.currentTime > 3);
    assert.ok(polls >= 2, 'playback spans a status poll');
    assert.notEqual(await elapsed(), before, 'elapsed time advances during an unfinished stage');
    assert.equal(await page.evaluate(() => document.querySelector('#taskPlayers audio') === originalAudio
      && originalAudio.isConnected && !originalAudio.paused), true);

    // Failed status requests also keep the player alive and polling retries.
    failPoll = true;
    const beforeFailure = polls;
    await page.waitForFunction(() => originalAudio.currentTime > 5.5);
    assert.ok(polls > beforeFailure);
    assert.equal(await page.evaluate(() => originalAudio.isConnected && !originalAudio.paused), true);
    failPoll = false;
    state = { ...state, status: 'done', stage: '', elapsed_seconds: 17,
      outputs: { ...state.outputs, score: '/v1/audio?kind=score',
        lead: '/v1/audio?kind=lead', clip: '/v1/audio?kind=clip' } };
    await page.waitForFunction(() => document.querySelector('#taskStatus').textContent === 'done · 17.0s');
    assert.equal(await page.locator('#taskPlayers audio').count(), 4);
    assert.equal(await page.evaluate(() => document.querySelector('[data-kind="mimic"] audio') === originalAudio
      && !originalAudio.paused && originalAudio.currentTime > 5.5), true);
    await page.waitForTimeout(300);
    assert.equal(await elapsed(), 'done · 17.0s');

    // Task switches clear old players, queue time is excluded, failures freeze.
    await page.evaluate(() => {
      current = 'next';
      renderDetail({ task_id: 'next', status: 'queued', spec: {}, timings: {}, outputs: {} });
    });
    assert.equal(await page.locator('#taskPlayers audio').count(), 0);
    await page.waitForTimeout(300);
    assert.equal(await elapsed(), 'queued · 0.0s');
    await page.evaluate(() => renderDetail({ task_id: 'next', status: 'failed',
      elapsed_seconds: 12.3, spec: {}, timings: {}, outputs: {} }));
    await page.waitForTimeout(300);
    assert.equal(await elapsed(), 'failed · 12.3s');

    // Legacy servers without timestamps must not reset the clock at each poll.
    await page.evaluate(() => {
      current = 'legacy';
      window.legacyState = { task_id: 'legacy', status: 'running', spec: {}, timings: { fetch: 2 }, outputs: {} };
      renderDetail(legacyState);
    });
    await page.waitForTimeout(300);
    await page.evaluate(() => renderDetail(legacyState));
    assert.notEqual(await elapsed(), 'running · 2.0s');
    assert.deepEqual(errors, []);
    runtime.sessions = [{ model: 'acoustic', backend: 'cpu',
      providers: ['CPUExecutionProvider'], fallback_reasons: ['webgpu: no device'] }];
    await page.locator('#refresh').click();
    await page.waitForFunction(() => document.querySelector('#runtime').textContent === '合成后端：CPU（已回退）');
    assert.match(await page.locator('#runtime').getAttribute('title'), /webgpu: no device/);
  } finally {
    await browser.close();
  }
});
