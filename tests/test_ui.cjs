// Run with `node --test tests/test_ui.cjs` with playwright on NODE_PATH.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { chromium } = require('playwright');
const launchOptions = { headless: true, ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}) };

function silentWav() {
  const data = Buffer.alloc(44 + 8000 * 2 * 30);
  data.write('RIFF'); data.writeUInt32LE(data.length - 8, 4); data.write('WAVEfmt ', 8);
  data.writeUInt32LE(16, 16); data.writeUInt16LE(1, 20); data.writeUInt16LE(1, 22);
  data.writeUInt32LE(8000, 24); data.writeUInt32LE(16000, 28);
  data.writeUInt16LE(2, 32); data.writeUInt16LE(16, 34);
  data.write('data', 36); data.writeUInt32LE(data.length - 44, 40);
  return data;
}

test('switching API keys clears private details and ignores an old in-flight poll', { timeout: 30000 }, async () => {
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage();
    const html = readFileSync(join(__dirname, '../src/dsapi/static/index.html'), 'utf8');
    const state = { task_id: 'alice-task', status: 'done', spec: {}, timings: {},
      outputs: { mimic: '/v1/audio?task_id=alice-task&kind=mimic' } };
    let delayPoll = false, signal;
    const stalled = new Promise(resolve => { signal = resolve; });
    await page.addInitScript(() => localStorage.setItem('dsapi.key', 'alice-secret'));
    await page.route('http://dsapi.test/**', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/ui/') return route.fulfill({ contentType: 'text/html', body: html });
      if (url.pathname === '/tasks') return route.fulfill({ json: url.searchParams.get('key') === 'alice-secret'
        ? [state] : [{ task_id: 'bob-task', status: 'queued' }] });
      if (url.pathname === '/query_result') {
        if (delayPoll) { signal(route); return; }
        return route.fulfill({ json: state });
      }
      if (url.pathname === '/v1/audio') return route.fulfill({ contentType: 'audio/wav', body: silentWav() });
      return route.fulfill({ status: 404 });
    });
    await page.goto('http://dsapi.test/ui/');
    await page.locator('[data-id="alice-task"]').click();
    await page.locator('#taskPlayers audio').waitFor();
    await page.evaluate(async () => {
      window.oldPlayer = document.querySelector('#taskPlayers audio');
      oldPlayer.muted = true;
      await oldPlayer.play();
    });
    delayPoll = true;
    await page.evaluate(() => { window.pendingPoll = poll(); });
    const pending = await stalled;
    await page.locator('#apikey').fill('bob-secret');
    await page.locator('#apikey').dispatchEvent('change');
    await page.locator('[data-id="bob-task"]').waitFor();
    await pending.fulfill({ json: state });
    await page.evaluate(() => window.pendingPoll);
    assert.equal(await page.locator('#taskPlayers').count(), 0);
    assert.equal(await page.locator('[data-id="alice-task"]').count(), 0);
    assert.equal(await page.evaluate(() => current === null && elapsedClock === null && oldPlayer.paused), true);
    assert.match(await page.locator('#detail').textContent(), /请选择或提交/);
  } finally { await browser.close(); }
});

test('delete confirms, handles errors, clears selection and protects active tasks', { timeout: 30000 }, async () => {
  const browser = await chromium.launch(launchOptions);
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
  const browser = await chromium.launch(launchOptions);
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


test('audio-only option is submitted and has no score controls', { timeout: 30000 }, async () => {
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage();
    const html = readFileSync(join(__dirname, '../src/dsapi/static/index.html'), 'utf8');
    let submitted;
    const state = { task_id: 'fast', status: 'done', spec: { modes: ['mimic'], audio_only: true },
      timings: { fetch: 0, select: 0, separate: 0, align: 0, notes_f0: 0, build: 0, render_mimic: 1 },
      outputs: {}, song: { name: 'test' }, alignment: [], jianpu: null };
    await page.route('http://dsapi.test/**', route => {
      const path = new URL(route.request().url()).pathname;
      if (path === '/ui/') return route.fulfill({ contentType: 'text/html', body: html });
      if (path === '/tasks') return route.fulfill({ json: [] });
      if (path === '/voices') return route.fulfill({ json: { default_voice_id: 'unknown', voices: [{id:'unknown',name:'测试音源',available:true}] } });
      if (path === '/release_task') {
        submitted = route.request().postDataJSON();
        return route.fulfill({ json: { task_id: 'fast' } });
      }
      if (path === '/query_result') return route.fulfill({ json: state });
      return route.fulfill({ status: 404 });
    });
    await page.goto('http://dsapi.test/ui/');
    await page.locator('input[name=srcMode][value=id]').check();
    await page.locator('#songId').fill('1');
    await page.locator('#mScore').uncheck();
    await page.locator('#audioOnly').check();
    await page.locator('#submit').click();
    await page.locator('#taskSummary h2').waitFor();
    assert.equal(submitted.audio_only, true);
    assert.deepEqual(submitted.modes, ['mimic']);
    assert.equal(await page.locator('#toEditor').count(), 0);
    assert.equal(await page.locator('.stage').count(), 7);
  } finally { await browser.close(); }
});

test('fixed-line score preview, validation and queued submission', { timeout: 30000 }, async () => {
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    const html = readFileSync(join(__dirname, '../src/dsapi/static/index.html'), 'utf8');
    const state = { task_id: 'score-task', status: 'done', spec: {score_text: '1 2\n天晴', modes: ['score']},
      song: {name: '自填乐谱'}, timings: {parse_score: 0.01, render_score: 1}, outputs: {}, jianpu: '1 2\n天晴' };
    let submitted = false;
    await page.route('http://dsapi.test/**', route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/ui/') return route.fulfill({contentType:'text/html', body:html});
      if (url.pathname === '/voices') return route.fulfill({json:{default_voice_id:'default',voices:[{id:'default',name:'测试音源',available:true}]}});
      if (url.pathname === '/tasks') return route.fulfill({json:submitted ? [state] : []});
      if (url.pathname === '/query_result') return route.fulfill({json:state});
      if (url.pathname === '/parse_score') {
        const body = route.request().postDataJSON();
        assert.equal(body.voice_id, 'default');
        if (body.text === '1 2\n天') return route.fulfill({status:422,json:{detail:{line:2,message:'需要 2 个演唱音节，实际 1 个'}}});
        return route.fulfill({json:{duration_seconds:1.8, rows:[{lyric_line:2,units:[{text:'天',pinyin:'tian'},{text:'晴',pinyin:'qing'}]}]}});
      }
      if (url.pathname === '/release_score_task') {
        assert.deepEqual(route.request().postDataJSON(), {text:'1 2\n天晴',voice_id:'default'});
        submitted = true;
        return route.fulfill({json:{task_id:'score-task'}});
      }
      return route.fulfill({status:404});
    });
    await page.goto('http://dsapi.test/ui/');
    await page.locator('#renderVoice option').waitFor({state:'attached'});
    await page.locator('#notationText').fill('1 2\n天');
    await page.locator('#parseScoreBtn').click();
    await page.waitForFunction(() => document.querySelector('#scorePreview').textContent.includes('需要 2'));
    assert.match(await page.locator('#scorePreview').textContent(), /第 2 行/);
    await page.locator('#notationText').fill('1 2\n天晴');
    assert.equal(await page.locator('#scorePreview').isHidden(), true);
    await page.locator('#parseScoreBtn').click();
    await page.waitForFunction(() => document.querySelector('#scorePreview').textContent.includes('天(tian)'));
    await page.locator('#submitScoreBtn').click();
    await page.locator('#taskStatus').waitFor();
    assert.equal(submitted, true);
    assert.equal(await page.locator('#taskSummary .stage').count(), 2);
    assert.equal(await page.locator('#evalBtn').count(), 0);
    assert.match(await page.locator('#taskExtras').textContent(), /自填乐谱/);
    await page.locator('#notationText').fill('1\n改');
    await page.locator('#toEditor').click();
    assert.equal(await page.locator('#notationText').inputValue(), '1 2\n天晴');
    assert.equal(await page.locator('#renderVoice').inputValue(), 'default');
    assert.equal(await page.locator('#scorePreview').isHidden(), true);
    await page.locator('#apikey').fill('another-key');
    await page.locator('#apikey').dispatchEvent('change');
    assert.equal(await page.locator('#scorePreview').isHidden(), true);
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
});

test('song and lyric lookup supports voice aliases, accompaniment presets and explicit range', {timeout:30000}, async () => {
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage({viewport:{width:390,height:844}});
    const html = readFileSync(join(__dirname,'../src/dsapi/static/index.html'),'utf8');
    const errors=[]; page.on('pageerror', e=>errors.push(e.message));
    let resolved, submitted;
    const state={task_id:'cover',status:'done',spec:{modes:['mimic']},timings:{},outputs:{},song:{name:'test'}};
    await page.route('http://dsapi.test/**', route=>{
      const path=new URL(route.request().url()).pathname;
      if(path==='/ui/') return route.fulfill({contentType:'text/html',body:html});
      if(path==='/voices') return route.fulfill({json:{default_voice_id:'first',voices:[
        {id:'first',name:'测试音源',aliases:['测试','默认'],available:true},
        {id:'second',name:'另一音源',aliases:['另一'],available:true},
        {id:'offline',name:'不可用',available:false}]}});
      if(path==='/tasks') return route.fulfill({json:[]});
      if(path==='/resolve_song') {
        resolved=route.request().postDataJSON();
        return route.fulfill({json:{song:{id:'123'},lyric_query:'第一句\n第二句\n第三句'}});
      }
      if(path==='/release_task') {submitted=route.request().postDataJSON();return route.fulfill({json:{task_id:'cover'}});}
      if(path==='/query_result') return route.fulfill({json:state});
      return route.fulfill({status:404});
    });
    await page.goto('http://dsapi.test/ui/');
    await page.locator('#voice option[value=second]').waitFor({state:'attached'});
    assert.equal(await page.locator('#voice').inputValue(),'first');
    assert.match(await page.locator('#voiceHint').textContent(),/测试.*默认/);
    assert.equal(await page.locator('#voice option[value=offline]').isDisabled(),true);
    await page.locator('#voice').selectOption('second');
    assert.match(await page.locator('#voiceHint').textContent(),/另一/);
    await page.locator('#songQuery').fill('第一句\n第二句\n第三句');
    await page.locator('#coverPresets button[data-mode=mimic][data-mix=true]').click();
    await page.locator('#start').fill('0'); await page.locator('#end').fill('120');
    await page.locator('#submit').click(); await page.locator('#taskStatus').waitFor();
    assert.deepEqual(resolved,{q:'第一句\n第二句\n第三句',lyric_extent:'auto'});
    assert.deepEqual(submitted.modes,['mimic']);
    assert.equal(submitted.with_accompaniment,true); assert.equal(submitted.voice_id,'second');
    assert.equal(submitted.start,0); assert.equal(submitted.end,120);
    assert.equal(submitted.lyric_query,resolved.q); assert.equal(submitted.song_id,'123');
    await page.locator('#scoreExample').selectOption('slur');
    await page.locator('#loadScoreExample').click();
    assert.match(await page.locator('#notationText').inputValue(),/\(2:0.5 3:0.5\)/);
    assert.match(await page.locator('#notationHint').textContent(),/可省略/);
    await page.locator('#scoreGuide').evaluate(el=>el.open=true);
    assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    assert.deepEqual(errors,[]);
  } finally {await browser.close();}
});

test('changing account while resolving a song does not submit under the new key', {timeout:30000}, async () => {
  const browser=await chromium.launch(launchOptions);
  try {
    const page=await browser.newPage();
    const html=readFileSync(join(__dirname,'../src/dsapi/static/index.html'),'utf8');
    let signal,submitted=0;
    const stalled=new Promise(resolve=>{signal=resolve;});
    await page.route('http://dsapi.test/**',route=>{
      const path=new URL(route.request().url()).pathname;
      if(path==='/ui/') return route.fulfill({contentType:'text/html',body:html});
      if(path==='/tasks') return route.fulfill({json:[]});
      if(path==='/resolve_song') {signal(route);return;}
      if(path==='/release_task') {submitted++;return route.fulfill({json:{task_id:'wrong-owner'}});}
      return route.fulfill({status:404});
    });
    await page.goto('http://dsapi.test/ui/');
    await page.locator('#songQuery').fill('晴天 周杰伦');
    await page.locator('#submit').click(); const pending=await stalled;
    await page.locator('#apikey').fill('new-owner'); await page.locator('#apikey').dispatchEvent('change');
    await pending.fulfill({json:{song:{id:'123'}}});
    await page.waitForFunction(()=>!document.querySelector('#submit').disabled);
    assert.equal(submitted,0);
  } finally {await browser.close();}
});

test('unconfigured installation prompts for a local import without a preset singer', async () => {
  const browser = await chromium.launch(launchOptions);
  try {
    const page = await browser.newPage();
    const html = readFileSync(join(__dirname, '../src/dsapi/static/index.html'), 'utf8');
    await page.route('http://dsapi.test/**', route => {
      const path = new URL(route.request().url()).pathname;
      if (path === '/ui/') return route.fulfill({ contentType: 'text/html', body: html });
      if (path === '/voices') return route.fulfill({ json: { default_voice_id: 'default', voices: [
        { id: 'default', name: '用户导入音源', available: false, aliases: [] }
      ] } });
      if (path === '/tasks') return route.fulfill({ json: [] });
      return route.fulfill({ status: 404 });
    });
    await page.goto('http://dsapi.test/ui/');
    await page.waitForFunction(() => document.querySelector('#voiceHint').textContent.includes('导入'));
    for (const id of ['voice', 'renderVoice']) {
      assert.equal(await page.locator('#' + id).inputValue(), '');
      assert.equal(await page.locator('#' + id + ' option').count(), 1);
      assert.equal(await page.locator('#' + id + ' option').isDisabled(), true);
    }
  } finally { await browser.close(); }
});
