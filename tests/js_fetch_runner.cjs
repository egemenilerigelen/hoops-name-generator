// test_web_model.py tarafından çağrılır: model indirme/yeniden deneme mantığını sahte fetch ile sınar.
//   node js_fetch_runner.cjs <model.json> <senaryo>
// Senaryolar: flaky (2 ağ hatası sonra başarı), network (hep ağ hatası),
//             http (503), truncated (yarım JSON, sonra başarı)
// Çıktı: {"ok": bool, "kind": hata türü, "calls": fetch çağrı sayısı}
const fs = require('fs');
const path = require('path');
const { generateName } = require(path.join(__dirname, '..', 'static', 'name-generator.js'));

const [modelPath, scenario] = process.argv.slice(2);
const body = fs.readFileSync(modelPath, 'utf-8');
let calls = 0;

const okResponse = () => ({ ok: true, status: 200, json: async () => JSON.parse(body) });
const brokenResponse = () => ({ ok: true, status: 200, json: async () => JSON.parse(body.slice(0, 100)) });

global.fetch = async () => {
    calls++;
    if (scenario === 'flaky') {
        if (calls <= 2) throw new TypeError('Failed to fetch');
        return okResponse();
    }
    if (scenario === 'network') throw new TypeError('Failed to fetch');
    if (scenario === 'http') return { ok: false, status: 503 };
    if (scenario === 'truncated') return calls === 1 ? brokenResponse() : okResponse();
    throw new Error(`bilinmeyen senaryo: ${scenario}`);
};

generateName(0.7, 'model.json', { retries: 2, retryDelayMs: 1 })
    .then(() => console.log(JSON.stringify({ ok: true, kind: null, calls })))
    .catch((err) => console.log(JSON.stringify({ ok: false, kind: err.kind, calls })));
