/*
 * Tarayıcıda çalışan isim üretici: src/generator.py'nin JavaScript karşılığı.
 * model.json (python -m src.web_export) ağırlıklarıyla CharLSTM'in ileri geçişini
 * ve temperature örneklemesini sunucuya ihtiyaç duymadan yapar.
 */
(function (root) {
    'use strict';

    const VALID_WORD = /^[a-z]+(?:[ -][a-z]+)*$/;

    function decodeTensor({ shape, data }) {
        const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
        return { shape, data: new Float32Array(bytes.buffer) };
    }

    // y = W·x + b  (W: [rows, cols], satır öncelikli)
    function affine(W, x, b, out) {
        const [rows, cols] = W.shape;
        for (let r = 0; r < rows; r++) {
            let sum = b ? b.data[r] : 0;
            const offset = r * cols;
            for (let c = 0; c < cols; c++) sum += W.data[offset + c] * x[c];
            out[r] = sum;
        }
        return out;
    }

    const sigmoid = (v) => 1 / (1 + Math.exp(-v));

    function createGenerator(model) {
        const t = {};
        for (const [name, tensor] of Object.entries(model.tensors)) t[name] = decodeTensor(tensor);

        const { PAD, BOS_FIRST, BOS_LAST, EOS } = model.special;
        const { hidden_dim: H, num_layers: L, embedding_dim: E } = model.config;
        const { min_word_len, max_word_len, max_name_len } = model.limits;
        const numSpecial = Object.keys(model.special).length;
        const vocabSize = numSpecial + model.vocab.length;
        const knownNames = new Set(model.known_names);

        const layers = [];
        for (let l = 0; l < L; l++) {
            layers.push({
                wIh: t[`lstm.weight_ih_l${l}`], wHh: t[`lstm.weight_hh_l${l}`],
                bIh: t[`lstm.bias_ih_l${l}`], bHh: t[`lstm.bias_hh_l${l}`],
                gi: new Float32Array(4 * H), gh: new Float32Array(4 * H),
            });
        }
        const logits = new Float32Array(vocabSize);

        function initialState() {
            return Array.from({ length: L }, () => ({ h: new Float32Array(H), c: new Float32Array(H) }));
        }

        // Tek zaman adımı: PyTorch LSTM kapı sırası i, f, g, o
        function step(tokenIdx, state) {
            let x = t['embedding.weight'].data.subarray(tokenIdx * E, (tokenIdx + 1) * E);
            for (let l = 0; l < L; l++) {
                const { wIh, wHh, bIh, bHh, gi, gh } = layers[l];
                const { h, c } = state[l];
                affine(wIh, x, bIh, gi);
                affine(wHh, h, bHh, gh);
                const hNext = new Float32Array(H);
                for (let j = 0; j < H; j++) {
                    const i = sigmoid(gi[j] + gh[j]);
                    const f = sigmoid(gi[H + j] + gh[H + j]);
                    const g = Math.tanh(gi[2 * H + j] + gh[2 * H + j]);
                    const o = sigmoid(gi[3 * H + j] + gh[3 * H + j]);
                    c[j] = f * c[j] + i * g;
                    hNext[j] = o * Math.tanh(c[j]);
                }
                state[l].h = hNext;
                x = hNext;
            }
            return affine(t['fc.weight'], x, t['fc.bias'], logits);
        }

        function sample(rawLogits, temperature, allowEos) {
            const scaled = Array.from(rawLogits, (v) => v / temperature);
            scaled[PAD] = scaled[BOS_FIRST] = scaled[BOS_LAST] = -Infinity;
            if (!allowEos) scaled[EOS] = -Infinity;

            const max = Math.max(...scaled);
            const weights = scaled.map((v) => Math.exp(v - max));
            let r = Math.random() * weights.reduce((a, b) => a + b, 0);
            for (let i = 0; i < weights.length; i++) {
                r -= weights[i];
                if (r <= 0 && weights[i] > 0) return i;
            }
            return weights.findLastIndex((w) => w > 0);
        }

        function generateWord(bos, temperature) {
            const state = initialState();
            let token = bos;
            let word = '';
            for (let n = 0; n <= max_word_len; n++) {
                const next = sample(step(token, state), temperature, word.length >= min_word_len);
                if (next === EOS) return VALID_WORD.test(word) ? word : null;
                word += model.vocab[next - numSpecial];
                token = next;
            }
            return null;
        }

        const titleCase = (s) => s.replace(/\b[a-z]/g, (ch) => ch.toUpperCase());

        function generate(temperature = 0.7, maxAttempts = 50) {
            for (let attempt = 0; attempt < maxAttempts; attempt++) {
                const first = generateWord(BOS_FIRST, temperature);
                const last = generateWord(BOS_LAST, temperature);
                if (first === null || last === null) continue;

                const fullName = `${first} ${last}`;
                if (fullName.length <= max_name_len && !knownNames.has(fullName)) return titleCase(fullName);
            }
            throw new Error(`${maxAttempts} denemede özgün bir isim üretilemedi`);
        }

        return { generate, step, initialState };
    }

    let generatorPromise = null;

    // Hata nedenini UI'ın ayırt edebilmesi için etiketli hata:
    // 'network' (bağlantı koptu), 'http' (sunucu hata kodu), 'parse' (bozuk/yarım dosya)
    function modelError(kind, message, cause) {
        const err = new Error(message);
        err.kind = kind;
        err.cause = cause;
        return err;
    }

    async function fetchModel(modelUrl) {
        let res;
        try {
            res = await fetch(modelUrl);
        } catch (err) {
            throw modelError('network', 'model.json indirilemedi (bağlantı hatası)', err);
        }
        if (!res.ok) throw modelError('http', `model.json yüklenemedi (${res.status})`);
        try {
            return await res.json();
        } catch (err) {
            // Mobil ağda gövde yarıda kesilirse JSON eksik gelir
            throw modelError('parse', 'model.json eksik veya bozuk indi', err);
        }
    }

    // Geçici (ağ/parse) hatalarda birkaç kez yeniden dener; 4xx/5xx'te denemez
    async function fetchModelWithRetry(modelUrl, retries, delayMs) {
        for (let attempt = 0; ; attempt++) {
            try {
                return await fetchModel(modelUrl);
            } catch (err) {
                if (err.kind === 'http' || attempt >= retries) throw err;
                await new Promise((resolve) => setTimeout(resolve, delayMs * (attempt + 1)));
            }
        }
    }

    // Model bir kez indirilir; sonraki çağrılar aynı üreticiyi kullanır
    async function generateName(temperature, modelUrl = 'model.json', { retries = 2, retryDelayMs = 700 } = {}) {
        if (!generatorPromise) {
            generatorPromise = fetchModelWithRetry(modelUrl, retries, retryDelayMs)
                .then(createGenerator)
                .catch((err) => { generatorPromise = null; throw err; });
        }
        return (await generatorPromise).generate(temperature);
    }

    const api = { createGenerator, generateName };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    else root.HoopsGenerator = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
