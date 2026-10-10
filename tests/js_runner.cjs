// test_web_model.py tarafından çağrılır: tarayıcı üreticisini Node.js'te çalıştırır.
//   node js_runner.cjs <model.json> logits <token,token,...>   -> her adımın logit'leri
//   node js_runner.cjs <model.json> generate <temperature> <n> -> n adet isim
const fs = require('fs');
const path = require('path');
const { createGenerator } = require(path.join(__dirname, '..', 'static', 'name-generator.js'));

const [modelPath, mode, arg, count] = process.argv.slice(2);
const generator = createGenerator(JSON.parse(fs.readFileSync(modelPath, 'utf-8')));

if (mode === 'logits') {
    const state = generator.initialState();
    const steps = arg.split(',').map(Number).map((token) => Array.from(generator.step(token, state)));
    console.log(JSON.stringify(steps));
} else {
    const names = Array.from({ length: Number(count) }, () => generator.generate(Number(arg)));
    console.log(JSON.stringify(names));
}
