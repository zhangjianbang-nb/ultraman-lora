// Scrape Bing image results for many queries, reusing ONE browser instance.
// Usage: node scrape_many.js <queries.json> <outJson> [pagesPerQuery]
// queries.json: [{"id":"tiga","q":"迪迦奥特曼"}, ...]
const puppeteer = require('/home/whu/workspace/servers/chatgpt-browser/node_modules/puppeteer-core');
const fs = require('fs');

const qfile = process.argv[2];
const outfile = process.argv[3];
const pages = parseInt(process.argv[4] || '2', 10);
const queries = JSON.parse(fs.readFileSync(qfile, 'utf8'));
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/usr/bin/chromium-browser',
    headless: 'new',
    args: ['--no-sandbox', '--disable-dev-shm-usage', '--lang=zh-CN',
           '--disable-blink-features=AutomationControlled'],
  });
  const page = await browser.newPage();
  await page.setUserAgent(UA);
  await page.setViewport({ width: 1440, height: 900 });
  const result = {};
  for (const item of queries) {
    const urls = new Set();
    for (let p = 0; p < pages; p++) {
      const first = p * 35 + 1;
      const url = `https://www.bing.com/images/search?q=${encodeURIComponent(item.q)}&first=${first}&count=35&mkt=zh-CN`;
      try {
        await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
        await new Promise(r => setTimeout(r, 1200));
        for (let s = 0; s < 3; s++) {
          await page.evaluate(() => window.scrollBy(0, 1300));
          await new Promise(r => setTimeout(r, 600));
        }
        const got = await page.evaluate(() => {
          const res = [];
          for (const a of document.querySelectorAll('a.iusc')) {
            const m = a.getAttribute('m');
            if (!m) continue;
            try { const j = JSON.parse(m); if (j.murl) res.push(j.murl); } catch (e) {}
          }
          return res;
        });
        got.forEach(u => urls.add(u));
      } catch (e) {
        console.error('ERR', item.id, p, e.message);
      }
    }
    result[item.id] = { q: item.q, urls: [...urls] };
    console.log(`${item.id}: ${urls.size} urls`);
    fs.writeFileSync(outfile, JSON.stringify(result, null, 1));
  }
  await browser.close();
  console.log('DONE', Object.keys(result).length, 'queries');
})();
