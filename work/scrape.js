// Scrape Bing image results for a query using a real Chromium instance.
// Usage: node scrape.js "<query>" <outJson> [pages]
const puppeteer = require('/home/whu/workspace/servers/chatgpt-browser/node_modules/puppeteer-core');
const fs = require('fs');

const query = process.argv[2];
const out = process.argv[3];
const pages = parseInt(process.argv[4] || '3', 10);
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
  const urls = new Set();
  try {
    for (let p = 0; p < pages; p++) {
      const first = p * 35 + 1;
      const url = `https://www.bing.com/images/search?q=${encodeURIComponent(query)}&first=${first}&count=35&mkt=zh-CN`;
      await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
      await new Promise(r => setTimeout(r, 1500));
      for (let s = 0; s < 4; s++) {
        await page.evaluate(() => window.scrollBy(0, 1200));
        await new Promise(r => setTimeout(r, 700));
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
      console.log(`page ${p}: +${got.length}, total ${urls.size}`);
    }
  } catch (e) {
    console.error('ERR', e.message);
  }
  await browser.close();
  fs.writeFileSync(out, JSON.stringify([...urls], null, 1));
  console.log('wrote', out, urls.size);
})();
