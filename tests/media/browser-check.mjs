import process from 'node:process';
import { chromium } from 'playwright';

const [fixtureUrl, screenshotPath] = process.argv.slice(2);
if (!fixtureUrl) throw new Error('fixture URL is required');
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 800, height: 520 } });
const websocketPorts = new Set();
const errors = [];
page.on('websocket', (socket) => websocketPorts.add(new URL(socket.url()).port));
page.on('pageerror', (error) => errors.push(error.message));
page.on('console', (message) => {
  if (message.type() === 'error') errors.push(message.text());
});

try {
  await page.goto(fixtureUrl, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(
    () => window.__RKNODE_MEDIA_RESULT__?.passed === true,
    undefined,
    { timeout: 30_000 },
  );
  const result = await page.evaluate(() => window.__RKNODE_MEDIA_RESULT__);
  process.stdout.write(JSON.stringify({ ...result, websocketPorts: [...websocketPorts], errors }));
} catch (error) {
  if (screenshotPath) await page.screenshot({ path: screenshotPath, fullPage: true });
  const result = await page.evaluate(() => window.__RKNODE_MEDIA_RESULT__).catch(() => null);
  process.stderr.write(JSON.stringify({ result, websocketPorts: [...websocketPorts], errors }));
  throw error;
} finally {
  await browser.close();
}
