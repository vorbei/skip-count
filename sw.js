// Jump-rope counter service worker: installable + offline.
// Shell (index.html) = network-first so updates show; heavy immutable assets
// (MediaPipe wasm + model, icons) = cache-first so it's fast and works offline.
const CACHE = 'skip-v3';
const SHELL = ['./', './index.html', './manifest.webmanifest', './icon-192.png', './icon-512.png', './icon-180.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const sameOrigin = url.origin === location.origin;
  // app code (page, scripts, manifest) = network-first so updates always show;
  // heavy immutable assets (versioned CDN wasm/model, icons) = cache-first.
  const isAppCode = req.mode === 'navigate' ||
    (sameOrigin && !url.pathname.includes('/mediapipe/') &&
     /\.(html?|js|webmanifest|json)$/.test(url.pathname));   // mediapipe/* stays cache-first (immutable)
  if (isAppCode) {
    // network-first: latest code online, fall back to cache offline
    e.respondWith(
      fetch(req).then((res) => { const c = res.clone(); caches.open(CACHE).then((k) => k.put(req, c)); return res; })
        .catch(() => caches.match(req).then((h) => h || caches.match('./index.html')))
    );
  } else {
    // cache-first: model/wasm/cdn/icons are immutable and large
    e.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        if (res && (res.ok || res.type === 'opaque')) { const c = res.clone(); caches.open(CACHE).then((k) => k.put(req, c)); }
        return res;
      }))
    );
  }
});
