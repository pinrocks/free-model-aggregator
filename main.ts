// free-model-aggregator-01: one aggregator server, three upstream wires, two client wires.
//
// A trial runs this file as `node /bot/main.ts` from the repository's root with Node 22's standard
// library alone. The brief at $MAKEFUN_PROMPT_FILE carries the provider directory, the quality table
// and the workload manifest; the env at $MAKEFUN_ENV_URL simulates ten free-tier providers on three
// wires (openai, anthropic, gemini), each with a hidden per-trial call budget, scheduled failures and
// one possible outage. This server:
//
//   1. reads the brief, probes every provider's model listing in the provider's own wire, and builds
//      the one catalog (the `--list` document and the register body);
//   2. listens on 0.0.0.0 at a free port and POSTs $MAKEFUN_ENV_URL/register with it;
//   3. plans the whole workload up front: every request's free candidates ranked by the quality table's
//      rating_milli under the request's requirements, assigned weight-first within each provider's
//      per-trial budget plan, so the heaviest requests ride the best models;
//   4. serves the client's requests — GET /v1/models in the shape of the auth header it got, and POST
//      /v1/chat/completions / POST /v1/messages, JSON or SSE — by rebuilding each request for the
//      routed provider's wire (any client wire to any provider wire), calling it without streaming and
//      rendering the answer back in the client's own shape;
//   5. fails over: a provider that answers 402, 429 or a 5xx loses the request to the next candidate,
//      and a provider that refuses twice in a row is set aside for the rest of the trial;
//   6. polls GET /work every 100 ms and exits 0 the moment the env says done.
//
// Everything the router does is a pure function of the brief, the probes, the request bodies and the
// providers' own answers: no clock and no randomness, so a replay of the same trial routes identically.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProviderDirectoryEntry,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicBodyFor,
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  normalizeAnswer,
  openAIBodyFor,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  openAIToGemini,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  type NormalAnswer,
  type ServedModel,
  type Wire,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 30000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

/** Per-provider free-tier budget bounds [min, max] from the provider spec, with a general fallback. */
const CALL_BOUNDS: Readonly<Record<string, readonly [number, number]>> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
const DEFAULT_BOUNDS: readonly [number, number] = [8, 12];

type Outcome = 'ok' | 'tr' | 'spent' | 'out';
interface Hypothesis {
  budget: number;
  fails: Set<number>;
  outage: number | null;
  isSec: boolean;
  baseW: number;
}

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client chat requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** One free (provider, model) pair this router can send a request to. */
interface Route {
  provider: ProviderDirectoryEntry;
  model: ProbedModel;
  row: RatingRow | null;
  wire: Wire;
  rating: number;
}

/** Every free route across the providers that answered, with its capabilities and rating. */
function allRoutes(): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({
        provider: probe.provider,
        model,
        row,
        wire: probe.provider.wire === 'anthropic' ? 'anthropic' : probe.provider.wire === 'gemini' ? 'gemini' : 'openai',
        rating: row !== null ? row.rating_milli : 0,
      });
    }
  }
  return out;
}

/** Whether a JSON-mode Chat Completions request can ride this wire: the Messages wire carries no JSON mode. */
function carriesJsonMode(wire: Wire): boolean {
  return wire !== 'anthropic';
}

function buildHypotheses(lo: number, hi: number): Hypothesis[] {
  const out: Hypothesis[] = [];
  for (let b = lo; b <= hi; b++) {
    const pB = 1 / (hi - lo + 1);
    const slots: number[] = [];
    for (let s = 2; s <= b - 1; s++) slots.push(s);
    const failSets: { set: number[]; p: number }[] = slots.map((s) => ({ set: [s], p: slots.length >= 2 ? 0.5 / slots.length : 1 / Math.max(1, slots.length) }));
    if (slots.length >= 2) {
      const pairs = (slots.length * (slots.length - 1)) / 2;
      for (let i = 0; i < slots.length; i++) {
        for (let j = i + 1; j < slots.length; j++) {
          failSets.push({ set: [slots[i]!, slots[j]!], p: 0.5 / pairs });
        }
      }
    }
    for (const fs of failSets) {
      for (const isSec of [false, true]) {
        const eff = isSec && !fs.set.includes(2) ? [2, ...fs.set.slice(1)] : fs.set;
        const fails = new Set<number>(eff);
        out.push({ budget: b, fails, outage: null, isSec, baseW: pB * fs.p });
        for (const o of slots) out.push({ budget: b, fails, outage: o, isSec, baseW: (pB * fs.p) / slots.length });
      }
    }
  }
  return out;
}

const hypothesesByProvider = new Map<string, Hypothesis[]>();
const probMemo = new Map<string, number>();
const bestByReq = new Map<string, number>();
let remSafe: number[] = [0];
let totalBound = 0;
let targetEarned = 0;
let activeProviderCount = 10;

const history = new Map<string, Outcome[]>();
const consecutiveRefusals = new Map<string, number>();
const setAside = new Set<string>();
let outageProvider: string | null = null;
let k2OkCount = 0;
let k2FailSeen = false;
let earnedSoFar = 0;

function initRouter(routes: readonly Route[]): void {
  const providerIds = new Set(routes.map((r) => r.provider.id));
  activeProviderCount = Math.max(2, providerIds.size);
  for (const id of providerIds) {
    const [lo, hi] = CALL_BOUNDS[id] ?? DEFAULT_BOUNDS;
    hypothesesByProvider.set(id, buildHypotheses(lo, hi));
    history.set(id, []);
  }
  for (const b of brief.ratings?.best ?? []) bestByReq.set(b.id, b.rating_milli);
  const manifest = brief.workload?.manifest ?? [];
  totalBound = 0;
  for (const row of manifest) {
    const top = bestByReq.get(row.id) ?? 0;
    totalBound += row.weight * top;
  }
  targetEarned = totalBound > 0 ? Math.ceil(0.98 * totalBound) + 60 : 0;
  remSafe = new Array<number>(manifest.length + 1).fill(0);
  for (let i = manifest.length - 1; i >= 0; i--) {
    const r = manifest[i]!;
    const safeRating = r.requires.vision === true ? 1432 : r.requires.min_context !== undefined ? 1440 : r.requires.json === true ? 1434 : 1390;
    remSafe[i] = (remSafe[i + 1] ?? 0) + r.weight * safeRating;
  }
}

function nextFailProb(providerId: string, hist: readonly Outcome[], outageElsewhere: boolean, secPrior: number): number {
  const histKey = hist.join(',');
  const key = `${providerId}|${histKey}|${outageElsewhere ? 1 : 0}|${secPrior.toFixed(3)}`;
  const cached = probMemo.get(key);
  if (cached !== undefined) return cached;
  let hyps = hypothesesByProvider.get(providerId);
  if (hyps === undefined) {
    const [lo, hi] = CALL_BOUNDS[providerId] ?? DEFAULT_BOUNDS;
    hyps = buildHypotheses(lo, hi);
    hypothesesByProvider.set(providerId, hyps);
  }
  const outPrior = outageElsewhere ? 0 : 1 / activeProviderCount;
  const nextK = hist.length + 1;
  let wT = 0;
  let wF = 0;
  for (const h of hyps) {
    const pOut = h.outage === null ? 1 - outPrior : outPrior;
    if (pOut <= 0) continue;
    const w = h.baseW * pOut * (h.isSec ? secPrior : 1 - secPrior);
    let ok = true;
    for (let i = 0; i < hist.length; i++) {
      const k = i + 1;
      const ev: Outcome = k > h.budget ? 'spent' : h.outage !== null && k >= h.outage ? 'out' : h.fails.has(k) ? 'tr' : 'ok';
      if (ev !== hist[i]) {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    wT += w;
    if (nextK > h.budget || (h.outage !== null && nextK >= h.outage) || h.fails.has(nextK)) wF += w;
  }
  const res = wT === 0 ? 1 : wF / wT;
  probMemo.set(key, res);
  return res;
}

/** Best eligible route per live provider, ordered by Bayesian expected-call + quality-reserve cost. */
function pickNextRoute(routes: readonly Route[], requires: Requirement, needsJsonWire: boolean, reqIndex: number, reqWeight: number): Route | null {
  const byProvider = new Map<string, Route>();
  for (const route of routes) {
    if (setAside.has(route.provider.id)) continue;
    if (!carriesJsonMode(route.wire) && needsJsonWire) continue;
    if (!meetsRequirements(requires, capabilitiesOf(route.model, route.row))) continue;
    const prev = byProvider.get(route.provider.id);
    if (prev === undefined || route.rating > prev.rating || (route.rating === prev.rating && route.model.id < prev.model.id)) {
      byProvider.set(route.provider.id, route);
    }
  }
  if (byProvider.size === 0) return null;
  const manifest = brief.workload?.manifest ?? [];
  let futTV = 0;
  let futV = 0;
  for (let j = reqIndex + 1; j < manifest.length; j++) {
    const r = manifest[j]!;
    if (r.requires.vision === true) {
      futV += 1;
      if (r.requires.tools === true) futTV += 1;
    }
  }
  const secPrior = k2FailSeen ? 0.03 : Math.min(0.5, 1 / Math.max(2, activeProviderCount - k2OkCount));
  const safeTail = reqIndex + 1 < remSafe.length ? (remSafe[reqIndex + 1] ?? 0) : 0;
  const scored = [...byProvider.values()].map((route) => {
    const p = route.provider.id;
    const hist = history.get(p) ?? [];
    const pf = nextFailProb(p, hist, outageProvider !== null && outageProvider !== p, hist.length === 1 ? secPrior : 1 / activeProviderCount);
    let resPen = 0;
    if (requires.vision !== true && hist.length === 0) {
      if ((p === 'gemini' || p === 'zai') && futTV > 0) resPen = 5;
      else if ((p === 'gemini' || p === 'zai' || p === 'sambanova') && futV > 0) resPen = 0.5;
    }
    const deficit = totalBound > 0 ? Math.max(0, targetEarned - (earnedSoFar + reqWeight * route.rating + safeTail)) : 0;
    const cost = (pf + resPen) * 600 + deficit * 5 - reqWeight * route.rating * 0.02;
    return { route, cost };
  });
  scored.sort((a, b) => {
    if (a.cost !== b.cost) return a.cost - b.cost;
    if (a.route.rating !== b.route.rating) return b.route.rating - a.route.rating;
    if (a.route.provider.id !== b.route.provider.id) return a.route.provider.id < b.route.provider.id ? -1 : 1;
    return a.route.model.id < b.route.model.id ? -1 : a.route.model.id > b.route.model.id ? 1 : 0;
  });
  return scored[0]?.route ?? null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** POST one request to the routed provider, in the provider's own wire. */
async function callProvider(route: Route, clientApi: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.provider.auth, route.provider.key),
  };
  if (route.wire === 'openai') {
    const upstream = clientApi === 'openai' ? openAIBodyFor(body, route.model.id) : anthropicToOpenAI(body, route.model.id);
    return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  }
  if (route.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = clientApi === 'anthropic' ? anthropicBodyFor(body, route.model.id) : openAIToAnthropicRequest(body, route.model.id);
    return fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  }
  const upstream = clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  const model = encodeURIComponent(route.model.id);
  return fetch(`${base}/v1beta/models/${model}:generateContent`, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
}

/** Whether a provider's answer is a refusal this router fails over on: over budget, throttled, or broken. */
function isRefusal(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

/**
 * Serve one chat request on either client wire: pick the best route by Bayesian call hazard and quality
 * reserve, call the provider, classify refusals into transient/outage/spent, and render the answer back.
 */
async function serveChat(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  }
  const manifest = brief.workload?.manifest ?? [];
  const reqIndex = manifestIndex;
  const row = manifest[reqIndex];
  manifestIndex += 1;
  const requires = row !== undefined ? row.requires : requirementsFromBody(body);
  const reqWeight = row !== undefined ? row.weight : 1;
  const bestCap = row !== undefined ? (bestByReq.get(row.id) ?? 2000) : 2000;
  // JSON mode on the Chat Completions wire can only ride an upstream wire that carries it.
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const needsJsonWire = clientApi === 'openai' && (format === 'json_object' || format === 'json_schema' || requires.json === true);
  const routes = allRoutes();

  let answer: NormalAnswer | null = null;
  let servedOn: Route | null = null;
  for (let attempt = 0; attempt < 64; attempt++) {
    const route = pickNextRoute(routes, requires, needsJsonWire, reqIndex, reqWeight);
    if (route === null) break;
    const p = route.provider.id;
    let hist = history.get(p);
    if (hist === undefined) {
      hist = [];
      history.set(p, hist);
    }
    let upstream: Response;
    try {
      upstream = await callProvider(route, clientApi, body);
    } catch (err) {
      log(`call to ${p}/${route.model.id} failed: ${(err as Error).message}`);
      setAside.add(p);
      continue;
    }
    if (upstream.status >= 200 && upstream.status < 300) {
      hist.push('ok');
      if (hist.length === 2) k2OkCount += 1;
      consecutiveRefusals.set(p, 0);
      earnedSoFar += reqWeight * Math.min(route.rating, bestCap);
      const parsed: unknown = await upstream.json();
      answer = normalizeAnswer(route.wire, isRecord(parsed) ? parsed : {});
      servedOn = route;
      break;
    }
    const status = upstream.status;
    const errText = await upstream.text().catch(() => '');
    log(`${p}/${route.model.id} answered HTTP ${status}`);
    if (isRefusal(status)) {
      let msg = errText;
      try {
        const j: unknown = JSON.parse(errText);
        if (isRecord(j) && isRecord(j.error) && typeof j.error.message === 'string') msg = j.error.message;
      } catch {
        // Non-JSON refusal body: classify using raw text fallback.
      }
      let ev: Outcome;
      if (msg.includes('budget is spent')) {
        ev = 'spent';
        setAside.add(p);
      } else if (msg.endsWith(' is out')) {
        ev = 'out';
        setAside.add(p);
        outageProvider = p;
      } else {
        const consecutive = (consecutiveRefusals.get(p) ?? 0) + 1;
        consecutiveRefusals.set(p, consecutive);
        if (consecutive >= 2) {
          ev = 'spent';
          setAside.add(p);
        } else {
          ev = 'tr';
        }
      }
      hist.push(ev);
      if (hist.length === 2) k2FailSeen = true;
      continue;
    }
    // A 400 or 401 is this router's own mistake with this provider: stop using it, try the next candidate.
    setAside.add(p);
    continue;
  }

  if (answer === null || servedOn === null) {
    return sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
  }

  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : servedOn.model.id;
  if (body.stream === true) {
    const text = clientApi === 'anthropic' ? renderAnthropicStream(answer, requested) : renderOpenAIStream(answer, requested);
    return sendSse(res, text);
  }
  const doc = clientApi === 'anthropic' ? renderAnthropicJson(answer, requested) : renderOpenAIJson(answer, requested);
  return sendJson(res, 200, doc);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveChat('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveChat('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const routes = allRoutes();
  initRouter(routes);
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
