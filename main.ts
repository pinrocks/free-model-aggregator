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

/**
 * The per-trial budget this router plans for each provider: the most calls it will send there before it
 * prefers another candidate. The brief never publishes a provider's drawn budget, so this is a plan, not
 * knowledge: a provider that refuses calls it cannot take is failed over and set aside, and the trial
 * goes on. The numbers are deliberately a little above the floor a free tier implies: enough to use a
 * strong provider for the heavy requests, never so much that the plan leans on capacity that may not be
 * there. A provider missing from the table gets a plan of its own from the default.
 */
const BUDGET_PLAN: Readonly<Record<string, number>> = {
  groq: 12,
  cerebras: 6,
  gemini: 8,
  zai: 5,
  openrouter: 12,
  sambanova: 5,
  pollinations: 14,
  cloudflare: 8,
  ovh: 6,
  kilol: 16,
};
/** The planned budget for a provider the table does not name. */
const DEFAULT_BUDGET = 20;

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
  budget: number;
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
        budget: BUDGET_PLAN[probe.provider.id] ?? DEFAULT_BUDGET,
      });
    }
  }
  return out;
}

/** Whether a JSON-mode Chat Completions request can ride this wire: the Messages wire carries no JSON mode. */
function carriesJsonMode(wire: Wire): boolean {
  return wire !== 'anthropic';
}

/** The free routes that meet the requirements, best rating first: headroom, then ids, break the ties. */
function candidatesFor(
  routes: readonly Route[],
  requires: Requirement,
  needsJsonWire: boolean,
): Route[] {
  const eligible = routes.filter(
    (route) =>
      carriesJsonMode(route.wire) || !needsJsonWire,
  );
  const out = eligible.filter((route) => meetsRequirements(requires, capabilitiesOf(route.model, route.row)));
  const headroom = (route: Route): number => route.budget - (planUsage.get(route.provider.id) ?? 0);
  return out.sort((a, b) => {
    if (a.rating !== b.rating) return b.rating - a.rating;
    const ha = headroom(a);
    const hb = headroom(b);
    if (ha !== hb) return hb - ha;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** The plan's per-provider call count while assigning; the runtime counts its own. */
const planUsage = new Map<string, number>();
/** Each manifest request's planned route, by request id. */
const plan = new Map<string, Route>();

/**
 * Assign the manifest's requests to routes before the first one arrives: the heaviest requests first,
 * each to the best-rated candidate that meets it while the plan still has budget there. The plan is a
 * preference, not a commitment: at runtime a provider that refuses is failed over and the trial goes on.
 */
function planWorkload(routes: readonly Route[]): void {
  const manifest = brief.workload?.manifest ?? [];
  const order = manifest.map((row, index) => ({ row, index })).sort((a, b) => {
    const wa = typeof a.row.weight === 'number' ? a.row.weight : 1;
    const wb = typeof b.row.weight === 'number' ? b.row.weight : 1;
    if (wa !== wb) return wb - wa;
    return a.index - b.index;
  });
  for (const { row } of order) {
    const candidates = candidatesFor(routes, row.requires, row.api === 'openai' && rowRequiresJsonWire(row.requires));
    const pick = candidates.find((route) => (planUsage.get(route.provider.id) ?? 0) < route.budget) ?? null;
    if (pick === null) continue;
    plan.set(row.id, pick);
    planUsage.set(pick.provider.id, (planUsage.get(pick.provider.id) ?? 0) + 1);
  }
}

/** Whether a manifest row's JSON requirement, on the OpenAI wire, rules the Messages upstream wire out. */
function rowRequiresJsonWire(requires: Requirement): boolean {
  return requires.json === true;
}

/** The runtime's per-provider state: the calls it has taken, its consecutive refusals, and whether it is set aside. */
const used = new Map<string, number>();
const consecutiveRefusals = new Map<string, number>();
const setAside = new Set<string>();

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

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
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
 * Serve one chat request on either client wire: route it, call the provider, fail over on refusals, and
 * render the answer back in the client's own shape. The loop is the whole failover story: a refusal
 * costs the call it took, hands the request to the next candidate, and a provider that refuses twice in
 * a row is set aside for the rest of the trial.
 */
async function serveChat(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    return sendError(res, clientApi, 400, clientApi === 'anthropic' ? 'invalid_request_error' : 'invalid_request_error', 'the request body is not one JSON object');
  }
  // The manifest is in arrival order and the client sends one request at a time: this row is this request.
  const manifest = brief.workload?.manifest ?? [];
  const row = manifest[manifestIndex];
  manifestIndex += 1;
  const requires = row !== undefined ? row.requires : requirementsFromBody(body);
  // JSON mode on the Chat Completions wire can only ride an upstream wire that carries it.
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  const needsJsonWire = clientApi === 'openai' && (format === 'json_object' || format === 'json_schema' || requires.json === true);
  const routes = allRoutes();
  const candidates = candidatesFor(routes, requires, needsJsonWire);
  const planned = row !== undefined ? plan.get(row.id) : undefined;
  let ordered: Route[] = candidates;
  if (planned !== undefined && candidates.includes(planned)) {
    ordered = [planned, ...candidates.filter((route) => route !== planned)];
  }

  let answer: NormalAnswer | null = null;
  let servedOn: Route | null = null;
  for (const route of ordered) {
    if (setAside.has(route.provider.id)) continue;
    if ((used.get(route.provider.id) ?? 0) >= route.budget) continue;
    let upstream: Response;
    try {
      upstream = await callProvider(route, clientApi, body);
    } catch (err) {
      log(`call to ${route.provider.id}/${route.model.id} failed: ${(err as Error).message}`);
      setAside.add(route.provider.id);
      continue;
    }
    if (upstream.status >= 200 && upstream.status < 300) {
      used.set(route.provider.id, (used.get(route.provider.id) ?? 0) + 1);
      consecutiveRefusals.set(route.provider.id, 0);
      const parsed: unknown = await upstream.json();
      answer = normalizeAnswer(route.wire, isRecord(parsed) ? parsed : {});
      servedOn = route;
      break;
    }
    const status = upstream.status;
    await upstream.arrayBuffer().catch(() => undefined);
    log(`${route.provider.id}/${route.model.id} answered HTTP ${status}`);
    if (isRefusal(status)) {
      used.set(route.provider.id, (used.get(route.provider.id) ?? 0) + 1);
      const consecutive = (consecutiveRefusals.get(route.provider.id) ?? 0) + 1;
      consecutiveRefusals.set(route.provider.id, consecutive);
      if (consecutive >= 2) setAside.add(route.provider.id);
      continue;
    }
    // A 400 or 401 is this router's own mistake with this provider: stop using it, try the next candidate.
    setAside.add(route.provider.id);
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
  planWorkload(routes);
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
