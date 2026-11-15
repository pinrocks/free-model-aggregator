// The wires this entry speaks: two on the client side (OpenAI Chat Completions and Anthropic Messages,
// each as JSON or as server-sent events) and three upstream (openai, anthropic, gemini).
//
// Every request the client sends is rebuilt for the routed provider's wire with its semantic content
// preserved exactly as the brief's echo comparison reads it: the system text, the last user message's
// text, the tool names in order, the output cap, the stop sequences, the image bytes and JSON mode.
// The provider is always called without streaming; the client's answer is rendered from the provider's
// final answer, in the client's own shape — a JSON document, or the wire's full event sequence when the
// client asked to stream. Everything is a pure function of its arguments: no clock, no randomness, so
// the same bodies and the same provider answers give the same bytes on a replay.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the free models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The three upstream wires a provider can speak. */
export type Wire = 'openai' | 'anthropic' | 'gemini';

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool result's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as a JSON string. */
function argumentsText(value: unknown): string {
  if (isRecord(value)) return JSON.stringify(value);
  if (typeof value === 'string') return value;
  return '{}';
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(url);
    if (dataUrl !== null) return { type: 'image', source: { type: 'base64', media_type: dataUrl[1], data: dataUrl[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** An image content part of either client shape as a Gemini inlineData part; null when there is none. */
function geminiImageOf(part: Json): Json | null {
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const dataUrl = /^data:([^;,]+);base64,(.*)$/s.exec(part.image_url.url);
    if (dataUrl !== null) return { inlineData: { mimeType: dataUrl[1], data: dataUrl[2] } };
    return { fileData: { fileUri: part.image_url.url } };
  }
  if (part.type === 'image' && isRecord(part.source) && part.source.type === 'base64' && typeof part.source.data === 'string') {
    return { inlineData: { mimeType: typeof part.source.media_type === 'string' ? part.source.media_type : 'image/png', data: part.source.data } };
  }
  if (part.type === 'image' && isRecord(part.source) && part.source.type === 'url' && typeof part.source.url === 'string') {
    return { fileData: { fileUri: part.source.url } };
  }
  return null;
}

/** An OpenAI Chat Completions request body for the same conversation on the routed model, never streaming. */
export function openAIBodyFor(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.stream;
  delete out.stream_options;
  return out;
}

/** An Anthropic Messages request body for the same conversation on the routed model, never streaming. */
export function anthropicBodyFor(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.stream;
  return out;
}

/** An Anthropic Messages request rewritten for a Gemini provider, never streaming. */
export function anthropicToGemini(body: Json): Json {
  const systemText = textOf(body.system);
  const contents: Json[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const role = raw.role === 'assistant' ? 'model' : 'user';
    if (typeof raw.content === 'string') {
      contents.push({ role, parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const block of raw.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else {
        const image = geminiImageOf(block);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const json: Json = { contents };
  if (systemText !== '') json.systemInstruction = { parts: [{ text: systemText }] };
  const declarations: Json[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    declarations.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  if (declarations.length > 0) json.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences;
  if (Object.keys(config).length > 0) json.generationConfig = config;
  return json;
}

/** An OpenAI Chat Completions request rewritten for a Gemini provider, never streaming. */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemTexts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    const role = raw.role === 'assistant' ? 'model' : 'user';
    if (raw.role === 'tool') {
      contents.push({ role: 'user', parts: [{ text: toolResultText(raw.content) }] });
      continue;
    }
    if (typeof raw.content === 'string') {
      contents.push({ role, parts: [{ text: raw.content }] });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const parts: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
      else {
        const image = geminiImageOf(part);
        if (image !== null) parts.push(image);
      }
    }
    if (parts.length > 0) contents.push({ role, parts });
  }
  const json: Json = { contents };
  if (systemTexts.length > 0) json.systemInstruction = { parts: systemTexts.map((text) => ({ text })) };
  const declarations: Json[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    declarations.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  if (declarations.length > 0) json.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (cap !== undefined) config.maxOutputTokens = cap;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop;
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) json.generationConfig = config;
  return json;
}

/** An OpenAI Chat Completions request rewritten for an Anthropic Messages provider, never streaming. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemTexts: string[] = [];
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') {
      const text = textOf(raw.content);
      if (text !== '') systemTexts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: JSON.parse(argumentsText(call.function.arguments)) as Json,
        });
      }
      messages.push({ role: 'assistant', content: blocks.length > 0 ? blocks : '' });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (Array.isArray(raw.content)) {
      const blocks: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        const block = anthropicBlockOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }
  const out: Json = { model, messages };
  if (systemTexts.length > 0) out.system = systemTexts.join('\n');
  const cap = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (cap !== undefined) out.max_tokens = cap;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  const tools: Json[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  return out;
}

/** An Anthropic Messages request rewritten for an OpenAI Chat Completions provider, never streaming. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: argumentsText(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  const tools: Json[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      type: 'function',
      function: {
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      },
    });
  }
  if (tools.length > 0) out.tools = tools;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  return out;
}

/** A provider's final answer, normalized across the three upstream wires. */
export interface NormalAnswer {
  text: string;
  input: number;
  output: number;
}

function numberOf(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as the normalized final answer. */
export function normalizeOpenAI(body: Json): NormalAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, input: numberOf(usage.prompt_tokens), output: numberOf(usage.completion_tokens) };
}

/** A Messages answer as the normalized final answer. */
export function normalizeAnthropic(body: Json): NormalAnswer {
  const blocks = Array.isArray(body.content) ? body.content : [];
  let text = '';
  for (const block of blocks) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') text += block.text;
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, input: numberOf(usage.input_tokens), output: numberOf(usage.output_tokens) };
}

/** A Gemini generateContent answer as the normalized final answer. */
export function normalizeGemini(body: Json): NormalAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  let text = '';
  for (const part of Array.isArray(content.parts) ? content.parts : []) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, input: numberOf(usage.promptTokenCount), output: numberOf(usage.candidatesTokenCount) };
}

/** The normalized answer of a provider on one of the three upstream wires. */
export function normalizeAnswer(wire: Wire, body: Json): NormalAnswer {
  if (wire === 'anthropic') return normalizeAnthropic(body);
  if (wire === 'gemini') return normalizeGemini(body);
  return normalizeOpenAI(body);
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** A Chat Completions answer document for the client. */
export function renderOpenAIJson(answer: NormalAnswer, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      { index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: 'stop', logprobs: null },
    ],
    usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
  };
}

/** A Chat Completions stream for the client: the role chunk, the content, the finish, the usage, [DONE]. */
export function renderOpenAIStream(answer: NormalAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    });
  const out = [chunk({ role: 'assistant', content: '' }, null)];
  if (answer.text !== '') out.push(chunk({ content: answer.text }, null));
  out.push(chunk({}, 'stop'));
  out.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: answer.input, completion_tokens: answer.output, total_tokens: answer.input + answer.output },
    }),
  );
  out.push('data: [DONE]\n\n');
  return out.join('');
}

/** A Messages answer document for the client. */
export function renderAnthropicJson(answer: NormalAnswer, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  };
}

/** A Messages stream for the client: the full event sequence, the usage carried where Anthropic carries it. */
export function renderAnthropicStream(answer: NormalAnswer, model: string): string {
  const out = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregator',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: answer.text } }),
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  ];
  return out.join('');
}
