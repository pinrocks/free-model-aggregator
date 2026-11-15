# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/pinrocks/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/pinrocks/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that turns ten simulated free-tier model providers into a single OpenAI- and Anthropic-compatible endpoint: every request is routed to the best-rated free model that meets its requirements, translated across all three upstream wires (openai, anthropic, gemini) with budgets planned per provider, and failed over to the next candidate when a provider refuses — the client's own wire and streaming shape answered back either way.

## Run it

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=/path/to/prompt.txt node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=zc8m81ne&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
