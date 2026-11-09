# Pydantic AI 版と ADK 版が Gemini に送るリクエストの比較（API は呼ばずに取得）

取得日: 2026-11-09。モデル: gemini-3.5-flash-lite。質問は Q5「VLAとは何ですか？」。
同じ `INSTRUCTIONS` と同じ3つのツール関数を渡しているのに、比較本番では ADK 版の入力トークンが
1回の実行（LLM 呼び出し2回の合計）あたり 123〜126 多かった（Q1・Q2・Q4・Q5。Q3 は make_quiz の
用語が乱数で変わるので比較対象外）。その原因を、実際に送られる直前のリクエストを並べて確かめる。

## 取得方法

- **ADK**: `root_agent.model_copy(update={"before_model_callback": ...})` でコールバックを付け、
  渡ってきた `LlmRequest` の `config.system_instruction`・`config.tools`・`contents` を保存し、ダミーの
  `LlmResponse` を返して API を呼ばない。1回目は `search_glossary(query="VLA")` の function call を、
  2回目はテキストを返し、ツール結果を含む2回目のリクエストも取得した。
- **Pydantic AI**: `TestModel(call_tools=["search_glossary"])` で `agent.run_sync` を実行し、
  `last_model_request_parameters.function_tools`（ツール定義）と `all_messages()`（instructions と履歴）を取得。
  さらに、それを Gemini 形式に変換する `GoogleModel._build_content_and_config()` を、ダミーのキーで作った
  `GoogleModel` に対して呼び（ネットワークは使わない）、実際に送られる `system_instruction`・`tools`・`contents` を得た。

## 1. system_instruction

| | 内容 |
|---|---|
| Pydantic AI | `INSTRUCTIONS` そのまま（217 文字） |
| ADK | `INSTRUCTIONS` + 次の文（277 文字） |

ADK が末尾に足している文:

```text
'\n\n\nYou are an agent. Your internal name is "glossary_agent".'
```

ADK は `Agent(name=...)` の名前を使って「You are an agent. Your internal name is "glossary_agent".」を
自動で追加する（マルチエージェントで転送先を名乗らせるための仕組み）。Pydantic AI には相当するものが無い。

## 2. ツール宣言（function_declarations）

名前と description（docstring）は3つとも完全に一致（確認済み: True）。違うのは構造と JSON Schema の細部。

| 観点 | Pydantic AI | ADK |
|---|---|---|
| Tool の個数 | ツールごとに1つの `Tool`（3個） | 1つの `Tool` に3つの宣言をまとめる |
| パラメータの schema | `additionalProperties: false` を付ける。`title` は無い | `title` を付ける（引数ごとに `"title": "Query"` など、全体に `"title": "search_glossaryParams"` など）。`additionalProperties` は無い |
| 文字数（tools 全体、整形 JSON） | 1394 文字 | 1402 文字 |

Pydantic AI 側（実際に送られる形）:

```json
[
 {
  "function_declarations": [
   {
    "name": "search_glossary",
    "description": "講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。\n\nAIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について\n答える前に使うこと。query には調べたい用語を1つだけ入れる（例: \"LoRA\"）。",
    "parameters_json_schema": {
     "additionalProperties": false,
     "properties": {
      "query": {
       "type": "string"
      }
     },
     "required": [
      "query"
     ],
     "type": "object"
    }
   }
  ]
 },
 {
  "function_declarations": [
   {
    "name": "get_lecture_summary",
    "description": "指定した回（1始まりの整数）の講義の要点を返す。\n\n「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。",
    "parameters_json_schema": {
     "additionalProperties": false,
     "properties": {
      "lecture": {
       "type": "integer"
      }
     },
     "required": [
      "lecture"
     ],
     "type": "object"
    }
   }
  ]
 },
 {
  "function_declarations": [
   {
    "name": "make_quiz",
    "description": "指定した回（1始まりの整数）の用語集から用語を1つ選び、確認問題と答えを作る。\n\n「第2回の確認問題を出して」「第3回の内容からクイズを1問出して」のように、\n利用者が特定の回の理解度を確かめる問題を求めたときに使う。回の番号を lecture に入れる。\n返り値は JSON 文字列で、question（「○○」の意味は？）、answer（その用語の定義）、\nterm_id、lecture を含む。問題を出すときは answer をそのまま見せず、利用者が答えたあとに使う。",
    "parameters_json_schema": {
     "additionalProperties": false,
     "properties": {
      "lecture": {
       "type": "integer"
      }
     },
     "required": [
      "lecture"
     ],
     "type": "object"
    }
   }
  ]
 }
]
```

ADK 側（実際に送られる形）:

```json
[
 {
  "function_declarations": [
   {
    "description": "講座の用語集から用語を検索し、定義と、どの回で扱うかを返す。\n\nAIエンジニアリングの用語や、この講座（AIE2026）に固有の言葉について\n答える前に使うこと。query には調べたい用語を1つだけ入れる（例: \"LoRA\"）。",
    "name": "search_glossary",
    "parameters_json_schema": {
     "properties": {
      "query": {
       "title": "Query",
       "type": "string"
      }
     },
     "required": [
      "query"
     ],
     "title": "search_glossaryParams",
     "type": "object"
    }
   },
   {
    "description": "指定した回（1始まりの整数）の講義の要点を返す。\n\n「第3回では何を学ぶ？」のように、回の内容や関係を聞かれたときに使う。",
    "name": "get_lecture_summary",
    "parameters_json_schema": {
     "properties": {
      "lecture": {
       "title": "Lecture",
       "type": "integer"
      }
     },
     "required": [
      "lecture"
     ],
     "title": "get_lecture_summaryParams",
     "type": "object"
    }
   },
   {
    "description": "指定した回（1始まりの整数）の用語集から用語を1つ選び、確認問題と答えを作る。\n\n「第2回の確認問題を出して」「第3回の内容からクイズを1問出して」のように、\n利用者が特定の回の理解度を確かめる問題を求めたときに使う。回の番号を lecture に入れる。\n返り値は JSON 文字列で、question（「○○」の意味は？）、answer（その用語の定義）、\nterm_id、lecture を含む。問題を出すときは answer をそのまま見せず、利用者が答えたあとに使う。",
    "name": "make_quiz",
    "parameters_json_schema": {
     "properties": {
      "lecture": {
       "title": "Lecture",
       "type": "integer"
      }
     },
     "required": [
      "lecture"
     ],
     "title": "make_quizParams",
     "type": "object"
    }
   }
  ]
 }
]
```

## 3. それ以外の config

| | Pydantic AI | ADK |
|---|---|---|
| config の他の項目 | {  "tool_config": {   "function_calling_config": {    "mode": "VALIDATED"   }  },  "response_modalities": [   "TEXT"  ] } | （なし。temperature などは未設定） |

Pydantic AI は `tool_config.function_calling_config.mode = "VALIDATED"` と `response_modalities = ["TEXT"]` を
付ける。これらはプロンプトの一部ではなく API の設定なので、入力トークンには数えられないと考えられる（推測）。

## 4. contents（会話履歴）

1回目（質問だけ）は両者とも同じ: `{"role": "user", "parts": [{"text": "VLAとは何ですか？"}]}`。

2回目（ツール結果を返したあと）:

| | Pydantic AI | ADK |
|---|---|---|
| function_call パート | `name`・`args` に加えて `id` と `thought_signature`（モデルが返したものを送り返す） | `name`・`args` のみ（ADK もモデルが返した `thought_signature` はパートごと保存するので、本番では付く可能性がある） |
| function_response の包み方 | `{"return_value": <ツールの返り値の文字列>}` + `id` | `{"result": <ツールの返り値の文字列>}` |

Pydantic AI（2回目）:

```json
[
 {
  "role": "user",
  "parts": [
   {
    "text": "VLAとは何ですか？"
   }
  ]
 },
 {
  "role": "model",
  "parts": [
   {
    "function_call": {
     "name": "search_glossary",
     "args": {
      "query": "a"
     },
     "id": "pyd_ai_tool_call_id__search_glossary"
    },
    "thought_signature": "b'skip_thought_signature_validator'"
   }
  ]
 },
 {
  "role": "user",
  "parts": [
   {
    "function_response": {
     "name": "search_glossary",
     "response": {
      "return_value": "[{\"term\": \"AIエンジニアリング\", \"lecture\": 1, \"definition\": \"学習済みの基盤モデル（主にLLM）を部品として使い、実際に役立つアプリケーションやシステムを設計・構築・評価・運用する技術領域。モデルを一から作ることより、モデルの選択、データやツールとの接続、評価と改善のサイクルを回すことに重心がある。\", \"related\": [\"llm\", \"evaluation-loop\"]}, {\"term\": \"評価と改善のサイクル\", \"lecture\": 1, \"definition\": \"作る→測る→直すを繰り返す開発の進め方。AIアプリは出力が確率的に揺れるため、感覚ではなく評価データや指標で良し悪しを判断し、その結果をもとにプロンプト・データ・モデル・ツールを改善する。本講座では学習支援エージェントを題材に、回をまたいでこのサイクルを回す。\", \"related\": [\"ai-engineering\", \"hallucination\"]}, {\"term\": \"LLM（大規模言語モデル）\", \"lecture\": 1, \"definition\": \"大量のテキストで学習し、入力に続くトークンを予測することで文章を生成するモデル。質問応答、要約、コード生成など多くのタスクを、指示（プロンプト）を変えるだけでこなせる。\", \"related\": [\"token\", \"prompt\", \"pretraining\"]}]"
     },
     "id": "pyd_ai_tool_call_id__search_glossary"
    }
   }
  ]
 }
]
```

ADK（2回目）:

```json
[
 {
  "parts": [
   {
    "text": "VLAとは何ですか？"
   }
  ],
  "role": "user"
 },
 {
  "parts": [
   {
    "function_call": {
     "args": {
      "query": "VLA"
     },
     "name": "search_glossary"
    }
   }
  ],
  "role": "model"
 },
 {
  "parts": [
   {
    "function_response": {
     "name": "search_glossary",
     "response": {
      "result": "用語集に「VLA」は見つかりませんでした。"
     }
    }
   }
  ],
  "role": "user"
 }
]
```

（TestModel はダミーの引数 `query="a"` でツールを呼ぶため、Pydantic AI 側の検索結果は VLA ではなく「a」の部分一致。
包み方の比較が目的なので中身の違いは無視してよい。）

## 5. 差がどこから来るか

**確かめたこと**
- 両者とも、`system_instruction` と `tools` は1回目・2回目のリクエストで同一（ADK: True/True、Pydantic AI: True/True）。
  つまり、固定の差は LLM 呼び出し1回ごとに同じだけ乗る。比較本番の 1 実行あたりの差 123〜126 トークンは、
  LLM 呼び出し2回ぶんなので、1 リクエストあたり約 62 トークンの差になる。
- ADK だけにある文字列: 「You are an agent. Your internal name is "glossary_agent".」（57 文字）と、
  schema の `title` 6 か所。Pydantic AI だけにある文字列: `additionalProperties: false` 3 か所、`id`、`thought_signature`。

**推測（トークナイザで数えていない）**
- 1 リクエストあたり約 62 トークンのうち、「You are an agent...」の文が 15 前後、`title` 6 か所が 30〜40 前後で
  大半を説明できると考えられる。`additionalProperties: false` は Pydantic AI 側を 10 前後増やす方向に働く。
- Gemini がツール宣言をどう文字列化してトークンに数えるかは公開されていないので、正確な内訳は
  `models.count_tokens` に両方の config を渡して測るしかない（API を使うので今回はやっていない）。
- 出力トークンの差（ADK が Q1〜Q3・Q5 で少なめ、Q4 で多め）は、入力の差ではなく回答文の長さの揺れによるもので、
  rep 2回では傾向と言えるほどではない。
