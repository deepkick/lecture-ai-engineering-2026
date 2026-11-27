# 第6回 解答

## 演習

実行日：2026-11-27。Mac ローカル（uv の `.venv`）、モデルは `gemini-3.5-flash-lite`。

- STEP 1 のツールの名前：`glossary_search_glossary`
- STEP 2 の [tool call]：`glossary_search_glossary({'query': '3レベル'})`
- STEP 3 の [tool call]：`fetch_fetch({'url': 'https://modelcontextprotocol.io/'})`
- STEP 4 の [tool call]：次の2行（1回目の結果が質問と合わず、モデルが検索語を変えてもう一度呼んだ）
  - `aws_aws___search_documentation({'search_phrase': 'Amazon S3 バージョニング 概要', 'topics': ['general']})`
  - `aws_aws___search_documentation({'search_phrase': '"Using versioning" "Amazon S3" overview what is versioning', 'topics': ['general']})`
- 気づき・疑問：
  - 設定に数行足すだけで、ツールが 1 個から 7 個になった。ただし、使わないツールの定義も毎回モデルに送られる。STEP 2 と同じ質問を 7 個の状態で実行すると、ツール呼び出しと回答は同じまま、入力トークンだけが約 7.5 倍になった（各1回の実行）。

    | 設定ファイルのサーバー | ツール | 入力トークン（応答 2 回の合計） |
    |---|---|---|
    | glossary | 1 個 | 715 |
    | glossary、fetch、aws | 7 個 | 5,383 |

  - `fetch_fetch` の説明文には「このツールでインターネットに接続できるようになった」とモデルに語りかける文があった。説明文は、サーバーの作者が書いた文がそのままモデルへの指示になる。出所を確認してから足す、という補足の意味が具体的に分かった。

## 宿題

- 取り組んだレベル：発展（対象は、普段使っている Claude Code 2.1.198）

### 目的

Claude Code の MCP サーバーの設定方法を公式ドキュメントで調べ、演習の `app/mcp_servers.json` と比べる。読むだけでなく、演習の用語集サーバーと AWS のサーバーを実際に登録して、記載どおりに動くかを確かめる。調べることは、講座が指定した4つ（設定を書く場所、コマンドで起動する書き方、URL で接続する書き方、有効になる範囲）。

### やったこと

- 調べた日：2026-11-27
- 参照した公式ドキュメント：
  - https://code.claude.com/docs/en/mcp-quickstart
  - https://code.claude.com/docs/en/mcp
- 実験（リポジトリのルートで実行。終わったあと、登録と `.mcp.json` は削除した）：
  - A：`claude mcp add glossary -- uv run python -m app.mcp_server` で用語集サーバーを登録し、`claude mcp list` と `claude mcp get glossary` を見た
  - B：`claude` の会話で `/mcp` を開き、そのあと「AIE2026 の宿題の3レベルとは」と質問した
  - C：演習の `"aws": {"url": "…"}` を `.mcp.json` にそのまま書き、次に `"type": "http"` を足し、最後に承認して接続した
  - D：別のフォルダで `claude mcp list` を実行した

### 結果

| 調べること | 演習（`app/mcp_servers.json`） | Claude Code（公式ドキュメントの記載） | 実験で確かめたこと |
|---|---|---|---|
| 設定を書く場所 | リポジトリの `app/mcp_servers.json` | `claude mcp add` で登録する。local と user は `~/.claude.json`、project はプロジェクト直下の `.mcp.json` | A：登録すると `File modified: ~/.claude.json [project: …]` と表示された |
| コマンドで起動 | `{"command": "uv", "args": […]}` | `claude mcp add glossary -- uv run python -m app.mcp_server`。JSON は演習と同じ形（`type` を省くと stdio） | A・B：演習と同じ起動コマンドで `✔ Connected`。会話から `search_glossary(query: "3レベル")` が呼ばれた。引数は演習の Gemini と同じ |
| URL で接続 | `{"url": "…"}` | `claude mcp add --transport http aws <URL>`。JSON は `{"type": "http", "url": "…"}` で、`type` は必須 | C：演習の行をそのまま書くと読み飛ばされた。`type` を足すと承認待ちになり、承認すると接続できた（ツール 5 個） |
| 有効になる範囲 | このエージェント（`app.mcp_agent`）だけ | local（既定）：自分だけ・このプロジェクト<br>project：clone した全員・承認が必要<br>user：自分だけ・全プロジェクト | A・D：指定しないと local になった。別のフォルダでは glossary も aws も出なかった |
| （参考）ツール名 | `glossary_search_glossary` | `mcp__<サーバー名>__<ツール名>` | B：`mcp__glossary__search_glossary` |
| （参考）呼ぶ前の承認 | なし | 通常のモードでは実行前に尋ねる | B：承認の画面が出た（普段の auto モードから切り替えて確認） |

実験 C の表示（`claude mcp list`）：

```text
# 演習の aws の行をそのまま書いたとき
[Warning] [aws] mcpServers.aws: Skipped — invalid MCP server config for "aws": command: expected string, received undefined

# "type": "http" を足したあと
aws: https://knowledge-mcp.global.api.aws (HTTP) - ⏸ Pending approval (run `claude` to approve)
```

### 気づき・疑問

- `mcpServers`・`command`・`args` の形は共通で、stdio のサーバーは設定をそのまま持っていけた。一方、URL の項目は `type` の有無で互換性がなかった。共通なのはプロトコルで、設定ファイルの書式はクライアントごとに決まっている。
- 演習の設定には「誰に有効か」と「人の承認」がなかった。Claude Code はスコープを3つに分け、project のサーバーは承認するまで接続せず、ツールを呼ぶ前にも尋ねる。演習の補足「出所の分からないサーバーは設定ファイルに書かない」を、仕組みとして持っている。また、claude.ai のコネクタのように設定ファイルにないサーバーも有効になるので、何が有効かは `claude mcp list` で確かめる必要がある。
- 手元の 2.1.198 のエラーは「`command` がない」としか言わず、`type` を足せばよいことが分からなかった。ドキュメントによると、新しい版は直し方まで示す。講義 Part 3 の「エラーには直し方を書く」は、人向けの設定エラーにも当てはまる。版と日付を記録する意味もここにある。
- 疑問：多数が fork するリポジトリに、project スコープの `.mcp.json` を含めるべきかどうかの基準は何か。
