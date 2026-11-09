"""第3回 宿題（発展 a）: 演習のエージェントを Google ADK で組み立て直す。

指示（INSTRUCTIONS）とツール（tools.py の関数）は Pydantic AI 版と同じものを使い、
エージェントを組み立てる部分だけを ADK（2.x）の書き方にする。

使い方（リポジトリ直下で）:
    PYTHONPATH=. uv run --with google-adk adk run adk_agent/glossary_agent

API キーは環境変数 GEMINI_API_KEY がそのまま使われる（ADK が内部で作る google-genai の
Client が読む）。モデル名は app/config.py の MODEL（環境変数 AIE_MODEL で切り替え可）。
"""

from google.adk.agents import Agent

from app import tools
from app.agent import INSTRUCTIONS
from app.config import MODEL

# adk run は、このモジュールの root_agent という名前の変数を探して動かす
root_agent = Agent(
    model=MODEL,
    name="glossary_agent",
    instruction=INSTRUCTIONS,
    # 関数をそのまま渡すと、ADK が docstring と型ヒントからツールの説明文を作る
    tools=[tools.search_glossary, tools.get_lecture_summary, tools.make_quiz],
)
