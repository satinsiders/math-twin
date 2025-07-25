---
search:
  exclude: true
---
# 複数エージェントのオーケストレーション

オーケストレーションとは、アプリ内でのエージェントの流れを指します。どのエージェントを実行するか、実行順序、そして次に何を行うかをどのように決定するかということです。エージェントをオーケストレーションする方法は主に 2 つあります。

1.  LLM に意思決定を任せる: これは  LLM  の知能を活用して計画・推論を行い、その結果に基づいて次のステップを決定させる方法です。  
2.  コードによるオーケストレーション: コードでエージェントの流れを制御する方法です。

これらのパターンは組み合わせて使用できます。それぞれにメリットとデメリットがあり、以下で説明します。

## LLM によるオーケストレーション

エージェントとは、 instructions、 tools、 handoffs を備えた  LLM  です。つまり、オープンエンドなタスクが与えられたとき、  LLM  はツールを使ってアクションを実行しデータを取得し、 handoffs でサブエージェントにタスクを委任しながら、自律的にタスクをこなす計画を立てられます。たとえば、リサーチエージェントには次のようなツールを装備できます。

-   Web 検索によりオンライン情報を取得  
-   ファイル検索と取得による社内データや接続先の検索  
-   コンピュータ操作でコンピュータ上のアクションを実行  
-   コード実行でデータ分析を実施  
-   計画立案やレポート作成などに優れた専門エージェントへの handoffs  

このパターンはタスクがオープンエンドで、  LLM  の知能に依存したい場合に最適です。重要な戦略は次のとおりです。

1.  良いプロンプトに投資する。利用可能なツール、その使い方、および動作パラメーターを明確に記述します。  
2.  アプリをモニタリングして改善を繰り返す。問題が発生した箇所を確認し、プロンプトを継続的に調整します。  
3.  エージェントに内省させて改善させる。たとえばループで実行し、自らの結果を批評させる、またはエラーメッセージを渡して改善させるなどです。  
4.  何でもこなす汎用エージェントではなく、特定タスクに特化したエージェントを用意します。  
5.  [evals](https://platform.openai.com/docs/guides/evals) に投資する。これによりエージェントを訓練し、タスク遂行能力を向上できます。  

## コードによるオーケストレーション

LLM によるオーケストレーションは強力ですが、コードによるオーケストレーションは速度、コスト、パフォーマンスの面でより決定論的かつ予測可能になります。代表的なパターンは以下のとおりです。

-   [structured outputs](https://platform.openai.com/docs/guides/structured-outputs) を使って、コードで検査可能な適切な形式のデータを生成する。たとえば、エージェントにタスクをいくつかのカテゴリーに分類させ、そのカテゴリーに応じて次のエージェントを選択する。  
-   複数エージェントを連鎖させ、一方の出力を次の入力に変換する。ブログ記事執筆タスクをリサーチ → アウトライン作成 → 記事執筆 → クリティーク → 改善という一連のステップに分解するなど。  
-   タスクを実行するエージェントを `while` ループで動かし、その出力を評価しフィードバックを返すエージェントと組み合わせ、評価者が基準を満たしたと判断するまで繰り返す。  
-   `asyncio.gather` のような  Python  の基本コンポーネントを使って複数エージェントを並列実行する。互いに依存しない複数タスクを高速に処理したい場合に有効です。  

`examples/agent_patterns` ディレクトリーに複数のコード例があります。