# 🇯🇵 日本株ファンダメンタル分析トレーディングボット

ファンダメンタル分析を用いて日本株を分析し、複数の取引戦略を搭載した包括的なPythonベースのトレーディングボットです。高度なバックテスト機能と戦略比較機能を提供します。

## 🌟 最新アップデート

- **複数戦略フレームワーク**: 8種類の異なる取引戦略（保守的、積極的、バリュー投資、グロース投資など）
- **戦略比較ツール**: 複数の戦略を同時に比較し、パフォーマンスを評価
- **タイムスタンプ付き結果管理**: `yyyyMMddHHmmSS`形式のディレクトリ構造で結果を整理
- **強化された可視化**: 包括的なチャート、リスク・リターン分析、戦略比較プロット
- **Docker対応**: 一貫した結果を得るためのコンテナ化実行環境

## 🎯 機能

### 📊 ファンダメンタル分析
- **多指標スコアリングシステム**: PER、PBR、ROE、負債比率、利益率、配当利回りを使用
- **重み付け複合スコアリング**: 銘柄のランキング付け
- **自動フィルタリング**: ファンダメンタルデータが不十分な銘柄の除外
- **セクター分散**: 分析機能
- **カスタマイズ可能な重み**: 異なるファンダメンタル指標の重み調整

### 🎯 複数取引戦略
- **8種類の内蔵戦略**: 保守的、積極的、バリュー投資、グロース投資、配当重視、品質重視、モメンタム、バランス型アプローチ
- **戦略比較**: 複数の戦略を同時にテストし、パフォーマンスを比較
- **カスタマイズ可能なパラメータ**: 戦略ごとに異なるポートフォリオサイズ、リバランス頻度、リスク管理ルール
- **戦略最適化**: リスクプロファイルに最適な戦略を発見

### 💰 高度なポートフォリオ管理
- **中長期投資アプローチ**: 3ヶ月から1年の保有期間
- **柔軟なポートフォリオサイズ**: 戦略に応じて5-15銘柄
- **動的リバランス**: 戦略に応じて30-120日間隔
- **高度なリスク管理**: カスタマイズ可能なストップロスとテイクプロフィットルール
- **ポートフォリオ最適化**: ファンダメンタルスコアと戦略固有の基準に基づく

### 🔬 バックテストエンジン
- **過去データシミュレーション**: 3-5年間のデータ使用
- **包括的なパフォーマンス指標**:
  - 総リターンと年率リターン
  - シャープレシオ
  - 最大ドローダウン
  - 勝率
  - 日経225ベンチマークに対するアルファ
- **取引追跡**: ポートフォリオ履歴の記録

### 📈 可視化・レポート機能
- **パフォーマンスチャート**: ベンチマーク（日経225）との比較
- **戦略比較チャート**: リスク・リターン分析付き
- **ポートフォリオ推移**: 戦略間の比較
- **ドローダウン分析**: リスク指標
- **月次リターンヒートマップ**
- **ファンダメンタル分析スコア可視化**
- **取引活動分析**
- **戦略特性レーダーチャート**
- **タイムスタンプ付き結果ディレクトリ**: 整理された出力
- **包括的なテキストおよびCSVレポート**

## 🚀 クイックスタート

### 前提条件
- Python 3.8以上
- Docker（オプション、推奨）
- インターネット接続（データ取得用）

### オプション1: Docker（推奨）

1. **リポジトリのクローン**
```bash
git clone https://github.com/your-username/japanese-stock-analysis-bot.git
cd japanese-stock-analysis-bot
```

2. **Dockerで実行**
```bash
docker build -t stock-analysis-bot .
docker run -v $(pwd)/results:/app/results stock-analysis-bot
```

### オプション2: ローカルインストール

1. **リポジトリのクローン**
```bash
git clone https://github.com/your-username/japanese-stock-analysis-bot.git
cd japanese-stock-analysis-bot
```

2. **依存関係のインストール**
```bash
pip install -r requirements.txt
```

3. **単一戦略バックテストの実行**
```bash
python main.py
```

4. **戦略比較の実行**
```bash
python compare_strategies.py
```

ボットは自動的に以下を実行します：
- 日本株の過去データをダウンロード
- ファンダメンタル分析を実行
- バックテストシミュレーション（複数）を実行
- パフォーマンスレポートと可視化を生成
- タイムスタンプ付きディレクトリに結果を保存

## 📋 使用例

### 単一戦略バックテスト
```bash
# デフォルト設定で実行（100万円、2023年から現在まで）
python main.py

# カスタム日付範囲を指定
python main.py --start-date 2020-01-01 --end-date 2023-12-31

# 異なる初期資本を使用（500万円）
python main.py --capital 5000000

# 可視化をスキップ（高速実行）
python main.py --no-plots
```

### 戦略比較
```bash
# 利用可能な全戦略を比較
python compare_strategies.py

# 特定の戦略を比較
python compare_strategies.py --strategies value_investing,growth_investing,dividend_focus

# 異なる資本でカスタム比較
python compare_strategies.py --capital 2000000 --start-date 2023-01-01

# 高速実行のためプロットをスキップ
python compare_strategies.py --no-plots
```

### Docker使用例
```bash
# Dockerで戦略比較を実行
docker run -v $(pwd)/results:/app/results stock-analysis-bot

# カスタムパラメータで単一戦略を実行
docker run -v $(pwd)/results:/app/results stock-analysis-bot python main.py --capital 2000000
```

### 高度な設定
`config.py`を編集してカスタマイズ:
- 株式ユニバース（銘柄の追加/削除）
- ファンダメンタル分析の重み
- 取引パラメータ（ストップロス、テイクプロフィット）
- ポートフォリオサイズとリバランス頻度
- スコアリング閾値

## 📁 プロジェクト構造

```
japanese-stock-analysis-bot/
│
├── main.py                    # 単一戦略実行スクリプト
├── compare_strategies.py      # 戦略比較スクリプト
├── config.py                 # 設定ファイル
├── requirements.txt          # Python依存関係
├── Dockerfile               # Docker設定
├── README.md               # 英語版README
├── README_JP.md           # 日本語版README（このファイル）
│
├── data_fetcher.py           # yfinanceを使用した株価データ取得
├── fundamental_analyzer.py   # ファンダメンタル分析とスコアリング
├── trading_strategy.py       # ポートフォリオ管理と取引ロジック
├── backtester.py            # バックテストエンジン
├── visualizer.py            # チャートと可視化
├── strategy_factory.py      # 複数戦略定義
├── strategy_comparison.py   # 戦略比較エンジン
│
├── data/                    # ダウンロードされた株価データ（自動作成）
├── results/                 # 分析結果（タイムスタンプ付きディレクトリ）
│   ├── 20231215143022/     # タイムスタンプ付き結果ディレクトリの例
│   ├── 20231215143155/     # 別のタイムスタンプ付き結果ディレクトリ
│   └── ...
└── logs/                   # アプリケーションログ（自動作成）
```

## 🔧 設定

### 株式ユニバース
デフォルトで25の主要日本株を分析対象とします：
- トヨタ自動車 (7203.T)
- ソニーグループ (6758.T)
- ソフトバンクグループ (9984.T)
- 任天堂 (7974.T)
- その他21銘柄...

### ファンダメンタル指標と重み
- **PER（20%）**: バリュー投資では低い方が良い
- **PBR（15%）**: 簿価に対する価格
- **ROE（20%）**: 自己資本利益率の効率性
- **負債比率（15%）**: 財務安定性
- **利益率（15%）**: 収益性
- **配当利回り（15%）**: 収益創出

### 取引パラメータ（デフォルト戦略）
- **ポートフォリオサイズ**: 8銘柄（設定可能）
- **リバランス**: 90日ごと
- **ストップロス**: -15%
- **テイクプロフィット**: +30%
- **初期資本**: 100万円（設定可能）

### 利用可能な取引戦略
1. **保守的**: 10銘柄、120日リバランス、-10%ストップロス、+25%テイクプロフィット
2. **積極的**: 5銘柄、30日リバランス、-25%ストップロス、+60%テイクプロフィット
3. **バリュー投資**: 10銘柄、低PER・PBRに重点
4. **グロース投資**: 6銘柄、ROEと利益率を重視
5. **配当重視**: 12銘柄、配当利回りを優先
6. **品質重視**: 8銘柄、品質指標のバランス型アプローチ
7. **モメンタム**: 6銘柄、短期リバランス
8. **バランス**: 8銘柄、全指標の均等重み

## 📊 サンプル出力

### 単一戦略出力
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║           🇯🇵 Japanese Stock Fundamental Analysis Trading Bot 🤖              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

📋 設定:
==================================================
開始日:               2023-01-01
終了日:               現在の日付
初期資本:             ¥1,000,000
ポートフォリオサイズ:   8銘柄
リバランス:           90日ごと
==================================================

🎯 パフォーマンス概要:
初期資本:             ¥1,000,000
最終価値:             ¥1,277,034
総リターン:           +27.70%
年率リターン:         +66.11%
シャープレシオ:       6.36
最大ドローダウン:     -2.05%
勝率:                100.0%

🏆 ベンチマーク比較:
戦略リターン:         +66.11%
日経225リターン:      +43.25%
アルファ（超過収益）: +22.86%
```

### 戦略比較出力
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║        🇯🇵 Japanese Stock Strategy Comparison Tool 📊                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

戦略パフォーマンス概要:
----------------------------------------------------------------------------------------------------
        戦略       総リターン(%) 年率リターン(%) シャープレシオ 最大DD(%) 日経225アルファ(%)
growth_investing    31.69         77.04         4.75       -8.15        33.79
 value_investing    26.16         61.97         6.84       -1.85        18.72

トップパフォーマー:
--------------------------------------------------
最高年率リターン:    growth_investing (77.04%)
最高シャープレシオ:  value_investing (6.84)
最高アルファ:        growth_investing (33.79%)
```

## 📈 生成されるレポート

ボットは自動的にタイムスタンプ付き結果ディレクトリ（`results/yyyyMMddHHmmSS/`）を生成し、以下を含みます：

### 単一戦略結果
1. **パフォーマンスレポート**（`performance_report.txt`）
   - 詳細な指標と統計
   - 主要保有銘柄分析
   - 取引概要

2. **CSVデータファイル**
   - `portfolio_history.csv` - 日次ポートフォリオ価値
   - `trade_history.csv` - 全実行取引
   - `fundamental_analysis.csv` - 株式スコアとランキング
   - `performance_metrics.csv` - 概要統計

3. **可視化**（PNGファイル）
   - ベンチマーク対ポートフォリオパフォーマンス
   - ドローダウン分析
   - 月次リターンヒートマップ
   - ファンダメンタルスコア比較
   - 取引活動分析

### 戦略比較結果
1. **比較レポート**（`strategy_comparison_report.txt`）
   - 全戦略のパフォーマンス概要
   - トップパフォーマー特定
   - 戦略特性比較

2. **戦略データファイル**
   - `strategy_comparison_summary.csv` - 全戦略指標
   - 詳細結果を含む個別戦略サブディレクトリ

3. **比較可視化**
   - 戦略パフォーマンス比較チャート
   - リスク・リターン散布図
   - ポートフォリオ推移比較
   - 戦略特性レーダーチャート

## ⚠️ 重要な免責事項

- **教育目的のみ**: このボットは研究・教育目的で設計されています
- **投資アドバイスではありません**: 十分なテストと検証なしに実際の取引に使用しないでください
- **過去のパフォーマンス**: 過去の成績は将来の結果を保証するものではありません
- **データの制限**: 公開データを使用しており、遅延や不正確さがある可能性があります
- **リスク警告**: すべての投資には損失のリスクが伴います

## 🛠️ カスタマイズ

### 新しい銘柄の追加
`config.py`を編集し、`JAPANESE_STOCKS`リストに株式シンボルを追加:
```python
JAPANESE_STOCKS = [
    '7203.T',  # トヨタ自動車
    '6758.T',  # ソニーグループ
    'XXXX.T',  # 新しい銘柄（東京証券取引所）
    # ... その他の銘柄
]
```

### 分析重みの変更
`config.py`でファンダメンタル分析の重みを調整:
```python
FUNDAMENTAL_WEIGHTS = {
    'per_score': 0.25,      # PER重みを増加
    'pbr_score': 0.10,      # PBR重みを減少
    'roe_score': 0.25,      # ROE重みを増加
    # ... その他の重み
}
```

### 取引ルールの変更
`config.py`で取引パラメータを変更:
```python
STOP_LOSS_THRESHOLD = -0.20      # -20%ストップロス
TAKE_PROFIT_THRESHOLD = 0.50     # +50%テイクプロフィット
REBALANCE_FREQUENCY = 60         # 60日ごとにリバランス
```

### カスタム戦略の作成
`strategy_factory.py`に新しい戦略を追加:
```python
def create_custom_strategy(self):
    return {
        'name': 'custom_strategy',
        'description': 'カスタム戦略の説明',
        'portfolio_size': 10,
        'rebalance_frequency': 45,
        'stop_loss': -0.15,
        'take_profit': 0.35,
        'weights': {
            'per_score': 0.30,
            'pbr_score': 0.20,
            'roe_score': 0.25,
            'debt_ratio_score': 0.10,
            'profit_margin_score': 0.10,
            'dividend_yield_score': 0.05
        }
    }
```

## 🔍 利用可能なコマンド

### 戦略比較コマンド
```bash
# 利用可能な全戦略を一覧表示
python compare_strategies.py --help

# 特定の戦略を実行
python compare_strategies.py --strategies conservative,aggressive,value_investing

# カスタム日付範囲と資本
python compare_strategies.py --start-date 2022-01-01 --capital 5000000
```

### 利用可能な戦略
- `conservative` - 低リスク、安定リターン
- `aggressive` - 高リスク、高潜在リターン
- `value_investing` - 割安株に焦点
- `growth_investing` - 成長ポテンシャルに焦点
- `dividend_focus` - 配当収入を優先
- `quality_focus` - バランス型品質アプローチ
- `momentum` - 短期モメンタム戦略
- `balanced` - 指標間の均等重み

## 🤝 貢献

貢献を歓迎します！改善分野：
- 追加のファンダメンタル指標
- 機械学習統合
- リアルタイムデータフィード
- オプション取引戦略
- 国際市場対応
- 追加戦略タイプ
- パフォーマンス最適化

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細はLICENSEファイルを参照してください。

## 🙏 謝辞

- **yfinance**: Yahoo Financeデータへの無料アクセスを提供
- **日本株式市場**: データプロバイダー
- **オープンソースコミュニティ**: 優秀なPythonライブラリ

---

**ハッピートレーディング！ 🚀📈**

*注意: これは教育目的のみです。投資決定を行う前に、必ず独自の調査を行い、ファイナンシャルアドバイザーに相談することを検討してください。*