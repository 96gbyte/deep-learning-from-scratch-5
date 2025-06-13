
## 実装内容

lib.rs:
- Encoder, Decoder, VAE 構造体
- train_vae_model() - 学習ロジック
- plot_results() - 可視化
- train_vae() - 統合関数
- テスト関数

main.rs:
- clapによるCLI引数解析
- -e/--epochs と --no-plots オプション

## 実行方法
cargo run -p step07                    # デフォルト30 epochs
cargo run -p step07 -- -e 10           # 10 epochs
cargo run -p step07 -- --no-plots      # プロットなし
cargo test -p step07                   # テスト実行
