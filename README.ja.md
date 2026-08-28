# nanoGPT

![nanoGPT](assets/nanogpt.jpg)


---

**2025年11月更新** nanoGPTには、新しく改良された親戚である[nanochat](https://github.com/karpathy/nanochat)があります。おそらくあなたが使用/探しているのはnanochatの方です。nanoGPT（このリポジトリ）は現在非常に古く非推奨ですが、記録のために残しておきます。

---

中規模GPTの訓練/ファインチューニングのための、最もシンプルで高速なリポジトリです。[minGPT](https://github.com/karpathy/minGPT)の書き換えで、教育よりも実用性を優先しています。まだ活発に開発中ですが、現在は`train.py`がGPT-2（124M）をOpenWebTextで再現し、単一の8XA100 40GBノードで約4日の訓練で実行できます。コード自体は平易で読みやすく：`train.py`は約300行のボイラープレート訓練ループで、`model.py`は約300行のGPTモデル定義であり、オプションでOpenAIからGPT-2の重みを読み込めます。以上です。

![repro124m](assets/gpt2_124M_loss.png)

コードが非常にシンプルなため、自分のニーズに合わせてハックしたり、スクラッチから新しいモデルを訓練したり、事前訓練済みチェックポイントをファインチューニングしたりするのが非常に簡単です（現在利用可能な最大のものはOpenAIのGPT-2 1.3Bモデルです）。

## インストール

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

依存関係：

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
- `transformers`：Hugging Face Transformers用 <3（GPT-2チェックポイントの読み込みのため）
- `datasets`：Hugging Face Datasets用 <3（OpenWebTextのダウンロード+前処理をする場合）
- `tiktoken`：Open高速BPEコード用 <3
- `wandb`：オプションのログ記録用 <3
- `tqdm`：プログレスバー用 <3

## クイックスタート

ディープラーニングの専門家でなく、ただ魔法を感じて手始めに試してみたい場合、最速のスタート方法は、シェイクスピアの作品で文字レベルのGPTを訓練することです。まず、1つの（1MB）ファイルとしてダウンロードし、生テキストから大きな整数ストリームに変換します：

```sh
python data/shakespeare_char/prepare.py
```

これにより、そのデータディレクトリに`train.bin`と`val.bin`が作成されます。さて、GPTを訓練する時です。そのサイズはシステムの計算リソースに大きく依存します：

**GPUがあります**。素晴らしい、[config/train_shakespeare_char.py](config/train_shakespeare_char.py)の設定ファイルで提供される設定で、ベビーGPTを素早く訓練できます：

```sh
python train.py config/train_shakespeare_char.py
```

中身を見ると、最大256文字のコンテキストサイズ、384の特徴チャネル、各層6ヘッドの6層Transformerを訓練していることがわかります。1つのA100 GPUでこの訓練は約3分かかり、最良の検証損失は1.4697です。設定に基づき、モデルチェックポイントは`--out_dir`ディレクトリ`out-shakespeare-char`に書き込まれます。訓練が終わると、サンプリングスクリプトをこのディレクトリに向けて最良のモデルからサンプリングできます：

```sh
python sample.py --out_dir=out-shakespeare-char
```

これによりいくつかのサンプルが生成されます、例えば：

```
ANGELO:
And cowards it be strawn to my bed,
And thrust the gates of my threats,
Because he that ale away, and hang'd
An one with him.

DUKE VINCENTIO:
I thank your eyes against it.

DUKE VINCENTIO:
Then will answer him to save the malm:
And what have you tyrannous shall do this?

DUKE VINCENTIO:
If you have done evils of all disposition
To end his power, the day of thrust for a common men
That I leave, to fight with over-liking
Hasting in a roseman.
```

lol  `¯\_(ツ)_/¯`。GPUで3分の訓練後の文字レベルモデルとしては悪くありません。より良い結果は、代わりに事前訓練済みGPT-2モデルをこのデータセットでファインチューニングすることで得られる可能性が高いです（後のファインチューニングセクションを参照）。

**MacBookしかありません**（またはその他の安価なコンピュータ）。心配ありません、GPTを訓練できますが、少し設定を下げる必要があります。最新のPyTorch nightly（[こちらで選択](https://pytorch.org/get-started/locally/)）を入手することをお勧めします。現在はコードの効率を大幅に向上させる可能性があります。なくても、シンプルな訓練は以下のようになります：

```sh
python train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

ここではCPUで実行するため、`--device=cpu`とPyTorch 2.0コンパイルを`--compile=False`で無効にする必要があります。評価時は少しノイジーですが高速な推定を得るため（`--eval_iters=20`、200から減少）、コンテキストサイズは256文字ではなく64文字、バッチサイズもイテレーションあたり64例ではなく12例のみです。さらに小さなTransformer（4層、4ヘッド、埋め込みサイズ128）を使用し、イテレーション数を2000に減らし（対応して学習率も`--lr_decay_iters`でmax_iters付近まで減衰）、ネットワークが小さいため正則化も緩和します（`--dropout=0.0`）。これでも約3分で実行されますが、損失は1.88のみで、サンプルも悪くなりますが、それでも十分楽しめます：

```sh
python sample.py --out_dir=out-shakespeare-char --device=cpu
```

以下のようなサンプルが生成されます：

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

CPUで約3分でこの文字の雰囲気が出るのは悪くありません。もっと待つ気があるなら、ハイパーパラメータを調整したり、ネットワークサイズ、コンテキスト長（`--block_size`）、訓練期間などを増やしたりしてください。

最後に、Apple Silicon MacBookと最近のPyTorchバージョンでは、`--device=mps`（「Metal Performance Shaders」の略）を追加してください。PyTorchがチップ内蔵GPUを使用し、訓練を*大幅に*高速化（2-3倍）し、より大きなネットワークを使用できるようになります。詳細は[Issue 28](https://github.com/karpathy/nanoGPT/issues/28)を参照してください。

## GPT-2の再現

より本格的なディープラーニングの専門家は、GPT-2の結果再現により関心があるかもしれません。では始めましょう - まずデータセットをトークナイズします。この場合は[OpenWebText](https://openwebtext2.readthedocs.io/en/latest/)、OpenAIの（非公開の）WebTextのオープン再現です：

```sh
python data/openwebtext/prepare.py
```

これにより[OpenWebText](https://huggingface.co/datasets/openwebtext)データセットがダウンロードされトークナイズされます。`train.bin`と`val.bin`が作成され、GPT2 BPEトークンIDが1つのシーケンスとして保持され、生のuint16バイトとして保存されます。そして訓練を開始する準備が整います。GPT-2（124M）を再現するには、少なくとも8X A100 40GBノードが必要で、以下を実行します：

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

これはPyTorch Distributed Data Parallel（DDP）を使用して約4日間実行され、損失は約2.85まで下がります。なお、GPT-2モデルをOWTで評価すると検証損失は約3.11ですが、ファインチューニングすると約2.85の領域まで下がります（ドメインギャップが明らかにあるため）、これにより2つのモデルがほぼ一致します。

複数のGPUノードに恵まれたクラスタ環境にいる場合、GPUをフル活用できます。例えば2ノードで：

```sh
# 最初の（マスター）ノードで実行（例：IP 123.456.123.456）：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# ワーカーノードで実行：
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

相互接続のベンチマーク（例：iperf3）を取るのは良い考えです。特に、Infinibandがない場合は、上記の起動コマンドの前に`NCCL_IB_DISABLE=1`を付けることもできます。マルチノード訓練は動作しますが、おそらく_非常に遅く_なります。デフォルトではチェックポイントが定期的に`--out_dir`に書き込まれます。`python sample.py`を実行するだけでモデルからサンプリングできます。

最後に、単一GPUで訓練するには単に`python train.py`スクリプトを実行します。すべての引数を見てください。スクリプトは非常に読みやすく、ハックしやすく、透明性が高いよう設計されています。ニーズに応じてそれらの変数の多くを調整したくなるでしょう。

## ベースライン

OpenAI GPT-2チェックポイントにより、openwebtextでのベースラインを設定できます。以下のように数値を取得できます：

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

そして、訓練と検証での以下の損失を観察できます：

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

ただし、GPT-2は（非公開で未リリースの）WebTextで訓練されたのに対し、OpenWebTextはこのデータセットのベストエフォートなオープン再現に過ぎません。これはデータセットのドメインギャップを意味します。実際、GPT-2（124M）チェックポイントを取り、OWTで直接少しファインチューニングすると損失が約2.85まで下がります。これが再現に関してより適切なベースラインとなります。

## ファインチューニング

ファインチューニングは訓練と変わりません。事前訓練済みモデルから初期化し、より小さな学習率で訓練することを確認するだけです。GPTを新しいテキストでファインチューニングする例として、`data/shakespeare`に移動し`prepare.py`を実行して、小さなシェイクスピアデータセットをダウンロードし、GPT-2のOpenAI BPEトークナイザーを使用して`train.bin`と`val.bin`にレンダリングします。OpenWebTextとは異なり、これは数秒で完了します。ファインチューニングは非常に短時間で済み、例えば単一GPUでわずか数分です。ファインチューニングの例を実行するには：

```sh
python train.py config/finetune_shakespeare.py
```

これは`config/finetune_shakespeare.py`の設定パラメータオーバーライドを読み込みます（あまり調整していませんが）。基本的に、`init_from`でGPT2チェックポイントから初期化し、通常通り訓練しますが、より短く、小さな学習率で行います。メモリ不足の場合はモデルサイズを減らしてみてください（`{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`）、または`block_size`（コンテキスト長）を減らすことも可能です。最良のチェックポイント（最低検証損失）は`out_dir`ディレクトリにあり、デフォルトでは設定ファイルにより`out-shakespeare`になります。その後`sample.py --out_dir=out-shakespeare`のコードを実行できます：

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

うわ、GPT、暗いところに入り込んでますね。設定のハイパーパラメータをあまり調整していません、自由に試してみてください！

## サンプリング / 推論

`sample.py`スクリプトを使用して、OpenAIがリリースした事前訓練済みGPT-2モデルから、または自分で訓練したモデルからサンプリングします。例えば、利用可能な最大の`gpt2-xl`モデルからサンプリングする方法は以下の通りです：

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

自分で訓練したモデルからサンプリングする場合は、`--out_dir`でコードを適切に指定してください。ファイルからテキストでモデルにプロンプトを与えることもできます、例：```python sample.py --start=FILE:prompt.txt```。

## 効率に関するメモ

シンプルなモデルベンチマークとプロファイリングには、`bench.py`が役立つかもしれません。これは`train.py`の訓練ループの核心部分と同一ですが、他の複雑さの多くを省略しています。

デフォルトではコードは[PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/)を使用します。執筆時点（2022年12月29日）では、これはnightlyリリースで`torch.compile()`を利用可能にします。この1行のコードによる改善は顕著で、例えばイテレーション時間を約250ms/iterから135ms/iterに短縮します。PyTorchチームの素晴らしい仕事です！

## TODO

- DDPの代わりにFSDPを調査・追加
- 標準評価でのゼロショットパープレキシティ評価（例：LAMBADA？HELM？など）
- ファインチューニングスクリプトの調整、ハイパーパラメータが良くないと思う
- 訓練中の線形バッチサイズ増加スケジュール
- 他の埋め込みの組み込み（rotary、alibi）
- チェックポイントでオプティマイザバッファをモデルパラメータから分離
- ネットワーク健全性に関する追加ログ（例：勾配クリップイベント、大きさ）
- より良い初期化などに関する更なる調査

## トラブルシューティング

デフォルトではこのリポジトリはPyTorch 2.0（つまり`torch.compile`）を使用します。これはかなり新しく実験的で、すべてのプラットフォームで利用可能とは限りません（例：Windows）。関連するエラーメッセージが発生した場合は、`--compile=False`フラグを追加して無効にしてみてください。コードは遅くなりますが、少なくとも動作します。

このリポジトリ、GPT、言語モデリングに関する背景については、私の[Zero To Heroシリーズ](https://karpathy.ai/zero-to-hero.html)を見ると参考になるかもしれません。特に、[GPTの動画](https://www.youtube.com/watch?v=kCc8FmEb1nY)は、言語モデリングの事前知識がある場合に人気です。

さらなる質問/議論は、Discordの**#nanoGPT**にお気軽にお立ち寄りください：

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## 謝辞

nanoGPTのすべての実験は、私のお気に入りのクラウドGPUプロバイダーである[Lambda labs](https://lambdalabs.com)のGPUによって支えられています。nanoGPTをスポンサーしてくれたLambda labsに感謝します！

---

maeshiro追記
ChatGPT からのヒント
- https://www.youtube.com/watch?v=kCc8FmEb1nY
- https://nlp.seas.harvard.edu/annotated-transformer/?trk=public_post_comment-text

```
文字列
  ↓
token ID
  ↓
Embedding
  ↓
┌──────────────────────────┐
│ Transformer Block        │
│                          │
│  LayerNorm               │
│      ↓                   │
│  Self Attention          │
│      ↓                   │
│  Residual                │
│      ↓                   │
│  LayerNorm               │
│      ↓                   │
│  MLP / Feed Forward      │
│      ↓                   │
│  Residual                │
└──────────────────────────┘
  ↓
同じBlockをN回
  ↓
Linear
  ↓
各tokenのlogit
  ↓
Softmax
  ↓
次token
```

```
Karpathy mini GPT
        ↓
GPT / Decoder-only Transformer
        ↓
Annotated Transformer
        ↓
Encoder / Decoderの違い
        ↓
Llama / Qwen / MiniMax等の実モデル

```


```
Level 1
token → embedding → Linear → softmax

Level 2
token → embedding
          ↓
       Attention
          ↓
       FFN / MLP
          ↓
        logits

Level 3
          Transformer Block
     ┌─────────────────────┐
x ──→ Attention ── + x
          ↓
        MLP ─────── + x
     └─────────────────────┘
             × N

Level 4
実際のLLM
  RMSNorm
  RoPE
  GQA / MQA
  SwiGLU
  KV Cache
  MoE
  FlashAttention
  etc.
```


```
入力:
"hello"

↓ tokenizer

[7, 4, 11, 11, 14]

↓ embedding

[5 tokens, 32 dimensions]

↓ Q,K,V

Q = XWq
K = XWk
V = XWv

↓ Attention

softmax(QK^T)V

↓ MLP

↓ logits

[5 tokens, vocabulary_size]

↓ 最後の位置

次の文字 = " "
```


