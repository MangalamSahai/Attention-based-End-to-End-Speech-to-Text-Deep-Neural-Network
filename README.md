# Attention-based-End-to-End-Speech-to-Text-Deep-Neural-Network

Reached Minimum Levenshtein Distance of 12

Model Used-

Seq2Seq(
  (encoder): Encoder(
    (embedding): Sequential(
      (0): Conv1d(13, 256, kernel_size=(3,), stride=(2,), padding=(1,), bias=False)
      (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): GELU()
      (3): Block(
        (layers): Sequential(
          (0): Conv1d(256, 256, kernel_size=(3,), stride=(1,), padding=(1,), groups=256)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): Conv1d(256, 1024, kernel_size=(1,), stride=(1,))
          (3): GELU()
          (4): Conv1d(1024, 256, kernel_size=(1,), stride=(1,))
        )
      )
      (4): Dropout(p=0.4, inplace=False)
    )
    (lstm): LSTM(256, 256, batch_first=True, bidirectional=True)
    (pBLSTMs): Sequential(
      (0): pBLSTM(
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (1): LD()
      (2): pBLSTM(
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (3): LD()
      (4): pBLSTM(
        (blstm): LSTM(1024, 256, batch_first=True, bidirectional=True)
      )
      (5): LD()
    )
    (key_network): Linear(in_features=512, out_features=128, bias=True)
    (value_network): Linear(in_features=512, out_features=128, bias=True)
  )
  (decoder): Decoder(
    (embedding): Embedding(30, 256, padding_idx=29)
    (lstm1): LSTMCell(384, 512)
    (lstm2): LSTMCell(512, 128)
    (attention): Attention()
    (character_prob): Linear(in_features=256, out_features=30, bias=True)
  )
)

