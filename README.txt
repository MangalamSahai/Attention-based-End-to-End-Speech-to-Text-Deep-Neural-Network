How to run the Code:

1) Install Levenshtein Library
2) Import all libraries
3) run the cell having functions: create_dictionaries, transform_index_to_letter 
4) Download Dataset from Kaggle
5) Run Cell having LibriSamples & LibriSamplesTest
6) Run the cell having dataloaders
7) Run the model cell
8) Run the Levenshtein Distance cell
9) Run the block cell
10) Run the Encoder
11) Run the Attention Class
12) Run the Decoder Class
13) Run the Seq2Seq class
14) Run the model cell
15) Initilize Optimizer, Scheduler & Criterion
16) Run the Distance function Cell(Levenshtein Distance)
17) Run the cell having both train & val functions defined.
18) Run the epochs
You can load saved models 
19) Run test function from model.eval()
20) Save final_submisison.csv

Different Variants tried:
1) Tried Dropout of 0.2 and 4x input channels in pbLSTMs

Result- Overfitting and after 80 epochs the Levenshtein distance started increasing after 17, even though the loss kept on decreasing.

2) Tried Masking all the sequence lengths after lens.  mask[j, lens:, :]

Result - Model reached LD of 14 and started increasing after 112 epochs.

3) Criterion , with reduction = "None" & did not mask the predictions.

Result- Model reached LD of 12 minimum & overfitted after that.

Best Model:
2x input channels in pblstms.(To prevent overfitting)
Criterion= nn.CrossEntropyLoss(ignore_index=int(-1000))
Dropout in pblstms = 0.4
Masking is 2 Dimensional
Masked the predictions; predictions = predictions.masked_fill_(mask,int(-1000))

Model Used:

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

