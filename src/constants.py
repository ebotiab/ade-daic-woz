import torch

CONSTANTS = {"max_question_grade": 3,
             "min_question_grade": 0,
             "async_transcripts": [318, 321, 341, 362],
             "diff_time_async_tr": [33, ],
             "init_spec_features": {"global_mean": torch.tensor(0.), "global_var": torch.tensor(1.)}
             }
