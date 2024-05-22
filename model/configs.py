class www_Config():
    def __init__(self):
        self.sen_len = 128
        self.entity_len = 4
        self.img_hidden_size = 2048
        self.hidden_dim = 150
        self.num_layers = 1
        self.dropout = 0.5  ## 之前是0.5
        self.train = 0.6
        self.val = 0.2
        self.test = 0.2
        self.val_en = 0.5
        self.test_en = 0.5


        self.fix_embedding = True