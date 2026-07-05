class EarlyStopping:
    def __init__(self, patience: int=15, min_delta: float=1e-4, mode: str="max"):
        self.patience=patience
        self.min_delta=min_delta
        self.mode=mode
        self.best=None
        self.n_bad=0
        self.should_stop=False
        
    def step(self, value: float) -> bool:
        if self.best is None:
            self.best=value
            return False
        improved=(value>self.best+self.min_delta) if self.mode=="max" else (value<self.best-self.min_delta)
        if improved:
            self.best=value
            self.n_bad=0
        else:
            self.n_bad+=1
            if self.n_bad>=self.patience:
                self.should_stop=True
        return self.should_stop