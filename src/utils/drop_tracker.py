class DropTracker:
    def __init__(self, save_path=None):
        from collections import OrderedDict
        self.history = OrderedDict()
        self.save_path = save_path

    def drop(self, df, cols, stage_name):
        """
        df: üzerinde işlem yapılan DataFrame
        cols: drop edilecek sütunlar listesi
        stage_name: bu drop işleminin adı/çözüm mantığı
        """
        # 1. Aşama kaydını tut
        self.history[stage_name] = list(cols)

        # 2. DataFrame'den düş
        return df.drop(columns=cols), cols

    def save(self, path):
        """history dict’ini pickle olarak kaydet"""
        import os, pickle
        if not self.save_path:
            raise ValueError("DropTracker.save_path tanımlı değil!")
        import os, pickle
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(self.history, f)
    def load(self, path):
        """Daha önce kaydettiğin history’yi geri yükle"""
        import pickle
        with open(path, "rb") as f:
            self.history = pickle.load(f)
        return self.history
