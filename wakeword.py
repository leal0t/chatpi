import openwakeword

class WakeWordDetector:
    def __init__(self, name="maple"):
        self.model = openwakeword.Model(wakeword=name)

    def detect(self):
        result = self.model.predict()
        if result.get("maple", 0) > 0.8:
            return True
        return False
