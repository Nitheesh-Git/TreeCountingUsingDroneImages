class Display:
    max_columns = 10
    width = 100
class Options:
    display = Display()
options = Options()
class DataFrame:
    def __init__(self, *a, **k): pass
    def to_csv(self, *a, **k): pass
def read_csv(*a, **k): return DataFrame()
