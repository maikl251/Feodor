try:
    import pymorphy2
except ImportError:
    print("pymorphy2 не установлен. Используется упрощённая лемматизация.")
    pymorphy2 = None

import inspect

if pymorphy2 and not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

_morph = None

def get_morph():
    global _morph
    if _morph is None:
        if pymorphy2:
            print("Инициализация MorphAnalyzer (один раз)...")
            _morph = pymorphy2.MorphAnalyzer()
        else:
            print("MorphAnalyzer недоступен, возвращается None")
            _morph = None
    return _morph
