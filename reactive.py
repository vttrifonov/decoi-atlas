# %%
class Observable:
    def __init__(self):
        self._observers = set()

    def _notify(self):
        for o in self._observers:
            o()

    def observe(self, o):
        self._observers.add(o)

    def unobserve(self, o):
        self._observers.remove(o)

class ValueProvider:
    def __init__(self):
        self._dirty = Observable()

class _reactive_trait(ValueProvider):
    def __init__(self, trait):
        super().__init__()
        self._trait = trait
        trait[0].observe(lambda _: self._dirty._notify(), trait[1])

    def __call__(self):
        return getattr(*self._trait)

def _fix_args(args):
    args1 = []
    for a in args:
        if not isinstance(a, ValueProvider):
            if not isinstance(a, tuple):
                a = a, 'value'
            a = _reactive_trait(a)
        args1.append(a)
    return args1

def observe(*args):
    args = _fix_args(args)
    def wrap(f):
        def observer():
            f(*[a() for a in args])        
        for a in args:
            a._dirty.observe(observer)
        return f
    return wrap

class _None:
    pass
_None = _None()
        
class Value(ValueProvider):
    def __init__(self, value=_None):
        super().__init__()
        self._value = value

    def __call__(self, value=_None):
        if value is _None:
            value = self._value
            if value is _None:
                raise(ValueError('value is not initialized'))
            return value
        self._value = value
        self._dirty._notify()

class Reactive(ValueProvider):
    def __init__(self, args, f):
        super().__init__()
        self._is_dirty = True
        self._value = _None

        for a in args:
            a._dirty.observe(self._dirty_notify)
        self._update = lambda: f(*[a() for a in args])

    def _dirty_notify(self):
        self._is_dirty = True
        self._dirty._notify()

    def __call__(self):
        if self._is_dirty:
            self._value = self._update()
            self._is_dirty = False
        return self._value

def reactive(*args):                    
    return lambda f: Reactive(_fix_args(args), f)

# %%
import ipywidgets as w
import IPython.display as ipd

def _display(x, out, clear_output, wait):
    if x is None:
        return
    def _display():
        if clear_output:
            ipd.clear_output(wait=wait)
        ipd.display(x)
    if out is None:
        _display()
        return
    with out:
        _display()

def display(x, out=None, clear_output=True, wait=True):
    if not isinstance(x, ValueProvider):
        _display(x, out, clear_output, wait)
        return
    observe(x)(lambda x: _display(x, out, clear_output, wait))
    out.on_displayed(lambda *_: _display(x(), out, clear_output, wait))

class _named_children:
    def __init__(self, children):
        for k, v in children.items():
            self.__dict__[k] = v

class VBox(_named_children, w.VBox):
    def __init__(self, children=dict(), **kwargs):        
        w.VBox.__init__(self, children=list(children.values()), **kwargs)
        _named_children.__init__(self, children)

class HBox(_named_children, w.HBox):    
    def __init__(self, children=dict(), **kwargs):
        w.HBox.__init__(self, list(children.values()), **kwargs)
        _named_children.__init__(self, children)

class Output(w.Output):
    def __init__(self):
        self._output = w.Output()
    
    def capture(self, clear_output=True, wait=True):
        def wrap(x):
            if isinstance(x, ValueProvider):
                display(x, self._output, clear_output, wait)
                return x
            else:
                def wrapper(*args, **kwargs):
                    v = x(*args, **kwargs)
                    display(v, self._output, clear_output, wait)
                    return v
                return wrapper
        return wrap

    class _display:
        def __set__(self, obj, x):
            display(x, obj._output)

    display = _display()

# %%
if __name__ == '__main__':
    x = 1

    # %%
