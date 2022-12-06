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

def _display(out, x, clear_output, wait):
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

def display(out, x, clear_output=True, wait=True):
    if not isinstance(x, ValueProvider):
        _display(out, x, clear_output, wait)
        return
    observe(x)(lambda x: _display(out, x, clear_output, wait))
    out.on_displayed(lambda *_: _display(out, x(), clear_output, wait))

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

# %%
if False:
    from types import SimpleNamespace as namespace
    import traitlets as t

    class Value(t.HasTraits):
        value = t.Any()

        def update(self, v):
            self.value = v

    def observe(*args):
        def wrap(f):
            def observer():
                f(*[a.value for a in args])
            for a in args:
                a.observe(lambda _: observer(), 'value')    

        return wrap

    def observer(*args):
        def wrap(f):
            observe(*args)(f)
            return f
        return wrap

    def reactive(*args):
        def wrap(f, v = None):
            if v is None:
                v = Value()        
            observe(*args)(lambda *a: v.update(f(*a)))
            v.update(f(*[a.value for a in args]))
            return v
        return wrap

    def react(*args):
        return lambda f: reactive(*args)(f)

    class Output(w.Output):
        def display(self, x, clear_output=True, wait=True):
            with self:
                if clear_output:
                    ipd.clear_output(wait=wait)
                ipd.display(x)

        def react(self, x, clear_output=True, wait=True):
            observe(x)(lambda x: self.display(x, clear_output, wait))
            self.on_displayed(lambda *_: self.display(x.value, clear_output, wait))


    class VBox(w.VBox):
        def __init__(self, children, **kwargs):
            w.VBox.__init__(self, list(children.values()), **kwargs)
            self.c = namespace(**children)


    class HBox(w.HBox):
        def __init__(self, children, **kwargs):
            w.HBox.__init__(self, list(children.values()), **kwargs)
            self.c = namespace(**children)

# %%
if __name__ == '__main__':
    x = 1

    # %%
