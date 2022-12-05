# %%
import ipywidgets as w
import traitlets as t
import IPython.display as ipd
from types import SimpleNamespace as namespace

# %%
class ValueProvider:
    pass

class ValueConsumer:
    pass

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

def _is_observable_value_provider(x):
    return isinstance(x, Observable) and isinstance(x, ValueProvider)

def observe(*args):
    def wrap(f):
        def observer():
            f(*[a() for a in args])
        for a in args:
            a.observe(observer)
        return f
    return wrap

class _None:
    pass
_None = _None()

class Value(Observable, ValueProvider, ValueConsumer):
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
        self._notify()

class Reactive(Observable, ValueProvider):
    def __init__(self, args, f):
        super().__init__()
        self._dirty = True
        self._value = _None

        for a in args:
            a.observe(self._notify)
        self._update = lambda: f(*[a() for a in args])

    def _notify(self):
        self._dirty = True
        super()._notify()

    def __call__(self):
        if self._dirty:
            self._value = self._update()
            self._dirty = False
        return self._value

class _reactive_trait(Observable, ValueProvider):
    def __init__(self, trait):
        self.trait = trait
        trait[0].observe(lambda _: self._notify(), trait[1])

    def __call__(self):
        return getattr(*self.trait)

def reactive(*args):
    args1 = []
    for a in args:
        if not _is_observable_value_provider(a):
            if not isinstance(a, tuple):
                a = a, 'value'
            a = _reactive_trait(a)
        args1.append(a)                        
    return lambda f: Reactive(args1, f)

def _display(out, x, clear_output=True, wait=True):
    with out:
        if clear_output:
            ipd.clear_output(wait=wait)
        ipd.display(x)

def display(out, x, clear_output=True, wait=True):
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
ui = VBox(dict(
    i = w.IntText(2),
    o1 = w.Output(),
    o2 = w.Output()
))

j = Value(2)

@reactive(ui.i, j)
def s1(i, j):
    print('ip1')
    return i+j

@reactive(ui.i)
def s2(i):
    print('ip2')
    return i+2

@reactive(ui.i)
def s3(i):
    print('x5')
    return i+3

display(ui.o1, s1)

display(ui.o2, s2)

ui

# %%
s3()

# %%
j(5)


# %%
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

# %%
if __name__ == '__main__':
    x = 1

    # %%
