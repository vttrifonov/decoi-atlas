# %%
import ipywidgets as w
import traitlets as t
import IPython.display as ipd
from types import SimpleNamespace as namespace

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

class NamedChildren:
    def __init__(self, children):
        self.c = namespace(**children)

class VBox(NamedChildren, w.VBox):
    def __init__(self, children, **kwargs):
        super().__init__(children)
        super(NamedChildren, self).__init__(tuple(children.values()), **kwargs)

class HBox(NamedChildren, w.HBox):
    def __init__(self, children, **kwargs):
        super().__init__(children)
        super(NamedChildren, self).__init__(tuple(children.values()), **kwargs)

# %%