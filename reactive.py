# %%
class Observable:
    class Observer:
        class Item:
            def __init__(self, observable, observer):
                self._observable = observable
                self._observer = observer
                self._paused = False

            def stop(self):
                if self._observer is None:
                    return
                self._observable._observers.remove(self)
                self._observer = None
                self._paused = False

            def pause(self):
                self._paused = True

            def resume(self):
                self._paused = False

            @property
            def paused(self):
                return self._paused

            def __call__(self, *args, **kwargs):
                if self._paused:
                    return
                self._observer(*args, **kwargs)

        class List:
            def __init__(self, l):
                self._l = l
                self._paused = False

            def stop(self):
                for i in self._l:
                    i.stop()
                self._l = []
                self._paused = False

            def pause(self):
                self._paused = True
                for i in self._l:
                    i.pause()

            def resume(self):
                self._paused = False
                for i in self._l:
                    i.resume()

            @property
            def paused(self):
                return self._paused

    def __init__(self):
        self._observers = set()

    def notify(self, *args, **kwargs):
        for o in self._observers:
            o(*args, **kwargs)

    def observe(self, observer):
        observer = Observable.Observer.Item(self, observer)
        self._observers.add(observer)
        return observer
        
class ValueProvider:
    def __init__(self):
        self._dirty = Observable()

class _reactive_trait(ValueProvider):
    def __init__(self, trait):
        super().__init__()
        self._trait = trait
        trait[0].observe(lambda _: self._dirty.notify(), trait[1])

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
        l = [a._dirty.observe(observer) for a in args]
        return Observable.Observer.List(l)
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
        self._dirty.notify()

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
        self._dirty.notify()

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_observable = None

    def _display(self, x, clear_output, wait):
        with self:
            if clear_output:
                ipd.clear_output(wait=wait)
            if x is None:
                return
            ipd.display(x)
    
    def display(self, x=_None, clear_output=True, wait=True):
        def wrap(x):
            if self._output_observable is not None:
                self._output_observable.stop()
                self._output_observable = None
            if x is None:
                return
            self._output_observable = observe(x)(lambda x: self._display(x, clear_output, wait))
            self.on_displayed(
                lambda *_: self._display(x(), clear_output, wait) if not(self._output_observable is None or self._output_observable.paused) else None
            )
            return x
        if x is _None:
            return wrap
        if isinstance(x, ValueProvider):
            wrap(x)
            return
        wrap(None)
        self._display(x, clear_output, wait)

    def pause(self):
        self._output_observable.pause()

    def resume(self):
        self._output_observable.resume()

    class _output:
        def __set__(self, obj, x):
            if x is tuple:
                if len(x)==1:
                    x = x + (True,True)
                elif len(x)==2:
                    x = x + (True,)
            else:
                x = (x,True,True)                    
            obj.display(*x)

    output = _output()

# %%
if __name__ == '__main__':
    pass

    # %%
    x1 = Value()
    x2 = observe(x1)(print)
    x1(3)
    x2.pause()
    x1(4)
    x2.resume()
    x1(5)
    x2.stop()
    x1(6)
    x1._dirty._observers

    # %%
    ui = VBox(dict(
        i = w.IntText(),
        j = w.IntText(),
        c = w.Dropdown(value='i', options=['i', 'j', 'pause', 'resume']),
        o = Output()        
    ))

    @reactive(ui.i)
    def i(i):
        return i+1

    @reactive(ui.j)
    def j(j):
        return j+1

    ui.o.output = i

    @observe(ui.c)
    def o(c):
        if c=='i':
            ui.o.output = i
        elif c=='j':
            ui.o.output = j
        elif c=='pause':
            ui.o.pause()
        elif c=='resume':
            ui.o.resume()

    ui

# %%
