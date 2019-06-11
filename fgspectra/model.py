from abc import ABC, abstractmethod
import types
import inspect
import yaml
import numpy as np

class Model(ABC):
    """ Abstract class for model definition

    A model is a class that has one purpose: evaluating the model. What
    evaluation means is defined by the `eval` method that any child class has to
    override.

    If the `eval` method of a hypotetical ``Child`` class calls the `eval`
    method of other `Model`s, it is likely that ``Child`` has also to override
    `set_defaults`, `defaults` and `_get_repr`
    """

    def __init__(self, **kwargs):
        """ You can set defaults at construction time """
        self.set_defaults(**kwargs)

    def set_defaults(self, **kwargs):
        """ Change the defaults of the evaluation

        Call ``set_defaults`` with the same arguments (or a subset) of ``eval``
        (or ``__call__``).
        This will change the default value of those arguments in ``eval``. If
        `kwargs` has keys that are not arguments of `eval`, these keys are
        ignored.

        Examples
        --------

        >>> class PrintABC(Model):
        ...     def eval(self, a=1, b=2, c=3):
        ...         print(a, b, c)
        ...
        >>> print_abc = PrintABC()
        >>> print_abc.eval()
        1 2 3
        >>> print_ab10c = PrintABC()
        >>> print_ab10c.set_defaults(b=10)
        >>> print_ab10c.eval()  # The default of b has changed
        1 10 3
        >>> print_abc.eval()  # Without affecting other instances
        1 2 3

        """
        # This is a dirty business, and has limitations. The most important
        # known one is that it breaks with decorated eval
        # To change the default values of the eval method
        # we have to bind a new method to the instance
        # (not the class, otherwise we change the defaults of all the instances)

        if not kwargs:  # No default to be replaced
            return

        # Make a deep copy of the eval function, assuming no variadic arguments
        func = self.eval.__func__

        new_func = types.FunctionType(
            func.__code__, func.__globals__, name=func.__name__,
            argdefs=func.__defaults__, closure=func.__closure__)

        # Build the defaults from scratch, assuming no positional argument
        # Update the defaults that have a corresponding key in kwargs
        # Note that if kwargs has keys that are not in the arguments,
        # these keys are ignored
        fas = inspect.getfullargspec(new_func)
        defaults = []
        for arg, default in zip(fas.args[1:], fas.defaults):
            #[1:] skips self, the only positional argument
            if arg in kwargs:
                defaults.append(kwargs[arg])
            else:
                defaults.append(default)

        new_func.__defaults__ = tuple(defaults)

        # Bind the new method to the instance
        self.eval = types.MethodType(new_func, self)

    @property
    def defaults(self):
        """ The current defaults

        The current defaults as you would pass them to the eval method.

        In particular, since all the arguments have to be keyword arguments, it
        is a nested set of dictionaries with a key for each argument of the
        `eval` methods in the model.

        >>> model.eval() == model.eval(**model_child.defaults)
        True

        """
        return {name: par.default
                for name, par in inspect.signature(self.eval).parameters.items()}

    def __call__(self, *args, **kwargs):
        # __call__ can't be rebound, that's why we use eval
        return self.eval(*args, **kwargs)

    @abstractmethod
    def eval(self, **kwargs):
        """ Evaluation of the model

        Note
        ----
        The signature must contain

        - only keyword arguments (except for ``self``)
        - no variadic argument (*args, **kwargs)

        and

        - no decoration

        """
        pass

    def _get_repr(self):
        """ Readable version of defaults

        In particular, it is also a nested set of dictionaries with a key for
        each argument of the `eval` methods in the model.

        If `eval` calls the `eval` of other classes, you typically want to
        call the _get_repr method of those classes and collect the result in
        some meaningful dictionary
        """
        res = {}
        for key, val in self.defaults.items():
            res[key] = val.tolist() if isinstance(val, np.ndarray) else val
        return {type(self).__name__: res}

    def __repr__(self):
        # Turns the nested set of dictionaries into a string to be printed
        return yaml.dump(self._get_repr())
