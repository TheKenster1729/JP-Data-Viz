"""Custom variables as a parsed expression tree.

A custom variable arrives from the UI as a JSON string, for example

    {"operation": "division",
     "output1": "elec_prod_Renewables_TWh",
     "output2": {"operation": "addition", "outputs": [...], "name": "Total"},
     "name": "Renewable Share"}

The old layer carried that string all the way down and re-decoded it at every
level of the recursion, so a two-level variable was parsed three times per
fetch and a malformed one failed somewhere in the middle of retrieval. Parsing
once at the boundary means the shape is known before any query runs, and the
set of outputs it depends on can be validated against the dataset up front.

Expressions are session-scoped and transient. Nothing here writes them
anywhere; a repository keeps a bounded cache of recently parsed strings and
that is all.

The arithmetic is delegated to global_classes.VariableOutput, which owns the
edge cases that matter to the science: division drops runs whose denominator is
zero, and operands with different run sets are intersected because a model run
that crashed for one output has no counterpart in the other.
"""

import json

BINARY_OPERATIONS = ("subtraction", "multiplication", "division")


class Expression:
    """One node of a parsed custom variable."""

    display_name = None

    def outputs(self):
        """Every output name this expression reads, in evaluation order."""
        raise NotImplementedError

    def evaluate(self, fetch, region, scenario, year=None):
        """Reduce to a Run #/Year/Value frame.

        fetch(output_name) returns the frame for one leaf, already scoped to
        region, scenario and year by the caller.
        """
        raise NotImplementedError


class OutputRef(Expression):
    """A plain output name."""

    def __init__(self, name):
        self.name = name
        self.display_name = name

    def outputs(self):
        return [self.name]

    def evaluate(self, fetch, region, scenario, year=None):
        return fetch(self.name)

    def __repr__(self):
        return "OutputRef({!r})".format(self.name)


class Addition(Expression):
    """A sum over two or more operands."""

    def __init__(self, terms, display_name=None):
        if len(terms) < 1:
            raise ValueError("addition needs at least one output")
        self.terms = terms
        self.display_name = display_name or "nested_operation"

    def outputs(self):
        return [name for term in self.terms for name in term.outputs()]

    def evaluate(self, fetch, region, scenario, year=None):
        variable_output = _variable_output()
        result = None
        for term in self.terms:
            current = term.evaluate(fetch, region, scenario, year)
            if result is None:
                result = current
                continue
            left = variable_output(self.display_name, self.display_name,
                                   region, scenario, result, year=year)
            right = variable_output(term.display_name, term.display_name,
                                    region, scenario, current, year=year)
            result = left + right
        return result

    def __repr__(self):
        return "Addition({!r})".format(self.terms)


class BinaryOperation(Expression):
    """Subtraction, multiplication or division of two operands."""

    def __init__(self, operation, left, right, display_name=None):
        self.operation = operation
        self.left = left
        self.right = right
        self.display_name = display_name or "nested_operation"

    def outputs(self):
        return self.left.outputs() + self.right.outputs()

    def evaluate(self, fetch, region, scenario, year=None):
        variable_output = _variable_output()
        left = variable_output(self.left.display_name, self.left.display_name,
                               region, scenario,
                               self.left.evaluate(fetch, region, scenario, year), year=year)
        right = variable_output(self.right.display_name, self.right.display_name,
                                region, scenario,
                                self.right.evaluate(fetch, region, scenario, year), year=year)
        if self.operation == "subtraction":
            return left - right
        if self.operation == "multiplication":
            return left * right
        return left / right

    def __repr__(self):
        return "BinaryOperation({!r}, {!r}, {!r})".format(
            self.operation, self.left, self.right)


def parse(spec):
    """Build an Expression from a JSON string, a dict, or an output name.

    Raises ValueError for anything that is neither.
    """
    return _build(_decode(spec))


def is_expression(spec):
    """True if spec describes a custom variable rather than an output name."""
    return isinstance(_decode(spec), dict)


def _decode(value):
    """Unwrap JSON strings, including the nested ones the UI produces.

    An operand can itself be a JSON string rather than an object, so decoding
    has to recurse rather than run once at the top.
    """
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return value
        # json.loads("2050") is an int, which is not an output name either;
        # only a further string or object is worth unwrapping again.
        if isinstance(decoded, (str, dict, list)):
            return _decode(decoded)
        return value
    return value


def _build(node):
    if isinstance(node, str):
        return OutputRef(node)
    if not isinstance(node, dict):
        raise ValueError("expected an output name or a custom variable, got {!r}".format(node))

    operation = node.get("operation")
    if operation == "addition":
        terms = node.get("outputs")
        if not isinstance(terms, list):
            raise ValueError("addition requires an 'outputs' list")
        return Addition([_build(_decode(term)) for term in terms], node.get("name"))

    if operation in BINARY_OPERATIONS:
        for key in ("output1", "output2"):
            if key not in node:
                raise ValueError("{} requires '{}'".format(operation, key))
        return BinaryOperation(operation,
                               _build(_decode(node["output1"])),
                               _build(_decode(node["output2"])),
                               node.get("name"))

    raise ValueError("Unsupported operation: {}".format(operation))


def _variable_output():
    # Imported on use: global_classes imports sql_utils, which imports this
    # module, so a module-level import would close the cycle.
    from global_classes import VariableOutput
    return VariableOutput
