class Value:
    def __init__(self, data, _children=()):
        self.data = float(data)
        self.grad = 0.0
        # Internal variable storing the backward function for this node
        self._backward = lambda: None
        # Set of child nodes that produced this node
        self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            # Sum rule: gradient flows equally to both addends
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            # Product rule: local gradient of self is other.data, and vice versa
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, (self,))

        def _backward():
            # Local derivative of ReLU is 1 if input > 0, else 0
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        # 1. Topological sort to order nodes from input to output
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # 2. Base case: set output node gradient to 1
        self.grad = 1.0

        # 3. Process nodes in reverse topological order (backpropagation)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        # Formatted to match expected integer outputs when clean
        data_str = int(self.data) if self.data.is_integer() else self.data
        grad_str = int(self.grad) if self.grad.is_integer() else self.grad
        return f"Value(data={data_str}, grad={grad_str})"