class UPat:
    CONST = "const"
    VAR = "var"

    def __init__(self, op=None, src=None, name=None):
        self.op = op
        self.src = src
        self.name = name

    def match(self, expr, store):
        temp_store = store.copy()
        if self._match_impl(expr, temp_store):
            store.clear()
            store.update(temp_store)
            return True
        return False

    def _match_impl(self, expr, store):
        # Determine op and children based on expression type
        if isinstance(expr, (int, float, complex)):
            expr_op = UPat.CONST
            expr_children = ()
        elif isinstance(expr, str):
            expr_op = UPat.VAR
            expr_children = ()
        elif isinstance(expr, tuple):
            expr_op = expr[0]
            expr_children = expr[1:]
        else:
            expr_op = type(expr)
            expr_children = ()

        # Match op
        if self.op is not None:
            if isinstance(self.op, (set, list, tuple)):
                if expr_op not in self.op:
                    return False
            else:
                if self.op != expr_op:
                    return False

        # Match src (children)
        if self.src is not None:
            if len(expr_children) != len(self.src):
                return False
            for p, child in zip(self.src, expr_children):
                if not p._match_impl(child, store):
                    return False

        # Match/bind name
        if self.name is not None:
            if self.name in store:
                if store[self.name] != expr:
                    return False
            else:
                store[self.name] = expr

        return True


def rewrite(expr, rules):
    memo = {}
    total_rewrites = 0

    def _rewrite(e):
        nonlocal total_rewrites
        if e in memo:
            res, delta = memo[e]
            total_rewrites += delta
            return res

        orig_e = e
        start_rewrites = total_rewrites

        # Bottom-up: rewrite children first if tuple
        if isinstance(e, tuple):
            op = e[0]
            children = e[1:]
            new_children = tuple(_rewrite(c) for c in children)
            e = (op, *new_children)

        # Apply rules until fixed point at this node
        changed = True
        while changed:
            changed = False
            for pat, fn in rules:
                store = {}
                if pat.match(e, store):
                    replacement = fn(**store)
                    if replacement is not None and replacement != e:
                        total_rewrites += 1
                        e = _rewrite(replacement)
                        changed = True
                        break

        delta = total_rewrites - start_rewrites
        memo[orig_e] = (e, delta)
        return e

    result = _rewrite(expr)
    return result, total_rewrites