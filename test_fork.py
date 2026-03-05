import multiprocessing as mp

def make_closure(x):
    def _fn(y):
        return x + y
    return _fn

def run():
    c = make_closure(10)
    
    # helper must be picklable, but what about the closure it calls?
    # if it's top-level, we can just pass the closure. Wait, passing the closure to map pickles it!
    # P.map picks the function AND the arguments.
    # We shouldn't map over the closure. We should set the closure in globals before forking.
    pass

# We can set global state before forking
closure_fn = None
def _worker(y):
    return closure_fn(y)

def test():
    global closure_fn
    closure_fn = make_closure(10)
    
    ctx = mp.get_context('fork')
    with ctx.Pool(2) as p:
        res = p.map(_worker, [1, 2, 3])
    print("res:", res)

if __name__ == '__main__':
    test()
