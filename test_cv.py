import sys
from collections import deque

def _clone(g):
    return [list(r) for r in g]

def gmap_largest_cc(g):
    R, C = len(g), len(g[0])
    visited = set()
    components = []
    
    for r in range(R):
        for c in range(C):
            if g[r][c] != 0 and (r, c) not in visited:
                comp = []
                q = deque([(r, c)])
                color = g[r][c]
                visited.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    comp.append((cr, cc, color))
                    for dr, dc in [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1), (1,-1), (-1,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            q.append((nr, nc))
                components.append((len(comp), comp))
                
    if not components:
        return _clone(g)
        
    components.sort(key=lambda x: x[0], reverse=True)
    largest = components[0][1]
    
    out = [[0]*C for _ in range(R)]
    for r, c, color in largest:
        out[r][c] = color
    return out

g1 = [
    [0, 1, 0, 0, 0],
    [1, 1, 0, 4, 0],
    [0, 0, 0, 0, 4],
]
print(gmap_largest_cc(g1))
