import sys
from PIL import Image
im=Image.open(sys.argv[1]).convert("L"); w,h=im.size; px=im.load()
def dark_cols(y0,y1):
    return [x for x in range(w) if sum(px[x,y]<60 for y in range(y0,y1,7))>(y1-y0)/7*0.8]
def runs(v):
    out=[];s=None;p=None
    for x in v:
        if s is None: s=p=x
        elif x==p+1: p=x
        else: out.append((s,p)); s=p=x
    if s is not None: out.append((s,p))
    return out
# border lines: full-length dark columns/rows over middle region
cols=[x for x in range(w) if sum(px[x,y]<70 for y in range(0,h,5))>h/5*0.6]
rows=[y for y in range(h) if sum(px[x,y]<70 for x in range(0,w,5))>w/5*0.6]
print("cols",runs(cols)); print("rows",runs(rows))
