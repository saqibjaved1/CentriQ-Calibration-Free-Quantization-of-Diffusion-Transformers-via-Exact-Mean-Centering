from PIL import Image
X=[(4,476),(524,996),(1044,1516)]; Y=[(124,596),(644,1116),(1164,1636),(1684,2156)]
for m in ["flux","sana","pixart"]:
    im=Image.open(f"src/figures/figures-qualitative_w4a4_{m}.jpg").convert("RGB")
    im.crop((0,0,1520,118)).save(f"tools/hdr_{m}.png")
    for r,(y0,y1) in enumerate(Y):
        for c,(x0,x1) in enumerate(X):
            im.crop((x0,y0,x1,y1)).save(f"assets/img/cmp/{m}_r{r}_c{c}.jpg",quality=88)
