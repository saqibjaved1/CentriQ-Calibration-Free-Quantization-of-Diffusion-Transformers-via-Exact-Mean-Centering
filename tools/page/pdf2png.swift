import PDFKit; import AppKit
let a=CommandLine.arguments; let d=PDFDocument(url: URL(fileURLWithPath:a[1]))!; let p=d.page(at:0)!
let s=Double(a[3])!; let r=p.bounds(for:.mediaBox)
let W=Int(r.width*s), H=Int(r.height*s)
let rep=NSBitmapImageRep(bitmapDataPlanes:nil,pixelsWide:W,pixelsHigh:H,bitsPerSample:8,samplesPerPixel:4,hasAlpha:true,isPlanar:false,colorSpaceName:.deviceRGB,bytesPerRow:0,bitsPerPixel:0)!
NSGraphicsContext.saveGraphicsState(); let ctx=NSGraphicsContext(bitmapImageRep:rep)!; NSGraphicsContext.current=ctx
let cg=ctx.cgContext; cg.setFillColor(NSColor.white.cgColor); cg.fill(CGRect(x:0,y:0,width:W,height:H)); cg.scaleBy(x:s,y:s); cg.translateBy(x:-r.minX,y:-r.minY)
p.draw(with:.mediaBox,to:cg); NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using:.png,properties:[:])!.write(to:URL(fileURLWithPath:a[2]))
print(W,H)
