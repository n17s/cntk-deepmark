import json

spliceDepth="""
SpliceDepth (_, tag='') = {
     ArrayTransposeDimensions (_, axis1, axis2) = {
         inputsT[i:0..Length(_)-1] = TransposeDimensions (_[i], axis1, axis2)
     }.inputsT
     out = [tag1=tag; out=TransposeDimensions (RowStack (ArrayTransposeDimensions (_, 1, 3)), 1, 3, tag=tag)].out
}.out
"""

def dumplayer(layer,spaces):
    out = []
    white = ' '*spaces
    if layer['type']=='inception' or layer['type']=='tower':
        name = 'Parallel' if layer['type']=='inception' else 'Sequential'
        oper = ', SpliceDepth' if layer['type']=='inception' else ''
        out.append(white+name+'(\n')
        spaces+=4
        kids = []
        for k in sorted(layer):
            if isinstance(layer[k],dict):
                kids.append(dumplayer(layer[k],spaces))
        out.append(' :\n'.join(kids))
        out.append('\n'+white+oper+')')
    elif layer['type']=='maxpool2d' or layer['type']=='avgpool2d':
        name = 'MaxPoolingLayer' if layer['type']=='maxpool2d' else 'AveragePoolingLayer'
        p = "true " if layer['kind']=="same" else "false"
        out.append(white+'%s{(%d:%d), stride = (%d:%d), pad=%s}'%(name,layer['kW'],layer['kH'],layer['strideW'],layer['strideH'],p))
    elif layer['type']=='conv2d':
        pad = "true " if layer['padW']> 0 or layer['padH']>0 else "false"
        out.append(white+'ConvolutionalLayer{%d, (%d:%d), stride=(%d:%d), pad=%s, lowerPad=(%d:%d), bias=false} :\n'%(layer['oC'],layer['kW'],layer['kH'],layer['strideW'],layer['strideH'],pad,layer['padW'],layer['padH']) +
                white+'BatchNormalizationLayer{spatialRank = 2} :\n' +
                white+'ReLU')
    else:
        raise(Exception('Unknown layer type'))
    return ''.join(out)

with open('inceptionv3cntk.json') as f:
    s=f.read()
model =' :\n'.join([dumplayer(layer,4) for layer in json.loads(s)])
bscontents = """
%s
inceptionv3 = Sequential(
%s
)
"""%(spliceDepth,model)
print(bscontents)
